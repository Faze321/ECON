# src/controllers/baseline_mac.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from modules.llm.llm_wrapper import ImprovedLLMWrapper
from utils.answer_extraction import _normalize_number as extract_answer_number


class BaselineMAC:
    """
    Inference-only 1-coordinator + N-agent controller.

    This controller keeps the project runner/env structure, but avoids BNE
    refinement networks, mixer optimization, learned belief updates, and action
    optimization. Each episode is plain LLM orchestration:
      coordinator strategy -> N executor answers -> coordinator commitment.
    """

    def __init__(self, scheme: Dict, groups: Dict, args: Any, logger):
        self.scheme = scheme
        self.groups = groups
        self.args = args
        self.logger = logger
        self.n_agents = int(getattr(args, "n_agents", 3))
        self.n_actions = int(getattr(args, "n_actions", 2))

        use_cuda = getattr(args.system, "use_cuda", False) and torch.cuda.is_available()
        device_num = getattr(args.system, "device_num", 0)
        self.device = torch.device(f"cuda:{device_num}" if use_cuda else "cpu")

        model_name = getattr(args, "llm_model_name", "gpt2")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.logger.info(f"[BaselineMAC] Loaded tokenizer for {model_name}")
        except Exception as exc:
            self.logger.warning(f"[BaselineMAC] Load tokenizer failed: {exc}; using minimal tokenizer")
            self.tokenizer = self._create_minimal_tokenizer()
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        api_key = self._get_opt("llm_api_key", "")
        base_url = self._get_opt("base_url", "https://openrouter.ai/api/v1")
        coordinator_model = self._get_opt("coordinator_model", "meta-llama/llama-3.3-70b-instruct")
        executor_model = self._get_opt("executor_model", "meta-llama/llama-3.1-8b-instruct")
        timeout_s = int(self._get_opt("llm_timeout_s", 60))
        max_retries = int(self._get_opt("llm_max_retries", 3))
        debug = bool(getattr(args, "debug", getattr(getattr(args, "system", object()), "debug", False)))

        self.coordinator = ImprovedLLMWrapper(
            api_key=api_key,
            model_name=coordinator_model,
            base_url=base_url,
            timeout_s=timeout_s,
            max_retries=max_retries,
            debug=debug,
        )
        self.agents = [
            ImprovedLLMWrapper(
                api_key=api_key,
                model_name=executor_model,
                base_url=base_url,
                timeout_s=timeout_s,
                max_retries=max_retries,
                debug=debug,
            )
            for _ in range(self.n_agents)
        ]

    def _get_opt(self, key: str, default=None):
        if hasattr(self.args, key) and getattr(self.args, key) is not None:
            return getattr(self.args, key)
        if hasattr(self.args, "llm") and hasattr(self.args.llm, key) and getattr(self.args.llm, key) is not None:
            return getattr(self.args.llm, key)
        return default

    def reset_token_usage(self):
        for wrapper in [self.coordinator] + list(self.agents):
            if hasattr(wrapper, "reset_usage"):
                wrapper.reset_usage()

    def get_token_usage(self) -> Dict[str, Dict[str, int]]:
        agents = self._empty_role_usage()
        for wrapper in self.agents:
            usage = wrapper.get_usage_summary() if hasattr(wrapper, "get_usage_summary") else {}
            agents["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            agents["completion_tokens"] += int(usage.get("completion_tokens", 0))
            agents["total_tokens"] += int(usage.get("total_tokens", 0))
            agents["requests"] += int(usage.get("requests", 0))

        coordinator = self._empty_role_usage()
        coord_usage = self.coordinator.get_usage_summary() if hasattr(self.coordinator, "get_usage_summary") else {}
        coordinator["prompt_tokens"] = int(coord_usage.get("prompt_tokens", 0))
        coordinator["completion_tokens"] = int(coord_usage.get("completion_tokens", 0))
        coordinator["total_tokens"] = int(coord_usage.get("total_tokens", 0))
        coordinator["requests"] = int(coord_usage.get("requests", 0))

        total = {
            "requests": agents["requests"] + coordinator["requests"],
            "prompt_tokens": agents["prompt_tokens"] + coordinator["prompt_tokens"],
            "completion_tokens": agents["completion_tokens"] + coordinator["completion_tokens"],
            "total_tokens": agents["total_tokens"] + coordinator["total_tokens"],
        }
        return {
            "agents": agents,
            "coordinator": coordinator,
            "total": total,
        }

    def _empty_role_usage(self) -> Dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }

    def preprocess_observation(self, observation_text: str, max_length: Optional[int] = None) -> torch.Tensor:
        if max_length is None:
            max_length = getattr(self.args.env_args, "max_question_length", 1024)
        enc = self.tokenizer(
            observation_text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=False,
        )
        return enc.input_ids.squeeze(0).to(self.device)

    def select_actions(
        self,
        ep_batch: Any,
        t_ep: int,
        t_env: int,
        raw_observation_text: Optional[str] = None,
        bs: slice = slice(None),
        test_mode: bool = False,
        agent_memory: Optional[torch.Tensor] = None,
        strategy_override: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        question = raw_observation_text or ""
        strategy = strategy_override if strategy_override is not None else self._get_strategy_and_format(question)
        if not strategy or not strategy.strip():
            strategy = "Solve the problem step by step and give the final answer in \\boxed{}."

        executor_outputs = []
        for idx, agent in enumerate(self.agents):
            prompt = self._build_agent_prompt(question, strategy, idx)
            text = agent.generate_response(
                prompt=prompt,
                temperature=float(self._get_opt("executor_temperature", 0.2)),
                top_p=float(self._get_opt("executor_top_p", 0.9)),
                max_tokens=int(self._get_opt("executor_max_tokens", 1024)),
            )
            executor_outputs.append(self._ensure_boxed_format(self._post_sanitize_text(text)))

        commitment = self._generate_commitment(question, strategy, executor_outputs)
        chosen_actions = torch.zeros((1, self.n_agents), dtype=torch.long, device=self.device)

        return chosen_actions, {
            "executor_responses": executor_outputs,
            "commitment": commitment,
            "commitment_text": commitment,
            "commitment_embedding": None,
            "strategy": strategy,
            "format": "",
            "selected_actions": chosen_actions.detach().clone(),
        }

    def _get_strategy_and_format(self, question: str) -> str:
        prompt = f"""You are the Coordinator. Provide a clear, step-by-step STRATEGY for solving this math problem.

Problem:
{question}

Your Response Format:

STRATEGY:
1. [First conceptual step ]
2. [Second conceptual step ]
3. [Final calculation approach ]

EXECUTION RULES:
- Show your reasoning for each step
- End with exactly: \\boxed{{<final_number>}}
- The number in \\boxed{{}} must be the complete final answer

Keep your strategy clear and under 80 tokens.
"""
        out = self.coordinator.generate_response(
            prompt=prompt,
            temperature=float(self._get_opt("coordinator_temperature", 0.2)),
            top_p=float(self._get_opt("coordinator_top_p", 0.9)),
            max_tokens=int(self._get_opt("strategy_max_tokens", 256)),
        )
        return self._post_sanitize_text(out)

    def _build_agent_prompt(self, question: str, strategy: str, agent_idx: int) -> str:
        return f"""You are a specialist Executor agent within a collaborative team. Your work will be critically reviewed by a Coordinator to determine the final answer. Therefore, absolute clarity and accuracy are paramount.

Problem:
{question}

High-Level Strategy to Follow:
{strategy}

Your Task:
1.  **Adhere strictly to the Strategy**: Address each point in the strategy in order.
2.  **Show Your Work**: For each step, explicitly state the numbers you are using and show the calculation (e.g., "Step 2: Calculate the total cost. 5 items * $3.50/item = $17.50").
3.  **Self-Correction**: Before concluding, briefly double-check your arithmetic.
4.  **Final Answer Format**: The final line of your entire response MUST be the answer enclosed in `\\boxed{{...}}`. Do not add any text after it.

Begin your detailed solution now.
"""

    def _generate_commitment(self, question: str, strategy: str, responses: List[str]) -> str:
        formatted = "\n".join([f"Agent {idx + 1}: {text}" for idx, text in enumerate(responses)])
        prompt = f"""You are the COORDINATOR. Review the question, strategy and all executor solutions to aggregate and produce a structured final answer.

Problem:
{question}

Strategy:
{strategy}

Executor Solutions (review each carefully):
{formatted}

Your Task:
1. Extract the final answer expression (numbers, fractions, radicals, units, or complex forms) from each executor's \boxed{{}} output
2. Compare all answers - if they agree, use that answer
3. If they disagree, analyze the reasoning to identify the mathematically correct answer
4. Verify the arithmetic step-by-step for the chosen answer (re-derive if needed)
5. If information is insufficient, return "undetermined" and explain briefly
6. Output a JSON object with verification checklist

Output Format (JSON only, no other text):
{{
  "final_value": "<answer expression or undetermined>",
  "reasoning": "<1-sentence explanation>",
  "confidence": <0.0-1.0>,
  "checklist": {{
    "all_agree": <true/false>,
    "arithmetic_verified": <true/false>,
    "units_correct": <true/false>
  }}
}}

Critical Requirements:
- Output MUST be valid JSON (no markdown code blocks)
- "final_value" must exactly match the chosen answer (fractions, radicals, complex numbers, or units allowed)
- If the answer is undetermined, set "final_value" to "undetermined" and "confidence" <= 0.2
- Re-check the problem statement instead of guessing when executor work is incomplete
- "confidence" should reflect agreement level (1.0 if all agree, lower if conflict or uncertainty)
- Keep reasoning concise (max 20 words)
"""
        out = self.coordinator.generate_response(
            prompt=prompt,
            temperature=float(self._get_opt("coordinator_temperature", 0.1)),
            top_p=float(self._get_opt("coordinator_top_p", 0.9)),
            max_tokens=int(self._get_opt("commitment_max_tokens", 256)),
        )
        return self._ensure_boxed_format(self._post_sanitize_text(out))

    def _post_sanitize_text(self, text: str) -> str:
        if text is None:
            return ""
        text = str(text).replace("\x08", "\\b").replace("\x0c", "\\f")
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _ensure_boxed_format(self, text: str) -> str:
        text = str(text or "").strip()
        boxed = self._extract_boxed_content(text)
        if boxed:
            return f"\\boxed{{{boxed}}}"
        candidate = extract_answer_number(text)
        if candidate is None:
            nums = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
            candidate = nums[-1] if nums else "undetermined"
        return f"\\boxed{{{candidate}}}"

    def _extract_boxed_content(self, text: str) -> Optional[str]:
        if not isinstance(text, str):
            return None
        match = re.search(r"\\boxed\{([\s\S]*?)\}", text)
        return match.group(1).strip() if match else None

    def _create_minimal_tokenizer(self):
        class MinimalTokenizer:
            def __init__(self):
                self.vocab = {chr(i): i for i in range(32, 127)}
                self.vocab.update({"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3})
                self.pad_token = "[PAD]"
                self.eos_token = "[EOS]"
                self.pad_token_id = self.vocab[self.pad_token]
                self.eos_token_id = self.vocab[self.eos_token]
                self.vocab_size = len(self.vocab)

            def __call__(self, text, max_length=None, padding=True, truncation=True, return_tensors="pt", **kwargs):
                if isinstance(text, str):
                    text = [text]
                rows = []
                for item in text:
                    limit = max_length - 1 if max_length else None
                    tokens = [self.vocab.get(ch, 1) for ch in item[:limit]]
                    tokens.append(self.eos_token_id)
                    if max_length and padding and len(tokens) < max_length:
                        tokens += [self.pad_token_id] * (max_length - len(tokens))
                    rows.append(tokens[:max_length] if max_length else tokens)
                return type("Enc", (), {"input_ids": torch.tensor(rows)})

        return MinimalTokenizer()


BasicBaselineMAC = BaselineMAC
