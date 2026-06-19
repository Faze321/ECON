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
    When baseline_rounds > 1, later rounds feed the previous executor answers
    and coordinator commitment back to the executors for another plain LLM pass.
    """

    def __init__(self, scheme: Dict, groups: Dict, args: Any, logger):
        self.scheme = scheme
        self.groups = groups
        self.args = args
        self.logger = logger
        self.n_agents = int(getattr(args, "n_agents", 3))
        self.n_actions = int(getattr(args, "n_actions", 2))
        self._last_commitment_metadata = None

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

        discussion = self.run_multi_round_discussion(question, strategy, self._get_discussion_rounds())
        executor_outputs = discussion["outputs_final"]
        commitment = discussion["commitment_final"]
        chosen_actions = torch.zeros((1, self.n_agents), dtype=torch.long, device=self.device)

        return chosen_actions, {
            "executor_responses": executor_outputs,
            "commitment": commitment,
            "commitment_text": commitment,
            "commitment_embedding": None,
            "strategy": strategy,
            "format": "",
            "selected_actions": chosen_actions.detach().clone(),
            "commitment_metadata": self._last_commitment_metadata,
            "baseline_rounds": discussion["n_rounds"],
            "baseline_discussion_history": discussion["history"],
        }

    def _get_discussion_rounds(self) -> int:
        configured = self._get_opt("baseline_rounds", None)
        if configured is None:
            configured = self._get_opt("discussion_rounds", 1)
        try:
            return max(1, int(configured))
        except (TypeError, ValueError):
            self.logger.warning(f"[BaselineMAC] Invalid baseline_rounds={configured!r}; using 1")
            return 1

    def run_multi_round_discussion(
        self,
        question: str,
        strategy: str,
        n_rounds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run plain no-BNE executor/coordinator discussion for one or more rounds."""
        rounds = max(1, int(n_rounds if n_rounds is not None else self._get_discussion_rounds()))
        history: List[Dict[str, Any]] = []
        feedback = ""
        executor_outputs: List[str] = []
        commitment = ""

        for round_idx in range(rounds):
            executor_outputs = []
            for agent_idx, agent in enumerate(self.agents):
                if round_idx == 0:
                    prompt = self._build_agent_prompt(question, strategy, agent_idx)
                else:
                    prompt = self._build_agent_revision_prompt(
                        question=question,
                        strategy=strategy,
                        agent_idx=agent_idx,
                        previous_rounds=history,
                        coordinator_feedback=feedback,
                    )
                executor_outputs.append(self._generate_executor_response(agent, prompt))

            commitment = self._generate_commitment(question, strategy, executor_outputs)
            metadata = dict(self._last_commitment_metadata or {})
            round_record = {
                "round": round_idx + 1,
                "executor_outputs": executor_outputs,
                "commitment": commitment,
                "commitment_metadata": metadata,
            }
            history.append(round_record)
            feedback = self._build_coordinator_feedback(commitment, metadata)

        return {
            "outputs_final": executor_outputs,
            "commitment_final": commitment,
            "history": history,
            "n_rounds": len(history),
        }

    def _generate_executor_response(self, agent: ImprovedLLMWrapper, prompt: str) -> str:
        text = agent.generate_response(
            prompt=prompt,
            temperature=float(self._get_opt("executor_temperature", 0.2)),
            top_p=float(self._get_opt("executor_top_p", 0.9)),
            repetition_penalty=float(self._get_opt("executor_repetition_penalty", 1.05)),
            max_tokens=int(self._get_opt("executor_max_tokens", 1024)),
        )
        return self._ensure_boxed_format(self._post_sanitize_text(text))

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
            temperature=float(self._get_opt("strategy_temperature", 0.3)),
            top_p=float(self._get_opt("strategy_top_p", 0.4)),
            repetition_penalty=float(self._get_opt("strategy_repetition_penalty", 1.1)),
            max_tokens=int(self._get_opt("strategy_max_tokens", 180)),
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

    def _build_agent_revision_prompt(
        self,
        question: str,
        strategy: str,
        agent_idx: int,
        previous_rounds: List[Dict[str, Any]],
        coordinator_feedback: str,
    ) -> str:
        history = self._format_discussion_history(previous_rounds)
        return f"""You are Executor {agent_idx + 1} in a no-BNE multi-round discussion. Revise your solution using the shared history and the Coordinator's latest review.

Problem:
{question}

High-Level Strategy:
{strategy}

Previous Discussion:
{history}

Coordinator Review:
{coordinator_feedback}

Your Task:
1. Re-check the problem from scratch, not just your previous answer.
2. Compare your reasoning against the other executors and the Coordinator review.
3. Correct any arithmetic, interpretation, or formatting mistakes.
4. The final line MUST be the answer enclosed in `\\boxed{{...}}`. Do not add any text after it.

Begin your revised solution now.
"""

    def _format_discussion_history(self, previous_rounds: List[Dict[str, Any]]) -> str:
        if not previous_rounds:
            return "No previous rounds."

        blocks = []
        max_chars = int(self._get_opt("baseline_history_max_chars", 6000))
        per_output_chars = max(400, max_chars // max(1, len(previous_rounds) * max(1, self.n_agents)))

        for record in previous_rounds:
            lines = [
                f"Round {record.get('round', '?')} Coordinator commitment: {record.get('commitment', '')}"
            ]
            for idx, output in enumerate(record.get("executor_outputs", [])):
                lines.append(f"Executor {idx + 1}: {self._truncate_text(str(output), per_output_chars)}")
            blocks.append("\n".join(lines))

        return self._truncate_text("\n\n".join(blocks), max_chars)

    def _build_coordinator_feedback(self, commitment: str, metadata: Dict[str, Any]) -> str:
        lines = [f"Current commitment: {commitment}"]
        reasoning = metadata.get("reasoning")
        if reasoning:
            lines.append(f"Coordinator reasoning: {reasoning}")
        confidence = metadata.get("confidence")
        if confidence is not None:
            lines.append(f"Coordinator confidence: {confidence}")
        checklist = metadata.get("checklist")
        if isinstance(checklist, dict) and checklist:
            checklist_text = ", ".join(f"{key}={value}" for key, value in checklist.items())
            lines.append(f"Coordinator checklist: {checklist_text}")
        lines.append("Use this review to verify or revise the next answer.")
        return "\n".join(lines)

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 20)].rstrip() + "\n...[truncated]"

    def _generate_commitment(self, question: str, strategy: str, responses: List[str]) -> str:
        formatted = "\n".join([f"Executor {idx + 1}: {text}" for idx, text in enumerate(responses)])
        prompt = f"""You are the COORDINATOR. Review the question, strategy and all executor solutions to aggregate and produce a structured final answer.

Problem:
{question}

Strategy:
{strategy}

Executor Solutions (review each carefully):
{formatted}

Your Task:
1. Extract the final answer expression (numbers, fractions, radicals, units, or complex forms) from each executor's \\boxed{{}} output
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
            temperature=float(self._get_opt("commitment_temperature", 0.1)),
            top_p=float(self._get_opt("commitment_top_p", 0.3)),
            repetition_penalty=float(self._get_opt("commitment_repetition_penalty", 1.05)),
            max_tokens=int(self._get_opt("commitment_max_tokens", 150)),
        )
        final_answer, metadata = self._parse_structured_commitment(self._post_sanitize_text(out))
        self._last_commitment_metadata = metadata
        return f"\\boxed{{{final_answer}}}"

    def _parse_structured_commitment(self, raw_output: str) -> Tuple[str, Dict[str, Any]]:
        import json

        raw_output = str(raw_output or "")
        metadata = {
            "parse_method": "fallback",
            "reasoning": "",
            "confidence": 0.0,
            "checklist": {},
            "raw_output": raw_output[:200],
        }

        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join([line for line in lines if not line.strip().startswith("```")])

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                final_candidate = data.get("final")
                if final_candidate is None:
                    final_candidate = data.get("final_value")
                if final_candidate is not None:
                    final_raw = str(final_candidate).strip()
                    final_norm = extract_answer_number(final_raw)
                    final_out = final_norm if final_norm is not None else final_raw

                    metadata["parse_method"] = "json"
                    metadata["reasoning"] = data.get("reasoning", "")
                    metadata["confidence"] = float(data.get("confidence", 0.5))
                    metadata["checklist"] = data.get("checklist", {})
                    return final_out, metadata
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            metadata["parse_error"] = str(exc)[:100]

        boxed_content = self._extract_boxed_content(raw_output)
        if boxed_content:
            final_norm = extract_answer_number(boxed_content)
            metadata["parse_method"] = "boxed"
            return final_norm if final_norm is not None else boxed_content, metadata

        final_value_match = re.search(r'"final(?:_value)?"\s*:\s*"([^"]+)"', raw_output)
        if final_value_match:
            extracted = final_value_match.group(1).strip()
            final_norm = extract_answer_number(extracted)
            metadata["parse_method"] = "json_field_regex"
            return final_norm if final_norm is not None else extracted, metadata

        nums = re.findall(r"[+-]?\d+(?:\.\d+)?", raw_output)
        if nums:
            num = extract_answer_number(nums[-1])
            if num is not None:
                metadata["parse_method"] = "regex"
                return num, metadata

        metadata["parse_method"] = "undetermined_fallback"
        return "undetermined", metadata

    def _post_sanitize_text(self, text: str) -> str:
        if text is None:
            return ""
        text = str(text).replace("\x08", "\\b").replace("\x0c", "\\f")
        return self._repair_boxed(text.replace("\r\n", "\n").replace("\r", "\n"))

    def _ensure_boxed_format(self, text: str) -> str:
        text = self._repair_boxed(str(text or "")).strip()
        if "\\boxed{" in text:
            return text
        candidate = extract_answer_number(text)
        if candidate is None:
            candidate = "undetermined"
        if text:
            return f"{text}\n\\boxed{{{candidate}}}"
        return f"\\boxed{{{candidate}}}"

    def _repair_boxed(self, text: str) -> str:
        text = str(text or "")
        text = text.replace("\x08oxed{", "\\boxed{")
        text = re.sub(r"(?<!\\)boxed\{", r"\\boxed{", text)
        return text

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
