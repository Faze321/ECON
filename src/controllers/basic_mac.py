# src/controllers/basic_mac.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from modules.agents.transformer_agent import LLMTransformerAgent
from modules.agents.belief_policy_network import BeliefPolicyNetwork
from modules.agents.refine_module import RefineModule
from modules.text_encoders.output_encoder import OutputEncoder
from modules.llm.llm_wrapper import ImprovedLLMWrapper, LLMConfig
from modules.llm.commitment_embedder import CommitmentEmbedder
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.belief_encoder import BeliefEncoder
from utils.answer_extraction import _normalize_number as extract_answer_number


class LLMBasicMAC:
    """
    Multi-Agent Controller for ECON.
    - Agent: belief + local Q + prompt embeddings (T_i, p_i)
    - Coordinator LLM:
        1) Generate STRATEGY & FORMAT (no numbers)
        2) Aggregate executors to produce COMMITMENT (only \\boxed{<number>})
    """
    def __init__(self, scheme: Dict, groups: Dict, args: Any, logger):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.hidden_dim = getattr(args.arch, "entity_dim", 256)
        self.memory_dim = int(getattr(args, "memory_dim", getattr(args, "belief_dim", 128) + 5))
        use_cuda = getattr(args.system, "use_cuda", False) and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.logger = logger

        model_name = getattr(args, "llm_model_name", "gpt2")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.logger.info(f"[MAC] Loaded tokenizer for {model_name}")
        except Exception as e:
            self.logger.warning(f"[MAC] Load tokenizer failed: {e}; using fallback minimal tokenizer")
            self.tokenizer = self._create_minimal_tokenizer()
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bne_cfg = getattr(args, "bne", object())
        self.bne_commitment_match_threshold = float(getattr(bne_cfg, "commitment_match_threshold", 0.995))
        self.bne_commitment_osc_threshold = float(getattr(bne_cfg, "commitment_osc_threshold", 0.98))
        self.bne_detect_oscillation = bool(getattr(bne_cfg, "detect_oscillation", True))
        self.bne_min_oscillation_history = int(getattr(bne_cfg, "oscillation_min_history", 5))

        max_token_len = getattr(args.env_args, "max_question_length", 512)
        self.max_question_length = int(max_token_len)

        # BNE: K individual Belief Policy Networks
        self.bne_enabled = getattr(getattr(args, "bne", object()), "enabled", False)
        if self.bne_enabled:
            self.logger.info(f"[MAC] BNE enabled: creating {self.n_agents} independent policy networks")
            entity_dim = getattr(args.arch, "entity_dim", 256)
            self.policy_networks = nn.ModuleList([
                BeliefPolicyNetwork(
                    observation_dim=max_token_len,
                    belief_dim=args.belief_dim,
                    hidden_dim=entity_dim,
                    n_heads=getattr(args.arch, "attention_heads", 4),
                    n_layers=getattr(args.arch, "transformer_blocks", 2),
                    dropout=getattr(args.arch, "dropout_rate", 0.1),
                    memory_dim=self.memory_dim,
                    T_min=getattr(args.sampling, "temperature_min", 0.1),
                    T_max=getattr(args.sampling, "temperature_max", 2.0),
                    p_min=getattr(args.sampling, "p_min", 0.1),
                    p_max=getattr(args.sampling, "p_max", 0.9),
                    vocab_size=self.tokenizer.vocab_size,
                    device=self.device
                ).to(self.device)
                for _ in range(self.n_agents)
            ])

            # BNE Refine Module
            commitment_dim = getattr(args, "commitment_embedding_dim", 768)
            self.commitment_dim = int(commitment_dim)
            self.refine_module = RefineModule(
                belief_dim=args.belief_dim,
                commitment_dim=self.commitment_dim,
                n_agents=self.n_agents,
                hidden_dim=256,
                max_delta=0.3
            ).to(self.device)

            self.logger.info(f"[MAC] BNE components initialized:")
            self.logger.info(f"  - Policy networks: {sum(sum(p.numel() for p in net.parameters()) for net in self.policy_networks)} params")
            self.logger.info(f"  - Refine module: {sum(p.numel() for p in self.refine_module.parameters())} params")

            # Still need LLM agent for text generation (but not its belief network)
            self.agent = LLMTransformerAgent(input_shape=max_token_len, args=self.args)
        else:
            self.logger.info("[MAC] BNE disabled: using legacy single agent")
            self._build_agents(max_token_len)
            self.commitment_dim = int(getattr(args, "commitment_embedding_dim", 768))

        # Output Encoder (shared)
        self.output_encoder = OutputEncoder(
            embedding_dim=self.commitment_dim,
            model_name=getattr(args, "commitment_embedding_model_name", "BAAI/bge-large-en-v1.5"),
            device=self.device
        )

        self.action_selector = action_REGISTRY[getattr(args, "action_selector", "multinomial")](args)

        self.coordinator = ImprovedLLMWrapper(
            api_key=args.llm_api_key,
            model_name=args.coordinator_model,
            belief_dim=args.belief_dim,
            debug=getattr(args, "debug", False)
        )

        self.belief_encoder = BeliefEncoder(
            belief_dim=args.belief_dim,
            n_agents=args.n_agents,
            n_heads=getattr(args.arch, "attention_heads", 4),
            key_dim=getattr(args.arch, "key_dim", 64),
            device=self.device
        ).to(self.device)
        self.commitment_embedder = CommitmentEmbedder(args, LLMConfig(
            api_key=args.llm_api_key,
            model_name=args.coordinator_model,
            debug=getattr(args, "debug", False)
        ))

        self.setup_attention_masks()
        self.last_commitment_embedding = None
        self.last_commitment_text = None


        self.eval_correct_count = 0
        self.eval_total_count = 0

    def reset_token_usage(self):
        for wrapper in self._usage_wrappers().values():
            if hasattr(wrapper, "reset_usage"):
                wrapper.reset_usage()

    def get_token_usage(self) -> Dict[str, Dict[str, int]]:
        usage = {}
        for name, wrapper in self._usage_wrappers().items():
            if hasattr(wrapper, "get_usage_summary"):
                usage[name] = wrapper.get_usage_summary()
            else:
                usage[name] = self._empty_token_usage()

        total = self._empty_token_usage()
        for item in usage.values():
            for key in total:
                total[key] += int(item.get(key, 0))
        usage["total"] = total
        return usage

    def _usage_wrappers(self) -> Dict[str, Any]:
        wrappers = {}
        if hasattr(self, "agent") and hasattr(self.agent, "llm_wrapper"):
            wrappers["agents"] = self.agent.llm_wrapper
        if hasattr(self, "coordinator"):
            wrappers["coordinator"] = self.coordinator
        return wrappers

    def _empty_token_usage(self) -> Dict[str, int]:
        return {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _build_agents(self, max_token_len: int):
        self.agent = LLMTransformerAgent(input_shape=max_token_len, args=self.args)

    def setup_attention_masks(self):
        max_seq = getattr(self.args, "max_seq_length", 512)
        self.base_attention_mask = torch.zeros((1, max_seq), dtype=torch.bool, device=self.device)
        if getattr(self.args, "use_causal_mask", False):
            m = torch.triu(torch.ones(max_seq, max_seq), 1)
            self.causal_mask = m.bool().to(self.device)

    def preprocess_observation(self, observation_text: str, max_length: Optional[int] = None) -> torch.Tensor:
        if max_length is None:
            max_length = getattr(self.args.env_args, "max_question_length", 512)
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

    def _build_inputs(self, batch: Any, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = batch.batch_size

        obs_tokens = batch["obs"][:, t].detach().clone().contiguous()  # (B, N, L)
        inputs = obs_tokens.reshape(bs * self.n_agents, -1)            # (B*N, L)
        pad_id = getattr(self.tokenizer, "pad_token_id", getattr(self.tokenizer, "eos_token_id", 50256))
        key_padding_mask = (inputs == pad_id)                           # (B*N, L) bool
        memory = None
        try:
            if "agent_memory" in batch.data:
                mem = batch["agent_memory"][:, t]  # (B, N, M)
                memory = mem.reshape(bs * self.n_agents, -1)
        except Exception:
            memory = None
        return inputs, key_padding_mask, memory

    def init_hidden(self, batch_size: int):
        # GRU hidden placeholder for compatibility
        return torch.zeros((batch_size, self.n_agents, self.hidden_dim), device=self.device)


    @torch.no_grad()
    def encode_step_nograd(self, ep_batch: Any, t: int) -> Dict[str, torch.Tensor]:

        inputs, key_padding_mask, memory = self._build_inputs(ep_batch, t)
        outputs, h_n = self.agent(inputs, key_padding_mask, test_mode=True, memory=memory)
        B = ep_batch.batch_size
        N = self.n_agents
        belief_states = outputs["belief_state"].view(B, N, -1)
        prompt_embeddings = outputs["prompt_embedding"].view(B, N, -1)
        q_local = outputs["q_value"].view(B, N, 1)
        group_representation = self.belief_encoder(belief_states)  # (B, d)
        return {
            "q_local": q_local,                          # (B, N, 1)
            "belief_states": belief_states,              # (B, N, d)
            "prompt_embeddings": prompt_embeddings,      # (B, N, 2)
            "group_repr": group_representation,          # (B, d)
            "hidden_states": h_n.view(B, N, -1),
        }

    def encode_step(self, ep_batch: Any, t: int) -> Dict[str, torch.Tensor]:
        inputs, key_padding_mask, memory = self._build_inputs(ep_batch, t)
        outputs, h_n = self.agent(inputs, key_padding_mask, test_mode=True, memory=memory)
        B = ep_batch.batch_size
        N = self.n_agents
        belief_states = outputs["belief_state"].view(B, N, -1)
        prompt_embeddings = outputs["prompt_embedding"].view(B, N, -1)
        q_local = outputs["q_value"].view(B, N, 1)
        group_representation = self.belief_encoder(belief_states)  # (B, d)
        return {
            "q_local": q_local,                          # (B, N, 1)
            "belief_states": belief_states,              # (B, N, d)
            "prompt_embeddings": prompt_embeddings,      # (B, N, 2)
            "group_repr": group_representation,          # (B, d)
            "hidden_states": h_n.view(B, N, -1),
        }

    def forward(self, ep_batch: Any, t: int, test_mode: bool = False, train_mode: bool = False,
                agent_memory: Optional[torch.Tensor] = None):
        """
          action_q_values: (B, N, A)
          info: {
             belief_states: (B, N, d),
             prompt_embeddings: (B, N, 2),
             q_values: (B, N, 1),
             group_repr: (B, d)
          }
        """
        actual_test = test_mode if not train_mode else False
        B = ep_batch.batch_size
        N = self.n_agents

        inputs, key_padding_mask, memory = self._build_inputs(ep_batch, t)     # (B*N, L)
        if agent_memory is not None:
            memory = agent_memory.view(B * N, -1)
        outputs, hidden = self.agent(inputs, key_padding_mask, test_mode=actual_test, memory=memory)

        b = outputs["belief_state"].view(B, N, -1)                     # (B,N,d)
        e = outputs["prompt_embedding"].view(B, N, -1)                 # (B,N,2)
        q = outputs["q_value"].view(B, N, -1)                          # (B,N,1)
        a_logits = outputs["action_q_values"].view(B, N, -1)           # (B,N,A)

        group_representation = self.belief_encoder(b)                  # (B,d)

        info = {
            "belief_states": b,
            "prompt_embeddings": e,
            "q_values": q,
            "action_q_values": a_logits,
            "group_repr": group_representation,
            "hidden_states": hidden.view(B, N, -1) if hidden is not None else None
        }
        return a_logits, info

    def select_actions(self, ep_batch: Any, t_ep: int, t_env: int,
                       raw_observation_text: Optional[str] = None,
                       bs: slice = slice(None), test_mode: bool = False,
                       agent_memory: Optional[torch.Tensor] = None,
                       strategy_override: Optional[str] = None) -> Tuple[torch.Tensor, Dict]:
        avail_actions = ep_batch["avail_actions"][:, t_ep]        # (B=1, N, A)
        agent_outputs, agent_info = self.forward(ep_batch, t_ep, test_mode=test_mode, agent_memory=agent_memory)

        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )  # (B=1, N)

        # Blend discrete actions into prompt embeddings so env/reward reflect action ids
        if "prompt_embeddings" in agent_info:
            try:
                pe = agent_info["prompt_embeddings"]
                pe = self._apply_discrete_to_prompt(pe, chosen_actions)
                agent_info["prompt_embeddings"] = pe
            except Exception as ex:
                self.logger.warning(f"[MAC] apply discrete->prompt failed: {ex}")

        # Tie discrete actions to prompt embeddings via simple bins over [T_min, p_max] ranges
        prompt_embeds = agent_info.get("prompt_embeddings")
        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.clone()
            prompt_embeds = self._apply_discrete_to_prompt(prompt_embeds, chosen_actions)
            agent_info["prompt_embeddings"] = prompt_embeds

        if bool(getattr(getattr(self.args, "bne", object()), "refine_at_infer", False)) and raw_observation_text is not None:
            try:
                e0 = agent_info["prompt_embeddings"][0].detach().clone()          # (N,2)
                b  = agent_info["belief_states"][0].detach().clone()              # (N,d)
                g  = agent_info["group_repr"][0].detach().clone()                 # (d,)
                from modules.mixer.mix_llm import LLMQMixer
                tmp_mixer = LLMQMixer(self.args).to(self.device).eval()
                with torch.enable_grad():
                    e = e0.clone().detach().to(self.device).requires_grad_(True)
                    opt = torch.optim.SGD([e], lr=0.3)
                    for _ in range(int(getattr(getattr(self.args, "bne", object()), "K", 2))):
                        opt.zero_grad()
                        q_local = agent_info["q_values"][0].squeeze(-1).detach()  # (N,)
                        q_local_in = q_local.view(1,1,-1)
                        b_in = b.view(1,1,self.n_agents,-1)
                        e_in = e.view(1,1,self.n_agents,2)
                        g_in = g.view(1,1,-1)
                        q_tot, _ = tmp_mixer(q_local=q_local_in, belief_states=b_in, prompt_embeddings=e_in,
                                             group_repr=g_in, commitment_embedding=None, reward_ctx=None)
                        loss = -q_tot.squeeze()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_([e], 1.0)
                        opt.step()

                        e.data[..., 0].clamp_(float(getattr(self.args.sampling, "temperature_min", 0.1)),
                                              float(getattr(self.args.sampling, "temperature_max", 2.0)))
                        e.data[..., 1].clamp_(float(getattr(self.args.sampling, "p_min", 0.1)),
                                              float(getattr(self.args.sampling, "p_max", 0.95)))
                agent_info["prompt_embeddings"] = e.detach().unsqueeze(0)  # (B=1,N,2)
            except Exception as ex:
                self.logger.warning(f"[MAC] BNE refine skipped: {ex}")


        strategy_text = ""
        exec_responses = []
        commitment_text = ""
        commitment_embed = None


        bne_infer_data = None
        if (self.bne_enabled and test_mode and
            bool(getattr(getattr(self.args, "bne", object()), "refine_at_infer", False)) and
            raw_observation_text is not None):

            try:

                strategy_text = self._get_strategy_and_format(raw_observation_text)
                if not strategy_text or not strategy_text.strip():
                    strategy_text = "Break down the problem into steps and solve each part."


                obs_text = raw_observation_text
                tokenized = self.tokenizer(
                    obs_text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_question_length
                )
                obs_tokens = tokenized["input_ids"].to(self.device)  # (1, L)
                obs_tokens_agents = obs_tokens.expand(self.n_agents, -1)  # (N, L)

                # 调用多轮BNE
                max_iter = int(getattr(getattr(self.args, "bne", object()), "max_iterations_infer", 2))
                threshold = float(getattr(getattr(self.args, "bne", object()), "convergence_threshold", 0.05))

                bne_infer_data = self._run_bne_multi_round_infer(
                    obs_tokens=obs_tokens_agents,
                    obs_text=obs_text,
                    strategy_text=strategy_text,
                    max_iterations=max_iter,
                    convergence_threshold=threshold,
                    agent_memory=agent_memory
                )


                exec_responses = bne_infer_data["outputs_final"]
                commitment_text = bne_infer_data["commitment_final"]


                agent_info.update({
                    "bne_n_iterations": bne_infer_data["n_iterations"],
                    "bne_converged": bne_infer_data["converged"],
                    "bne_iteration_history": bne_infer_data["iteration_history"]
                })

                self.logger.info(f"[BNE Infer] Completed {bne_infer_data['n_iterations']} iterations, converged={bne_infer_data['converged']}")

            except Exception as e:
                self.logger.warning(f"[BNE Infer] Multi-round BNE failed: {e}, falling back to normal flow")
                bne_infer_data = None

        if raw_observation_text is not None and bne_infer_data is None:
            strategy_text = strategy_override if strategy_override is not None else self._get_strategy_and_format(raw_observation_text)
            if not strategy_text or not strategy_text.strip():
                self.logger.warning(f"[MAC] Strategy generation returned empty (got: {repr(strategy_text)}), using fallback")
                strategy_text = "Break down the problem into steps and solve each part."
            else:
                self.logger.debug(f"[MAC] Strategy generated: {strategy_text[:100]}...")
            e_now = agent_info["prompt_embeddings"][0]  # (N,2)
            for i in range(self.n_agents):
                try:
                    # Extract temperature and top_p (stateless)
                    T_i = float(e_now[i, 0].item())
                    p_i = float(e_now[i, 1].item())

                    # Generate with explicit parameters
                    resp = self.agent.generate_answer(
                        question=raw_observation_text,
                        strategy=strategy_text,
                        temperature=T_i,
                        top_p=p_i
                    )
                except Exception as e:
                    self.logger.warning(f"Executor {i} failed: {e}")
                    resp = ""
                resp = self._post_sanitize_text(resp)
                resp = self._ensure_boxed_format(resp)
                exec_responses.append(resp)


            commitment_text = self._generate_commitment(raw_observation_text, strategy_text, exec_responses)

            commitment_metadata = getattr(self, "_last_commitment_metadata", None)

            try:
                commitment_embed = self._encode_commitment_vector(commitment_text).to(self.device)
            except Exception as e:
                self.logger.warning(f"Commitment embedding failed: {e}")
                commitment_embed = None


            if test_mode and hasattr(self, 'eval_total_count'):
                self.eval_total_count += 1


        agent_info.update({
            "executor_responses": exec_responses,
            "commitment": commitment_text,
            "commitment_text": commitment_text,
            "commitment_embedding": commitment_embed,
            "strategy": strategy_text,
            "format": "",
            "q_values": agent_info.get("q_values"),
            "selected_actions": chosen_actions.detach().clone() if isinstance(chosen_actions, torch.Tensor) else chosen_actions,
        })

        if raw_observation_text is not None and hasattr(self, "_last_commitment_metadata"):
            agent_info["commitment_metadata"] = self._last_commitment_metadata


        return chosen_actions, agent_info

    def _apply_discrete_to_prompt(self, prompt_embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Map discrete action ids to target [T, p] bins and blend with network outputs.
        This makes the discrete action space influence generation and rewards.
        """
        if prompt_embeddings is None or actions is None:
            return prompt_embeddings
        try:
            e = prompt_embeddings
            a = actions
            if a.dim() == 3 and a.size(-1) == 1:
                a = a.squeeze(-1)
            if a.dim() == 1:
                a = a.unsqueeze(0)
            n_act = max(1, int(getattr(self.args, "n_actions", 2)))
            frac = a.float() / max(1, n_act - 1)

            T_min = float(getattr(self.args.sampling, "temperature_min", 0.1))
            T_max = float(getattr(self.args.sampling, "temperature_max", 2.0))
            p_min = float(getattr(self.args.sampling, "p_min", 0.8))   # repetition_penalty min
            p_max = float(getattr(self.args.sampling, "p_max", 1.3))   # repetition_penalty max

            T_target = T_min + frac * (T_max - T_min)
            p_target = p_max - frac * (p_max - p_min)
            target = torch.stack([T_target, p_target], dim=-1)  # (B,N,2)

            e = 0.5 * e + 0.5 * target
            e[..., 0] = torch.clamp(e[..., 0], T_min, T_max)
            e[..., 1] = torch.clamp(e[..., 1], p_min, p_max)
            return e
        except Exception as ex:
            self.logger.warning(f"[MAC] Discrete→prompt mapping failed: {ex}")
            return prompt_embeddings
    def run_bne_refinement(
        self,
        mode: str,  # "train" or "infer"
        obs_tokens: torch.Tensor,
        obs_text: str,
        strategy_text: str,
        agent_memory: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """

        Args:
            mode: "train"  "infer" 
            obs_tokens: (N, L) observation tokens
            obs_text: raw question text
            strategy_text: strategy from coordinator

        Returns:
            Dict containing:
                - beliefs: (N, d_belief)
                - e_init: (N, 2) 
                - e_final: (N, 2) 
                - e_path: List[List[Tensor]] 
                - commitments: List[str] 
                - commitment_emb_final: (d_c,) 
                - iterations_run: int 
                - early_stop: bool 
                - outputs_history: List[List[str]] 
        """
        if not self.bne_enabled:
            raise RuntimeError("BNE not enabled")


        if mode == "train":
            max_iterations = int(getattr(getattr(self.args, "bne", object()), "max_iterations_train", 1))
        else:  # mode == "infer"
            max_iterations = int(getattr(getattr(self.args, "bne", object()), "max_iterations_infer", 2))

        convergence_eps = float(getattr(getattr(self.args, "bne", object()), "convergence_threshold", 0.05))
        update_eta = float(getattr(getattr(self.args, "bne", object()), "update_rate", 1.0))  # 可选: 学习率

        # Sampling
        T_min = float(getattr(getattr(self.args, "sampling", object()), "temperature_min", 0.1))
        T_max = float(getattr(getattr(self.args, "sampling", object()), "temperature_max", 2.0))
        p_min = float(getattr(getattr(self.args, "sampling", object()), "p_min", 0.8))   # repetition_penalty min
        p_max = float(getattr(getattr(self.args, "sampling", object()), "p_max", 1.3))   # repetition_penalty max

        N = self.n_agents

        beliefs = []
        e_init = []

        mem_local = None
        if agent_memory is not None:
            mem_local = agent_memory.detach().clone()
        with torch.no_grad() if mode == "infer" else torch.enable_grad():
            for i in range(N):
                mem_i = mem_local[i:i+1] if mem_local is not None else None
                output = self.policy_networks[i](obs_tokens[i:i+1], memory=mem_i)
                beliefs.append(output["belief_state"].squeeze(0))
                e_init.append(output["prompt_embedding"].squeeze(0))

        beliefs = torch.stack(beliefs)  # (N, d_belief)
        e_init = torch.stack(e_init)    # (N, 2)

        group_repr = self.belief_encoder(beliefs.unsqueeze(0)).squeeze(0)  # (d_belief,)

        # Track embeddings for training
        output_embs_0 = None  # First iteration output embeddings
        commitment_emb_0 = None  # First iteration commitment embedding


        def generate_with_params(e_mat: torch.Tensor) -> List[str]:
            """显式传递temperature和top_p参数"""
            texts = []
            for i in range(N):
                T_i = float(e_mat[i, 0].item())
                p_i = float(e_mat[i, 1].item())

                
                T_i = max(T_min, min(T_max, T_i))
                p_i = max(p_min, min(p_max, p_i))

             
                txt = self.agent.generate_answer(
                    question=obs_text,
                    strategy=strategy_text,
                    temperature=T_i,
                    top_p=p_i
                )

                
                if not txt or not isinstance(txt, str):
                    txt = "0"

                txt = self._post_sanitize_text(txt)
                txt = self._ensure_boxed_format(txt)

                texts.append(txt)

            return texts


        e_current = e_init.clone()
        e_path = [[e_init[i].detach().clone()] for i in range(N)]  

        outputs = generate_with_params(e_current)
        commitment = self._generate_commitment(obs_text, strategy_text, outputs)
        commitment_emb = self._encode_output_vector(commitment)

        commitments_history = [commitment]
        outputs_history = [outputs]
        commitment_text_history = [self._normalize_commitment_text(commitment)]
        commitment_emb_history: List[torch.Tensor] = [commitment_emb.detach().clone()]

        iterations_run = 0
        early_stop = False

        for iteration in range(max_iterations):
            iterations_run = iteration + 1


            output_embs = torch.stack([
                self._encode_output_vector(out) for out in outputs
            ])  # (N, d_c)
            output_embs = self._ensure_vector_dim(output_embs, self.commitment_dim)
            commitment_emb = self._ensure_vector_dim(commitment_emb, self.commitment_dim)

            # Store first iteration embeddings for training
            if iteration == 0:
                output_embs_0 = output_embs.detach().clone()
                commitment_emb_0 = commitment_emb.detach().clone()


            context_manager = torch.no_grad() if mode == "infer" else torch.enable_grad()
            with context_manager:
                delta_e_list = []
                for i in range(N):
                    output_i_emb = self._ensure_vector_dim(output_embs[i:i+1], self.commitment_dim)
                    commitment_emb_batch = self._ensure_vector_dim(commitment_emb.unsqueeze(0), self.commitment_dim)
                    delta_e = self.refine_module(
                        belief_i=beliefs[i:i+1],
                        output_i_emb=output_i_emb,
                        commitment_emb=commitment_emb_batch,
                        group_repr=group_repr.unsqueeze(0),
                        e_prev=e_current[i:i+1]
                    ).squeeze(0)
                    delta_e_list.append(delta_e)

                delta_e_all = torch.stack(delta_e_list, dim=0)  # (N, 2)
                e_next_raw = e_current + update_eta * delta_e_all
                e_next = torch.stack(
                    [
                        torch.clamp(e_next_raw[:, 0], T_min, T_max),  # Temperature
                        torch.clamp(e_next_raw[:, 1], p_min, p_max),  # repetition_penalty
                    ],
                    dim=-1
                )


            for i in range(N):
                e_path[i].append(e_next[i].detach().clone())

        
            delta_magnitude = torch.norm(e_next - e_current).item()

            
            outputs_new = generate_with_params(e_next)
            commitment_new = self._generate_commitment(obs_text, strategy_text, outputs_new)
            commitment_emb_new = self._encode_output_vector(commitment_new)
            commitment_emb_new = self._ensure_vector_dim(commitment_emb_new, self.commitment_dim)
            normalized_commitment_new = self._normalize_commitment_text(commitment_new)

            outputs_history.append(outputs_new)
            commitments_history.append(commitment_new)
            commitment_emb_history.append(self._ensure_vector_dim(commitment_emb_new.detach().clone(), self.commitment_dim))
            commitment_text_history.append(normalized_commitment_new)

            if self._has_commitment_converged(commitment, commitment_new, commitment_emb, commitment_emb_new):
                self.logger.info(f"[BNE {mode}] Converged at iteration {iteration+1}: commitment equivalent")
                e_current = e_next
                commitment = commitment_new
                outputs = outputs_new
                commitment_emb = commitment_emb_new
                early_stop = True
                break

            if self._has_commitment_oscillated(commitment_emb_history, commitment_text_history):
                self.logger.warning(f"[BNE {mode}] Oscillation detected at iteration {iteration+1}; terminating refinement")
                e_current = e_next
                commitment = commitment_new
                outputs = outputs_new
                commitment_emb = commitment_emb_new
                early_stop = True
                break

            if convergence_eps > 0 and delta_magnitude < convergence_eps:
                self.logger.info(f"[BNE {mode}] Converged at iteration {iteration+1}: delta={delta_magnitude:.4f} < {convergence_eps}")
                e_current = e_next
                commitment = commitment_new
                outputs = outputs_new
                commitment_emb = commitment_emb_new
                early_stop = True
                break

            e_current = e_next
            outputs = outputs_new
            commitment = commitment_new
            commitment_emb = commitment_emb_new

        commitment_emb_final = self._ensure_vector_dim(commitment_emb_history[-1].detach().clone(), self.commitment_dim)

        # Compute final output embeddings
        output_embs_final = torch.stack([
            self._encode_output_vector(out) for out in outputs_history[-1]
        ])  # (N, d_c)
        output_embs_final = self._ensure_vector_dim(output_embs_final, self.commitment_dim)

        return {
            "beliefs": beliefs,                             # (N, d_belief)
            "e_init": e_init,                              # (N, 2)
            "e_final": e_current,                          # (N, 2)
            "e_path": e_path,                              # List[List[Tensor(2,)]]
            "commitments": commitments_history,            # List[str]
            "commitment_emb_final": commitment_emb_final,  # (d_c,)
            "commitment_emb_0": commitment_emb_0,          # (d_c,) - first iteration
            "output_embs_0": output_embs_0,                # (N, d_c) - first iteration
            "output_embs_final": output_embs_final,        # (N, d_c) - final iteration
            "iterations_run": iterations_run,              # total iterations run
            "early_stop": early_stop,
            "outputs_history": outputs_history,            # List[List[str]]
            "group_repr": group_repr,                      # (d_belief,)
            "mode": mode
        }


    def _run_bne_single_round(
        self,
        obs_tokens: torch.Tensor,
        obs_text: str,
        strategy_text: str,
        agent_memory: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """

        1. K policy networks -> beliefs, e_init
        2. first round -> exec_outputs_0
        3. Coordinator -> commitment_0
        4. Refine: calculate delta_e
        5. second round -> exec_outputs_1
        6. Coordinator outpuit commitment_final

        Args:
            obs_tokens: (N, L) token IDs for each agent
            obs_text: raw question text
            strategy_text: strategy from coordinator

        Returns:
            dict with:
                - beliefs: (N, belief_dim)
                - e_init: (N, 2)
                - e_refined: (N, 2)
                - exec_outputs_0: List[str] - first round responses
                - exec_outputs_1: List[str] - refined responses
                - commitment_0: str
                - commitment_final: str
                - commitment_emb_0: (commitment_dim,)
                - commitment_emb_final: (commitment_dim,)
                - output_embs_0: (N, commitment_dim)
                - output_embs_1: (N, commitment_dim)
                - group_repr: (belief_dim,)
        """
        if not self.bne_enabled:
            raise RuntimeError("BNE not enabled, cannot run _run_bne_single_round")


        beliefs = []
        e_init = []

        mem_local = agent_memory.detach().clone() if agent_memory is not None else None
        with torch.no_grad():
            for i in range(self.n_agents):
                mem_i = mem_local[i:i+1] if mem_local is not None else None
                output = self.policy_networks[i](obs_tokens[i:i+1], memory=mem_i)  # (1, ...)
                beliefs.append(output["belief_state"].squeeze(0))     # (belief_dim,)
                e_init.append(output["prompt_embedding"].squeeze(0))  # (2,)

        beliefs = torch.stack(beliefs)  # (N, belief_dim)
        e_init = torch.stack(e_init)    # (N, 2)

    
        group_repr = self.belief_encoder(beliefs.unsqueeze(0)).squeeze(0)  # (belief_dim,)

        
        exec_outputs_0 = []
        for i in range(self.n_agents):
            try:
                T_i = float(e_init[i, 0].item())
                p_i = float(e_init[i, 1].item())
                resp = self.agent.generate_answer(
                    question=obs_text,
                    strategy=strategy_text,
                    temperature=T_i,
                    top_p=p_i
                )
            except Exception as e:
                self.logger.warning(f"[BNE] Executor {i} round 0 failed: {e}")
                resp = ""
            resp = self._post_sanitize_text(resp)
            resp = self._ensure_boxed_format(resp)
            exec_outputs_0.append(resp)

 
        output_embs_0 = torch.stack([self._encode_output_vector(out) for out in exec_outputs_0])  # (N, commitment_dim)
        output_embs_0 = self._ensure_vector_dim(output_embs_0, self.commitment_dim)

        commitment_0 = self._generate_commitment(obs_text, strategy_text, exec_outputs_0)
        try:
            commitment_emb_0 = self._encode_commitment_vector(commitment_0).to(self.device)
        except Exception as e:
            self.logger.warning(f"[BNE] Commitment_0 embedding failed: {e}")
            commitment_emb_0 = torch.zeros(self.output_encoder.dim, device=self.device)
        commitment_emb_0 = self._ensure_vector_dim(commitment_emb_0, self.commitment_dim)

        e_refined = []
        for i in range(self.n_agents):
            with torch.no_grad():
                output_i_emb = self._ensure_vector_dim(output_embs_0[i:i+1], self.commitment_dim)
                commitment_emb_batch = self._ensure_vector_dim(commitment_emb_0.unsqueeze(0), self.commitment_dim)
                delta_e = self.refine_module(
                    belief_i=beliefs[i:i+1],
                    output_i_emb=output_i_emb,
                    commitment_emb=commitment_emb_batch,
                    group_repr=group_repr.unsqueeze(0),
                    e_prev=e_init[i:i+1]
                ).squeeze(0)  # (2,)

            e_new = e_init[i] + delta_e

           
            T_min = float(getattr(self.args.sampling, "temperature_min", 0.1))
            T_max = float(getattr(self.args.sampling, "temperature_max", 2.0))
            p_min = float(getattr(self.args.sampling, "p_min", 0.1))
            p_max = float(getattr(self.args.sampling, "p_max", 0.9))

            e_new[0] = torch.clamp(e_new[0], T_min, T_max)
            e_new[1] = torch.clamp(e_new[1], p_min, p_max)

            e_refined.append(e_new)

        e_refined = torch.stack(e_refined)  # (N, 2)


        exec_outputs_1 = []
        for i in range(self.n_agents):
            try:
                T_i = float(e_refined[i, 0].item())
                p_i = float(e_refined[i, 1].item())
                resp = self.agent.generate_answer(
                    question=obs_text,
                    strategy=strategy_text,
                    temperature=T_i,
                    top_p=p_i
                )
            except Exception as e:
                self.logger.warning(f"[BNE] Executor {i} round 1 failed: {e}")
                resp = ""
            resp = self._post_sanitize_text(resp)
            resp = self._ensure_boxed_format(resp)
            exec_outputs_1.append(resp)

        # Encode outputs
        output_embs_1 = torch.stack([self._encode_output_vector(out) for out in exec_outputs_1])  # (N, commitment_dim)
        output_embs_1 = self._ensure_vector_dim(output_embs_1, self.commitment_dim)

   
        commitment_final = self._generate_commitment(obs_text, strategy_text, exec_outputs_1)
        try:
            commitment_emb_final = self._encode_commitment_vector(commitment_final).to(self.device)
        except Exception as e:
            self.logger.warning(f"[BNE] Commitment_final embedding failed: {e}")
            commitment_emb_final = torch.zeros(self.output_encoder.dim, device=self.device)
        commitment_emb_final = self._ensure_vector_dim(commitment_emb_final, self.commitment_dim)

        return {
            "beliefs": beliefs,                           # (N, belief_dim)
            "e_init": e_init,                             # (N, 2)
            "e_refined": e_refined,                       # (N, 2)
            "exec_outputs_0": exec_outputs_0,             # List[str]
            "exec_outputs_1": exec_outputs_1,             # List[str]
            "commitment_0": commitment_0,                 # str
            "commitment_final": commitment_final,         # str
            "commitment_emb_0": commitment_emb_0,         # (commitment_dim,)
            "commitment_emb_final": commitment_emb_final, # (commitment_dim,)
            "output_embs_0": output_embs_0,               # (N, commitment_dim)
            "output_embs_1": output_embs_1,               # (N, commitment_dim)
            "group_repr": group_repr,                     # (belief_dim,)
        }

    def _run_bne_multi_round_infer(
        self,
        obs_tokens: torch.Tensor,
        obs_text: str,
        strategy_text: str,
        max_iterations: int = 2,
        convergence_threshold: float = 0.05,
        agent_memory: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        BNE Multi-Round Inference

        Args:
            obs_tokens: (N, L) observation tokens
            obs_text: raw question text
            strategy_text: strategy from coordinator
            max_iterations: maximum BNE refinement rounds
            convergence_threshold: convergence threshold for early stopping
        """
        if not self.bne_enabled:
            raise RuntimeError("BNE not enabled")

        N = self.n_agents

        # Get sampling bounds from config (unified with training)
        T_min = float(getattr(getattr(self.args, "sampling", object()), "temperature_min", 0.1))
        T_max = float(getattr(getattr(self.args, "sampling", object()), "temperature_max", 2.0))
        p_min = float(getattr(getattr(self.args, "sampling", object()), "p_min", 0.1))
        p_max = float(getattr(getattr(self.args, "sampling", object()), "p_max", 0.9))
        update_eta = float(getattr(getattr(self.args, "bne", object()), "update_rate", 1.0))

        # Policy networks: generate initial beliefs and prompt embeddings
        beliefs = []
        e_init = []
        mem_local = agent_memory.detach().clone() if agent_memory is not None else None
        with torch.no_grad():
            for i in range(N):
                mem_i = mem_local[i:i+1] if mem_local is not None else None
                output = self.policy_networks[i](obs_tokens[i:i+1], memory=mem_i)
                beliefs.append(output["belief_state"].squeeze(0))
                e_init.append(output["prompt_embedding"].squeeze(0))

        beliefs = torch.stack(beliefs)  # (N, belief_dim)
        e_init = torch.stack(e_init)    # (N, 2)

        # Helper: generate responses with explicit temperature/top_p
        def generate_with_params(e_mat: torch.Tensor) -> List[str]:

            texts = []
            for i in range(N):
                T_i = float(e_mat[i, 0].item())
                p_i = float(e_mat[i, 1].item())

                # Clamp to valid range
                T_i = max(T_min, min(T_max, T_i))
                p_i = max(p_min, min(p_max, p_i))  # repetition_penalty

                # Generate with explicit parameters (stateless)
                resp = self.agent.generate_answer(
                    question=obs_text,
                    strategy=strategy_text,
                    temperature=T_i,
                    top_p=p_i
                )

                # Sanitize
                if not resp or not isinstance(resp, str):
                    resp = "0"
                if len(resp) > 2048:
                    resp = resp[:2048]

                texts.append(resp)

            return texts

        # Round 0: Initial responses
        exec_outputs_0 = generate_with_params(e_init)

        commitment_0 = self._generate_commitment(obs_text, strategy_text, exec_outputs_0)

        output_embs_0 = torch.stack([
            self._encode_output_vector(out) for out in exec_outputs_0
        ])  # (N, commitment_dim)
        output_embs_0 = self._ensure_vector_dim(output_embs_0, self.commitment_dim)
        commitment_emb_0 = self._ensure_vector_dim(self._encode_output_vector(commitment_0), self.commitment_dim)  # (commitment_dim,)

      
        group_repr = self.belief_encoder(beliefs.unsqueeze(0)).squeeze(0)  # (belief_dim,)

      
        e_prev = e_init
        commitment_prev = commitment_0
        commitment_emb_prev = commitment_emb_0
        output_embs_prev = output_embs_0
        commitment_text_history = [self._normalize_commitment_text(commitment_0)]
        commitment_emb_history: List[torch.Tensor] = [self._ensure_vector_dim(commitment_emb_0.detach().clone(), self.commitment_dim)]

        iteration_history = []
        converged = False
        commitment_new = commitment_0
        exec_outputs_new = exec_outputs_0
        e_new = e_init

        for iter_idx in range(max_iterations):
            # Refine prompt embeddings
            e_new = []
            for i in range(N):
                with torch.no_grad():
                    output_i_emb = self._ensure_vector_dim(output_embs_prev[i:i+1], self.commitment_dim)
                    commitment_emb_batch = self._ensure_vector_dim(commitment_emb_prev.unsqueeze(0), self.commitment_dim)
                    delta_e = self.refine_module(
                        belief_i=beliefs[i:i+1],
                        output_i_emb=output_i_emb,
                        commitment_emb=commitment_emb_batch,
                        group_repr=group_repr.unsqueeze(0),
                        e_prev=e_prev[i:i+1]
                    ).squeeze(0)

                e_new_i = e_prev[i] + update_eta * delta_e

                # Clamp to configured bounds (unified with training)
                e_new_i[0] = torch.clamp(e_new_i[0], T_min, T_max)  # Temperature
                e_new_i[1] = torch.clamp(e_new_i[1], p_min, p_max)  # repetition_penalty

                e_new.append(e_new_i)

            e_new = torch.stack(e_new)  # (N, 2)

            delta_magnitude = torch.norm(e_new - e_prev).item()

            # Generate new responses with refined embeddings (stateless)
            exec_outputs_new = generate_with_params(e_new)

            # Generate new commitment
            commitment_new = self._generate_commitment(obs_text, strategy_text, exec_outputs_new)
            normalized_commitment_new = self._normalize_commitment_text(commitment_new)

            output_embs_new = torch.stack([
                self._encode_output_vector(out) for out in exec_outputs_new
            ])
            output_embs_new = self._ensure_vector_dim(output_embs_new, self.commitment_dim)
            commitment_emb_new = self._ensure_vector_dim(self._encode_output_vector(commitment_new), self.commitment_dim)
            commitment_emb_history.append(commitment_emb_new.clone())
            commitment_text_history.append(normalized_commitment_new)

            iteration_history.append({
                "iter": iter_idx + 1,
                "e": e_new.detach().cpu(),
                "outputs": exec_outputs_new,
                "commitment": commitment_new,
                "delta_magnitude": delta_magnitude
            })

            if self._has_commitment_converged(commitment_prev, commitment_new, commitment_emb_prev, commitment_emb_new):
                self.logger.info(f"[BNE Infer] Converged at iteration {iter_idx+1}: commitment equivalent")
                converged = True
                commitment_prev = commitment_new
                commitment_emb_prev = commitment_emb_new
                output_embs_prev = output_embs_new
                e_prev = e_new
                break

            if self._has_commitment_oscillated(commitment_emb_history, commitment_text_history):
                self.logger.warning(f"[BNE Infer] Oscillation detected at iteration {iter_idx+1}; terminating refinement")
                commitment_prev = commitment_new
                commitment_emb_prev = commitment_emb_new
                output_embs_prev = output_embs_new
                e_prev = e_new
                break

            if delta_magnitude < convergence_threshold:
                self.logger.info(f"[BNE Infer] Converged at iteration {iter_idx+1}: delta={delta_magnitude:.4f} < {convergence_threshold}")
                converged = True
                commitment_prev = commitment_new
                commitment_emb_prev = commitment_emb_new
                output_embs_prev = output_embs_new
                e_prev = e_new
                break

            e_prev = e_new
            commitment_prev = commitment_new
            commitment_emb_prev = commitment_emb_new
            output_embs_prev = output_embs_new

        return {
            "beliefs": beliefs,
            "e_init": e_init,
            "e_final": e_new if len(iteration_history) > 0 else e_init,
            "commitment_init": commitment_0,
            "commitment_final": commitment_new if len(iteration_history) > 0 else commitment_0,
            "outputs_init": exec_outputs_0,
            "outputs_final": exec_outputs_new if len(iteration_history) > 0 else exec_outputs_0,
            "iteration_history": iteration_history,
            "n_iterations": len(iteration_history) + 1,  # +1 for round 0
            "converged": converged
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
            prompt=prompt, temperature=0.3, top_p=0.4, repetition_penalty=1.1, max_tokens=180
        )
        return self._post_sanitize_text(out)

    def _generate_commitment(self, question: str, strategy: str, responses: List[str],
                             group_repr: Optional[torch.Tensor] = None,
                             prompt_embeddings: Optional[torch.Tensor] = None) -> str:

        formatted = "\n".join([f"Executor {i+1}: {r}" for i, r in enumerate(responses)])
        commit_prompt = f"""You are the COORDINATOR. Review the question, strategy and all executor solutions to aggregate and produce a structured final answer.

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
            prompt=commit_prompt, temperature=0.1, top_p=0.3, repetition_penalty=1.05, max_tokens=150
        )

    
        final_answer, metadata = self._parse_structured_commitment(out)

   
        self._last_commitment_metadata = metadata

      
        return f"\\boxed{{{final_answer}}}"

    def _parse_structured_commitment(self, raw_output: str) -> Tuple[str, Dict[str, Any]]:
        """

        Args:
            raw_output

        Returns:
            (final_answer: str, metadata: dict)
            - final_answer
            - metadata
        """
        import json

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
        except (json.JSONDecodeError, ValueError, KeyError) as e:
          
            metadata["parse_error"] = str(e)[:100]

      
        boxed_content = self._extract_boxed_content(raw_output)
        if boxed_content:
            num = extract_answer_number(boxed_content)
            if num is not None:
                metadata["parse_method"] = "boxed"
                return num, metadata

        # Try to extract "final_value" or "final" from malformed JSON
        final_value_match = re.search(r'"final(?:_value)?"\s*:\s*"([^"]+)"', raw_output)
        if final_value_match:
            extracted = final_value_match.group(1)
            num = extract_answer_number(extracted)
            if num is not None:
                metadata["parse_method"] = "json_field_regex"
                return num, metadata

        nums = re.findall(r'[+-]?\d+(?:\.\d+)?', raw_output)
        if nums:
            num = extract_answer_number(nums[-1])
            if num is not None:
                metadata["parse_method"] = "regex"
                return num, metadata

      
        metadata["parse_method"] = "zero_fallback"
        return "0", metadata


    def _post_sanitize_text(self, text: str) -> str:
        if text is None:
            return ""
        text = text.replace("\x08", "\\b").replace("\x0c", "\\f")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return self._repair_boxed(text)

    def _repair_boxed(self, s: str) -> str:
        s = re.sub(r'(?<![\\b])oxed\{', r'\\boxed{', s)
        s = re.sub(r'(?<!\\)boxed\{', r'\\boxed{', s)
        return s

    def _ensure_boxed_format(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text or "")
        stripped = text.strip()
        if "\\boxed{" in stripped:
            return stripped
        candidate = extract_answer_number(stripped)
        if candidate is None:
            candidate = "undetermined"
        if stripped:
            return f"{stripped}\n\\boxed{{{candidate}}}"
        return f"\\boxed{{{candidate}}}"

    def _extract_boxed_content(self, text: str) -> Optional[str]:
        if not isinstance(text, str):
            return None
        m = re.search(r'\\boxed\{([\s\S]*?)\}', text)
        return m.group(1).strip() if m else None

    def _normalize_number(self, s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        normalized = extract_answer_number(str(s))
        if normalized is not None:
            return normalized
        cleaned = str(s).strip()
        return cleaned if cleaned else None

    def _strip_non_box_numbers(self, text: str) -> str:
        if text is None:
            return ""
        return str(text)

    def _create_minimal_tokenizer(self):
        class MinimalTokenizer:
            def __init__(self):
                self.vocab = {chr(i): i for i in range(32, 127)}
                self.vocab.update({'[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3})
                self.pad_token = '[PAD]'
                self.eos_token = '[EOS]'
                self.pad_token_id = self.vocab[self.pad_token]
                self.eos_token_id = self.vocab[self.eos_token]
                self.unk_token_id = self.vocab['[UNK]']
                self.bos_token_id = self.vocab['[BOS]']
                self.vocab_size = len(self.vocab)
            def __call__(self, text, max_length=None, padding=True, truncation=True, return_tensors="pt", **kw):
                if isinstance(text, str): text = [text]
                import torch
                out = []
                for t in text:
                    toks = [self.vocab.get(c, 1) for c in t[:(max_length-1 if max_length else None)]]
                    toks.append(self.eos_token_id)
                    if max_length and padding and len(toks) < max_length:
                        toks += [self.pad_token_id] * (max_length - len(toks))
                    out.append(toks[:max_length] if max_length else toks)
                return type("Enc", (), {"input_ids": torch.tensor(out)})
        return MinimalTokenizer()

    def _normalize_commitment_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip().lower()

    def _commitment_similarity(self, emb_a: Optional[torch.Tensor], emb_b: Optional[torch.Tensor]) -> float:
        if emb_a is None or emb_b is None:
            return 0.0
        if emb_a.dim() == 1:
            emb_a = emb_a.unsqueeze(0)
        if emb_b.dim() == 1:
            emb_b = emb_b.unsqueeze(0)
        if emb_a.size(-1) != emb_b.size(-1):
            min_dim = min(emb_a.size(-1), emb_b.size(-1))
            emb_a = emb_a[..., :min_dim]
            emb_b = emb_b[..., :min_dim]
        try:
            return float(F.cosine_similarity(emb_a, emb_b, dim=-1).mean().item())
        except Exception:
            return 0.0

    def _has_commitment_converged(
        self,
        prev_text: str,
        new_text: str,
        prev_emb: Optional[torch.Tensor],
        new_emb: Optional[torch.Tensor],
    ) -> bool:
        normalized_prev = self._normalize_commitment_text(prev_text)
        normalized_new = self._normalize_commitment_text(new_text)
        if normalized_prev and normalized_prev == normalized_new:
            return True
        similarity = self._commitment_similarity(prev_emb, new_emb)
        return similarity >= self.bne_commitment_match_threshold

    def _has_commitment_oscillated(
        self,
        emb_history: List[torch.Tensor],
        text_history: List[str],
    ) -> bool:
        if not self.bne_detect_oscillation:
            return False
        min_hist = max(2, int(getattr(self, "bne_min_oscillation_history", 5)))
        if len(text_history) < min_hist or len(emb_history) < min_hist:
            return False
        if len(text_history) >= min_hist:
            if text_history[-1] == text_history[-3] and text_history[-2] == text_history[-4] and text_history[-1] != text_history[-2]:
                return True
        if len(emb_history) >= min_hist:
            sim_13 = self._commitment_similarity(emb_history[-1], emb_history[-3])
            sim_24 = self._commitment_similarity(emb_history[-2], emb_history[-4])
            sim_12 = self._commitment_similarity(emb_history[-1], emb_history[-2])
            if sim_13 >= self.bne_commitment_osc_threshold and sim_24 >= self.bne_commitment_osc_threshold and sim_12 < self.bne_commitment_osc_threshold:
                return True
        return False

    def _ensure_vector_dim(self, vec: Optional[torch.Tensor], target_dim: int) -> Optional[torch.Tensor]:
        if vec is None:
            return None
        if vec.dim() == 0:
            vec = vec.unsqueeze(0)
        squeeze = False
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)
            squeeze = True
        current_dim = vec.size(-1)
        if current_dim > target_dim:
            vec = vec[..., :target_dim]
        elif current_dim < target_dim:
            pad = target_dim - current_dim
            vec = F.pad(vec, (0, pad), mode="constant", value=0.0)
        if squeeze:
            vec = vec.squeeze(0)
        return vec

    def _encode_output_vector(self, text: str) -> torch.Tensor:
        vec = self.output_encoder.encode_output(text)
        vec = self._ensure_vector_dim(vec, self.commitment_dim)
        return vec.detach()

    def _encode_commitment_vector(self, text: str) -> torch.Tensor:
        vec = self.commitment_embedder.embed_commitments([text])
        if isinstance(vec, torch.Tensor) and vec.numel() > 0:
            vec = vec.squeeze(0)
        else:
            vec = torch.zeros(self.commitment_dim, device=self.device, dtype=torch.float32)
        vec = self._ensure_vector_dim(vec, self.commitment_dim)
        return vec.detach()


    def cuda(self):
        self.agent.cuda()
        self.belief_encoder.cuda()
        self.commitment_embedder.cuda()

BasicMAC = LLMBasicMAC
