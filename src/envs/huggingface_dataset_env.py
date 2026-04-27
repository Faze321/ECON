# src/envs/huggingface_dataset_env.py
# -*- coding: utf-8 -*-
import gym
from gym import spaces
import numpy as np
from datasets import load_dataset
from typing import Dict, Any, Optional, Tuple, List
import re
import torch
import random

from utils.answer_extraction import extract_numeric_answer, _normalize_number as _norm_num
from modules.text_encoders.output_encoder import OutputEncoder

class HuggingFaceDatasetEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, logger, **kwargs):
        super().__init__()
        self.logger = logger
        self.dataset_path = kwargs.get("hf_dataset_path", "gsm8k")
        self.dataset_config_name = kwargs.get("hf_dataset_config_name", None)
        self.dataset_split = kwargs.get("dataset_split", "train")
        self.is_streaming = kwargs.get("dataset_streaming", False)
        self.n_actions = int(kwargs.get("n_actions", 2))

        self.use_random_sampling = kwargs.get("use_random_sampling", False)
        self.random_without_replacement = kwargs.get("random_without_replacement", True)
        self.loop_dataset = kwargs.get("loop_dataset", True)

        self.use_dataset_episode = kwargs.get("use_dataset_episode", False)
        self.max_rounds = int(kwargs.get("max_rounds", 1))

        self.question_field = kwargs.get("question_field_name", "question")
        self.answer_field = kwargs.get("answer_field_name", "answer")

        from types import SimpleNamespace as _SNS
        _reward_cfg = kwargs.get("reward_config", kwargs.get("reward", {}))
        self.reward_args = _SNS(**_reward_cfg) if isinstance(_reward_cfg, dict) else _reward_cfg
        self.compute_reward = bool(kwargs.get("compute_reward", getattr(self.reward_args, "compute_reward", True)))

        # Reward weights are only used when compute_reward=True.
        _alpha_init = []
        if hasattr(self.reward_args, "initial_weights"):
            _alpha_init = list(getattr(self.reward_args, "initial_weights", []))
        if not _alpha_init:
            _alpha_init = [
                float(getattr(self.reward_args, "al_weight", 0.3)),
                float(getattr(self.reward_args, "ts_weight", 0.5)),
                float(getattr(self.reward_args, "cc_weight", 0.2)),
            ]
        self.alpha_default = _alpha_init

        try:
            self.dataset = load_dataset(
                self.dataset_path,
                name=self.dataset_config_name,
                split=self.dataset_split,
                streaming=self.is_streaming
            )
            if self.is_streaming:
                self.dataset_iterator = iter(self.dataset)
                self.dataset_list = None
                self.num_samples = None
                self.logger.info(f"Loaded IterableDataset: {self.dataset_path}, split: {self.dataset_split}")
            else:
                self.dataset_list = list(self.dataset)
                self.num_samples = len(self.dataset_list)
                self.dataset_iterator = None
                self.logger.info(f"Loaded Dataset: {self.dataset_path}, split: {self.dataset_split}, num_samples: {self.num_samples}")
        except Exception as e:
            self.logger.error(f"Failed to load dataset '{self.dataset_path}' (config: {self.dataset_config_name}, split: {self.dataset_split}): {e}")
            raise

        self.current_data_idx = -1
        self._unused_indices: List[int] = list(range(self.num_samples)) if (not self.is_streaming and self.use_random_sampling) else None

        self.max_question_length = kwargs.get("max_question_length", 1024)
        self.max_answer_length = kwargs.get("max_answer_length", 1024)
        self.action_space = spaces.Text(max_length=self.max_answer_length)
        self.observation_space = spaces.Text(max_length=self.max_question_length)

        self.current_sample: Optional[Dict] = None
        self.current_question: Optional[str] = None
        self.current_ground_truth_answer: Optional[str] = None
        self.episode_count = 0

        self.episode_length = 0
        self.round_count = 0
        self.episode_limit = int(self.max_rounds) if self.max_rounds and self.max_rounds > 0 else 1
        self.round_history: List[Dict[str, Any]] = []

        # Embedding utilities for optional reward calculation.
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        if not self.compute_reward:
            self.output_encoder = None
        else:
            try:
                self.output_encoder = OutputEncoder(
                    embedding_dim=getattr(self.reward_args, "commitment_embedding_dim", 1024),
                    model_name=getattr(self.reward_args, "commitment_embedding_model_name", "BAAI/bge-large-en-v1.5"),
                    device=torch.device("cpu"),
                )
            except Exception as e:
                self.output_encoder = None
                self.logger.warning(f"[Env] OutputEncoder init failed; AL/CC rewards will fallback to 0. Error: {e}")

        if self.use_dataset_episode:
            self.step_count = 0
            self.current_episode_samples = []
            self.episode_results = []

    def _get_field_value(self, sample: Dict[str, Any], field_name: Optional[str]) -> Optional[Any]:
        if not sample or field_name is None:
            return None
        if field_name in sample:
            return sample[field_name]
        lower_map = {str(k).lower(): k for k in sample.keys()}
        key = lower_map.get(str(field_name).lower())
        if key is not None:
            return sample[key]
        return None

    def _compose_question(self, sample: Dict[str, Any]) -> str:
        dataset_hint = str(self.dataset_path).lower()

        candidate_fields = []
        if self.question_field:
            candidate_fields.append(self.question_field)

        default_candidates = [
            "question", "Question",
            "question_text", "question_concat",
            "prompt", "Prompt",
            "body", "Body",
            "problem", "Problem",
            "instruction", "Instruction",
            "text", "Text",
        ]
        if "math" in dataset_hint:
            candidate_fields = ["question_concat", "problem", "Problem", "question", "Question", "body", "Body"] + candidate_fields
        if "svamp" in dataset_hint:
            candidate_fields = ["question_concat", "Question_Concat", "question", "Question", "body", "Body"] + candidate_fields

        candidate_fields.extend(default_candidates)

        seen = set()
        ordered_fields = []
        for field in candidate_fields:
            if field and field not in seen:
                seen.add(field)
                ordered_fields.append(field)

        for field in ordered_fields:
            value = self._get_field_value(sample, field)
            if value is not None and str(value).strip():
                return str(value)

        # Fallbacks for composite fields (Body + Question)
        body = self._get_field_value(sample, "body") or self._get_field_value(sample, "Body")
        question = self._get_field_value(sample, "question") or self._get_field_value(sample, "Question")
        parts = [str(p).strip() for p in [body, question] if p and str(p).strip()]
        if parts:
            return "\n".join(parts)

        equation = self._get_field_value(sample, "equation") or self._get_field_value(sample, "Equation")
        if equation and str(equation).strip():
            return str(equation)

        self.logger.warning(f"[Env] Question text empty. Sample keys: {list(sample.keys())}")
        return ""

    def get_ground_truth_text(self) -> Optional[str]:
        """
        answer > solution > final_answer > gold > gt > ground_truth
        """

        if hasattr(self, '_cached_ground_truth') and self._cached_ground_truth:
            return self._cached_ground_truth
            
        if getattr(self, "_cur_item", None) is None:
            if self.current_sample is not None:
                self._cur_item = self.current_sample
            else:
                self.logger.warning("No current sample available for ground truth extraction")
                return None
        
        dataset_hint = str(self.dataset_path).lower()

        # Dataset-specific logic
        if "math" in dataset_hint:
            solution_text = (
                self._get_field_value(self._cur_item, "solution")
                or self._get_field_value(self._cur_item, "Solution")
                or self._get_field_value(self._cur_item, "solution_text")
            )
            if solution_text:
                num = extract_numeric_answer(str(solution_text), dataset_hint=self.dataset_path)
                if num:
                    self._cached_ground_truth = num
                    return self._cached_ground_truth

        if "svamp" in dataset_hint:
            answer_val = (
                self._get_field_value(self._cur_item, "answer")
                or self._get_field_value(self._cur_item, "Answer")
                or self._get_field_value(self._cur_item, "output")
            )
            if answer_val is not None:
                self._cached_ground_truth = str(answer_val).strip()
                return self._cached_ground_truth

        # General fallback
        for field_name in ["answer", "Answer", "solution", "Solution", "final_answer", "final answer", "gold", "gt", "ground_truth"]:
            value = self._get_field_value(self._cur_item, field_name)
            if value is not None:
                value_str = str(value).strip()
                if value_str:
                    self._cached_ground_truth = value_str
                    return self._cached_ground_truth

        self.logger.warning(f"Could not extract ground truth from sample. Available keys: {list(self._cur_item.keys())}")
        return None

   
    def _next_index(self) -> Optional[int]:
        if self.is_streaming:
            return None
        if self.use_random_sampling:
            if not self._unused_indices:
                if not self.loop_dataset:
                    return None
                self._unused_indices = list(range(self.num_samples))
            j = random.randrange(len(self._unused_indices))
            return self._unused_indices.pop(j)
        else:
            if self.current_data_idx < 0:
                next_idx = 0
            else:
                next_idx = self.current_data_idx + 1
            if next_idx >= self.num_samples:
                if not self.loop_dataset:
                    return None
                next_idx = 0
            return next_idx

    def _get_next_sample(self) -> Optional[Dict]:
        if self.is_streaming:
            try:
                return next(self.dataset_iterator)
            except StopIteration:
                self.logger.info("Streaming dataset iterator exhausted.")
                return None

        if self.use_dataset_episode:
            nxt = self.current_data_idx + 1
            if nxt >= self.num_samples:
                return None
            self.current_data_idx = nxt
            return self.dataset_list[self.current_data_idx]

        next_idx = self._next_index()
        if next_idx is None:
            return None
        self.current_data_idx = next_idx
        return self.dataset_list[self.current_data_idx]

    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if self.use_dataset_episode:
            self.current_data_idx = -1
            self.step_count = 0
            self.current_episode_samples = []
            self.episode_results = []

        prev_idx = self.current_data_idx
        self.current_sample = self._get_next_sample()
        if self.current_sample is None:
            raise StopIteration("Dataset exhausted (or loop disabled).")

  
        self._cur_item = self.current_sample  
        self._cached_ground_truth = None  
        
    
        self.current_question = self._compose_question(self.current_sample)
        
        
        self.current_ground_truth_answer = self.get_ground_truth_text()
        
       
        if not self.current_ground_truth_answer:
            self.current_ground_truth_answer = str(self.current_sample.get(self.answer_field, ""))
            if self.current_ground_truth_answer:
                self._cached_ground_truth = self.current_ground_truth_answer
   
        if not self.current_ground_truth_answer:
            self.logger.error(f"Failed to extract ground truth for sample {self.current_data_idx}")
            self.logger.error(f"Sample keys: {list(self.current_sample.keys())}")
     

        self.episode_length = 0
        self.round_count = 0
        self.episode_count += 1
        self.round_history = []

        q_preview = self.current_question[:100] + "..." if len(self.current_question) > 100 else self.current_question
        self.logger.info(f"Episode {self.episode_count}: idx={self.current_data_idx} | {q_preview}")
        self.logger.debug(f"[Env.reset] prev_idx={prev_idx} -> new_idx={self.current_data_idx} | split={self.dataset_split}")

        observation = self.current_question
        info = {
            "sample": self.current_sample, 
            "sample_idx": self.current_data_idx,
            "ground_truth": self.current_ground_truth_answer  
        }
        return observation, info

  
    def step(self, action: Any, extra_info: Optional[Dict[str, Any]] = None) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        if self.current_sample is None:
            raise RuntimeError("step() called before reset() or after dataset exhaustion.")

        self.round_count += 1

        if isinstance(action, dict):
            llm_answer_str = str(action.get("answer", ""))
            if extra_info is None:
                extra_info = action
        else:
            llm_answer_str = str(action)

        extra_info = extra_info or {}

        self.logger.info("=" * 80)
        self.logger.info(f"🔍 QUESTION: {self.current_question}")
        self.logger.info("=" * 80)
        self.logger.info(f"📖 GROUND TRUTH: {self.current_ground_truth_answer}")
        self.logger.info("=" * 80)

       
        pred_num = extract_numeric_answer(llm_answer_str, dataset_hint=self.dataset_path)
        gt_num = extract_numeric_answer(self.current_ground_truth_answer, dataset_hint=self.dataset_path)

        is_correct = False
        if (pred_num is not None) and (gt_num is not None):
            try:
                is_correct = abs(float(pred_num) - float(gt_num)) < 1e-5
            except Exception:
                is_correct = (pred_num == gt_num)

        reward_ts = 1.0 if is_correct else 0.0
        reward_al = 0.0
        reward_cc = 0.0

        alpha_weights = [0.0, 1.0, 0.0]
        action_bonus = 1.0
        if self.compute_reward:
            reward_al = self._calculate_action_likelihood_reward(extra_info, pred_num)
            reward_cc = self._calculate_collaborative_contribution_reward(extra_info, pred_num, is_correct)
            alpha_weights = self._resolve_alpha_weights(extra_info)
            action_bonus = self._action_bonus_from_prompts(extra_info)

        total_reward = (
            alpha_weights[0] * reward_al
            + alpha_weights[1] * reward_ts
            + alpha_weights[2] * reward_cc
        )
        # When compute_reward=False, total_reward is just the correctness score.
        total_reward = float(max(0.0, min(1.0, total_reward * (0.8 + 0.2 * action_bonus))))

        if self.compute_reward:
            self.logger.info(f"   REWARD BREAKDOWN:")
            self.logger.info(f"   alpha = {alpha_weights}")
            self.logger.info(f"   TS (Task-Specific): {reward_ts:.3f} * {alpha_weights[1]:.2f} = {reward_ts * alpha_weights[1]:.3f}")
            self.logger.info(f"   AL (Action Likelihood): {reward_al:.3f} * {alpha_weights[0]:.2f} = {reward_al * alpha_weights[0]:.3f}")
            self.logger.info(f"   CC (Collaborative): {reward_cc:.3f} * {alpha_weights[2]:.2f} = {reward_cc * alpha_weights[2]:.3f}")
            self.logger.info(f"   ACTION BONUS (from [T,p]): {action_bonus:.3f}")
            self.logger.info(f"   TOTAL REWARD: {total_reward:.3f}")
            self.logger.info("=" * 80)

        terminated = (self.round_count >= self.episode_limit)
        truncated = False

        next_observation = self.current_question
        info = {
            "is_correct": is_correct,
            "reward_ts": reward_ts,
            "reward_al": reward_al,
            "reward_cc": reward_cc,
            "extracted_answer": llm_answer_str,
            "llm_answer": llm_answer_str,
            "pred_num": pred_num,
            "gt_num": gt_num,
            "ground_truth_answer": self.current_ground_truth_answer,
            "parsed_pred": pred_num,
            "parsed_gt": gt_num,
            "round_count": self.round_count,
            "max_rounds": self.episode_limit,
            "sample_idx": self.current_data_idx
        }

        self.round_history.append({
            "round": self.round_count,
            "answer": llm_answer_str,
            "reward": total_reward,
            "alpha": alpha_weights,
            "action_bonus": action_bonus,
            "prompt_embeddings": extra_info.get("prompt_embeddings") if isinstance(extra_info, dict) else None
        })
        return next_observation, float(total_reward), bool(terminated), bool(truncated), info

 
    def _text_embedding(self, text: str) -> Optional[torch.Tensor]:
        if self.output_encoder is None or not text:
            return None
        key = text.strip()
        if key in self._embedding_cache:
            return self._embedding_cache[key]
        try:
            emb = self.output_encoder.encode_output(key)
            self._embedding_cache[key] = emb
            return emb
        except Exception as e:
            self.logger.warning(f"[Env] embedding failed: {e}")
            return None

    def _cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
        if a is None or b is None:
            return 0.0
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        a = a.float()
        b = b.float()
        try:
            sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
            return float(sim.mean().item())
        except Exception:
            return 0.0

    def _calculate_action_likelihood_reward(self, extra_info: Dict[str, Any], pred_num: Optional[str]) -> float:
        """
        Theoretical AL: cosine similarity between executor outputs and coordinator commitment embedding.
        """
        try:
            agent_responses = extra_info.get('agent_responses', [])
            commit_text = extra_info.get('commitment_text', '')
            if not agent_responses or not commit_text:
                return 0.0

            commit_emb = self._text_embedding(commit_text)
            if commit_emb is None:
                return 0.0

            sims = []
            for r in agent_responses:
                emb = self._text_embedding(str(r))
                if emb is None:
                    continue
                sims.append(self._cosine(emb, commit_emb))

            if not sims:
                return 0.0

            mean_sim = sum(sims) / len(sims)
            # Map cosine [-1,1] -> [0,1]
            return float(max(0.0, min(1.0, 0.5 * (mean_sim + 1.0))))
        except Exception:
            return 0.0

    def _calculate_collaborative_contribution_reward(self, extra_info: Dict[str, Any],
                                                     pred_num: Optional[str],
                                                     is_correct: bool) -> float:
        """
        Approximate quality(u_i, {u_j}) with two terms:
          - fidelity: similarity to final commitment (stay useful)
          - novelty: 1 - mean similarity to other executors (encourage diverse contributions)
        """
        try:
            agent_responses = extra_info.get('agent_responses', [])
            commit_text = extra_info.get('commitment_text', '')
            if not agent_responses or not commit_text:
                return 0.0

            commit_emb = self._text_embedding(commit_text)
            if commit_emb is None:
                return 0.0

            resp_embs = []
            for r in agent_responses:
                emb = self._text_embedding(str(r))
                if emb is not None:
                    resp_embs.append(emb)

            if not resp_embs:
                return 0.0

            fidelities = [self._cosine(e, commit_emb) for e in resp_embs]
            fidelities = [(f + 1.0) / 2.0 for f in fidelities]  # [0,1]

            # Novelty: 1 - avg peer similarity
            peer_sims = []
            for i, e_i in enumerate(resp_embs):
                sims = []
                for j, e_j in enumerate(resp_embs):
                    if i == j:
                        continue
                    sims.append(self._cosine(e_i, e_j))
                if sims:
                    peer_sims.append(1.0 - max(0.0, min(1.0, 0.5 * (sum(sims) / len(sims) + 1.0))))
            novelty = sum(peer_sims) / len(peer_sims) if peer_sims else 0.0

            quality = 0.7 * (sum(fidelities) / len(fidelities)) + 0.3 * novelty
            return float(max(0.0, min(1.0, quality)))
        except Exception:
            return 0.0

    def _resolve_alpha_weights(self, extra_info: Optional[Dict[str, Any]]) -> List[float]:
        weights = None
        if isinstance(extra_info, dict) and "alpha_weights" in extra_info and extra_info["alpha_weights"] is not None:
            weights = list(extra_info["alpha_weights"])
        if not weights:
            weights = list(getattr(self.reward_args, "initial_weights", []))
        if not weights:
            weights = self.alpha_default
        if len(weights) != 3:
            weights = [0.34, 0.33, 0.33]
        weights = [max(0.0, float(w)) for w in weights]
        s = sum(weights)
        if s <= 0:
            return [1/3, 1/3, 1/3]
        return [w / s for w in weights]

    def _action_bonus_from_prompts(self, extra_info: Optional[Dict[str, Any]]) -> float:
        """
        Use prompt embeddings [T, p] as direct actions: encourage stable, non-extreme choices.
        Returns a factor in [0, 1].
        """
        if not isinstance(extra_info, dict):
            return 0.5
        pe = extra_info.get("prompt_embeddings")
        if pe is None:
            return 0.5
        try:
            arr = np.array(pe, dtype=float).reshape(-1, 2)
            if arr.size == 0:
                return 0.5
            temp_span = arr[:, 0].max() - arr[:, 0].min() if arr.shape[0] > 1 else 0.0
            p_span = arr[:, 1].max() - arr[:, 1].min() if arr.shape[0] > 1 else 0.0
            span_norm = max(1e-6, temp_span + p_span)
            dispersion = float(span_norm / 4.0)  # scale down
            base = 1.0 - min(1.0, dispersion)
            clipped_mean = float(np.clip(arr.mean(axis=0), 0.0, 1.5).mean()) / 1.5
            return float(max(0.0, min(1.0, 0.5 * base + 0.5 * clipped_mean)))
        except Exception:
            return 0.5

    def get_env_info(self) -> Dict[str, Any]:
        return {
            "episode_limit": int(self.episode_limit),
            "n_actions": int(getattr(self, "n_actions", 2)),
            "obs_shape": (self.max_question_length,),
            "state_shape": (1,),
        }

    def render(self, mode='human'):
        if mode == 'human':
            if self.current_sample:
                print("-" * 30)
                print(f"[idx={self.current_data_idx}] Question: {self.current_question}")
                print(f"Ground Truth: {self.current_ground_truth_answer}")
                print("-" * 30)
            else:
                print("No current sample to render. Call reset() first.")

    def close(self):
        self.logger.info("Closing HuggingFaceDatasetEnv.")
