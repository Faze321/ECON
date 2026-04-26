
from typing import Any, Dict, Optional, Tuple, List

from types import SimpleNamespace

import torch
import numpy as np

from components.episode_buffer import EpisodeBatch

from envs.huggingface_dataset_env import HuggingFaceDatasetEnv


class EpisodeRunner:

    def __init__(self, args: SimpleNamespace, logger):
        self.args = args
        self.logger = logger
        self.alpha_provider = None

      
        use_cuda = getattr(args.system, "use_cuda", False) and torch.cuda.is_available()
        devnum = getattr(args.system, "device_num", 0)
        self.device = torch.device(f"cuda:{devnum}" if use_cuda and torch.cuda.is_available() else "cpu")

        
        env_kwargs = vars(args.env_args) if hasattr(args, "env_args") else {}
        self.env = HuggingFaceDatasetEnv(logger=self.logger, **env_kwargs)


        self.scheme = None
        self.groups = None
        self.preprocess = None
        self.mac = None

    
        self._last_trace: Optional[Dict[str, Any]] = None

        
        self._episode_idx = 0
        self._t_env = 0
        self.memory_dim = int(getattr(args, "memory_dim", getattr(args, "belief_dim", 128) + 9))


    def setup(self, scheme: Dict, groups: Dict, preprocess: Any, mac: Any):
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.mac = mac

    def set_alpha_provider(self, provider: Any):
        """
        provider: object exposing get_alpha_weights_list() -> List[float]
        """
        self.alpha_provider = provider

    def get_env_info(self) -> Dict[str, Any]:

        n_actions = int(getattr(self.args, "n_actions", 2))
        max_q_len = int(getattr(self.args.env_args, "max_question_length", 1024)) if hasattr(self.args, "env_args") else 1024
        return {
            "episode_limit": int(self.env.episode_limit) if hasattr(self.env, "episode_limit") else 1,
            "n_actions": n_actions,
            "obs_shape": (max_q_len,),
            "state_shape": (1,),
        }

    def get_last_trace(self) -> Optional[Dict[str, Any]]:
        return self._last_trace



    def run(self, test_mode: bool = False) -> EpisodeBatch:

        assert self.scheme is not None and self.groups is not None and self.mac is not None, "Runner not set up"

        n_agents = int(getattr(self.args, "n_agents", 3))
        n_actions = int(getattr(self.args, "n_actions", 2))
        obs_len = int(self.get_env_info()["obs_shape"][0])
        belief_dim = int(getattr(self.args, "belief_dim", 128))
        commit_dim = int(getattr(self.args, "commitment_embedding_dim", 1024))
        episode_limit = int(getattr(self.env, "episode_limit", 1))
        max_seq_len = episode_limit + 1  # final bootstrap slot

        obs_text, info = self.env.reset()
        gt_text = self.env.get_ground_truth_text() if hasattr(self.env, "get_ground_truth_text") else None

        ep_batch = EpisodeBatch(
            scheme=self.scheme,
            groups=self.groups,
            batch_size=1,
            max_seq_length=max_seq_len,
            device=self.device
        )

        state_vec = torch.zeros(self.scheme["state"]["vshape"], dtype=torch.float32, device=self.device)  # (1,)
        avail_actions_agents = torch.ones((n_agents, n_actions), dtype=torch.int64, device=self.device)    # (N,A)
        obs_tokens_agents = torch.zeros((n_agents, obs_len), dtype=torch.long, device=self.device)

        agent_memory = torch.zeros((n_agents, self.memory_dim), dtype=torch.float32, device=self.device)
        bne_enabled = getattr(getattr(self.args, "bne", object()), "enabled", False)
        total_reward = 0.0
        steps_run = 0
        trajectories = []
        last_env_info = {}
        mac_info: Dict[str, Any] = {}
        bne_graph_data = None

        for t in range(episode_limit):
            steps_run = t + 1
            # Coordinator strategy is part of each agent's observation (partial observability still holds via memory)
            strategy_text = self.mac._get_strategy_and_format(obs_text)
            if not strategy_text or not strategy_text.strip():
                strategy_text = "Break down the problem into steps and solve each part."
            obs_with_strategy = f"{obs_text}\n\nStrategy:\n{strategy_text}"

            obs_tokens_1d = self.mac.preprocess_observation(obs_with_strategy, max_length=obs_len)  # (L,)
            if obs_tokens_1d.dim() != 1:
                obs_tokens_1d = obs_tokens_1d.view(-1)
            obs_tokens_agents = obs_tokens_1d.unsqueeze(0).repeat(n_agents, 1).to(self.device)  # (N, L), dtype long

            ep_batch.update(
                {
                    "obs": obs_tokens_agents.long(),
                    "state": state_vec,
                    "avail_actions": avail_actions_agents,
                    "agent_memory": agent_memory,
                },
                ts=t
            )

            bne_data = None

            if bne_enabled and not test_mode:
                try:
                    bne_data_raw = self.mac.run_bne_refinement(
                        mode="train",
                        obs_tokens=obs_tokens_agents,
                        obs_text=obs_text,
                        strategy_text=strategy_text,
                        agent_memory=agent_memory
                    )

                    bne_graph_data = bne_data_raw
                    bne_data = {
                        "exec_outputs_1": bne_data_raw["outputs_history"][-1],
                        "commitment_final": bne_data_raw["commitments"][-1],
                        "commitment_emb_final": bne_data_raw["commitment_emb_final"],
                        "beliefs": bne_data_raw["beliefs"],
                        "e_init": bne_data_raw["e_init"],
                        "e_refined": bne_data_raw["e_final"],
                        "commitment_emb_0": bne_data_raw["commitment_emb_0"],
                        "output_embs_0": bne_data_raw["output_embs_0"],
                        "output_embs_1": bne_data_raw["output_embs_final"],
                        "group_repr": bne_data_raw["group_repr"],
                        "e_path": bne_data_raw["e_path"],
                        "commitments": bne_data_raw["commitments"],
                        "iterations_run": bne_data_raw["iterations_run"],
                        "early_stop": bne_data_raw["early_stop"],
                    }
                except Exception as e:
                    self.logger.warning(f"[Runner] BNE flow failed at t={t}: {e}, fallback to normal flow")
                    bne_data = None

            if bne_data is not None:
                avail_actions = ep_batch["avail_actions"][:, t]
                agent_outputs, agent_info = self.mac.forward(ep_batch, t=t, test_mode=test_mode, agent_memory=agent_memory)
                chosen_actions = self.mac.action_selector.select_action(
                    agent_outputs[slice(None)], avail_actions[slice(None)], self._t_env, test_mode=test_mode
                )
                mac_info = agent_info.copy()
                mac_info.update({
                    "executor_responses": bne_data["exec_outputs_1"],
                    "commitment": bne_data["commitment_final"],
                    "commitment_text": bne_data["commitment_final"],
                    "commitment_embedding": bne_data["commitment_emb_final"],
                    "strategy": strategy_text,
                    "format": "",
                    "bne_beliefs": bne_data["beliefs"],
                    "bne_e_init": bne_data["e_init"],
                    "bne_e_refined": bne_data["e_refined"],
                    "bne_commitment_emb_0": bne_data["commitment_emb_0"],
                    "bne_output_embs_0": bne_data["output_embs_0"],
                    "bne_output_embs_1": bne_data["output_embs_1"],
                    "bne_group_repr": bne_data["group_repr"],
                    "bne_e_path": bne_data["e_path"],
                    "bne_commitments": bne_data["commitments"],
                    "bne_iterations_run": bne_data["iterations_run"],
                    "bne_early_stop": bne_data["early_stop"],
                })
            else:
                chosen_actions, mac_info = self.mac.select_actions(
                    ep_batch, t_ep=t, t_env=self._t_env, raw_observation_text=obs_text, test_mode=test_mode,
                    agent_memory=agent_memory, strategy_override=strategy_text
                )
                if isinstance(mac_info, dict):
                    mac_info["strategy"] = strategy_text

            if chosen_actions.dim() == 2 and chosen_actions.size(0) == 1:
                actions_for_batch = chosen_actions.squeeze(0).unsqueeze(-1).long().to(self.device)  # (N,1)
            elif chosen_actions.dim() == 1:
                actions_for_batch = chosen_actions.unsqueeze(-1).long().to(self.device)  # (N,1)
            else:
                actions_for_batch = chosen_actions.long().to(self.device).unsqueeze(-1)

            commitment_text = mac_info.get("commitment_text", None) or mac_info.get("commitment", "")
            agent_responses = mac_info.get("executor_responses", mac_info.get("llm_responses", [])) or []
            final_answer_str = commitment_text if isinstance(commitment_text, str) else str(commitment_text)

            prompt_e = mac_info.get("prompt_embeddings", None)
            if prompt_e is None:
                prompt_agents = torch.zeros((n_agents, 2), dtype=torch.float32, device=self.device)
            else:
                pe = self._to_agent_2d(prompt_e)
                if pe.dim() == 1:
                    pe = pe.unsqueeze(-1)
                prompt_agents = pe.float().to(self.device).detach()

            belief_states = mac_info.get("belief_states", None)
            if belief_states is None:
                b_agents = torch.zeros((n_agents, belief_dim), dtype=torch.float32, device=self.device)
            else:
                b_agents = self._to_agent_2d(belief_states).float().to(self.device).detach()

            group_repr = mac_info.get("group_repr", None)
            if group_repr is None:
                g_vec = torch.zeros((belief_dim,), dtype=torch.float32, device=self.device)
            else:
                gv = self._to_1d(group_repr).float().to(self.device)
                if gv.numel() != belief_dim:
                    gv = torch.nn.functional.pad(gv, (0, max(0, belief_dim - gv.numel())))[:belief_dim]
                g_vec = gv.detach()

            commit_emb = mac_info.get("commitment_embedding", None)
            if commit_emb is None or (isinstance(commit_emb, torch.Tensor) and commit_emb.numel() == 0):
                c_vec = torch.zeros((commit_dim,), dtype=torch.float32, device=self.device)
            elif isinstance(commit_emb, torch.Tensor):
                c_vec = self._to_1d(commit_emb).float().to(self.device)
                if c_vec.numel() != commit_dim:
                    c_vec = torch.nn.functional.pad(c_vec, (0, max(0, commit_dim - c_vec.numel())))[:commit_dim]
                c_vec = c_vec.detach()
            else:
                c_vec = torch.tensor(commit_emb, dtype=torch.float32, device=self.device).view(-1)
                if c_vec.numel() != commit_dim:
                    c_vec = torch.nn.functional.pad(c_vec, (0, max(0, commit_dim - c_vec.numel())))[:commit_dim]
                c_vec = c_vec.detach()

            q_vals = mac_info.get("q_values", None)
            if q_vals is None:
                q_vals_agents = torch.zeros((n_agents, 1), dtype=torch.float32, device=self.device)
            else:
                q = self._to_agent_2d(q_vals)
                if q.dim() == 1:
                    q = q.unsqueeze(-1)
                q_vals_agents = q.float().to(self.device).detach()

            extra_for_env = {
                "agent_responses": agent_responses,
                "commitment_text": final_answer_str,
                "prompt_embeddings": prompt_agents.detach().cpu().tolist(),
                "chosen_actions": actions_for_batch.squeeze(-1).detach().cpu().tolist(),
                "alpha_weights": self._alpha_weights(),
            }
            next_obs_text, reward, terminated, truncated, env_info = self.env.step(
                {"answer": final_answer_str}, extra_for_env
            )
            last_env_info = env_info
            reward = float(reward)
            total_reward += reward
            is_correct = bool(env_info.get("is_correct", False))

            r_al = float(env_info.get("reward_al", 0.0))
            r_ts = float(env_info.get("reward_ts", 0.0))
            r_cc = float(env_info.get("reward_cc", 0.0))
            reward_al_agents = torch.full((n_agents, 1), r_al, dtype=torch.float32, device=self.device)
            reward_ts_agents = torch.full((n_agents, 1), r_ts, dtype=torch.float32, device=self.device)
            reward_cc_agents = torch.full((n_agents, 1), r_cc, dtype=torch.float32, device=self.device)

            bne_e_init_agents = torch.zeros((n_agents, 2), dtype=torch.float32, device=self.device)
            bne_e_refined_agents = torch.zeros((n_agents, 2), dtype=torch.float32, device=self.device)
            bne_commitment_emb_0_vec = torch.zeros((commit_dim,), dtype=torch.float32, device=self.device)
            bne_output_emb_0_agents = torch.zeros((n_agents, commit_dim), dtype=torch.float32, device=self.device)
            bne_output_emb_1_agents = torch.zeros((n_agents, commit_dim), dtype=torch.float32, device=self.device)
            bne_group_repr_vec = torch.zeros((belief_dim,), dtype=torch.float32, device=self.device)

            if bne_data is not None:
                if "bne_e_init" in mac_info:
                    ei = self._to_agent_2d(mac_info["bne_e_init"]).float().to(self.device).detach()
                    if ei.shape == (n_agents, 2):
                        bne_e_init_agents = ei
                if "bne_e_refined" in mac_info:
                    er = self._to_agent_2d(mac_info["bne_e_refined"]).float().to(self.device).detach()
                    if er.shape == (n_agents, 2):
                        bne_e_refined_agents = er
                if "bne_commitment_emb_0" in mac_info:
                    ce0 = self._to_1d(mac_info["bne_commitment_emb_0"]).float().to(self.device)
                    if ce0.numel() == commit_dim:
                        bne_commitment_emb_0_vec = ce0.detach()
                if "bne_output_embs_0" in mac_info:
                    oe0 = self._to_agent_2d(mac_info["bne_output_embs_0"]).float().to(self.device).detach()
                    if oe0.shape == (n_agents, commit_dim):
                        bne_output_emb_0_agents = oe0
                if "bne_output_embs_1" in mac_info:
                    oe1 = self._to_agent_2d(mac_info["bne_output_embs_1"]).float().to(self.device).detach()
                    if oe1.shape == (n_agents, commit_dim):
                        bne_output_emb_1_agents = oe1
                if "bne_group_repr" in mac_info:
                    bgr = self._to_1d(mac_info["bne_group_repr"]).float().to(self.device)
                    if bgr.numel() == belief_dim:
                        bne_group_repr_vec = bgr.detach()

            ep_batch.update(
                {
                    "actions": actions_for_batch,
                    "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                    "terminated": torch.tensor([1 if terminated else 0], dtype=torch.uint8, device=self.device),
                    "filled": torch.tensor([1], dtype=torch.long, device=self.device),
                    "is_correct": torch.tensor([1.0 if is_correct else 0.0], dtype=torch.float32, device=self.device),
                    "q_values": q_vals_agents,
                    "prompt_embeddings": prompt_agents,
                    "belief_states": b_agents,
                    "group_representation": g_vec,
                    "commitment_embedding": c_vec,
                    "bne_e_init": bne_e_init_agents,
                    "bne_e_refined": bne_e_refined_agents,
                    "bne_commitment_emb_0": bne_commitment_emb_0_vec,
                    "bne_output_emb_0": bne_output_emb_0_agents,
                    "bne_output_emb_1": bne_output_emb_1_agents,
                    "bne_group_repr": bne_group_repr_vec,
                    "reward_al": reward_al_agents,
                    "reward_ts": reward_ts_agents,
                    "reward_cc": reward_cc_agents,
                },
                ts=t
            )

            hidden_states = mac_info.get("hidden_states", None)
            if isinstance(hidden_states, torch.Tensor):
                hidden_agents = self._to_agent_2d(hidden_states).float().to(self.device)
            else:
                hidden_agents = None

            agent_memory = self._build_agent_memory(
                b_agents,
                prompt_agents,
                r_al,
                r_ts,
                r_cc,
                hidden_states=hidden_agents,
                commitment_emb=c_vec,
                agent_output_embs=bne_output_emb_1_agents if bne_data is not None else None
            )

            trajectories.append({
                "t": t,
                "question": obs_text,
                "commitment": final_answer_str,
                "executor_outputs": agent_responses,
                "reward": reward,
                "r_al": r_al,
                "r_ts": r_ts,
                "r_cc": r_cc,
            })

            obs_text = next_obs_text
            self._t_env += 1
            if terminated or truncated:
                break

        # Bootstrap slot
        ep_batch.update(
            {
                "obs": obs_tokens_agents.long(),  # reuse last obs tokens
                "state": state_vec,
                "avail_actions": avail_actions_agents,
                "agent_memory": agent_memory,
                "filled": torch.tensor([0], dtype=torch.long, device=self.device),
                "terminated": torch.tensor([1], dtype=torch.uint8, device=self.device),
            },
            ts=min(steps_run, max_seq_len - 1)
        )

        question_text = trajectories[-1]["question"] if trajectories else obs_text
        final_commitment = trajectories[-1]["commitment"] if trajectories else ""
        executor_outputs = trajectories[-1]["executor_outputs"] if trajectories else []
        pred_answer = last_env_info.get("extracted_answer", "") if last_env_info else ""
        if not pred_answer and last_env_info:
            pred_answer = last_env_info.get("pred_num", "")

        if "ground_truth" in info:
            gt_text = info.get("ground_truth")
        if not gt_text and last_env_info:
            gt_text = last_env_info.get("ground_truth")
        if not gt_text:
            gt_text = self.env.get_ground_truth_text()
        if not gt_text and hasattr(self.env, 'current_ground_truth_answer'):
            gt_text = self.env.current_ground_truth_answer

        strategy_text = mac_info.get("strategy", "") if isinstance(mac_info, dict) else ""
        format_text = mac_info.get("format", "") if isinstance(mac_info, dict) else ""
        r_al_val = float(last_env_info.get("reward_al", 0.0)) if last_env_info else 0.0
        r_ts_val = float(last_env_info.get("reward_ts", 0.0)) if last_env_info else 0.0
        r_cc_val = float(last_env_info.get("reward_cc", 0.0)) if last_env_info else 0.0
        is_correct_flag = bool(last_env_info.get("is_correct", False)) if last_env_info else False

        self._last_trace = {
            "episode": int(self._episode_idx),
            "question": str(question_text) if question_text else "",
            "strategy": str(strategy_text) if strategy_text else "",
            "format": str(format_text) if format_text else "",
            "executor_outputs": [str(x) for x in executor_outputs] if executor_outputs else [],
            "final_commitment": str(final_commitment) if final_commitment else "",
            "ground_truth": str(gt_text) if gt_text else "",
            "pred_answer": str(pred_answer) if pred_answer else "",
            "is_correct": is_correct_flag,
            "reward": float(total_reward),
            "reward_al": r_al_val,
            "reward_ts": r_ts_val,
            "reward_cc": r_cc_val,
            "t_env": int(self._t_env),
            "test_mode": bool(test_mode),
            "rounds": steps_run,
            "trajectory": trajectories,
        }

        if isinstance(mac_info, dict) and "commitment_metadata" in mac_info:
            self._last_trace["commitment_metadata"] = mac_info["commitment_metadata"]

        if bne_data is not None:
            e_init_np = bne_data["e_init"].detach().cpu().numpy()
            e_refined_np = bne_data["e_refined"].detach().cpu().numpy()
            e_delta_np = e_refined_np - e_init_np
            delta_magnitude = float(np.linalg.norm(e_delta_np))
            max_agent_delta = float(np.max(np.linalg.norm(e_delta_np, axis=1)))
            e_path_serializable = []
            if "bne_e_path" in mac_info:
                for agent_path in mac_info["bne_e_path"]:
                    agent_path_list = [tensor.detach().cpu().tolist() for tensor in agent_path]
                    e_path_serializable.append(agent_path_list)

            self._last_trace["bne"] = {
                "commitment_0": bne_data.get("commitment_0", bne_data.get("commitments", [""])[0]),
                "commitment_final": bne_data.get("commitment_final", ""),
                "executor_outputs_0": bne_data.get("exec_outputs_0", []),
                "executor_outputs_1": bne_data.get("exec_outputs_1", []),
                "e_init": e_init_np.tolist(),
                "e_refined": e_refined_np.tolist(),
                "e_delta": e_delta_np.tolist(),
                "delta_magnitude": delta_magnitude,
                "max_agent_delta": max_agent_delta,
                "e_path": e_path_serializable,
                "commitments": bne_data.get("commitments", []),
                "iterations_run": bne_data.get("iterations_run", 1),
                "early_stop": bne_data.get("early_stop", False),
            }

        # Attach autograd-aware BNE graph for the learner (only when available)
        setattr(ep_batch, "bne_graph", bne_graph_data if bne_graph_data is not None else None)

        if test_mode and isinstance(mac_info, dict) and "bne_iteration_history" in mac_info:
            iteration_history = mac_info.get("bne_iteration_history", [])
            n_iterations = mac_info.get("bne_n_iterations", 1)
            converged = mac_info.get("bne_converged", False)
            self._last_trace["bne_infer"] = {
                "n_iterations": n_iterations,
                "converged": converged,
                "iterations": [
                    {
                        "iter": h["iter"],
                        "commitment": h["commitment"],
                        "delta_magnitude": h["delta_magnitude"],
                        "e": h["e"].tolist() if hasattr(h["e"], "tolist") else h["e"],
                        "outputs": h.get("outputs", [])
                    }
                    for h in iteration_history
                ]
            }

        if not self._last_trace["question"]:
            self.logger.warning(f"Episode {self._episode_idx}: question is empty!")
        if not self._last_trace["ground_truth"]:
            self.logger.warning(f"Episode {self._episode_idx}: ground_truth is empty!")
        if not self._last_trace["final_commitment"]:
            self.logger.warning(f"Episode {self._episode_idx}: final_commitment is empty!")

        self._episode_idx += 1
        return ep_batch


    def _alpha_from_args(self) -> Optional[list]:
        # Prefer live weights from learner if provided
        if self.alpha_provider is not None and hasattr(self.alpha_provider, "get_alpha_weights_list"):
            w = self.alpha_provider.get_alpha_weights_list()
            if w:
                return w
        try:
            reward_cfg = getattr(self.args, "reward", None)
            if reward_cfg and hasattr(reward_cfg, "initial_weights"):
                weights = list(getattr(reward_cfg, "initial_weights"))
                if len(weights) == 3:
                    s = sum([max(0.0, float(w)) for w in weights])
                    if s > 0:
                        return [float(max(0.0, w)) / s for w in weights]
            return None
        except Exception:
            return None

    def _alpha_weights(self) -> Optional[list]:
        return self._alpha_from_args()

    def _build_agent_memory(self, belief_states: torch.Tensor, prompt_embeddings: torch.Tensor,
                            r_al: float, r_ts: float, r_cc: float,
                            hidden_states: Optional[torch.Tensor] = None,
                            commitment_emb: Optional[torch.Tensor] = None,
                            agent_output_embs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        τ_i^t = GRU hidden (preferred) or concat(belief, [T,p], reward components, commitment, own output)
        Only own output embedding is included; no peer outputs are injected to preserve partial observability.
        """
        N = belief_states.size(0)
        if hidden_states is not None:
            mem = hidden_states.detach()
        else:
            belief_states = belief_states.detach()
            prompt_embeddings = prompt_embeddings.detach()
            reward_vec = torch.tensor([r_al, r_ts, r_cc], dtype=torch.float32, device=self.device).view(1, 3).repeat(N, 1)

            def _summarize(vec: torch.Tensor) -> torch.Tensor:
                if vec.dim() == 1:
                    vec = vec.unsqueeze(0)
                return torch.stack([vec.mean(dim=-1), vec.norm(dim=-1)], dim=-1)

            commit_tile = None
            if commitment_emb is not None:
                ce = commitment_emb.detach()
                if ce.dim() == 1:
                    ce = ce.unsqueeze(0)
                if ce.size(0) == 1:
                    ce = ce.repeat(N, 1)
                commit_tile = _summarize(ce)

            outputs_self = None
            if agent_output_embs is not None:
                ao = agent_output_embs.detach()
                if ao.dim() == 1:
                    ao = ao.unsqueeze(0)
                if ao.size(0) != N and ao.size(0) == 1:
                    ao = ao.repeat(N, 1)
                outputs_self = _summarize(ao)

            pieces = [belief_states, prompt_embeddings, reward_vec]
            if commit_tile is not None:
                pieces.append(commit_tile)
            if outputs_self is not None:
                pieces.append(outputs_self)
            mem = torch.cat(pieces, dim=-1)
        if mem.size(-1) < self.memory_dim:
            pad = torch.zeros((N, self.memory_dim - mem.size(-1)), device=self.device)
            mem = torch.cat([mem, pad], dim=-1)
        elif mem.size(-1) > self.memory_dim:
            mem = mem[..., :self.memory_dim]
        return mem



    def _to_agent_2d(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        t = x
      
        while t.dim() >= 3 and t.size(0) == 1:
            t = t.squeeze(0)
        if t.dim() >= 2 and t.size(0) == 1 and t.size(1) != 1:
           
            t = t.squeeze(0)
      
        return t

    def _to_1d(self, x: torch.Tensor) -> torch.Tensor:
     
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        t = x
        while t.dim() > 1 and t.size(0) == 1:
            t = t.squeeze(0)
        if t.dim() > 1:
        
            t = t.view(-1)
        return t

    def _extract_number_from_boxed(self, text: str) -> Optional[str]:
     
        import re
        if not isinstance(text, str):
            return None
        m = re.search(r"\\boxed\{([^}]*)\}", text)
        if m:
            return self._normalize_num(m.group(1))
        m = re.search(r"####\s*([+-]?\d+(?:\.\d+)?)", text)
        if m:
            return self._normalize_num(m.group(1))
        nums = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
        return self._normalize_num(nums[-1]) if nums else None

    def _normalize_num(self, s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        s = s.replace(",", "").strip()
        if "." in s:
            try:
                v = float(s)
                if abs(v - int(v)) < 1e-9:
                    return str(int(v))
                return s
            except Exception:
                return s
        return s
