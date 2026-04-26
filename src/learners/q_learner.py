# src/learners/q_learner.py
import os
import copy
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from modules.mixer.mix_llm import LLMQMixer
from utils.logging import get_logger

# LOG = get_logger()


def _safe_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-9) -> torch.Tensor:
    if mask is None:
        return x.mean()
    num = (x * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den


class ECONLearner:


    def __init__(self, mac, scheme: Dict, logger, args: Any):
        self.mac = mac
        self.args = args
        self.logger = logger
        self.device = getattr(args, "device", torch.device("cpu"))
        self.train_steps = 0

        # Target update config
        self.target_update_tau = float(getattr(args, "target_update_tau", 0.01))
        self.target_update_interval = int(getattr(args, "target_update_interval", 8))

        # Mixer
        self.mixer = LLMQMixer(args).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        for p in self.target_mixer.parameters():
            p.requires_grad = False

    
        self.bne_enabled = getattr(getattr(args, "bne", object()), "enabled", False)

        if self.bne_enabled:
            # BNE Mode: K independent policy networks + refine module
            self.logger.info("[Learner] BNE mode: using policy_networks + refine_module")

            params = []
            n_agents = int(getattr(args, "n_agents", 3))

            # Policy networks (K agents)
            for i in range(n_agents):
                params += list(self.mac.policy_networks[i].parameters())

            # Refine module
            params += list(self.mac.refine_module.parameters())

            # Belief encoder
            params += list(self.mac.belief_encoder.parameters())

            # Mixer
            params += list(self.mixer.parameters())

            # Target networks
            self.target_policy_networks = nn.ModuleList(
                [copy.deepcopy(net).to(self.device) for net in self.mac.policy_networks]
            )
            self.target_belief_encoder = copy.deepcopy(self.mac.belief_encoder).to(self.device)
            for net in self.target_policy_networks:
                for p in net.parameters():
                    p.requires_grad = False
            for p in self.target_belief_encoder.parameters():
                p.requires_grad = False

            # Single optimizer for all BNE components
            lr = float(getattr(args, "lr", 1e-3))
            wd = float(getattr(args, "weight_decay", 0.0))

            self.optimizer = optim.Adam(params, lr=lr, weight_decay=wd)

            self.logger.info(f"[Learner] BNE optimizer: {sum(p.numel() for p in params)} params, lr={lr}")

            # Legacy optimizers set to None
            self.opt_belief = None
            self.opt_encoder = None
            self.opt_mixer = None

        else:
            # Legacy Mode: single shared belief network
            self.logger.info("[Learner] Legacy mode: using shared belief_network")

            self.params_belief = list(self.mac.agent.belief_network.parameters())
            self.params_encoder = list(self.mac.belief_encoder.parameters())
            self.params_mixer = list(self.mixer.parameters())

            lr_belief = float(getattr(args, "belief_net_lr", getattr(args, "lr", 1e-3)))
            lr_encoder = float(getattr(args, "encoder_lr", getattr(args, "lr", 1e-3)))
            lr_mixer = float(getattr(args, "mixer_lr", getattr(args, "lr", 1e-3)))
            wd = float(getattr(args, "weight_decay", 0.0))

            self.opt_belief = optim.Adam(self.params_belief, lr=lr_belief, weight_decay=wd)
            self.opt_encoder = optim.Adam(self.params_encoder, lr=lr_encoder, weight_decay=wd)
            self.opt_mixer = optim.Adam(self.params_mixer, lr=lr_mixer, weight_decay=wd)

            # BNE optimizer set to None
            self.optimizer = None

        # Loss
        self.w_belief = float(getattr(args.loss, "belief_weight", 0.1))
        self.w_encoder = float(getattr(args.loss, "encoder_weight", 0.1))
        self.w_mixer = float(getattr(args.loss, "mixing_weight", 0.1))

   
        self.gamma = float(getattr(getattr(args, "train", object()), "gamma", getattr(args, "gamma", 0.99)))

        
        self.reward_weighting = bool(getattr(getattr(args, "loss", object()), "reward_weighting", False))
        self.reward_weighting_alpha = float(getattr(getattr(args, "loss", object()), "reward_weighting_alpha", 1.0))

        # Reward combination weights (alpha_1, alpha_2, alpha_3)
        reward_cfg = getattr(args, "reward", SimpleNamespace())
        init_alpha = getattr(reward_cfg, "initial_weights", [0.4, 0.4, 0.2])
        if len(init_alpha) != 3:
            init_alpha = [0.4, 0.4, 0.2]
        init_alpha = torch.tensor(init_alpha, dtype=torch.float32, device=self.device).clamp_min(1e-6)
        self.alpha_logits = nn.Parameter(init_alpha.log())
        self.dynamic_alpha = bool(getattr(reward_cfg, "dynamic_alpha_update", False))
        self.alpha_lr = float(getattr(reward_cfg, "eta_alpha", 0.001))
        self.opt_alpha = optim.Adam([self.alpha_logits], lr=self.alpha_lr) if self.dynamic_alpha else None

        self.max_grad_norm = 10.0

    # ---------------- target helpers ----------------
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - tau).add_(sp.data * tau)

    def _maybe_soft_update_targets(self):
        if (self.train_steps % max(1, self.target_update_interval)) != 0:
            return
        self._soft_update(self.target_mixer, self.mixer, self.target_update_tau)
        if self.bne_enabled:
            for tgt, src in zip(self.target_policy_networks, self.mac.policy_networks):
                self._soft_update(tgt, src, self.target_update_tau)
            self._soft_update(self.target_belief_encoder, self.mac.belief_encoder, self.target_update_tau)


    def _get_alpha_weights(self) -> torch.Tensor:
        """Softmaxed reward weights α (shape: (3,))."""
        return torch.softmax(self.alpha_logits, dim=0)

    def get_alpha_weights_list(self) -> Optional[list]:
        try:
            return [float(x) for x in self._get_alpha_weights().detach().cpu().tolist()]
        except Exception:
            return None

    def _combine_reward(self, tensors: Dict[str, torch.Tensor], alpha: torch.Tensor, detach_alpha: bool = True) -> Optional[torch.Tensor]:
        """
        Combine reward_al/reward_ts/reward_cc using alpha weights.
        Returns (B,1,1) or None if components missing.
        """
        r_al = tensors.get("reward_al")
        r_ts = tensors.get("reward_ts")
        r_cc = tensors.get("reward_cc")
        if r_al is None or r_ts is None or r_cc is None:
            return None

        # Ensure shapes (B,1,1)
        def _reshape(x):
            if x is None:
                return None
            if x.dim() == 1:
                return x.view(-1, 1, 1)
            if x.dim() == 2:
                return x.unsqueeze(-1)
            return x

        r_al = _reshape(r_al)
        r_ts = _reshape(r_ts)
        r_cc = _reshape(r_cc)

        alpha_use = alpha.detach() if detach_alpha else alpha
        alpha_use = alpha_use.view(1, 1, -1)  # (1,1,3)

        stacked = torch.stack([r_al, r_ts, r_cc], dim=-1)  # (B,1,*,*,3)
        combined = (stacked * alpha_use).sum(dim=-1)       # (B,1,*,*)

        # If rewards are per-agent (B,1,N,1), reduce to team-level (B,1,1)
        if combined.dim() == 4:
            combined = combined.mean(dim=2)
        elif combined.dim() == 3 and combined.size(-1) != 1:
            combined = combined.mean(dim=-1, keepdim=True)
        return combined

    def _maybe_update_alpha(self, tensors: Dict[str, torch.Tensor], alpha: torch.Tensor,
                            q_expectation: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
        """
        Update α with L_dr = (r_actual - r_expected)^2, where r_expected≈q_expectation (detached).
        """
        if self.opt_alpha is None:
            return None

        reward_alpha = self._combine_reward(tensors, alpha, detach_alpha=False)
        if reward_alpha is None:
            return None

        q_expect = q_expectation.detach()
        loss_alpha = _safe_mean((reward_alpha - q_expect).pow(2), mask)

        if not torch.isfinite(loss_alpha):
            return None

        self.opt_alpha.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.opt_alpha.step()
        return float(loss_alpha.item())


    def train(self, batch, t_env: int, episode: int) -> Dict[str, float]:
        try:
           
            if self.bne_enabled:
                out = self._train_bne(batch, t_env, episode)
                self.train_steps += 1
                self._maybe_soft_update_targets()
                return out

          
            tensors = self._extract_basic(batch)
            if tensors is None:
                return {"status": "empty-batch"}

            alpha = self._get_alpha_weights()
            rewards_raw = tensors["reward"]        # (B,1,1)
            mask        = tensors["mask"]          # (B,1,1)
            commit      = tensors["commitment_embedding"]  # (B,1,d_c) or None
            terminated  = tensors.get("terminated")

            reward_combined = self._combine_reward(tensors, alpha, detach_alpha=True)
            rewards = reward_combined if reward_combined is not None else rewards_raw

            B = rewards.size(0)
            N = int(getattr(self.args, "n_agents", 3))

       
            feats = self.mac.encode_step(batch, t=0)  # 训练图与 runner 写回解耦
            q_local_bn   = feats["q_local"]           # (B, N, 1)
            belief_state = feats["belief_states"]     # (B, N, d)
            prompt_embed = feats["prompt_embeddings"] # (B, N, 2)
            group_repr   = feats["group_repr"]        # (B, d)

      
            q_local = q_local_bn.squeeze(-1).unsqueeze(1)        # (B,1,N)
            b_state = belief_state.unsqueeze(1)                  # (B,1,N,d)
            p_embed = prompt_embed.unsqueeze(1)                  # (B,1,N,2)
            g_repr  = group_repr.unsqueeze(1)                    # (B,1,d)
            c_emb   = commit                                      # (B,1,d_c) or None

            if self.reward_weighting:
                w = (self.reward_weighting_alpha * rewards + (1.0 - self.reward_weighting_alpha))  # (B,1,1)
                w = w.clamp_min(1e-3)
            else:
                w = torch.ones_like(rewards)

            # Local TD 
            not_done = torch.ones_like(rewards)
            if terminated is not None:
                not_done = 1.0 - terminated.float()

            td_target_local = rewards + self.gamma * not_done * q_local.detach()
            y_local = td_target_local.expand(B, 1, N)            # (B,1,N)
            td_err_local = q_local - y_local                     # (B,1,N)
            mask_local = (mask * w).expand(B, 1, N)              # (B,1,N)
            l_td_local = _safe_mean(td_err_local.pow(2), mask_local)  # scalar

            #  Mixer 
            q_tot_pred, aux = self.mixer(
                q_local=q_local,                  # (B,1,N)
                belief_states=b_state,           # (B,1,N,d)
                prompt_embeddings=p_embed,       # (B,1,N,2)
                group_repr=g_repr,               # (B,1,d)
                commitment_embedding=c_emb,      # (B,1,d_c) or None
                reward_ctx=None
            )  # q_tot_pred: (B,1,1)

            # Target mixer bootstrap
            with torch.no_grad():
                q_tot_target, _ = self.target_mixer(
                    q_local=q_local,
                    belief_states=b_state,
                    prompt_embeddings=p_embed,
                    group_repr=g_repr,
                    commitment_embedding=c_emb,
                    reward_ctx=None
                )
            td_target_tot = rewards + self.gamma * not_done * q_tot_target.detach()
            l_td_tot = _safe_mean((q_tot_pred - td_target_tot).pow(2), mask * w)  # (B,1,1) 对齐

            l_cons  = aux.get("consistency_loss", q_tot_pred.new_zeros(()))
            l_align = aux.get("align_loss", q_tot_pred.new_zeros(()))


            lambda_e = 0.5
            lambda_c = 0.1
            lambda_a = 0.05
            loss_belief  = l_td_local
            loss_encoder = l_td_tot + lambda_e * l_td_local
            loss_mixer   = l_td_tot + lambda_c * l_cons + lambda_a * l_align

            total_loss = self.w_belief * loss_belief + self.w_encoder * loss_encoder + self.w_mixer * loss_mixer


            if not torch.isfinite(total_loss):
                self.logger.warning("NaN/Inf detected in total_loss; skip step")
                return {"status": "nan", "loss_total": float("nan")}


            self.opt_belief.zero_grad(set_to_none=True)
            self.opt_encoder.zero_grad(set_to_none=True)
            self.opt_mixer.zero_grad(set_to_none=True)

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.params_belief, self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.params_encoder, self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.params_mixer, self.max_grad_norm)

            self.opt_belief.step()
            self.opt_encoder.step()
            self.opt_mixer.step()

            alpha_loss = self._maybe_update_alpha(tensors, alpha, q_tot_pred, mask * w)

            stats = {
                "status": "ok",
                "loss_total": float(total_loss.item()),
                "loss_belief": float(loss_belief.item()),
                "loss_encoder": float(loss_encoder.item()),
                "loss_mixer": float(loss_mixer.item()),
                "mixer_L_TD_tot": float(l_td_tot.item()),
                "mixer_L_cons": float(l_cons.item() if torch.is_tensor(l_cons) else l_cons),
                "mixer_L_align": float(l_align.item() if torch.is_tensor(l_align) else l_align),
                "q_local_mean": float(q_local.mean().item()),
                "reward_mean": float(rewards.mean().item()),
            }
            if reward_combined is not None:
                stats["reward_combined_mean"] = float(rewards.mean().item())
                stats["alpha_weights"] = [float(x) for x in alpha.detach().cpu().tolist()]
            if alpha_loss is not None:
                stats["alpha_loss"] = float(alpha_loss)

            self.train_steps += 1
            self._maybe_soft_update_targets()
            return stats

        except Exception as e:
            self.logger.warning(f"Learner.train exception: {e}")
            raise



    def _train_bne(self, batch, t_env: int, episode: int) -> Dict[str, float]:
        """
        procedure
        1. data extract
        2. policy_networks to calculate q_local
        3. Mixer feed forward (init and refined)
        4. 4个loss: TD, consistency, equilibrium, commit_improve
        5. parameter update
        """
        device = self.device
        N = int(getattr(self.args, "n_agents", 3))


        def get_field(name, default=None, cut=True):
            try:
                x = batch[name]
                if cut and x.size(1) >= 1:
                    x = x[:, :-1]  # Remove last timestep for bootstrap
                return x.to(device).float()
            except Exception:
                return default

        reward = get_field("reward")  # (B, 1, 1)
        reward_al = get_field("reward_al")
        reward_ts = get_field("reward_ts")
        reward_cc = get_field("reward_cc")
        terminated = get_field("terminated")
        filled = get_field("filled")  # (B, 1, 1)
        if reward is None or filled is None:
            return {"status": "empty-batch"}

        mask = filled.float()  # (B, 1, 1)
        alpha = self._get_alpha_weights()
        reward_dict = {
            "reward_al": reward_al,
            "reward_ts": reward_ts,
            "reward_cc": reward_cc,
        }
        reward_combined = self._combine_reward(reward_dict, alpha, detach_alpha=True)
        reward_use = reward_combined if reward_combined is not None else reward
        not_done = torch.ones_like(reward_use)
        if terminated is not None:
            not_done = 1.0 - terminated

        bne_graph = getattr(batch, "bne_graph", None)
        if bne_graph is not None:
            def _bt(x):
                if x is None:
                    return None
                if x.dim() == 1:
                    return x.view(1, 1, -1)
                if x.dim() == 2:
                    return x.unsqueeze(0).unsqueeze(0)
                if x.dim() == 3:
                    return x.unsqueeze(1)
                return x

            belief_states = _bt(bne_graph.get("beliefs"))
            bne_e_init = _bt(bne_graph.get("e_init"))
            e_final = bne_graph.get("e_final")
            e_refined = bne_graph.get("e_refined")
            bne_e_refined = _bt(e_final if e_final is not None else e_refined)
            bne_commitment_emb_0 = _bt(bne_graph.get("commitment_emb_0"))
            commitment_emb_final = _bt(bne_graph.get("commitment_emb_final"))
            group_repr = _bt(bne_graph.get("group_repr"))
        else:
            bne_e_init = get_field("bne_e_init")  # (B, 1, N, 2) or (B, N, 2)
            bne_e_refined = get_field("bne_e_refined")  # (B, 1, N, 2) or (B, N, 2)
            bne_commitment_emb_0 = get_field("bne_commitment_emb_0")  # (B, 1, d_c)
            commitment_emb_final = get_field("commitment_embedding")  # (B, 1, d_c)
            belief_states = get_field("belief_states")  # (B, 1, N, d_belief) or (B, N, d_belief)
            group_repr = None

            if bne_e_init is not None and bne_e_init.dim() == 3:
                bne_e_init = bne_e_init.unsqueeze(1)  # (B, 1, N, 2)
            if bne_e_refined is not None and bne_e_refined.dim() == 3:
                bne_e_refined = bne_e_refined.unsqueeze(1)  # (B, 1, N, 2)
            if belief_states is not None and belief_states.dim() == 3:
                belief_states = belief_states.unsqueeze(1)  # (B, 1, N, d_belief)


        if bne_e_init is None or bne_e_refined is None or belief_states is None:
            self.logger.warning("[Learner] BNE data not found in batch, falling back to zero losses")
            return {"status": "no-bne-data", "loss_total": 0.0}

        B, T = reward.size(0), reward.size(1)

        q_local_list = []
        q_local_target_list = []
        for i in range(N):
            belief_i = belief_states[:, :, i, :]  # (B, T, d_belief)
            belief_i_flat = belief_i.reshape(B * T, -1)   # (B*T, d_belief)
            with torch.enable_grad():
                q_i = self.mac.policy_networks[i].q_head(belief_i_flat)  # (B*T, 1)
            q_local_list.append(q_i.view(B, T, 1))  # (B, T, 1)

            # Target q using target policy network
            with torch.no_grad():
                q_i_tgt = self.target_policy_networks[i].q_head(belief_i_flat)
            q_local_target_list.append(q_i_tgt.view(B, T, 1))

        q_local = torch.cat(q_local_list, dim=-1)              # (B, T, N)
        q_local_target = torch.cat(q_local_target_list, dim=-1)  # (B, T, N)

        # === 3. Group representation ===
        # belief_encoder expects (B, N, d_belief)
        belief_for_encoder = belief_states.view(B * T, N, -1)  # (B*T, N, d_belief)
        if group_repr is None:
            group_repr = self.mac.belief_encoder(belief_for_encoder)  # (B*T, d_belief)
            group_repr = group_repr.view(B, T, -1)  # (B, T, d_belief)
        else:
            group_repr = group_repr.view(B, T, -1)

        with torch.no_grad():
            group_repr_tgt = self.target_belief_encoder(belief_for_encoder).view(B, T, -1)

        # === 4. Mixer forward ===
        # Prepare shapes for mixer
        # q_local: (B, 1, N) ✓
        # belief_states: (B, 1, N, d_belief) ✓
        # prompt_embeddings: (B, 1, N, 2)
        # group_repr: (B, 1, d_belief) ✓
        # commitment_embedding: (B, 1, d_c)

        # Initial state (before refinement)
        q_tot_init, aux_init = self.mixer(
            q_local=q_local,
            belief_states=belief_states,
            prompt_embeddings=bne_e_init,  # (B, 1, N, 2)
            group_repr=group_repr,
            commitment_embedding=bne_commitment_emb_0,
            reward_ctx=None
        )  # q_tot_init: (B, 1, 1)

        # Refined state (after refinement)
        q_tot_refined, aux_refined = self.mixer(
            q_local=q_local,
            belief_states=belief_states,
            prompt_embeddings=bne_e_refined,  # (B, 1, N, 2)
            group_repr=group_repr,
            commitment_embedding=commitment_emb_final,
            reward_ctx=None
        )  # q_tot_refined: (B, 1, 1)

        # Target mixer for bootstrap
        with torch.no_grad():
            q_tot_target, _ = self.target_mixer(
                q_local=q_local_target,
                belief_states=belief_states,
                prompt_embeddings=bne_e_refined,
                group_repr=group_repr_tgt,
                commitment_embedding=commitment_emb_final,
                reward_ctx=None
            )

        # === 5. Compute losses ===

        # L1: TD Loss - main training signal (with bootstrap over rounds)
        q_next = torch.cat([q_tot_target[:, 1:], torch.zeros_like(q_tot_target[:, -1:])], dim=1)
        td_target = reward_use + self.gamma * not_done * q_next.detach()
        l_td = _safe_mean((q_tot_refined - td_target).pow(2), mask)

        # L2: Consistency Loss - QMIX monotonicity
        q_local_sum = q_local.sum(dim=-1, keepdim=True)  # (B, T, 1)
        l_consistency = _safe_mean((q_tot_refined - q_local_sum).pow(2), mask)

        # L3: Equilibrium Loss - encourage stability
        delta_e = bne_e_refined - bne_e_init  # (B, T, N, 2)
        l_equilibrium = _safe_mean(delta_e.pow(2), mask.unsqueeze(-1).expand_as(delta_e))

        # L4: Commitment Improvement Loss (encourage refined commitment/q_tot to increase)
        q_improvement = (q_tot_refined - q_tot_init)  # (B, T, 1)
        l_commit_improve = -_safe_mean(q_improvement.squeeze(-1), mask.squeeze(-1))

        # === 6. Combine losses ===
        # Weights for each loss component
        w_td = 1.0
        w_consistency = 0.1
        w_equilibrium = 0.05
        w_commit_improve = 0.1

        total_loss = (
            w_td * l_td +
            w_consistency * l_consistency +
            w_equilibrium * l_equilibrium +
            w_commit_improve * l_commit_improve
        )

        alpha_loss = self._maybe_update_alpha(reward_dict, alpha, q_tot_refined, mask)

        # === 7. Guard against NaN/Inf ===
        if not torch.isfinite(total_loss):
            self.logger.warning("[Learner] NaN/Inf in BNE total_loss, skipping step")
            return {"status": "nan", "loss_total": float("nan")}

        # === 8. Backward and optimize ===
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.max_grad_norm)

        self.optimizer.step()
        self.train_steps += 1
        self._maybe_soft_update_targets()

        # === 9. Return metrics ===
        # BNE-specific metrics
        metrics_bne = {
            "loss_td": float(l_td.item()),
            "loss_consistency": float(l_consistency.item()),
            "loss_equilibrium": float(l_equilibrium.item()),
            "loss_commit_improve": float(l_commit_improve.item()),
            "q_tot_init_mean": float(q_tot_init.mean().item()),
            "q_tot_refined_mean": float(q_tot_refined.mean().item()),
            "q_improvement_mean": float(q_improvement.mean().item()),
        }

        # Legacy-compatible metrics (for CSV logging compatibility)
        # Map BNE losses to legacy field names
        metrics_legacy = {
            "loss_belief": float(l_td.item()),  # TD loss as belief loss
            "loss_encoder": float(l_consistency.item()),  # Consistency as encoder loss
            "loss_mixer": float((l_equilibrium + l_commit_improve).item()),  # BNE losses as mixer loss
            "mixer_L_TD_tot": float(l_td.item()),
            "mixer_L_cons": float(l_consistency.item()),
            "mixer_L_align": float(l_equilibrium.item()),  # Equilibrium as alignment
        }

        out = {
            "status": "ok",
            "loss_total": float(total_loss.item()),
            "q_local_mean": float(q_local.mean().item()),
            "reward_mean": float(reward_use.mean().item()),
            **metrics_bne,      # BNE-specific metrics
            **metrics_legacy,   # Legacy-compatible metrics
        }
        if reward_combined is not None:
            out["reward_combined_mean"] = float(reward_use.mean().item())
            out["alpha_weights"] = [float(x) for x in alpha.detach().cpu().tolist()]
        if alpha_loss is not None:
            out["alpha_loss"] = float(alpha_loss)
        return out



    def _extract_basic(self, batch) -> Optional[Dict[str, torch.Tensor]]:
        device = self.device

        def get_or_default(name, default=None, cut=True):
            try:
                x = batch[name]
                if cut and x.size(1) >= 1:
                    x = x[:, :-1]
                return x.to(device)
            except Exception:
                if default is None:
                    return None
                return default.to(device) if hasattr(default, "to") else default

        reward = get_or_default("reward")           # (B,1,1)
        filled = get_or_default("filled")           # (B,1,1)
        if reward is None or filled is None:
            return None

        reward = reward.float()
        mask = filled.float()                       # (B,1,1)

        reward_al = get_or_default("reward_al")
        reward_ts = get_or_default("reward_ts")
        reward_cc = get_or_default("reward_cc")
        terminated = get_or_default("terminated")


        cemb = get_or_default("commitment_embedding")  # (B,1,d_c) or None

        return {
            "reward": reward,
            "mask": mask,
            "reward_al": reward_al,
            "reward_ts": reward_ts,
            "reward_cc": reward_cc,
            "terminated": terminated,
            "commitment_embedding": cemb,
        }

    # ---------------- IO ----------------

    def save_models(self, path: str):
        os.makedirs(path, exist_ok=True)

        if self.bne_enabled:
          
            N = int(getattr(self.args, "n_agents", 3))
            for i in range(N):
                torch.save(
                    self.mac.policy_networks[i].state_dict(),
                    os.path.join(path, f"policy_network_{i}.th")
                )
            torch.save(self.mac.refine_module.state_dict(), os.path.join(path, "refine_module.th"))
            torch.save(self.mac.belief_encoder.state_dict(), os.path.join(path, "belief_encoder.th"))
            torch.save(self.mixer.state_dict(), os.path.join(path, "mixer.th"))
            torch.save(self.alpha_logits.detach().cpu(), os.path.join(path, "alpha_weights.th"))
            self.logger.info(f"[Learner] Saved BNE models to {path}")
        else:
            # Legacy mode: save single belief network
            torch.save(self.mac.agent.belief_network.state_dict(), os.path.join(path, "belief_network.th"))
            torch.save(self.mac.belief_encoder.state_dict(), os.path.join(path, "belief_encoder.th"))
            torch.save(self.mixer.state_dict(), os.path.join(path, "mixer.th"))
            torch.save(self.alpha_logits.detach().cpu(), os.path.join(path, "alpha_weights.th"))
            self.logger.info(f"[Learner] Saved legacy models to {path}")

    def load_models(self, path: str):
        if self.bne_enabled:
            # BNE mode: load K policy networks + refine module
            N = int(getattr(self.args, "n_agents", 3))
            for i in range(N):
                self.mac.policy_networks[i].load_state_dict(
                    torch.load(os.path.join(path, f"policy_network_{i}.th"), map_location=self.device)
                )
            self.mac.refine_module.load_state_dict(torch.load(os.path.join(path, "refine_module.th"), map_location=self.device))
            self.mac.belief_encoder.load_state_dict(torch.load(os.path.join(path, "belief_encoder.th"), map_location=self.device))
            self.mixer.load_state_dict(torch.load(os.path.join(path, "mixer.th"), map_location=self.device))
            # Sync targets to loaded weights
            if hasattr(self, "target_policy_networks"):
                for tgt, src in zip(self.target_policy_networks, self.mac.policy_networks):
                    tgt.load_state_dict(src.state_dict())
            if hasattr(self, "target_belief_encoder"):
                self.target_belief_encoder.load_state_dict(self.mac.belief_encoder.state_dict())
            if hasattr(self, "target_mixer"):
                self.target_mixer.load_state_dict(self.mixer.state_dict())
            alpha_path = os.path.join(path, "alpha_weights.th")
            if os.path.exists(alpha_path):
                self.alpha_logits.data = torch.load(alpha_path, map_location=self.device).to(self.device)
            self.logger.info(f"[Learner] Loaded BNE models from {path}")
        else:
            # Legacy mode: load single belief network
            self.mac.agent.belief_network.load_state_dict(torch.load(os.path.join(path, "belief_network.th"), map_location=self.device))
            self.mac.belief_encoder.load_state_dict(torch.load(os.path.join(path, "belief_encoder.th"), map_location=self.device))
            self.mixer.load_state_dict(torch.load(os.path.join(path, "mixer.th"), map_location=self.device))
            if hasattr(self, "target_mixer"):
                self.target_mixer.load_state_dict(self.mixer.state_dict())
            alpha_path = os.path.join(path, "alpha_weights.th")
            if os.path.exists(alpha_path):
                self.alpha_logits.data = torch.load(alpha_path, map_location=self.device).to(self.device)
            self.logger.info(f"[Learner] Loaded legacy models from {path}")
