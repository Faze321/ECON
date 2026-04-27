# src/learners/noop_learner.py
# -*- coding: utf-8 -*-
from typing import Any, Dict


class NoOpLearner:
    """
    Learner placeholder for execution-only baselines.

    It preserves the runner/train interface but performs no gradient updates,
    no target updates, and no model checkpoint serialization.
    """

    def __init__(self, mac, scheme: Dict, logger, args: Any):
        self.mac = mac
        self.scheme = scheme
        self.logger = logger
        self.args = args
        self.train_steps = 0
        self.logger.info("[NoOpLearner] Optimization disabled.")

    def train(self, batch, t_env: int, episode: int) -> Dict[str, Any]:
        self.train_steps += 1
        is_correct = self._extract_scalar(batch, "is_correct")
        return {
            "status": "no-op",
            "is_correct": is_correct,
        }

    def get_alpha_weights_list(self):
        """No reward weighting is used by the execution-only baseline."""
        return None

    def save_models(self, path: str):
        self.logger.info(f"[NoOpLearner] No model parameters to save at {path}.")

    def load_models(self, path: str):
        self.logger.info(f"[NoOpLearner] No model parameters to load from {path}.")

    def _extract_scalar(self, batch, key: str) -> float:
        try:
            value = batch[key]
            if hasattr(value, "detach"):
                return float(value.detach().float().mean().item())
            return float(value)
        except Exception:
            return 0.0
