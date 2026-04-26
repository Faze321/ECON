
import os
import json
import time
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from loguru import logger

@dataclass
class LLMConfig:
    api_key: str
    model_name: str
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: int = 60
    max_retries: int = 3
    debug: bool = False
    default_max_tokens: int = 2048
    default_temperature: float = 0.2
    default_top_p: float = 0.9
    default_repetition_penalty: float = 1.05

class ImprovedLLMWrapper:
    """
    Together chat-completions wrapper with robust postprocessing:
      - accept both `timeout` and legacy alias `timeout_s`
      - fix control chars (esp. JSON-decoded backspace \b -> keeps \\boxed intact)
    """
    def __init__(self,
                 api_key: str,
                 model_name: str,
                 belief_dim: Optional[int] = None,
                 base_url: str = "https://openrouter.ai/api/v1",
                 timeout: Optional[int] = None,
                 max_retries: int = 3,
                 debug: bool = False,
                 **kwargs):
        """
        Known kwargs that we ALLOW silently (forward/back compat):
          - timeout_s: alias of timeout (in seconds)
          - anything else is ignored (we just log once in debug)
        """
        # alias mapping
        if timeout is None and "timeout_s" in kwargs and kwargs["timeout_s"] is not None:
            try:
                timeout = int(kwargs["timeout_s"])
            except Exception:
                timeout = None

        if timeout is None:
            timeout = 60  # default

        if debug and kwargs:
            safe_keys = ", ".join(sorted(k for k in kwargs.keys()))
            logger.debug(f"[ImprovedLLMWrapper] Ignoring extra kwargs: {safe_keys}")

        self.cfg = LLMConfig(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            timeout=int(timeout),
            max_retries=int(max_retries),
            debug=bool(debug),
        )
        # 仅为接口兼容保留
        self.belief_dim = belief_dim
        self.last_usage = self._empty_usage()
        self.usage_history = []
        masked = (api_key[:4] + "*" * max(0, len(api_key) - 8) + api_key[-4:]) if api_key else "(empty)"
        logger.info(f"[APIHandler] OpenRouter API key resolved: {masked}")

    # ---------------------- public API ----------------------
    def generate_response(self,
                          prompt: str,
                          temperature: Optional[float] = None,
                          top_p: Optional[float] = None,
                          repetition_penalty: Optional[float] = None,
                          max_tokens: Optional[int] = None,
                          stop: Optional[List[str]] = None) -> str:
        """
        Call Together Chat Completions with a *single* user message.
        """
        url = f"{self.cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self._pick(temperature, self.cfg.default_temperature)),
            "top_p": float(self._pick(top_p, self.cfg.default_top_p)),
            "max_tokens": int(self._pick(max_tokens, self.cfg.default_max_tokens)),
            "repetition_penalty": float(self._pick(repetition_penalty, self.cfg.default_repetition_penalty)),
            "stream": False
        }
        if stop:
            payload["stop"] = stop

        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.cfg.timeout)
                if resp.status_code != 200:
                    if self.cfg.debug:
                        logger.warning(f"[OpenRouter] HTTP {resp.status_code}: {resp.text}")
                    last_err = RuntimeError(f"HTTP {resp.status_code}")
                    time.sleep(min(1.5 * (attempt + 1), 6.0))
                    continue
                data = resp.json()
                self._record_usage(data.get("usage", {}))
                text = (data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")).strip()
                return self._postprocess_text(text)
            except Exception as e:
                last_err = e
                if self.cfg.debug:
                    logger.warning(f"[OpenRouter] request failed (attempt {attempt+1}/{self.cfg.max_retries}): {e}")
                time.sleep(min(1.5 * (attempt + 1), 6.0))

        # Fallback: conservative
        logger.error(f"[OpenRouter] All retries failed, fallback empty output. Last error: {last_err}")
        return ""

    # ---------------------- helpers ----------------------
    def _pick(self, val, default):
        return default if val is None else val

    def reset_usage(self):
        self.last_usage = self._empty_usage()
        self.usage_history = []

    def get_usage_summary(self) -> Dict[str, int]:
        summary = self._empty_usage()
        summary["requests"] = len(self.usage_history)
        for usage in self.usage_history:
            summary["prompt_tokens"] += usage.get("prompt_tokens", 0)
            summary["completion_tokens"] += usage.get("completion_tokens", 0)
            summary["total_tokens"] += usage.get("total_tokens", 0)
        return summary

    def _empty_usage(self) -> Dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _record_usage(self, raw_usage: Optional[Dict[str, Any]]):
        usage = self._empty_usage()
        if isinstance(raw_usage, dict):
            usage["prompt_tokens"] = int(raw_usage.get("prompt_tokens") or 0)
            usage["completion_tokens"] = int(raw_usage.get("completion_tokens") or 0)
            usage["total_tokens"] = int(raw_usage.get("total_tokens") or 0)
        self.last_usage = usage
        self.usage_history.append(usage)

    def _postprocess_text(self, s: str) -> str:
        """
        Fix critical control characters & normalize lines.
        Key fix: JSON-decoded \\b -> backspace(0x08). Replace with literal \\b so '\\boxed' survives.
        """
        if s is None:
            return ""
        # Replace control chars that could mutate content
        # \b(backspace)=\x08, \f(formfeed)=\x0c
        s = s.replace("\x08", "\\b")  # critical for \boxed
        s = s.replace("\x0c", "\\f")
        # normalize CR/LF
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        # unicode separators
        s = s.replace("\u2028", " ").replace("\u2029", " ")
        return s
