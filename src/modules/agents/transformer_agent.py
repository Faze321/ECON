# src/modules/agents/transformer_agent.py
# -*- coding: utf-8 -*-
import math
import re
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.llm.llm_wrapper import ImprovedLLMWrapper

# --------------------------- Positional Encoding ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L,1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: (d_model // 2) + 1])
        pe = pe.unsqueeze(0)  # (1,L,D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

# --------------------------- Transformer Block -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

# ------------------------------ Belief Network -----------------------------
class BeliefNetwork(nn.Module):
    """
    个体置信网络 B_i：
      token ids → embedding → Transformer → pooled → GRU(hidden_{t-1}) → belief b_i
      b_i → (T, p_top)  （T: temperature；p_top: top_p）
      b_i → q_i
    """
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        belief_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        T_min: float = 0.1,
        T_max: float = 2.0,
        p_min: float = 0.1,
        p_max: float = 0.9,
        vocab_size: int = 50257,
        device: Optional[torch.device] = None,
        memory_dim: int = 0,
    ):
        super().__init__()
        self.belief_dim = belief_dim
        self.hidden_dim = hidden_dim
        self.T_min, self.T_max = float(T_min), float(T_max)
        self.p_min, self.p_max = float(p_min), float(p_max)
        self.device = device or torch.device("cpu")
        self.memory_dim = int(memory_dim) if memory_dim and memory_dim > 0 else hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, dropout, max_len=observation_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, n_heads, hidden_dim * 4, dropout=dropout) for _ in range(n_layers)]
        )

        self.pre_gru = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.belief_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, belief_dim),
        )

        self.temp_projection = nn.Linear(belief_dim, 1)
        self.top_p_projection = nn.Linear(belief_dim, 1)

        self.q_network = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.memory_projection = nn.Linear(self.memory_dim, hidden_dim) if self.memory_dim > 0 else None

    def to(self, *args, **kwargs):
        device = kwargs.get("device", None)
        if device is None and len(args) > 0:
            device = args[0]
        if device is not None:
            self.device = torch.device(device)
        return super().to(*args, **kwargs)

    def forward(self, token_ids: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                memory: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)

        device = self.token_embedding.weight.device
        token_ids = token_ids.to(device)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device)

        # Clamp token IDs to valid vocabulary range to prevent CUDA index out of bounds
        vocab_size = self.token_embedding.num_embeddings
        token_ids = torch.clamp(token_ids.long(), 0, vocab_size - 1)

        x = self.token_embedding(token_ids)
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x, key_padding_mask)

        if key_padding_mask is not None:
            valid_len = (~key_padding_mask).sum(dim=1).clamp(min=1)
            # Ensure index doesn't exceed sequence length to prevent CUDA index out of bounds
            max_seq_len = x.size(1)
            idx = (valid_len - 1).clamp(max=max_seq_len - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, x.size(-1))
            pooled = x.gather(1, idx).squeeze(1)
        else:
            pooled = x[:, -1, :]

        # Pre-MLP then GRU (single step) with hidden from memory
        x_pre = self.pre_gru(pooled)  # (B, hidden_dim)
        gru_in = x_pre.unsqueeze(1)   # (B,1,H)
        if memory is not None:
            # GRU expects a contiguous hidden state; memory may come from views/slices
            mem = memory.to(device=device, dtype=gru_in.dtype).contiguous().view(1, gru_in.size(0), -1)
            if mem.size(-1) != self.hidden_dim:
                mem = mem[..., :self.hidden_dim] if mem.size(-1) > self.hidden_dim else \
                    torch.nn.functional.pad(mem, (0, self.hidden_dim - mem.size(-1)))
            mem = mem.contiguous()
        else:
            mem = torch.zeros((1, gru_in.size(0), self.hidden_dim), device=device, dtype=gru_in.dtype)
        gru_out, h_n = self.gru(gru_in, mem)  # h_n: (1,B,H)
        pooled_gru = gru_out.squeeze(1)       # (B,H)

        if self.memory_projection is not None and memory is not None:
            memproj = self.memory_projection(memory.to(device=device, dtype=pooled_gru.dtype).view(pooled.size(0), -1))
            pooled_gru = pooled_gru + memproj

        b = self.belief_projection(pooled_gru)

        t_logit = self.temp_projection(b)
        p_logit = self.top_p_projection(b)
        T = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(t_logit)
        p_top = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(p_logit)
        prompt_embed = torch.cat([T, p_top], dim=1)  # (B,2)

        q_i = self.q_network(b)

        return {
            "belief_state": b,
            "prompt_embedding": prompt_embed,  # [T, top_p]
            "q_value": q_i,
            "temp_logit": t_logit,
            "top_p_logit": p_logit,
        }, h_n.squeeze(0)

# --------------------------- LLM Transformer Agent -------------------------
class LLMTransformerAgent(nn.Module):
    def __init__(self, input_shape: int, args: Any):
        super().__init__()
        self.args = args
        use_cuda = getattr(args.system, "use_cuda", False) and torch.cuda.is_available()
        devnum = getattr(getattr(args, "system", object()), "device_num", 0)
        self.device = torch.device(f"cuda:{devnum}" if use_cuda else "cpu")

        self.belief_dim = getattr(args, "belief_dim", 128)
        entity_dim = getattr(args.arch, "entity_dim", 256)
        self.memory_dim = int(getattr(args, "memory_dim", entity_dim))

        self.T_min = float(getattr(args.sampling, "temperature_min", 0.1))
        self.T_max = float(getattr(args.sampling, "temperature_max", 2.0))
        # p is repetition penalty per paper
        self.p_min = float(getattr(args.sampling, "p_min", 0.8))
        self.p_max = float(getattr(args.sampling, "p_max", 1.3))
        self.top_p_default = float(getattr(args.sampling, "top_p_default", 0.9))

        max_token_len = getattr(args.env_args, "max_question_length", 512)
        vocab_size = getattr(args, "vocab_size", 50257)

        self.belief_network = BeliefNetwork(
            observation_dim=max_token_len,
            action_dim=0,
            hidden_dim=entity_dim,
            belief_dim=self.belief_dim,
            n_heads=getattr(args.arch, "attention_heads", 4),
            n_layers=getattr(args.arch, "transformer_blocks", 2),
            dropout=getattr(args.arch, "dropout_rate", 0.1),
            T_min=self.T_min, T_max=self.T_max,
            p_min=self.p_min, p_max=self.p_max,
            vocab_size=vocab_size,
            device=self.device,
            memory_dim=self.memory_dim
        )

        self.n_actions = getattr(args, "n_actions", 2)
        self.output_network = nn.Linear(self.belief_dim, self.n_actions)

        def _get_opt(key: str, default=None):
            if hasattr(args, key) and getattr(args, key) is not None:
                return getattr(args, key)
            if hasattr(args, "llm") and hasattr(args.llm, key) and getattr(args.llm, key) is not None:
                return getattr(args.llm, key)
            return default

        exec_model = _get_opt("executor_model", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        api_key = _get_opt("llm_api_key", None)
        rep_penalty_default = float(getattr(args, "repetition_penalty_default", 1.1))

        self.llm_wrapper = ImprovedLLMWrapper(
            api_key=api_key,
            model_name=exec_model,
            belief_dim=self.belief_dim,
            debug=getattr(args, "debug", getattr(getattr(args, "system", object()), "debug", False)),
            max_retries=getattr(args, "llm_max_retries", 6),
            timeout_s=getattr(args, "llm_timeout_s", 60),
            repetition_penalty_default=rep_penalty_default,
        )

        # 删除隐式状态: current_prompt_embedding_tensor
        # 改为显式传参，见run_bne_refinement()
        self.to(self.device)

    def forward(self, inputs: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, test_mode: bool = False,
                memory: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        inputs = inputs.to(self.device)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(self.device)

        bn_out, h_n = self.belief_network(inputs, key_padding_mask, memory=memory)
        b = bn_out["belief_state"]
        e = bn_out["prompt_embedding"]
        q = bn_out["q_value"]

        # 删除副作用: 不再写入成员变量
        # (保持forward()纯函数化)

        action_q_values = self.output_network(b)

        outputs = {
            "action_q_values": action_q_values,
            "belief_state": b,
            "prompt_embedding": e,
            "q_value": q,
            "temp_logit": bn_out["temp_logit"],
            "top_p_logit": bn_out["top_p_logit"],
        }
        return outputs, h_n

# 在文件 src/modules/agents/transformer_agent.py 中，替换 LLMTransformerAgent 内的 generate_answer 函数

    def generate_answer(
        self,
        question: str,
        strategy: str,
        belief_state: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        显式传参版本 - 不依赖任何隐式状态

        Args:
            question: 问题文本
            strategy: 策略文本
            temperature: 显式Temperature参数 (必传或使用配置默认值)
            top_p: 显式top_p参数 (必传或使用配置默认值)
            repetition_penalty: 重复惩罚

        Returns:
            生成的答案文本 (\\boxed{...}格式)
        """
        # Fallback到配置的默认值 (中点)
        default_T = (self.T_min + self.T_max) / 2.0
        default_p = (self.p_min + self.p_max) / 2.0

        final_T = float(temperature) if temperature is not None else default_T
        final_rep_penalty = float(top_p) if top_p is not None else default_p  # reuse second dim as repetition penalty

        # Clamp到有效范围
        final_T = max(self.T_min, min(self.T_max, final_T))
        final_rep_penalty = max(self.p_min, min(self.p_max, final_rep_penalty))

        prompt = f"""You are a specialist Executor agent within a collaborative team. Your work will be critically reviewed by a Coordinator to determine the final answer. Therefore, absolute clarity and accuracy are paramount.

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

        txt = self.llm_wrapper.generate_response(
            prompt=prompt,
            temperature=final_T,
            top_p=self.top_p_default,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else final_rep_penalty,
            max_tokens=2048,
        )

        txt = self._ensure_boxed_format(txt)
        txt = self._strip_non_box_numbers_and_cleanup(txt)
        return txt

    # -------------------------- 清洗工具 ---------------------------
    def _ensure_boxed_format(self, text: str) -> str:
        s = str(text or "")
        if "\\boxed{" in s:
            return s.strip()
        m = re.search(r'\\boxed\{([^}]*)\}', s)
        if m:
            return s.strip()
        nums = re.findall(r'[+-]?\d+(?:\.\d+)?', s)
        if nums:
            return (s.strip() + f" \\boxed{{{nums[-1]}}}").strip()
        return (s.strip() + " \\boxed{0}").strip()

    def _normalize_number(self, s: str) -> Optional[str]:
        s = (s or "").replace(",", "").strip()
        if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', s):
            if "." in s:
                h, t = s.split(".", 1)
                if set(t) <= {"0"}:
                    return h
            return s
        m = re.fullmatch(r'(-?)\\frac\{\s*([0-9]+)\s*\}\{\s*([0-9]+)\s*\}', s)
        if m:
            sign = "-" if m.group(1) == "-" else ""
            a = int(m.group(2)); b = int(m.group(3)) if int(m.group(3)) != 0 else 1
            val = a / b
            out = f"{val:.10f}".rstrip("0").rstrip(".")
            return sign + (out if out else "0")
        return None

    def _strip_non_box_numbers_and_cleanup(self, text: str) -> str:
        s = str(text or "")
        if not s.strip():
            return s
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in s.split("\n")]
        
        # Remove prompt echo (first few lines starting with "Problem:" or "Strategy:")
        filtered = []
        for idx, line in enumerate(lines):
            lower = line.lstrip().lower()
            if idx < 3 and (lower.startswith("problem:") or lower.startswith("strategy:")):
                continue
            filtered.append(line)
        
        cleaned = "\n".join(filtered).strip()
        return cleaned

    def save_models(self, path: str):
        torch.save(self.belief_network.state_dict(), f"{path}/belief_network.th")
        torch.save(self.output_network.state_dict(), f"{path}/output_network.th")

    def load_models(self, path: str):
        self.belief_network.load_state_dict(torch.load(f"{path}/belief_network.th", map_location=self.device))
        self.output_network.load_state_dict(torch.load(f"{path}/output_network.th", map_location=self.device))

    def to(self, *args, **kwargs):
        device = kwargs.get("device", None)
        if device is None and len(args) > 0:
            device = args[0]
        if device is not None:
            self.device = torch.device(device)
        return super().to(*args, **kwargs)

    def cuda(self):
        return self.to("cuda")
