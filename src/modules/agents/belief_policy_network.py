# src/modules/agents/belief_policy_network.py
"""
独立的Belief Policy Network for BNE
每个agent有自己的一个实例，参数不共享
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """位置编码（复用）"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: (d_model // 2) + 1])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Transformer块（复用）"""
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


class BeliefPolicyNetwork(nn.Module):
    """
    独立的Belief Policy Network for BNE

    每个agent一个实例，参数独立
    输入: token_ids (B, L)
    输出: {
        "belief_state": (B, belief_dim),
        "prompt_embedding": (B, 2),  # [temperature, top_p]
        "q_value": (B, 1)
    }
    """
    def __init__(
        self,
        observation_dim: int,
        belief_dim: int,
        hidden_dim: int = 256,
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

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, dropout, max_len=observation_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, n_heads, hidden_dim * 4, dropout=dropout) for _ in range(n_layers)]
        )

        # Belief head with GRU
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
        self.belief_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, belief_dim),
        )

        # Prompt parameters head (temperature, top_p)
        self.prompt_head = nn.Sequential(
            nn.Linear(belief_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        # Local Q value head
        self.q_head = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.memory_projection = nn.Linear(self.memory_dim, hidden_dim) if self.memory_dim > 0 else None

    def forward(self, token_ids: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                memory: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            token_ids: (B, L) long tensor
            key_padding_mask: (B, L) bool tensor, True for padding positions

        Returns:
            dict with:
                - belief_state: (B, belief_dim)
                - prompt_embedding: (B, 2) [temperature, top_p]
                - q_value: (B, 1)
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)

        # Clamp token IDs to valid vocabulary range to prevent CUDA index out of bounds
        vocab_size = self.token_embedding.num_embeddings
        token_ids = torch.clamp(token_ids.long(), 0, vocab_size - 1)

        # Embed and encode
        x = self.token_embedding(token_ids)  # (B, L, hidden_dim)
        x = self.pos_enc(x)

        for blk in self.blocks:
            x = blk(x, key_padding_mask)

        # Pool: use last valid position
        if key_padding_mask is not None:
            valid_len = (~key_padding_mask).sum(dim=1).clamp(min=1)
            # Ensure index doesn't exceed sequence length to prevent CUDA index out of bounds
            max_seq_len = x.size(1)
            idx = (valid_len - 1).clamp(max=max_seq_len - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, x.size(-1))
            pooled = x.gather(1, idx).squeeze(1)  # (B, hidden_dim)
        else:
            pooled = x[:, -1, :]  # (B, hidden_dim)

        # GRU stage
        x_pre = self.pre_gru(pooled)  # (B, hidden_dim)
        gru_in = x_pre.unsqueeze(1)   # (B,1,H)
        if memory is not None:
            mem = memory.view(1, gru_in.size(0), -1)
            if mem.size(-1) != self.hidden_dim:
                mem = mem[..., :self.hidden_dim] if mem.size(-1) > self.hidden_dim else \
                    torch.nn.functional.pad(mem, (0, self.hidden_dim - mem.size(-1)))
        else:
            mem = torch.zeros((1, gru_in.size(0), self.hidden_dim), device=gru_in.device, dtype=gru_in.dtype)
        gru_out, h_n = self.gru(gru_in, mem)
        pooled_gru = gru_out.squeeze(1)

        if self.memory_projection is not None and memory is not None:
            memproj = self.memory_projection(memory.view(pooled.size(0), -1))
            pooled_gru = pooled_gru + memproj

        # Belief state
        belief = self.belief_head(pooled_gru)  # (B, belief_dim)

        # Prompt parameters (temperature, top_p)
        prompt_logits = self.prompt_head(belief)  # (B, 2)
        T = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(prompt_logits[:, 0:1])
        top_p = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(prompt_logits[:, 1:2])
        prompt_embedding = torch.cat([T, top_p], dim=-1)  # (B, 2) -> [temperature, top_p]

        # Local Q value
        q_value = self.q_head(belief)  # (B, 1)

        return {
            "belief_state": belief,
            "prompt_embedding": prompt_embedding,
            "q_value": q_value,
            "hidden_state": h_n.squeeze(0),
        }

    def to(self, device):
        self.device = device
        return super().to(device)
