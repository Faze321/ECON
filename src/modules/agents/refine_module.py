# src/modules/agents/refine_module.py
"""
BNE Refinement Module
根据全局信息（commitment + group_repr）调整prompt parameters
"""
import torch
import torch.nn as nn


class RefineModule(nn.Module):
    """
    BNE refinement: 基于全局信息调整prompt parameters

    关键特性：
    1. 输入包含agent自己的belief和输出
    2. 输入包含全局commitment和group representation
    3. 输出是对prompt parameters的调整量 delta_e = (delta_temperature, delta_top_p)

    这体现了BNE的核心：每个agent根据全局信息（包括其他agent的信息）
    来调整自己的策略参数
    """
    def __init__(
        self,
        belief_dim: int,
        commitment_dim: int,
        n_agents: int = 3,
        hidden_dim: int = 256,
        max_delta: float = 0.3,
    ):
        """
        Args:
            belief_dim: Belief state维度
            commitment_dim: Commitment embedding维度
            n_agents: Agent数量（用于归一化）
            hidden_dim: 隐藏层维度
            max_delta: 最大调整幅度
        """
        super().__init__()
        self.max_delta = max_delta

        # 输入维度:
        # - belief_i: belief_dim
        # - output_i_emb: commitment_dim
        # - commitment_emb: commitment_dim
        # - group_repr: belief_dim
        # - e_prev: 2
        input_dim = belief_dim + commitment_dim * 2 + belief_dim + 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # output (delta_temperature, delta_top_p)
            nn.Tanh()  # 限制在[-1, 1]
        )

    def forward(
        self,
        belief_i: torch.Tensor,
        output_i_emb: torch.Tensor,
        commitment_emb: torch.Tensor,
        group_repr: torch.Tensor,
        e_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            belief_i: (B, belief_dim) - 当前agent的belief state
            output_i_emb: (B, commitment_dim) - 当前agent的输出embedding
            commitment_emb: (B, commitment_dim) - 全局commitment embedding
            group_repr: (B, belief_dim) - 所有agent的聚合表示（包含其他agent信息）
            e_prev: (B, 2) - 当前的prompt parameters (temperature, top_p)

        Returns:
            delta_e: (B, 2) - 调整量 (delta_temperature, delta_top_p)
        """
        # 拼接所有输入
        x = torch.cat([
            belief_i,
            output_i_emb,
            commitment_emb,
            group_repr,
            e_prev
        ], dim=-1)  # (B, input_dim)

        # 通过网络计算调整量
        delta = self.net(x)  # (B, 2), range in [-1, 1]

        # Scale到实际调整范围
        delta = delta * self.max_delta  # (B, 2), range in [-max_delta, +max_delta]

        return delta
