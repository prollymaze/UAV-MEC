"""
Deterministic actor and dual Q-critic networks for TD3-style DDPG.

Uses two independent Q-critics and takes the minimum during target computation,
which prevents the overestimation bias that single-critic DDPG suffers from.
This is especially important for noisy, stochastic environments (like this UAV 
system with random task sizes and channel conditions).

Unlike PPO's ActorCritic (which outputs *distributions* to sample from),
DDPG's actor outputs one raw action vector directly -- it's a deterministic
policy. To handle this codebase's hybrid action space (continuous velocity
and offload ratio, discrete device/symbol choices) with a standard DDPG
critic, the discrete "choices" are represented as raw logit vectors.
"""

import torch
import torch.nn as nn

from .config import Config


def action_dim(cfg: Config) -> int:
    """Size of the flattened raw action vector: v(2) + device_logits(M) + R(1) + b(1) + symbol_logits(K)."""
    return 2 + cfg.M + 1 + 1 + len(cfg.symbol_set)


class DDPGActor(nn.Module):
    """Deterministic policy network: state -> one raw action vector."""

    def __init__(self, state_dim: int, cfg: Config):
        super().__init__()
        h1, h2 = cfg.hidden_layer_sizes
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, h1), nn.LayerNorm(h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
        )
        h = h2
        self.v_out = nn.Linear(h, 2)
        self.device_out = nn.Linear(h, cfg.M)
        self.R_out = nn.Linear(h, 1)
        # FIX (Bug 3): bandwidth-request output, matching R_out in structure.
        self.b_out = nn.Linear(h, 1)
        self.symbol_out = nn.Linear(h, len(cfg.symbol_set))

    def forward(self, s: torch.Tensor):
        z = self.trunk(s)
        v_raw = self.v_out(z)
        device_logits = self.device_out(z)
        R_raw = self.R_out(z)
        b_raw = self.b_out(z)
        symbol_logits = self.symbol_out(z)
        return v_raw, device_logits, R_raw, b_raw, symbol_logits


class DDPGCritic(nn.Module):
    """Q(s, a): state + raw action vector, concatenated, -> scalar Q-value."""

    def __init__(self, state_dim: int, act_dim: int, cfg: Config):
        super().__init__()
        h1, h2 = cfg.hidden_layer_sizes
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, h1), nn.LayerNorm(h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)
