"""
Actor-critic network with hybrid action heads:
  - continuous: UAV velocity (2D), offloading ratio (1D)
  - discrete:   device scheduling, semantic symbol count
  - critic:     scalar state-value estimate
"""

import torch
import torch.nn as nn

from .config import Config


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, cfg: Config):
        super().__init__()
        h1, h2 = cfg.hidden_layer_sizes
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, h1), nn.LayerNorm(h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
        )
        h = h2  # width of the trunk's output, used for every head below
        # continuous velocity head (2D mean + learned log-std)
        self.v_mu = nn.Linear(h, 2)
        self.v_log_std = nn.Parameter(torch.zeros(2) - 0.5)

        # discrete device-scheduling head
        self.device_logits = nn.Linear(h, cfg.M)

        # continuous offloading-ratio head (scalar mean + learned log-std)
        self.R_mu = nn.Linear(h, 1)
        self.R_log_std = nn.Parameter(torch.zeros(1) - 0.5)

        # FIX (Bug 3): continuous bandwidth-request head (scalar mean +
        # learned log-std), matching R_mu/R_log_std in structure. This is
        # hat_b_m[n] in the paper's Algorithm 1 -- the agent's raw bandwidth
        # request for whichever device it schedules this slot.
        self.b_mu = nn.Linear(h, 1)
        self.b_log_std = nn.Parameter(torch.zeros(1) - 0.5)

        # discrete semantic-symbol head
        self.symbol_logits = nn.Linear(h, len(cfg.symbol_set))

        # critic
        self.value_head = nn.Linear(h, 1)

    def forward(self, s: torch.Tensor):
        z = self.trunk(s)
        v_mu = self.v_mu(z)
        v_std = self.v_log_std.exp().expand_as(v_mu)
        device_logits = self.device_logits(z)
        R_mu = self.R_mu(z)
        R_std = self.R_log_std.exp().expand_as(R_mu)
        b_mu = self.b_mu(z)
        b_std = self.b_log_std.exp().expand_as(b_mu)
        symbol_logits = self.symbol_logits(z)
        value = self.value_head(z).squeeze(-1)
        return v_mu, v_std, device_logits, R_mu, R_std, b_mu, b_std, symbol_logits, value
