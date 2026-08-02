"""
Algorithm 1 -- Action Mapping.

Converts normalized/raw policy outputs (velocity, device-scheduling logits,
offloading ratio, symbol logits, bandwidth request) into feasible physical
actions that respect:
  C1 - device scheduling must pick exactly one device
  C2 - offloading ratio in [0, 1]
  C3 - UAV must stay inside the service-area boundary
  C4 - UAV speed must not exceed v_max
  C5 - bandwidth allocation coefficients b_m[n] in [0, 1], sum_m b_m[n] = 1
  C6/C8 - only symbol counts that meet the semantic-similarity threshold are
          eligible (with a max-symbols fallback if none qualify)
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict

from .config import Config
from .environment import UAVSemanticEnv


class ActionMapper:
    """Converts normalized policy outputs into feasible physical actions (Algorithm 1)."""

    def __init__(self, cfg: Config, env: UAVSemanticEnv):
        self.cfg = cfg
        self.env = env

    def map(self, v_raw: np.ndarray, device_logits: np.ndarray,
            R_raw: float, symbol_logits: np.ndarray, q_now: np.ndarray,
            b_raw: float, sample: bool = True) -> Dict:
        cfg = self.cfg

        # --- velocity de-normalization (C4) ---
        v = np.tanh(v_raw) * cfg.v_max
        speed = np.linalg.norm(v)
        if speed > cfg.v_max:
            v = v * (cfg.v_max / speed)

        # --- boundary projection (C3), keep v consistent with clipped move ---
        q_tent = q_now + v * cfg.delta
        q_next = np.clip(q_tent, [0, 0], [cfg.L, cfg.W])
        if not np.allclose(q_next, q_tent):
            v = (q_next - q_now) / cfg.delta

        # --- device scheduling (C1) ---
        probs_m = F.softmax(torch.as_tensor(device_logits), dim=-1)
        if sample:
            m_star = Categorical(probs_m).sample().item()
        else:
            m_star = int(torch.argmax(probs_m).item())

        # --- bandwidth allocation (C5, Algorithm 1 lines 8-10) ---
        # FIX (Bug 3): b_m[n] was previously a hardcoded 1/M split with no
        # policy control. Here the agent's raw output for the scheduled
        # device is mapped to [0, 1] (hat_b in the paper's Algorithm 1 input
        # list), added to that device's persistent long-term accumulator
        # bar_b_m[n], and the whole accumulator vector is renormalized to sum
        # to 1 -- exactly Algorithm 1 steps 8-10. This lets the policy learn
        # to shift bandwidth toward a jammed or bandwidth-starved device
        # instead of every device always getting an identical fixed share.
        b_hat = float(np.clip((math.tanh(b_raw) + 1) / 2, 0.0, 1.0))
        self.env.b_bar[m_star] += b_hat
        self.env.b_bar /= self.env.b_bar.sum()

        # BUG FIX: without a floor, only the *served* device's entry ever
        # gets a positive increment, while every other device's share is
        # divided down at every single step (since the sum keeps growing).
        # For a device not served for k consecutive slots, its share decays
        # like ~1/(1+b_hat)^k -- with M=15 devices and N=40 slots, most
        # devices go 20-35 slots between visits, so b_bar collapses to
        # ~1e-6-1e-9 by the time they ARE finally scheduled. effective_bandwidth
        # = b * bandwidth_hz then becomes numerically ~0, so rate ~0 and
        # T_tx = off_bits / rate explodes -- this is exactly what produced the
        # single-episode cost spike to ~385 (reward -439) in the DDPG
        # convergence run, and is a major source of the noisy/low reward
        # plateau vs. the paper's smooth curves in Fig. 3, plus outlier-driven
        # noise in the Fig. 4-8 sweeps. Flooring every device at a small
        # guaranteed minimum share (then renormalizing) keeps the "reward
        # frequently-served devices with more bandwidth" spirit of Algorithm 1
        # steps 8-10 while eliminating the numerical collapse.
        b_floor = 0.3 / cfg.M
        self.env.b_bar = np.maximum(self.env.b_bar, b_floor)
        self.env.b_bar /= self.env.b_bar.sum()
        b = float(self.env.b_bar[m_star])

        # --- semantic symbol masking (C6, C8 pre-check) then selection ---
        # FIX: pass b (already resolved above, per Eq. 8's b_m[n]) instead of
        # implicitly using b=1 -- otherwise the feasibility mask here is
        # computed against a different SINR than the one env.step() actually
        # uses for the reward and delay/energy calculation.
        snr = self.env.sinr(m_star, b)
        masked_logits = symbol_logits.copy()
        feasible = []
        for j, s_val in enumerate(cfg.symbol_set):
            eps_hat = self.env.semantic_similarity(s_val, snr)
            if eps_hat < cfg.epsilon_th:
                masked_logits[j] = -1e9
            else:
                feasible.append(j)
        if len(feasible) == 0:
            symbol_idx = len(cfg.symbol_set) - 1   # fallback: max symbols/word
        else:
            probs_s = F.softmax(torch.as_tensor(masked_logits), dim=-1)
            if sample:
                symbol_idx = Categorical(probs_s).sample().item()
            else:
                symbol_idx = int(torch.argmax(probs_s).item())
        num_symbols = cfg.symbol_set[symbol_idx]

        # --- offloading ratio de-normalization (C2) ---
        R = float(np.clip((math.tanh(R_raw) + 1) / 2, 0.0, 1.0))

        return dict(v=v.astype(np.float32), m_star=m_star, R=R,
                    symbol_idx=symbol_idx, num_symbols=num_symbols, b=b)
