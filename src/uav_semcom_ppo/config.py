"""
Configuration for the UAV-assisted MEC with semantic communication environment
and the PPO agent that controls it.

All physical constants (geometry, propulsion power model, comms/task params),
the semantic-symbol set, the energy budget, and PPO hyperparameters live here
so the rest of the codebase never hardcodes a "magic number" -- everything is
threaded through a single Config instance.
"""

import torch
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # --- system geometry (Table 2) ---
    L: float = 400.0            # service area length (m)
    W: float = 400.0            # service area width (m)
    H: float = 300.0            # fixed UAV altitude (m)
    v_max: float = 15.0         # max UAV velocity (m/s)
    delta: float = 1.0          # slot duration (s)
    N: int = 40                 # slots per episode / mission horizon (Table 2's "N")
    M: int = 15                 # number of IoT devices

    # --- semantic symbol set (Table 2: K = 1,2,...,20) ---
    symbol_set: Tuple[int, ...] = tuple(range(1, 21))
    epsilon_th: float = 0.9     # semantic similarity threshold

    # --- task / comms (Table 2) ---
    bandwidth_hz: float = 1e7           # B = 10 MHz (paper sweeps 6-12 MHz in Fig. 8; not a fixed table value; default to midpoint)
    noise_power_w: float = 1e-14        # sigma^2 = -140 dBW
    p_max_w: float = 0.2                # P_up, device max tx power
    # FIX: carrier_freq_hz removed. It was only used by an ad hoc Friis
    # free-space path-loss formula (20*log10(d) + 20*log10(f) - 147.55) that
    # doesn't appear anywhere in the paper and silently ignored the paper's
    # own channel-gain parameter (rho_0, Table 2). Replaced by rho0_db below,
    # feeding Eq. 6/7's actual formula: G_m[n] = rho_0 / (||l_u-l_m||^2 + H^2).
    rho0_db: float = -50.0               # Table 2: channel gain at 1m reference distance
    task_bits_mean: float = 5e5         # mean raw task size (bits); paper sweeps 0.4-1.6 Mbits in Figs. 7-8
    cpu_cycles_per_bit: float = 1000.0  # c_m

    f_loc_hz: float = 1.05e9     # f_device, IoT device computing capability
    f_uav_max_hz: float = 2.3e9  # f_UAV, MEC server computing capability
    kappa_device: float = 1e-27  # effective capacity coefficient of the processors, kappa
    kappa_uav: float = 1e-27

    # --- jammer (Eq. 6-8) ---
    jammer_power_dbm: float = 20.0            # P_J, jammer transmit power (dBm)
    jammer_pos_frac: Tuple[float, float] = (0.5, 0.5)  # fixed jammer location, as fraction of (L, W)
    penetration_loss_dbw: float = -100.0      # P_NLoS, obstruction interference power (dBW)
    obstruction_prob: float = 0.2             # probability a device's link is obstructed (xi_m) each episode

    # --- UAV propulsion (Eq. 12: E_fly = 0.5 * M_UAV * v^2 * delta) ---
    uav_mass_kg: float = 8.5    # M_UAV, Table 2

    # --- energy budget (Table 2) ---
    E_b: float = 8e4   # total mission energy budget (J) = 80 kJ

    # --- cost weights (Table 3) ---
    alpha: float = 2.0      # delay weight
    beta: float = 0.002     # energy consumption weight

    # --- fairness / queueing pressure ---
    fairness_weight: float = 0.02   # reward penalty on average device staleness; makes M causally matter

    # --- reward penalty coefficients ---
    # FIX: was 5.0. Eq. 23 defines F_epsilon = 10*min(0, max(-1, eps-eps_th)),
    # i.e. a penalty coefficient of 10 on the semantic-similarity shortfall,
    # not 5.
    eta_semantic: float = 10.0    # penalty coeff for residual semantic-QoS shortfall
    psi_boundary: float = 0.0     # boundary is handled structurally (Algorithm 1), kept for logging only

    # --- PPO hyperparameters (Table 3) ---
    gamma: float = 0.9              # reward discount factor
    gae_lambda: float = 0.95        # GAE parameter lambda
    clip_ratio: float = 0.2         # truncation factor epsilon (not given in Table 3; kept at the common default)
    lr_actor: float = 1e-3          # paper uses a single learning rate eta = 0.001 for both networks
    lr_critic: float = 1e-3
    lr_mu: float = 1e-3             # dual variable (Lagrange multiplier) learning rate -- not in the paper
    train_epochs_per_update: int = 4    # N_epoch
    minibatch_size: int = 256           # batch size for each update
    # FIX: Table 3 lists T_epi = 100 ("number of steps per episode") as a
    # PPO-specific hyperparameter, distinct from Table 2's N = 40 (mission
    # length in slots). This is the on-policy batch size PPO accumulates
    # before each update -- previously the code updated after every single
    # 40-step mission, giving PPO only ~40 samples and ~4 gradient steps per
    # update, which was too little to learn from (flat reward curve, see
    # convergence.png). PPOAgent.train() now batches ceil(steps_per_update / N)
    # episodes together before calling update().
    steps_per_update: int = 100
    entropy_coef: float = 0.01          # not given in Table 3; kept at a common default
    vf_coef: float = 0.5                # not given in Table 3; kept at a common default
    max_grad_norm: float = 0.5          # not given in Table 3; kept at a common default
    episodes: int = 2000                # N_epi (override with a smaller value via CLI for quicker runs)
    hidden_layer_sizes: Tuple[int, int] = (128, 64)   # L = [128, 64]
    hidden_dim: int = 256               # kept for backward compatibility; unused now that hidden_layer_sizes exists
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # --- DDPG hyperparameters (baseline for comparison; not in the paper's
    # tables since Table 3 only covers the PPO agent) ---
    ddpg_actor_lr: float = 1e-3
    ddpg_critic_lr: float = 1e-3
    ddpg_tau: float = 0.005              # soft target-network update rate
    ddpg_replay_capacity: int = 50_000
    ddpg_batch_size: int = 256
    ddpg_warmup_steps: int = 200         # fully-random-action steps before training starts
    ddpg_noise_std: float = 0.3          # initial Gaussian exploration noise (on raw actor outputs)
    ddpg_noise_decay: float = 0.9995     # multiplicative decay applied once per env step
    ddpg_min_noise_std: float = 0.05
    ddpg_policy_update_delay: int = 2    # TD3: update actor every N critic updates (reduces variance)
    # If True (default), DDPG uses TD3 stabilization tricks (double critics,
    # delayed policy updates, target-policy smoothing). These make it a
    # meaningfully stronger/more-stable algorithm than the paper's plain
    # DDPG-MEC-SC baseline -- fine if you just want a stable baseline to
    # compare against, but it breaks the paper's Fig. 3 narrative (DDPG
    # should show large early fluctuations and converge *below* PPO). Set
    # to False to run single-critic, undelayed, no-smoothing vanilla DDPG
    # that matches what the paper actually benchmarks against.
    ddpg_use_td3_tricks: bool = False
