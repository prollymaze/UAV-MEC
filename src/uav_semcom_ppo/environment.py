"""
Simulation environment for one UAV mission over N slots.

Devices have fixed random positions in the service area; channel gains follow
a simplified UAV air-to-ground path-loss model. Semantic accuracy is modeled
via a saturating function of symbols/word and channel SNR (stand-in for an
empirical DeepSC-style rate-distortion curve, since the paper's curve-fit
coefficients are not published).
"""

import math
import numpy as np

from .config import Config


class UAVSemanticEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.device_pos = self.rng.uniform(
            low=[0, 0], high=[cfg.L, cfg.W], size=(cfg.M, 2)
        )
        self.state_dim = 2 + cfg.M + cfg.M + 1 + 1   # q(2) + task_bits(M) + channel_gains(M) + E_used(1) + n/N(1)

        # --- jammer (Eq. 6-8): fixed position, transmit power in W ---
        self.jammer_pos = np.array(
            [cfg.L * cfg.jammer_pos_frac[0], cfg.W * cfg.jammer_pos_frac[1]], dtype=np.float32
        )
        self.jammer_power_w = 10 ** (cfg.jammer_power_dbm / 10) / 1000.0
        self.penetration_loss_w = 10 ** (cfg.penetration_loss_dbw / 10)

    def reset(self):
        self.n = 0
        self.q = np.array([self.cfg.L / 2, self.cfg.W / 2], dtype=np.float32)
        self.E_used = 0.0
        self.task_bits = self._sample_task_bits()
        # xi_m[n]: NLoS obstruction flag per device, resampled each episode (Eq. 8)
        self.obstruction = (self.rng.random(self.cfg.M) < self.cfg.obstruction_prob).astype(np.float32)
        # steps_since_served (queueing backlog per device); grows for devices
        # not currently selected, resets to 0 for whichever device is served
        self.wait_time = np.zeros(self.cfg.M, dtype=np.float32)
        # bar_b_m[n]: long-term bandwidth allocation coefficient per device
        # (Algorithm 1, steps 1-2 and 8-10). Initialized to 1/M at the start
        # of each mission, then nudged toward the agent's requested allocation
        # for whichever device is currently scheduled, renormalized so
        # sum_m bar_b_m[n] == 1 at all times.
        self.b_bar = np.full(self.cfg.M, 1.0 / self.cfg.M, dtype=np.float32)
        return self._get_state()

    def _sample_task_bits(self):
        return self.rng.exponential(self.cfg.task_bits_mean, size=self.cfg.M).astype(np.float32)

    def _channel_gains(self):
        """Channel gain per device, given current q (Eq. 6):
        G_m[n] = rho_0 / (||l_u[n+1] - l_m[n]||^2 + H^2).
        FIX: previously used an unrelated Friis free-space path-loss formula
        with a hardcoded, paper-undocumented carrier frequency (2 GHz) and
        the FSPL constant 147.55 -- that formula ignored rho_0 (Table 2:
        -50 dB) entirely and doesn't correspond to any equation in the
        paper. Eq. 6's model is just an inverse-square-law gain scaled by
        the reference-distance gain rho_0."""
        cfg = self.cfg
        rho0_linear = 10 ** (cfg.rho0_db / 10)
        d_sq = np.sum((self.device_pos - self.q) ** 2, axis=1) + cfg.H ** 2
        g_linear = rho0_linear / d_sq
        return g_linear.astype(np.float32)

    def _get_state(self):
        h = self._channel_gains()
        s = np.concatenate([
            self.q / np.array([self.cfg.L, self.cfg.W], dtype=np.float32),
            self.task_bits / self.cfg.task_bits_mean,
            h / (h.max() + 1e-9),
            np.array([self.E_used / self.cfg.E_b], dtype=np.float32),
            np.array([self.n / self.cfg.N], dtype=np.float32),
        ]).astype(np.float32)
        return s

    def _jammer_gain(self):
        """Channel gain from the jammer to the UAV (Eq. 7):
        G_J[n] = rho_0 / (||l_u[n+1] - l_J||^2 + H^2). Same FIX as
        _channel_gains(): use the paper's rho_0-based inverse-square model
        instead of the unrelated hardcoded-carrier-frequency FSPL formula."""
        cfg = self.cfg
        rho0_linear = 10 ** (cfg.rho0_db / 10)
        d_sq = np.sum((self.jammer_pos - self.q) ** 2) + cfg.H ** 2
        return float(rho0_linear / d_sq)

    def sinr(self, m: int, b: float = 1.0) -> float:
        """SINR for device m at the UAV's current position (Eq. 8):
        gamma_m[n] = P_up*G_m[n] / (b_m[n]*(sigma^2 + G_J[n]*P_J + xi_m[n]*P_NLoS)).
        FIX: previously omitted b_m[n] from the denominator entirely, even
        though Eq. 8 explicitly scales the whole noise+interference term by
        the device's bandwidth allocation coefficient. b defaults to 1.0 for
        call sites that only need a bandwidth-agnostic reference SNR (e.g.
        the state observation); pass the actual b_m[n] wherever the true
        Eq. 8 SINR is needed (semantic-similarity/masking, transmission
        delay/energy)."""
        cfg = self.cfg
        h = self._channel_gains()
        g_j = self._jammer_gain()
        interference = b * (
            cfg.noise_power_w
            + g_j * self.jammer_power_w
            + self.obstruction[m] * self.penetration_loss_w
        )
        return float((cfg.p_max_w * h[m]) / max(interference, 1e-30))

    def semantic_similarity(self, num_symbols: int, snr_linear: float) -> float:
        """Saturating rate-distortion proxy: more symbols/word and higher SNR -> higher similarity."""
        s_norm = num_symbols / max(self.cfg.symbol_set)
        snr_db = 10 * np.log10(snr_linear + 1e-12)
        snr_term = 1 / (1 + math.exp(-0.15 * (snr_db - 5)))   # sigmoid in SNR
        eps = 0.5 * s_norm + 0.5 * snr_term
        return float(np.clip(eps, 0.0, 0.999))

    def semantic_compressed_bits(self, raw_bits: float, num_symbols: int) -> float:
        """More symbols/word -> larger transmitted payload (less compression)."""
        ratio = num_symbols / max(self.cfg.symbol_set)
        return raw_bits * (0.15 + 0.35 * ratio)   # 15%-50% of raw size

    def flight_energy(self, v: np.ndarray) -> float:
        """UAV flight energy for this slot (Eq. 12 of the paper):
        E_fly[n] = 0.5 * M_UAV * v[n]^2 * delta_t. Note this model has zero
        cost for hovering (v=0) -- that's the paper's model, not an
        approximation; a real UAV would need hover power too, but this is
        what Eq. 12 actually specifies."""
        cfg = self.cfg
        speed = float(np.linalg.norm(v))
        return 0.5 * cfg.uav_mass_kg * speed ** 2 * cfg.delta

    def step(self, m_star: int, R: float, num_symbols: int, v: np.ndarray, b: float):
        cfg = self.cfg
        # FIX: pass b (the scheduled device's bandwidth coefficient) into
        # sinr() so gamma_m[n] matches Eq. 8 exactly, instead of silently
        # using b=1 (no bandwidth scaling of the interference term).
        snr = self.sinr(m_star, b)

        # Raw task bits (Eq. 1/2/3 in the paper).
        raw_bits = float(self.task_bits[m_star])
        eps = self.semantic_similarity(num_symbols, snr)

        # Semantic compression reduces ONLY the bits sent over the air.  The
        # device CPU (local fraction) and MEC CPU (offloaded fraction) both
        # operate on the original task information, so they use raw_bits for
        # their compute time / energy calculations.
        # FIX: previously D_sem was used for all three quantities (local
        # compute, transmission, MEC compute), cutting compute time and energy
        # by 15-50% and making the per-slot cost 10-50× smaller than the
        # paper's scale, which in turn caused the reward to plateau far above
        # the paper's Fig. 3 convergence values.
        D_sem = self.semantic_compressed_bits(raw_bits, num_symbols)

        # --- delay (Eqs. 11, 13, 14) ---
        local_compute_bits = (1 - R) * raw_bits   # device processes (1-R) of raw task
        off_tx_bits       = R * D_sem              # compressed bits transmitted over the air
        off_compute_bits  = R * raw_bits           # MEC processes R fraction of raw task
        T_loc = local_compute_bits * cfg.cpu_cycles_per_bit / cfg.f_loc_hz
        # FIX (Bug 3): b is the currently-scheduled device's bandwidth
        # allocation coefficient b_m[n] (Eq. 9), produced by ActionMapper via
        # Algorithm 1 steps 8-10 -- a real, policy-controlled, per-device
        # quantity -- rather than a hardcoded 1/M split. This lets the agent
        # shift bandwidth toward a jammed or bandwidth-starved device instead
        # of every device always getting an identical, fixed share.
        effective_bandwidth = b * cfg.bandwidth_hz
        rate = effective_bandwidth * math.log2(1 + snr)
        T_tx = off_tx_bits / max(rate, 1e-6)
        f_uav = cfg.f_uav_max_hz
        T_uav = off_compute_bits * cfg.cpu_cycles_per_bit / max(f_uav, 1e-6)
        T = max(T_loc, T_tx + T_uav)

        # --- energy ---
        E_device  = cfg.kappa_device * (cfg.f_loc_hz ** 2) * local_compute_bits * cfg.cpu_cycles_per_bit
        E_trans   = cfg.p_max_w * T_tx
        E_uav_comp = cfg.kappa_uav * (f_uav ** 2) * off_compute_bits * cfg.cpu_cycles_per_bit
        E_fly = self.flight_energy(v)
        E_total_slot = E_fly + E_uav_comp + E_device + E_trans

        # FIX (Bug 4): the paper's Eq. (19)/(23) reward/cost definition is
        #   varrho[n] = ... + beta*(E_UAV[n] + E_fly[n]) + E_device[n] + E_trans[n]
        # i.e. E_fly IS inside the beta-weighted bracket alongside E_UAV -- it was
        # incorrectly excluded here before. Previously the policy had zero
        # per-step incentive to conserve flight energy (only a slow Lagrangian
        # nudge if the whole-mission budget was exceeded), so it could move more
        # aggressively than the paper's optimum, skewing energy magnitudes/trends
        # in every figure. E_cost_slot now matches Eq. (19)/(23) exactly.
        E_cost_slot = E_uav_comp + E_fly + E_device + E_trans
        rho_cost = cfg.alpha * T + cfg.beta * E_cost_slot

        # --- move UAV (boundary/velocity already enforced by ActionMapper) ---
        self.q = np.clip(self.q + v * cfg.delta, [0, 0], [cfg.L, cfg.W]).astype(np.float32)
        self.E_used += (E_fly + E_uav_comp)

        # --- queueing / fairness bookkeeping ---
        # how long device m_star had been waiting since it was last served
        wait_before_service = float(self.wait_time[m_star])
        # every device accrues one more slot of backlog; the served device resets
        self.wait_time += cfg.delta
        self.wait_time[m_star] = 0.0
        # average current staleness across all devices -- this is what makes
        # M causally affect the policy: more devices sharing a fixed N-slot
        # mission forces a higher average backlog if service is spread evenly
        fairness_penalty = float(np.mean(self.wait_time))

        # --- reward pieces ---
        semantic_shortfall = max(0.0, cfg.epsilon_th - eps)
        energy_pace_budget = (self.n / cfg.N) * cfg.E_b
        energy_overrun = max(0.0, self.E_used - energy_pace_budget)

        info = dict(T=T, E=E_cost_slot, rho=rho_cost, eps=eps,
                    energy_overrun=energy_overrun, semantic_shortfall=semantic_shortfall,
                    E_used=self.E_used, wait_before_service=wait_before_service,
                    completion_time=wait_before_service + T, fairness_penalty=fairness_penalty,
                    E_total_slot=E_total_slot, b=b)

        self.n += 1
        self.task_bits = self._sample_task_bits()
        done = self.n >= cfg.N
        next_state = self._get_state()
        return next_state, rho_cost, semantic_shortfall, energy_overrun, done, info
