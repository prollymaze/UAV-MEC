"""
DDPG baseline agent, for comparison against PPOAgent (DDPG-MEC-SC vs.
PPO-MEC-SC, matching the paper's naming convention for its baseline
comparisons in Figs. 3-8).

DDPGAgent deliberately mirrors PPOAgent's public interface --
__init__(cfg, env), train() -> history, evaluate(episodes) -> metrics dict
-- so every sweep script can build/train/evaluate either algorithm through
the exact same code path (see scripts/sweep_utils.py).

Unlike PPO (on-policy, one big update per episode from freshly-collected
data), DDPG is off-policy: it keeps a persistent ReplayBuffer across the
whole run and performs one gradient update per environment step, sampling a
random batch of past transitions each time. Exploration comes from adding
Gaussian noise to the actor's raw (pre-ActionMapper) outputs, decayed over
training, plus a fully-random-action warmup period at the very start to
seed the replay buffer with diverse experience.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .environment import UAVSemanticEnv
from .action_mapper import ActionMapper
from .ddpg_networks import DDPGActor, DDPGCritic, action_dim
from .replay_buffer import ReplayBuffer


class DDPGAgent:
    def __init__(self, cfg: Config, env: UAVSemanticEnv):
        self.cfg = cfg
        print(f"[DDPGAgent] instantiated with ddpg_use_td3_tricks={cfg.ddpg_use_td3_tricks}", flush=True)
        self.env = env
        self.mapper = ActionMapper(cfg, env)
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)

        self.act_dim = action_dim(cfg)

        self.actor = DDPGActor(env.state_dim, cfg).to(self.device)
        self.actor_target = DDPGActor(env.state_dim, cfg).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # TD3-style: dual critics for overestimation prevention
        self.critic1 = DDPGCritic(env.state_dim, self.act_dim, cfg).to(self.device)
        self.critic1_target = DDPGCritic(env.state_dim, self.act_dim, cfg).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = DDPGCritic(env.state_dim, self.act_dim, cfg).to(self.device)
        self.critic2_target = DDPGCritic(env.state_dim, self.act_dim, cfg).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.ddpg_actor_lr)
        self.opt_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=cfg.ddpg_critic_lr)
        self.opt_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=cfg.ddpg_critic_lr)

        self.replay = ReplayBuffer(cfg.ddpg_replay_capacity, env.state_dim, self.act_dim)

        self.noise_std = cfg.ddpg_noise_std
        self.mu = 0.0          # Lagrange multiplier for the energy budget, same mechanism as PPOAgent
        self.total_steps = 0

    @staticmethod
    def _raw_action_vector(v_raw, device_logits, R_raw, b_raw, symbol_logits):
        return np.concatenate(
            [v_raw, device_logits, np.array([R_raw], dtype=np.float32),
             np.array([b_raw], dtype=np.float32), symbol_logits]
        ).astype(np.float32)

    def _normalize_action_torch(self, v_raw, device_logits, R_raw, b_raw, symbol_logits):
        """Torch equivalent of the normalization applied before storing actions
        in the replay buffer (see train()). CRITICAL: this must be used
        wherever the critic is *queried* with a freshly-produced actor output
        (target-Q computation, actor-loss computation) -- not just when
        writing to the buffer -- otherwise the critic is trained on bounded,
        consistently-scaled inputs but evaluated on raw unbounded logits it
        never saw during training, which is exactly the mismatch that caused
        DDPG's Q-value blowups in Fig. 3 and the M=20 spike in Fig. 4."""
        norm_v = torch.tanh(v_raw)
        norm_dev = torch.tanh(device_logits / 10.0)
        norm_r = torch.tanh(R_raw)
        norm_b = torch.tanh(b_raw)
        norm_sym = torch.tanh(symbol_logits / 10.0)
        return torch.cat([norm_v, norm_dev, norm_r, norm_b, norm_sym], dim=-1)

    def act(self, state: np.ndarray, q_now: np.ndarray, explore: bool = True):
        """Query the actor, optionally add exploration noise to its raw
        outputs, then map to a legal action. Returns (mapped_action,
        raw_action_vector) -- the latter is what gets stored in the replay
        buffer and what the critic conditions on."""
        s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            v_raw, device_logits, R_raw, b_raw, symbol_logits = self.actor(s_t)

        v_raw = v_raw.cpu().numpy().flatten()
        device_logits = device_logits.cpu().numpy().flatten()
        R_raw = float(R_raw.item())
        b_raw = float(b_raw.item())
        symbol_logits = symbol_logits.cpu().numpy().flatten()

        if explore:
            v_raw = v_raw + self.rng.normal(0, self.noise_std, size=v_raw.shape)
            device_logits = device_logits + self.rng.normal(0, self.noise_std, size=device_logits.shape)
            R_raw = R_raw + float(self.rng.normal(0, self.noise_std))
            b_raw = b_raw + float(self.rng.normal(0, self.noise_std))
            symbol_logits = symbol_logits + self.rng.normal(0, self.noise_std, size=symbol_logits.shape)

        raw_action = self._raw_action_vector(v_raw, device_logits, R_raw, b_raw, symbol_logits)
        mapped = self.mapper.map(v_raw, device_logits, R_raw, symbol_logits, q_now, b_raw, sample=False)
        return mapped, raw_action

    def _random_action(self, q_now: np.ndarray):
        """Fully random raw action, used only during the warmup period to
        seed the replay buffer with diverse experience before the actor is
        trustworthy enough to explore around."""
        v_raw = self.rng.normal(0, 1.0, size=2).astype(np.float32)
        device_logits = self.rng.normal(0, 1.0, size=self.cfg.M).astype(np.float32)
        R_raw = float(self.rng.normal(0, 1.0))
        b_raw = float(self.rng.normal(0, 1.0))
        symbol_logits = self.rng.normal(0, 1.0, size=len(self.cfg.symbol_set)).astype(np.float32)
        raw_action = self._raw_action_vector(v_raw, device_logits, R_raw, b_raw, symbol_logits)
        mapped = self.mapper.map(v_raw, device_logits, R_raw, symbol_logits, q_now, b_raw, sample=False)
        return mapped, raw_action

    def update(self):
        cfg = self.cfg
        if len(self.replay) < cfg.ddpg_batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(cfg.ddpg_batch_size, self.rng)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        use_td3 = cfg.ddpg_use_td3_tricks

        # --- critic update: TD3-style dual critic with target policy smoothing,
        # or plain single-critic DDPG when ddpg_use_td3_tricks=False (paper mode) ---
        with torch.no_grad():
            nv, ndl, nR, nb, nsl = self.actor_target(next_states_t)
            # FIX (Bug 1): normalize the target actor's raw output the same way
            # actions are normalized before being stored in the replay buffer.
            # Previously this concatenated raw, unbounded logits directly --
            # the critic was being queried far outside the input range it was
            # actually trained on.
            next_action = self._normalize_action_torch(nv, ndl, nR, nb, nsl)

            if use_td3:
                # Target policy smoothing: add small noise to target actions
                policy_noise = 0.2 * torch.randn_like(next_action)
                policy_noise = torch.clamp(policy_noise, -0.5, 0.5)  # clip to [-0.5, 0.5]
                next_action = torch.clamp(next_action + policy_noise, -1.0, 1.0)

            target_q1 = self.critic1_target(next_states_t, next_action)
            if use_td3:
                # Double Q-learning: take minimum of two critic estimates
                target_q2 = self.critic2_target(next_states_t, next_action)
                target_q = torch.min(target_q1, target_q2)
            else:
                target_q = target_q1
            y = rewards_t + cfg.gamma * (1.0 - dones_t) * target_q

        # Update critic 1 (always)
        q1 = self.critic1(states_t, actions_t)
        critic1_loss = F.mse_loss(q1, y)
        self.opt_critic1.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), cfg.max_grad_norm)
        self.opt_critic1.step()

        # Update critic 2 only in TD3 mode -- kept in sync with critic1's
        # target either way so switching modes mid-run doesn't leave a stale
        # network around, but its Q-estimate isn't used for targets/actor
        # gradient when use_td3=False.
        if use_td3:
            q2 = self.critic2(states_t, actions_t)
            critic2_loss = F.mse_loss(q2, y)
            self.opt_critic2.zero_grad()
            critic2_loss.backward()
            nn.utils.clip_grad_norm_(self.critic2.parameters(), cfg.max_grad_norm)
            self.opt_critic2.step()

        # --- actor update: delayed (every N critic updates) in TD3 mode,
        # every step in plain DDPG mode ---
        policy_update_delay = cfg.ddpg_policy_update_delay if use_td3 else 1
        if self.total_steps % policy_update_delay == 0:
            v, dl, R, b, sl = self.actor(states_t)
            # FIX (Bug 1): same normalization fix as the target-Q computation
            # above -- the actor's gradient must flow through the same bounded
            # representation the critic was trained on, or the policy-gradient
            # direction is computed against an out-of-distribution critic query.
            action_pred = self._normalize_action_torch(v, dl, R, b, sl)
            actor_loss = -self.critic1(states_t, action_pred).mean()
            self.opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
            self.opt_actor.step()

            # --- soft (Polyak) update of all target networks ---
            with torch.no_grad():
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.mul_(1 - cfg.ddpg_tau).add_(cfg.ddpg_tau * p)
                for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    tp.mul_(1 - cfg.ddpg_tau).add_(cfg.ddpg_tau * p)
                if use_td3:
                    for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                        tp.mul_(1 - cfg.ddpg_tau).add_(cfg.ddpg_tau * p)

    def train(self):
        cfg = self.cfg
        history = []

        for ep in range(cfg.episodes):
            state = self.env.reset()
            ep_cost, ep_energy, ep_shortfall, ep_reward = 0.0, 0.0, 0.0, 0.0

            for _ in range(cfg.N):
                q_now = self.env.q.copy()
                self.total_steps += 1

                if self.total_steps <= cfg.ddpg_warmup_steps:
                    mapped, raw_action = self._random_action(q_now)
                else:
                    mapped, raw_action = self.act(state, q_now, explore=True)

                next_state, rho_cost, shortfall, overrun, done, info = self.env.step(
                    mapped["m_star"], mapped["R"], mapped["num_symbols"], mapped["v"], mapped["b"]
                )

                reward = (
                    -rho_cost
                    - self.mu * overrun
                    - cfg.eta_semantic * shortfall
                    - cfg.fairness_weight * info["fairness_penalty"]
                )

                # CRITICAL FIX: Normalize the action before storing in replay buffer.
                # Raw actor outputs (logits) are unbounded; storing them as-is causes
                # Q-value explosion. Normalize all components to [-1, 1] so the critic
                # never sees values outside a reasonable range.
                norm_v = np.tanh(raw_action[0:2])  # v normalized to [-1, 1]
                norm_dev = np.tanh(raw_action[2:2+cfg.M] / 10)  # device_logits scaled then tanh
                norm_r = np.tanh(raw_action[2+cfg.M:3+cfg.M])   # R normalized
                # FIX (Bug 3): normalize the new bandwidth-request component the
                # same way as R (single scalar, plain tanh).
                norm_b = np.tanh(raw_action[3+cfg.M:4+cfg.M])   # b normalized
                norm_sym = np.tanh(raw_action[4+cfg.M:] / 10)   # symbol_logits scaled then tanh
                normalized_action = np.concatenate([norm_v, norm_dev, norm_r, norm_b, norm_sym]).astype(np.float32)
                
                self.replay.add(state, normalized_action, reward, next_state, done)
                self.update()
                self.noise_std = max(cfg.ddpg_min_noise_std, self.noise_std * cfg.ddpg_noise_decay)

                ep_cost += rho_cost
                ep_energy = info["E_used"]
                ep_shortfall += shortfall
                ep_reward += reward
                state = next_state

            self.mu = max(0.0, self.mu + cfg.lr_mu * (ep_energy - cfg.E_b))

            history.append(dict(episode=ep, total_reward=ep_reward, total_cost=ep_cost,
                                 energy_used=ep_energy, mu=self.mu, semantic_shortfall=ep_shortfall))

            if ep % 20 == 0:
                print(f"[DDPG ep {ep:4d}] reward={ep_reward:10.3f}  cost={ep_cost:10.3f}"
                      f"  E_used={ep_energy:10.1f}/{cfg.E_b:.0f}"
                      f"  mu={self.mu:8.4f}  noise_std={self.noise_std:.3f}")

        return history

    def evaluate(self, episodes: int = 5):
        """Run deterministic (no-noise) rollouts under the current policy
        and report average delay/energy -- same metrics and same method
        signature as PPOAgent.evaluate(), for apples-to-apples comparison."""
        cfg = self.cfg
        # FIX: use info["completion_time"] (per-task processing+transmission
        # delay, Eqs. 11/13/14) as the reported delay metric, matching what the
        # paper's Figs. 4-8 actually plot -- not fairness_penalty (queueing
        # backlog across all devices), which is dominated by N/M scheduling
        # cadence rather than the swept physical parameter. See PPOAgent.evaluate()
        # for the same fix and rationale; fairness_penalty is kept as a
        # secondary diagnostic only.
        delays, energies, fairness_vals = [], [], []
        for _ in range(episodes):
            state = self.env.reset()
            for _ in range(cfg.N):
                q_now = self.env.q.copy()
                mapped, _ = self.act(state, q_now, explore=False)
                next_state, rho_cost, shortfall, overrun, done, info = self.env.step(
                    mapped["m_star"], mapped["R"], mapped["num_symbols"], mapped["v"], mapped["b"]
                )
                delays.append(info["completion_time"])
                fairness_vals.append(info["fairness_penalty"])
                energies.append(info["E"])
                state = next_state
        n = max(1, len(delays))
        # FIX: same SEM fix as PPOAgent.evaluate() -- see that method's
        # comment for the rationale.
        return dict(
            avg_delay=float(np.mean(delays)), std_delay=float(np.std(delays)),
            sem_delay=float(np.std(delays) / np.sqrt(n)),
            avg_energy=float(np.mean(energies)), std_energy=float(np.std(energies)),
            sem_energy=float(np.std(energies) / np.sqrt(n)),
            avg_fairness=float(np.mean(fairness_vals)), std_fairness=float(np.std(fairness_vals)),
        )
