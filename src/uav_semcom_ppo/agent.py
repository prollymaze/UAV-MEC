"""
Algorithm 2 -- PPO Training Loop with Lagrangian energy-budget enforcement.

PPOAgent ties together the environment, the ActionMapper (Algorithm 1), the
ActorCritic network, and the RolloutBuffer to run full episodes, compute GAE
advantages, and perform clipped-surrogate PPO updates. A dual variable `mu`
is adapted on a slower timescale to softly enforce the mission-wide energy
budget constraint (C7).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from .config import Config
from .environment import UAVSemanticEnv
from .action_mapper import ActionMapper
from .networks import ActorCritic
from .buffer import RolloutBuffer


class PPOAgent:
    def __init__(self, cfg: Config, env: UAVSemanticEnv):
        self.cfg = cfg
        self.env = env
        self.mapper = ActionMapper(cfg, env)
        self.device = torch.device(cfg.device)

        self.net = ActorCritic(env.state_dim, cfg).to(self.device)
        self.opt_actor = torch.optim.Adam(
            list(self.net.trunk.parameters()) + list(self.net.v_mu.parameters()) +
            [self.net.v_log_std] + list(self.net.device_logits.parameters()) +
            list(self.net.R_mu.parameters()) + [self.net.R_log_std] +
            list(self.net.b_mu.parameters()) + [self.net.b_log_std] +
            list(self.net.symbol_logits.parameters()),
            lr=cfg.lr_actor,
        )
        self.opt_critic = torch.optim.Adam(self.net.value_head.parameters(), lr=cfg.lr_critic)

        self.mu = 0.0  # Lagrange multiplier for mission-wide energy budget (C7)

    # -- act: sample action, return everything needed for the buffer --
    def act(self, state: np.ndarray, q_now: np.ndarray, sample=True):
        s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            v_mu, v_std, device_logits, R_mu, R_std, b_mu, b_std, symbol_logits, value = self.net(s_t)

        v_dist = Normal(v_mu, v_std)
        v_raw = v_dist.sample() if sample else v_mu
        logp_v = v_dist.log_prob(v_raw).sum(-1)

        R_dist = Normal(R_mu, R_std)
        R_raw = R_dist.sample() if sample else R_mu
        logp_R = R_dist.log_prob(R_raw).sum(-1)

        # FIX (Bug 3): sample the bandwidth-request head the same way as R.
        b_dist = Normal(b_mu, b_std)
        b_raw = b_dist.sample() if sample else b_mu
        logp_b = b_dist.log_prob(b_raw).sum(-1)

        device_probs = F.softmax(device_logits, dim=-1)
        m_cat = Categorical(device_probs)

        symbol_probs_raw = F.softmax(symbol_logits, dim=-1)
        s_cat_raw = Categorical(symbol_probs_raw)

        mapped = self.mapper.map(
            v_raw.cpu().numpy().flatten(),
            device_logits.cpu().numpy().flatten(),
            float(R_raw.item()),
            symbol_logits.cpu().numpy().flatten(),
            q_now,
            float(b_raw.item()),
            sample=sample,
        )

        logp_m = m_cat.log_prob(torch.tensor(mapped["m_star"], device=self.device).float().long())
        logp_s = s_cat_raw.log_prob(torch.tensor(mapped["symbol_idx"], device=self.device).float().long())

        return mapped, dict(
            v_raw=v_raw.cpu().numpy().flatten(), R_raw=float(R_raw.item()), b_raw=float(b_raw.item()),
            logp_v=float(logp_v.item()), logp_m=float(logp_m.item()),
            logp_R=float(logp_R.item()), logp_b=float(logp_b.item()), logp_s=float(logp_s.item()),
            value=float(value.item()),
        )

    # -- recompute log-probs / value / entropy for a batch under current params --
    def evaluate_actions(self, states, v_raw, m_star, R_raw, b_raw, symbol_idx):
        v_mu, v_std, device_logits, R_mu, R_std, b_mu, b_std, symbol_logits, value = self.net(states)

        v_dist = Normal(v_mu, v_std)
        logp_v = v_dist.log_prob(v_raw).sum(-1)
        ent_v = v_dist.entropy().sum(-1)

        R_dist = Normal(R_mu, R_std)
        logp_R = R_dist.log_prob(R_raw).sum(-1)
        ent_R = R_dist.entropy().sum(-1)

        # FIX (Bug 3): bandwidth head, evaluated the same way as R.
        b_dist = Normal(b_mu, b_std)
        logp_b = b_dist.log_prob(b_raw).sum(-1)
        ent_b = b_dist.entropy().sum(-1)

        device_probs = F.softmax(device_logits, dim=-1)
        m_cat = Categorical(device_probs)
        logp_m = m_cat.log_prob(m_star)
        ent_m = m_cat.entropy()

        symbol_probs = F.softmax(symbol_logits, dim=-1)
        s_cat = Categorical(symbol_probs)
        logp_s = s_cat.log_prob(symbol_idx)
        ent_s = s_cat.entropy()

        total_logp = logp_v + logp_m + logp_R + logp_b + logp_s
        total_entropy = ent_v + ent_m + ent_R + ent_b + ent_s
        return total_logp, value, total_entropy

    def compute_gae(self, rewards, values, dones, last_value):
        cfg = self.cfg
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        values_ext = values + [last_value]
        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + cfg.gamma * values_ext[t + 1] * mask - values_ext[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def update(self, buf: RolloutBuffer, last_value: float):
        cfg = self.cfg
        advantages, returns = self.compute_gae(buf.rewards, buf.values, buf.dones, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.as_tensor(np.array(buf.states), dtype=torch.float32, device=self.device)
        v_raw = torch.as_tensor(np.array(buf.v_raw), dtype=torch.float32, device=self.device)
        R_raw = torch.as_tensor(np.array(buf.R_raw), dtype=torch.float32, device=self.device).unsqueeze(-1)
        b_raw = torch.as_tensor(np.array(buf.b_raw), dtype=torch.float32, device=self.device).unsqueeze(-1)
        m_star = torch.as_tensor(np.array(buf.m_star), dtype=torch.long, device=self.device)
        symbol_idx = torch.as_tensor(np.array(buf.symbol_idx), dtype=torch.long, device=self.device)
        old_logp = torch.as_tensor(
            np.array(buf.logp_v) + np.array(buf.logp_m) + np.array(buf.logp_R)
            + np.array(buf.logp_b) + np.array(buf.logp_s),
            dtype=torch.float32, device=self.device,
        )
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        n = states.shape[0]
        idx_all = np.arange(n)

        for _ in range(cfg.train_epochs_per_update):
            np.random.shuffle(idx_all)
            for start in range(0, n, cfg.minibatch_size):
                mb_idx = idx_all[start:start + cfg.minibatch_size]
                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=self.device)

                new_logp, value_pred, entropy = self.evaluate_actions(
                    states[mb_idx_t], v_raw[mb_idx_t], m_star[mb_idx_t],
                    R_raw[mb_idx_t], b_raw[mb_idx_t], symbol_idx[mb_idx_t],
                )

                ratio = torch.exp(new_logp - old_logp[mb_idx_t])
                surr1 = ratio * adv_t[mb_idx_t]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv_t[mb_idx_t]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value_pred, ret_t[mb_idx_t])
                entropy_loss = -entropy.mean()

                loss = policy_loss + cfg.vf_coef * value_loss + cfg.entropy_coef * entropy_loss

                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt_actor.step()
                self.opt_critic.step()

    def evaluate(self, episodes: int = 5):
        """Run deterministic (greedy) rollouts under the current policy and
        report average per-step task-completion delay and energy consumption
        -- the metrics plotted in Figs. 4-8 of the paper.
        Delay is info["T"] = max(T_loc, T_tx + T_uav), which is the task
        processing+transmission delay from the paper's Eqs. 11/13/14 and
        exactly what the reward's rho_cost term penalises."""
        cfg = self.cfg
        delays, energies, fairness_vals = [], [], []
        for _ in range(episodes):
            state = self.env.reset()
            for _ in range(cfg.N):
                q_now = self.env.q.copy()
                mapped, _ = self.act(state, q_now, sample=False)
                next_state, rho_cost, shortfall, overrun, done, info = self.env.step(
                    mapped["m_star"], mapped["R"], mapped["num_symbols"], mapped["v"], mapped["b"]
                )
                # Use info["T"] for task completion time: that is exactly
                # max(T_loc, T_tx + T_uav) from the paper's Eqs. 11/13/14 --
                # the processing+transmission delay for the task handled in
                # this slot. This is what the reward's rho_cost penalises and
                # what the paper's Figs. 4b/5b/6b/7/8 plot on their y-axes.
                # FIX: previously used info["completion_time"] = wait_before_
                # service + T, which adds per-device queueing backlog (time
                # since last served) and is NOT part of the paper's Eq. 11-14
                # definition of task completion time. The wait term grows
                # linearly with N/M (slots-per-device), dominating T and
                # making delay artificially increase with M for reasons
                # unrelated to the physics being swept.
                delays.append(info["T"])
                fairness_vals.append(info["fairness_penalty"])
                energies.append(info["E"])
                state = next_state
        n = max(1, len(delays))
        # FIX: return standard error of the mean alongside the raw population
        # std. Raw std is computed over individual per-slot samples (only one
        # device served per slot, with widely varying task size/channel gain),
        # so it can legitimately exceed the mean -- that's fine as a spread
        # diagnostic, but plotting it directly as a symmetric error bar around
        # the mean produces a nonsensical negative lower bound for a
        # strictly-nonnegative quantity (Joules, seconds). SEM = std/sqrt(n)
        # is the correct quantity for "how uncertain is this mean estimate",
        # which is what Figs. 4-8's error bars are meant to convey.
        return dict(
            avg_delay=float(np.mean(delays)), std_delay=float(np.std(delays)),
            sem_delay=float(np.std(delays) / np.sqrt(n)),
            avg_energy=float(np.mean(energies)), std_energy=float(np.std(energies)),
            sem_energy=float(np.std(energies) / np.sqrt(n)),
            avg_fairness=float(np.mean(fairness_vals)), std_fairness=float(np.std(fairness_vals)),
        )

    def train(self):
        """Algorithm 2: end-to-end PPO training loop with Lagrangian energy enforcement.

        FIX: previously called self.update() after every single cfg.N-step
        episode (~40 samples, ~4 gradient steps total per update), which was
        too little on-policy data for PPO's advantage estimates to be
        meaningful -- the reward curve stayed flat instead of improving (see
        convergence.png). Table 3 lists T_epi=100 as a separate PPO batch-size
        hyperparameter from Table 2's N=40 mission length, so we now
        accumulate ceil(steps_per_update / N) full episodes into one shared
        RolloutBuffer before each update, matching that intent. Each
        accumulated episode still ends with done=True, so GAE's per-step
        done-mask correctly resets the advantage bootstrap at every episode
        boundary within the batch -- concatenating several finished
        trajectories into one buffer is standard practice and doesn't
        require any change to compute_gae() or the mid-episode bootstrap
        value (still 0.0, since the batch always ends on a completed episode).
        """
        cfg = self.cfg
        history = []
        episodes_per_update = max(1, round(cfg.steps_per_update / cfg.N))
        print(f"[PPO] batching {episodes_per_update} episode(s) "
              f"(~{episodes_per_update * cfg.N} steps) per update "
              f"(target steps_per_update={cfg.steps_per_update})")

        buf = RolloutBuffer()
        episodes_since_update = 0

        for ep in range(cfg.episodes):
            state = self.env.reset()
            ep_cost, ep_energy, ep_shortfall, ep_reward = 0.0, 0.0, 0.0, 0.0

            for _ in range(cfg.N):
                q_now = self.env.q.copy()
                mapped, act_info = self.act(state, q_now, sample=True)

                next_state, rho_cost, shortfall, overrun, done, info = self.env.step(
                    mapped["m_star"], mapped["R"], mapped["num_symbols"], mapped["v"], mapped["b"]
                )

                reward = (
                    -rho_cost
                    - self.mu * overrun
                    - cfg.eta_semantic * shortfall
                    - cfg.fairness_weight * info["fairness_penalty"]
                )

                buf.add(
                    state, act_info["value"], reward,
                    act_info["v_raw"], act_info["R_raw"], act_info["b_raw"],
                    mapped["m_star"], mapped["symbol_idx"],
                    act_info["logp_v"], act_info["logp_m"], act_info["logp_R"],
                    act_info["logp_b"], act_info["logp_s"],
                    done,
                )

                ep_cost += rho_cost
                ep_energy = info["E_used"]
                ep_shortfall += shortfall
                ep_reward += reward
                state = next_state

            episodes_since_update += 1

            # --- slower-timescale dual ascent on mu (C7 enforcement) ---
            # kept per-episode: E_b is a per-mission energy budget, independent
            # of how many episodes get batched together for the PPO update.
            self.mu = max(0.0, self.mu + cfg.lr_mu * (ep_energy - cfg.E_b))

            history.append(dict(episode=ep, total_reward=ep_reward, total_cost=ep_cost,
                                 energy_used=ep_energy, mu=self.mu, semantic_shortfall=ep_shortfall))

            if ep % 20 == 0:
                print(f"[ep {ep:4d}] reward={ep_reward:10.3f}  cost={ep_cost:10.3f}"
                      f"  E_used={ep_energy:10.1f}/{cfg.E_b:.0f}"
                      f"  mu={self.mu:8.4f}  semantic_shortfall={ep_shortfall:6.3f}")

            is_last_episode = (ep == cfg.episodes - 1)
            if episodes_since_update >= episodes_per_update or is_last_episode:
                # bootstrap value for the (non-existent) next state after the
                # last episode in this batch -- always 0.0 since every
                # episode in the batch ends with done=True.
                last_value = 0.0
                self.update(buf, last_value)
                buf.clear()
                episodes_since_update = 0

        return history
