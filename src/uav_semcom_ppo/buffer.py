"""
Simple on-policy rollout buffer. Collects one episode's worth of transitions
(states, raw continuous actions, discrete action indices, log-probs, values,
rewards, done flags) to be consumed by PPOAgent.update().
"""


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states, self.values, self.rewards = [], [], []
        self.v_raw, self.R_raw, self.b_raw = [], [], []
        self.m_star, self.symbol_idx = [], []
        self.logp_v, self.logp_m, self.logp_R, self.logp_b, self.logp_s = [], [], [], [], []
        self.dones = []

    def add(self, s, value, reward, v_raw, R_raw, b_raw, m_star, symbol_idx,
            logp_v, logp_m, logp_R, logp_b, logp_s, done):
        self.states.append(s)
        self.values.append(value)
        self.rewards.append(reward)
        self.v_raw.append(v_raw)
        self.R_raw.append(R_raw)
        self.b_raw.append(b_raw)
        self.m_star.append(m_star)
        self.symbol_idx.append(symbol_idx)
        self.logp_v.append(logp_v)
        self.logp_m.append(logp_m)
        self.logp_R.append(logp_R)
        self.logp_b.append(logp_b)
        self.logp_s.append(logp_s)
        self.dones.append(done)
