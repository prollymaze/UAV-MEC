"""
Fixed-capacity replay buffer for off-policy algorithms (DDPG).

This is the off-policy counterpart to buffer.py's RolloutBuffer: instead of
being built fresh each episode and discarded after one PPO update, this
persists across the whole training run, holds a fixed-size window of past
transitions, and supports random-batch sampling -- exactly what DDPG needs
since it can (and should) reuse old experience for many gradient updates.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.states[idx], self.actions[idx], self.rewards[idx],
            self.next_states[idx], self.dones[idx],
        )

    def __len__(self):
        return self.size
