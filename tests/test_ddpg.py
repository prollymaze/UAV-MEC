"""
Minimal sanity tests for DDPGAgent. Run with: pytest tests/
These check that DDPG trains and evaluates without error and returns the
same result shape as PPOAgent -- not that it has converged to anything good.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uav_semcom_ppo import Config, UAVSemanticEnv, DDPGAgent, set_seed


def make_agent(episodes=4, M=4):
    cfg = Config(episodes=episodes, N=5, M=M, seed=0, ddpg_warmup_steps=5, ddpg_batch_size=8)
    set_seed(cfg.seed)
    env = UAVSemanticEnv(cfg)
    agent = DDPGAgent(cfg, env)
    return cfg, agent


def test_ddpg_trains_without_error():
    cfg, agent = make_agent()
    history = agent.train()
    assert len(history) == cfg.episodes
    assert "total_reward" in history[0]


def test_ddpg_evaluate_returns_expected_keys():
    cfg, agent = make_agent()
    agent.train()
    result = agent.evaluate(episodes=2)
    for key in ("avg_delay", "std_delay", "avg_energy", "std_energy"):
        assert key in result
        assert result[key] == result[key]  # not NaN


def test_ddpg_replay_buffer_fills_up():
    cfg, agent = make_agent(episodes=4, M=4)
    agent.train()
    # N=5 steps/episode * 4 episodes = 20 transitions added
    assert len(agent.replay) == cfg.N * cfg.episodes
