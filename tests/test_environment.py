"""
Minimal sanity tests. Run with:  pytest tests/
These check shapes and basic physical/logical invariants, not RL convergence.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uav_semcom_ppo import Config, UAVSemanticEnv, ActionMapper


def make_env():
    cfg = Config(episodes=1, N=5, M=3, seed=0)
    env = UAVSemanticEnv(cfg)
    return cfg, env


def test_reset_state_shape():
    cfg, env = make_env()
    s = env.reset()
    assert s.shape == (env.state_dim,)
    assert env.state_dim == 2 + cfg.M + cfg.M + 1 + 1


def test_step_runs_and_returns_expected_types():
    cfg, env = make_env()
    env.reset()
    v = np.array([1.0, -1.0], dtype=np.float32)
    # b is the scheduled device's bandwidth allocation coefficient (Eq. 9),
    # normally produced by ActionMapper.map() via Algorithm 1 steps 8-10;
    # here we pass a plausible standalone value since we're calling
    # env.step() directly without going through the mapper.
    next_state, rho_cost, shortfall, overrun, done, info = env.step(
        m_star=0, R=0.5, num_symbols=cfg.symbol_set[-1], v=v, b=1.0 / cfg.M
    )
    assert next_state.shape == (env.state_dim,)
    assert rho_cost >= 0.0
    assert shortfall >= 0.0
    assert overrun >= 0.0
    assert isinstance(done, (bool, np.bool_))
    assert 0.0 <= info["b"] <= 1.0


def test_action_mapper_respects_boundary_and_speed():
    cfg, env = make_env()
    env.reset()
    mapper = ActionMapper(cfg, env)

    # Push a huge raw velocity toward the corner to force boundary clipping.
    v_raw = np.array([100.0, 100.0], dtype=np.float32)
    device_logits = np.zeros(cfg.M, dtype=np.float32)
    symbol_logits = np.zeros(len(cfg.symbol_set), dtype=np.float32)
    q_now = np.array([cfg.L - 1.0, cfg.W - 1.0], dtype=np.float32)

    mapped = mapper.map(v_raw, device_logits, R_raw=0.0,
                         symbol_logits=symbol_logits, q_now=q_now,
                         b_raw=0.0, sample=False)

    speed = np.linalg.norm(mapped["v"])
    assert speed <= cfg.v_max + 1e-4
    assert 0 <= mapped["m_star"] < cfg.M
    assert mapped["num_symbols"] in cfg.symbol_set
    assert 0.0 <= mapped["R"] <= 1.0
    assert 0.0 <= mapped["b"] <= 1.0


def test_action_mapper_bandwidth_accumulator_stays_normalized():
    """Algorithm 1 steps 8-10: bar_b_m[n] must always sum to 1 across
    devices, both right after env.reset() and after repeated bandwidth
    requests for arbitrary scheduled devices."""
    cfg, env = make_env()
    env.reset()
    mapper = ActionMapper(cfg, env)

    assert np.isclose(env.b_bar.sum(), 1.0)

    v_raw = np.zeros(2, dtype=np.float32)
    symbol_logits = np.zeros(len(cfg.symbol_set), dtype=np.float32)
    q_now = np.array([cfg.L / 2, cfg.W / 2], dtype=np.float32)

    for m in range(cfg.M):
        device_logits = np.full(cfg.M, -1e9, dtype=np.float32)
        device_logits[m] = 0.0  # force scheduling device m
        mapper.map(v_raw, device_logits, R_raw=0.0, symbol_logits=symbol_logits,
                   q_now=q_now, b_raw=1.0, sample=False)
        assert np.isclose(env.b_bar.sum(), 1.0)
