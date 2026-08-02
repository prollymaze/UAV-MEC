"""
Shared helper for the figure-reproduction sweep scripts (Figs. 3-8).

Each sweep script varies one Config field (device count, UAV CPU capacity,
jammer power, bandwidth, task size) across a list of values, and for each
value: builds a fresh environment + agent (PPO or DDPG), trains it from
scratch, and evaluates the trained (deterministic) policy to get average
delay/energy.

NOTE: training a fresh agent per sweep point is what the paper implies too
(each parameter setting defines a different environment/optimization
problem), but it means runtime scales linearly with the number of sweep
points x algorithms x training episodes. Defaults here favor a reasonable
runtime over matching the paper's full 2000-episode training; pass
--train-episodes to scale up.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uav_semcom_ppo import Config, UAVSemanticEnv, PPOAgent, DDPGAgent, set_seed

# Every sweep script imports these two so the algorithm choice, its display
# label, and its plot color stay consistent across every figure.
AGENT_CLASSES = {"ppo": PPOAgent, "ddpg": DDPGAgent}
ALGO_LABELS = {"ppo": "PPO-MEC-SC", "ddpg": "DDPG-MEC-SC"}
ALGO_COLORS = {"ppo": "tab:blue", "ddpg": "tab:orange"}


def build_agent(cfg: Config, algo: str = "ppo"):
    """Build a fresh environment + agent for the given algorithm ('ppo' or 'ddpg')."""
    algo = algo.lower()
    if algo not in AGENT_CLASSES:
        raise ValueError(f"Unknown algo '{algo}', expected one of {list(AGENT_CLASSES)}")
    set_seed(cfg.seed)
    env = UAVSemanticEnv(cfg)
    agent_cls = AGENT_CLASSES[algo]
    return agent_cls(cfg, env)


def train_and_evaluate(cfg: Config, train_episodes: int, eval_episodes: int = 5, algo: str = "ppo"):
    """Train a fresh agent (PPO or DDPG) under the given Config, then run
    deterministic evaluation rollouts. Returns dict(avg_delay, std_delay,
    avg_energy, std_energy)."""
    cfg.episodes = train_episodes
    agent = build_agent(cfg, algo)
    agent.train()
    return agent.evaluate(episodes=eval_episodes)


def train_and_get_history(cfg: Config, algo: str = "ppo"):
    """Train a fresh agent and return its full per-episode training history
    (used by plot_convergence.py)."""
    agent = build_agent(cfg, algo)
    return agent.train()
