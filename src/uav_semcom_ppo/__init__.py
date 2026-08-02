"""
uav_semcom_ppo
==============

PPO-based joint UAV trajectory, device scheduling, task offloading, and
semantic symbol configuration for semantic-communication-enabled UAV-assisted
Mobile Edge Computing (MEC).

Public API:
    Config          - all hyperparameters / system constants (config.py)
    UAVSemanticEnv  - simulation environment (environment.py)
    ActionMapper    - Algorithm 1: Action Mapping (action_mapper.py)
    ActorCritic     - hybrid discrete/continuous actor-critic network (networks.py)
    RolloutBuffer   - on-policy trajectory storage (buffer.py)
    PPOAgent        - Algorithm 2: PPO training loop (agent.py)
    DDPGActor       - deterministic policy network for DDPG (ddpg_networks.py)
    DDPGCritic      - Q-value critic network for DDPG (ddpg_networks.py)
    ReplayBuffer    - off-policy replay buffer for DDPG (replay_buffer.py)
    DDPGAgent       - DDPG baseline agent, comparable to PPOAgent (ddpg_agent.py)
    set_seed        - reproducibility helper (utils.py)
"""

from .config import Config
from .environment import UAVSemanticEnv
from .action_mapper import ActionMapper
from .networks import ActorCritic
from .buffer import RolloutBuffer
from .agent import PPOAgent
from .ddpg_networks import DDPGActor, DDPGCritic
from .replay_buffer import ReplayBuffer
from .ddpg_agent import DDPGAgent
from .utils import set_seed

__all__ = [
    "Config",
    "UAVSemanticEnv",
    "ActionMapper",
    "ActorCritic",
    "RolloutBuffer",
    "PPOAgent",
    "DDPGActor",
    "DDPGCritic",
    "ReplayBuffer",
    "DDPGAgent",
    "set_seed",
]

__version__ = "0.1.0"
