#!/usr/bin/env python
"""
Entry point for training a single agent (PPO or DDPG) on the UAV
semantic-communication MEC environment.

Usage:
    python scripts/train.py
    python scripts/train.py --algo ddpg --episodes 500 --seed 7
    python scripts/train.py --output outputs/run1_history.json

Run from the project root with the package installed in editable mode
(`pip install -e .`), or with PYTHONPATH=src set (see README.md).
"""

import argparse
import json
import os
import sys

# Allow running directly from the repo without an editable install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uav_semcom_ppo import Config, UAVSemanticEnv, PPOAgent, DDPGAgent, set_seed

AGENT_CLASSES = {"ppo": PPOAgent, "ddpg": DDPGAgent}


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO or DDPG on the UAV-SemCom-MEC environment.")
    p.add_argument("--algo", type=str, default="ppo", choices=["ppo", "ddpg"], help="Which agent to train")
    p.add_argument("--episodes", type=int, default=None, help="Override Config.episodes")
    p.add_argument("--seed", type=int, default=None, help="Override Config.seed")
    p.add_argument("--devices", type=int, default=None, help="Override Config.M (number of IoT devices)")
    p.add_argument(
        "--output", type=str, default=None,
        help="Where to save the training history JSON (relative to project root); "
             "defaults to outputs/training_history_<algo>.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    if args.episodes is not None:
        cfg.episodes = args.episodes
    if args.seed is not None:
        cfg.seed = args.seed
    if args.devices is not None:
        cfg.M = args.devices

    set_seed(cfg.seed)

    env = UAVSemanticEnv(cfg)
    agent = AGENT_CLASSES[args.algo](cfg, env)
    history = agent.train()

    out_path = args.output or f"outputs/training_history_{args.algo}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        # default=float coerces any numpy scalar (e.g. np.float32 energy
        # values coming out of the env) into a plain JSON-serializable float.
        json.dump(history, f, indent=2, default=float)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
