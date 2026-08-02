#!/usr/bin/env python
"""
Train PPO and/or DDPG and plot their reward-convergence curves, in the style
of Fig. 3 in the paper ("Convergence performance of different approaches").

This reproduces the PPO-MEC-SC and DDPG-MEC-SC curves. The paper's Fig. 3
also compares against SAC-MEC-SC and a non-SemCom PPO baseline
(PPO-MEC-NSC) -- those aren't implemented here.

Usage:
    python scripts/plot_convergence.py
    python scripts/plot_convergence.py --algos ppo ddpg --episodes 300 --smooth 20
    python scripts/plot_convergence.py --algos ppo          # PPO only
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display needed
import matplotlib.pyplot as plt
import numpy as np

from sweep_utils import train_and_get_history, ALGO_LABELS, ALGO_COLORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from uav_semcom_ppo import Config


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO/DDPG and plot the reward-convergence curve.")
    p.add_argument("--algos", type=str, nargs="+", default=["ppo", "ddpg"], choices=["ppo", "ddpg"])
    p.add_argument("--episodes", type=int, default=300,
                    help="Number of training episodes (paper uses 2000; that's slow on CPU, "
                         "so default here is smaller for a quick first look).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--devices", type=int, default=None, help="Override Config.M")
    p.add_argument("--smooth", type=int, default=10,
                    help="Moving-average window (in episodes) applied to the reward curve, "
                         "similar to the smoothing visible in the paper's Fig. 3.")
    p.add_argument("--history-out-prefix", type=str, default="outputs/history")
    p.add_argument("--plot-out", type=str, default="outputs/convergence.png")
    p.add_argument("--ddpg-vanilla", action="store_true",
                    help="Run DDPG without TD3 stabilization tricks (single critic, "
                         "undelayed actor updates, no target-policy smoothing), matching "
                         "the paper's plain DDPG-MEC-SC baseline instead of a TD3-strength one.")
    return p.parse_args()


def moving_average(x, window):
    if window <= 1 or len(x) < window:
        return np.array(x)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def main():
    args = parse_args()

    fig, ax = plt.subplots(figsize=(7, 5))

    for algo in args.algos:
        cfg = Config()
        cfg.episodes = args.episodes
        if args.seed is not None:
            cfg.seed = args.seed
        if args.devices is not None:
            cfg.M = args.devices
        if args.ddpg_vanilla:
            cfg.ddpg_use_td3_tricks = False

        print(f"=== Training {ALGO_LABELS[algo]} for {args.episodes} episodes ===")
        history = train_and_get_history(cfg, algo=algo)

        history_out = f"{args.history_out_prefix}_{algo}.json"
        os.makedirs(os.path.dirname(history_out) or ".", exist_ok=True)
        with open(history_out, "w") as f:
            json.dump(history, f, indent=2, default=float)
        print(f"Saved {history_out}")

        episodes = np.array([h["episode"] for h in history])
        rewards = np.array([h["total_reward"] for h in history], dtype=float)
        steps = episodes * cfg.N   # x-axis in "training steps", matching Fig. 3's axis

        smoothed_rewards = moving_average(rewards, args.smooth)
        smoothed_steps = steps[len(steps) - len(smoothed_rewards):]

        color = ALGO_COLORS[algo]
        ax.plot(steps, rewards, color=color, alpha=0.2, linewidth=1)
        ax.plot(smoothed_steps, smoothed_rewards, color=color, linewidth=2, label=ALGO_LABELS[algo])

    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    title_algos = " vs. ".join(ALGO_LABELS[a] for a in args.algos)
    ax.set_title(f"Convergence performance ({title_algos})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    os.makedirs(os.path.dirname(args.plot_out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.plot_out, dpi=150)
    print(f"Saved {args.plot_out}")


if __name__ == "__main__":
    main()
