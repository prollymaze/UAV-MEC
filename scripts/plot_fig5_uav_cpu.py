#!/usr/bin/env python
"""
Reproduces Fig. 5 of the paper, comparing PPO-MEC-SC and DDPG-MEC-SC:
  (a) Energy consumption vs. UAV CPU computing capacity f_UAV
  (b) Task completion time vs. UAV CPU computing capacity f_UAV

Usage:
    python scripts/plot_fig5_uav_cpu.py
    python scripts/plot_fig5_uav_cpu.py --algos ppo ddpg --cpu-ghz 1.4 1.7 2.0 2.3 2.6 2.9 --train-episodes 300
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sweep_utils import train_and_evaluate, ALGO_LABELS, ALGO_COLORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from uav_semcom_ppo import Config


def parse_args():
    p = argparse.ArgumentParser(description="Reproduce Fig. 5: metrics vs. UAV CPU capacity.")
    p.add_argument("--algos", type=str, nargs="+", default=["ppo", "ddpg"], choices=["ppo", "ddpg"])
    p.add_argument("--cpu-ghz", type=float, nargs="+", default=[1.4, 1.7, 2.0, 2.3, 2.6, 2.9])
    # FIX: same class of bug as plot_fig4_devices.py -- 500 episodes falls
    # short of Table 3's N_epi=2000, and a single fixed seed means each
    # sweep point is one random draw (device layout, obstruction pattern,
    # network init) with no averaging, which is why DDPG (and to a lesser
    # extent PPO) zigzags instead of showing f_UAV's real monotone effect.
    p.add_argument("--train-episodes", type=int, default=2000)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--out", type=str, default="outputs/fig5_uav_cpu.png")
    return p.parse_args()


def main():
    args = parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for algo in args.algos:
        energies, delays, energy_sems, delay_sems = [], [], [], []
        for ghz in args.cpu_ghz:
            print(f"=== Fig5 [{ALGO_LABELS[algo]}]: f_UAV={ghz} GHz, "
                  f"averaging over {len(args.seeds)} seeds ===")
            per_seed_energy, per_seed_delay = [], []
            for seed in args.seeds:
                cfg = Config(f_uav_max_hz=ghz * 1e9, seed=seed)
                result = train_and_evaluate(cfg, args.train_episodes, args.eval_episodes, algo=algo)
                per_seed_energy.append(result["avg_energy"])
                per_seed_delay.append(result["avg_delay"])
            per_seed_energy = np.asarray(per_seed_energy)
            per_seed_delay = np.asarray(per_seed_delay)
            n_seeds = len(args.seeds)
            energies.append(float(per_seed_energy.mean()))
            delays.append(float(per_seed_delay.mean()))
            energy_sems.append(float(per_seed_energy.std() / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0)
            delay_sems.append(float(per_seed_delay.std() / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0)
            print(f"  -> mean avg_energy={energies[-1]:.4f} J  mean avg_delay={delays[-1]:.4f} s")

        color = ALGO_COLORS[algo]
        label = ALGO_LABELS[algo]

        def clipped_yerr(means, sems):
            # FIX: SEM instead of raw std, lower bound clipped at 0 -- see
            # plot_fig4_devices.py / PPOAgent.evaluate() for the rationale.
            means_a, sems_a = np.asarray(means), np.asarray(sems)
            lower = np.minimum(means_a, sems_a)
            upper = sems_a
            return np.vstack([lower, upper])
        axes[0].errorbar(args.cpu_ghz, energies, yerr=clipped_yerr(energies, energy_sems), marker="o",
                          color=color, label=label, capsize=3)
        axes[1].errorbar(args.cpu_ghz, delays, yerr=clipped_yerr(delays, delay_sems), marker="o",
                          color=color, label=label, capsize=3)

    axes[0].set_xlabel("CPU computing capacity of UAV (GHz)")
    axes[0].set_ylabel("Energy consumption (J)")
    axes[0].set_title("(a) Energy consumption vs. UAV CPU capacity")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_xlabel("CPU computing capacity of UAV (GHz)")
    axes[1].set_ylabel("Task completion time (s)")
    axes[1].set_title("(b) Task completion time vs. UAV CPU capacity")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
