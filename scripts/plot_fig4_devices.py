#!/usr/bin/env python
"""
Reproduces Fig. 4 of the paper, comparing PPO-MEC-SC and DDPG-MEC-SC:
  (a) Energy consumption vs. number of IoT devices M
  (b) Task completion time vs. number of IoT devices M

Usage:
    python scripts/plot_fig4_devices.py
    python scripts/plot_fig4_devices.py --algos ppo ddpg --devices 5 10 15 20 25 --train-episodes 300
    python scripts/plot_fig4_devices.py --algos ppo             # PPO only
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
    p = argparse.ArgumentParser(description="Reproduce Fig. 4: metrics vs. number of IoT devices.")
    p.add_argument("--algos", type=str, nargs="+", default=["ppo", "ddpg"], choices=["ppo", "ddpg"])
    p.add_argument("--devices", type=int, nargs="+", default=[5, 10, 15, 20, 25])
    p.add_argument("--train-episodes", type=int, default=2000,
                    help="FIX: raised to match Table 3's N_epi=2000. The previous "
                         "default of 500 (and 150 before that) left the policy "
                         "undertrained, especially for larger M where the "
                         "device-scheduling head (size M) has more to learn -- "
                         "undertrained runs are exactly where a single M value can "
                         "post an unrepresentative outlier.")
    p.add_argument("--eval-episodes", type=int, default=50,
                    help="FIX: raised from 5. At 5 episodes (200 per-slot samples), "
                         "the mean is dominated by whichever few slots happened to "
                         "have bad SINR/obstruction -- not enough for a stable "
                         "per-M estimate given task_bits is exponential (heavy-tailed).")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                    help="FIX (root cause of the non-monotonic zigzag): the old script "
                         "trained ONE random device layout per M value (fixed seed=42), "
                         "so each point on the curve was a single random draw with no "
                         "averaging -- a single unlucky layout (e.g. several obstructed "
                         "devices far from the UAV's path) shows up as a lone spike with "
                         "nothing keeping the curve monotonic. We now train/evaluate "
                         "once per seed per M and average the results, matching how a "
                         "Monte-Carlo sweep like the paper's Fig. 4 would be produced.")
    p.add_argument("--out", type=str, default="outputs/fig4_devices.png")
    return p.parse_args()


def main():
    args = parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for algo in args.algos:
        energies, delays, energy_sems, delay_sems = [], [], [], []
        for m in args.devices:
            print(f"=== Fig4 [{ALGO_LABELS[algo]}]: training with M={m} devices, "
                  f"averaging over {len(args.seeds)} seeds ===")
            # FIX: average over multiple random seeds (device layouts +
            # obstruction patterns + network init) instead of one fixed
            # seed per M. Each seed contributes one trained policy's
            # avg_energy/avg_delay; we then treat those per-seed means as
            # the samples for this M point, so the point estimate is a
            # mean-of-means and the error bar reflects seed-to-seed
            # variability (device layout luck), not just within-run
            # per-slot noise.
            per_seed_energy, per_seed_delay = [], []
            for seed in args.seeds:
                cfg = Config(M=m, seed=seed)
                result = train_and_evaluate(cfg, args.train_episodes, args.eval_episodes, algo=algo)
                per_seed_energy.append(result["avg_energy"])
                per_seed_delay.append(result["avg_delay"])
                print(f"    seed={seed}: avg_energy={result['avg_energy']:.4f} J  "
                      f"avg_delay={result['avg_delay']:.4f} s")

            per_seed_energy = np.asarray(per_seed_energy)
            per_seed_delay = np.asarray(per_seed_delay)
            n_seeds = len(args.seeds)

            energies.append(float(per_seed_energy.mean()))
            delays.append(float(per_seed_delay.mean()))
            # SEM across seeds (n=len(args.seeds)); falls back to 0 for a
            # single seed rather than dividing by zero.
            energy_sems.append(float(per_seed_energy.std() / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0)
            delay_sems.append(float(per_seed_delay.std() / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0)
            print(f"  -> mean avg_energy={energies[-1]:.4f} J  mean avg_delay={delays[-1]:.4f} s")

        color = ALGO_COLORS[algo]
        label = ALGO_LABELS[algo]

        def clipped_yerr(means, sems):
            # FIX: clip the lower error to the mean itself so the plotted
            # lower bound never dips below 0, since energy (J) and time (s)
            # can't be negative.
            means_a, sems_a = np.asarray(means), np.asarray(sems)
            lower = np.minimum(means_a, sems_a)
            upper = sems_a
            return np.vstack([lower, upper])

        axes[0].errorbar(args.devices, energies, yerr=clipped_yerr(energies, energy_sems), marker="o",
                          color=color, label=label, capsize=3)
        axes[1].errorbar(args.devices, delays, yerr=clipped_yerr(delays, delay_sems), marker="o",
                          color=color, label=label, capsize=3)

    axes[0].set_xlabel("Number of IoT devices M")
    axes[0].set_ylabel("Energy consumption (J)")
    axes[0].set_title("(a) Energy consumption vs. number of IoT devices")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_xlabel("Number of IoT devices M")
    axes[1].set_ylabel("Task completion time (s)")
    axes[1].set_title("(b) Task completion time vs. number of IoT devices")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
