#!/usr/bin/env python
"""
Reproduces Fig. 7 of the paper, comparing PPO-MEC-SC and DDPG-MEC-SC:
grouped bar charts of
  (a) Energy consumption vs. number of IoT devices, grouped by task size
  (b) Task completion time vs. number of IoT devices, grouped by task size
one row per algorithm.

This is the most expensive script: it trains one agent per
(algorithm x device count x task size) combination. With the defaults below
that's 2 x 5 x 4 = 40 training runs.

Usage:
    python scripts/plot_fig7_devices_tasksize.py
    python scripts/plot_fig7_devices_tasksize.py --algos ppo --devices 5 10 15 --task-sizes-mbits 0.4 0.8 --train-episodes 100
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sweep_utils import train_and_evaluate, ALGO_LABELS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from uav_semcom_ppo import Config


def parse_args():
    p = argparse.ArgumentParser(description="Reproduce Fig. 7: metrics vs. device count, grouped by task size.")
    p.add_argument("--algos", type=str, nargs="+", default=["ppo", "ddpg"], choices=["ppo", "ddpg"])
    p.add_argument("--devices", type=int, nargs="+", default=[5, 10, 15, 20, 25])
    p.add_argument("--task-sizes-mbits", type=float, nargs="+", default=[0.4, 0.8, 1.2, 1.6])
    # FIX: 100 episodes is far short of Table 3's N_epi=2000 -- convergence.png
    # shows reward is still around -100 to -150 (nowhere near converged) at
    # that point, so this was measuring an undertrained policy that hasn't
    # learned to offload yet (low energy / very high delay from doing
    # everything locally), not a units or config-threading bug.
    p.add_argument("--train-episodes", type=int, default=2000)
    p.add_argument("--eval-episodes", type=int, default=50)
    # FIX: 3 seeds (not 5, like Figs 4-6) to keep this grid's runtime
    # (2 algos x 5 devices x 4 task sizes x 3 seeds = 120 training runs)
    # reasonable; raise if you can afford the compute.
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--out", type=str, default="outputs/fig7_devices_tasksize.png")
    return p.parse_args()


def sweep_one_algo(algo, args):
    n_sizes = len(args.task_sizes_mbits)
    n_devices = len(args.devices)
    energy_grid = np.zeros((n_sizes, n_devices))
    delay_grid = np.zeros((n_sizes, n_devices))
    energy_err_grid = np.zeros((n_sizes, n_devices))
    delay_err_grid = np.zeros((n_sizes, n_devices))

    for si, size_mbits in enumerate(args.task_sizes_mbits):
        for di, m in enumerate(args.devices):
            print(f"=== Fig7 [{ALGO_LABELS[algo]}]: M={m}, task_size={size_mbits} Mbits, "
                  f"averaging over {len(args.seeds)} seeds ===")
            per_seed_energy, per_seed_delay = [], []
            for seed in args.seeds:
                cfg = Config(M=m, task_bits_mean=size_mbits * 1e6, seed=seed)
                result = train_and_evaluate(cfg, args.train_episodes, args.eval_episodes, algo=algo)
                per_seed_energy.append(result["avg_energy"])
                per_seed_delay.append(result["avg_delay"])
            per_seed_energy = np.asarray(per_seed_energy)
            per_seed_delay = np.asarray(per_seed_delay)
            n_seeds = len(args.seeds)
            energy_grid[si, di] = per_seed_energy.mean()
            delay_grid[si, di] = per_seed_delay.mean()
            # FIX: SEM across seeds, not the raw per-slot population std
            # the old code plugged in here (result["std_energy"]/
            # result["std_delay"]) -- that raw std is explicitly documented
            # in PPOAgent.evaluate() as a spread diagnostic that can exceed
            # the mean, not something meant to be plotted as a symmetric
            # error bar. That mismatch is why the bars were swamped by
            # whiskers several times taller than the bars themselves.
            energy_err_grid[si, di] = per_seed_energy.std() / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
            delay_err_grid[si, di] = per_seed_delay.std() / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
            print(f"  -> mean avg_energy={energy_grid[si, di]:.4f} J  "
                  f"mean avg_delay={delay_grid[si, di]:.4f} s")

    return energy_grid, delay_grid, energy_err_grid, delay_err_grid


def plot_grouped_bars(ax, x_labels, grid, err_grid, group_labels, ylabel, title):
    n_groups, n_x = grid.shape
    x = np.arange(n_x)
    bar_width = 0.8 / n_groups
    colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
    for gi, glabel in enumerate(group_labels):
        offset = (gi - (n_groups - 1) / 2) * bar_width
        ax.bar(x + offset, grid[gi], width=bar_width, yerr=err_grid[gi],
               capsize=2, color=colors[gi], label=glabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Number of IoT devices M")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)


def main():
    args = parse_args()
    n_rows = len(args.algos)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 5 * n_rows), squeeze=False)

    task_labels = [f"Task size = {s} Mbits" for s in args.task_sizes_mbits]

    for row, algo in enumerate(args.algos):
        energy_grid, delay_grid, energy_err, delay_err = sweep_one_algo(algo, args)
        plot_grouped_bars(axes[row][0], args.devices, energy_grid, energy_err, task_labels,
                           "Energy consumption (J)",
                           f"(a) Energy consumption vs. devices [{ALGO_LABELS[algo]}]")
        plot_grouped_bars(axes[row][1], args.devices, delay_grid, delay_err, task_labels,
                           "Task completion time (s)",
                           f"(b) Task completion time vs. devices [{ALGO_LABELS[algo]}]")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
