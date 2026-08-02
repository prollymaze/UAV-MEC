#!/usr/bin/env python
"""
Reproduces Fig. 8 of the paper, comparing PPO-MEC-SC and DDPG-MEC-SC:
grouped bar charts of
  (a) Energy consumption vs. bandwidth, grouped by task size
  (b) Task completion time vs. bandwidth, grouped by task size
one row per algorithm.

This is the most expensive script: it trains one agent per
(algorithm x bandwidth x task size) combination. With the defaults below
that's 2 x 4 x 4 = 32 training runs. Use fewer sweep points or
--train-episodes for a faster look.

Usage:
    python scripts/plot_fig8_bandwidth_tasksize.py
    python scripts/plot_fig8_bandwidth_tasksize.py --algos ppo --bandwidths-mhz 6 9 12 --task-sizes-mbits 0.4 0.8 --train-episodes 100
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
    p = argparse.ArgumentParser(description="Reproduce Fig. 8: metrics vs. bandwidth, grouped by task size.")
    p.add_argument("--algos", type=str, nargs="+", default=["ppo", "ddpg"], choices=["ppo", "ddpg"])
    p.add_argument("--bandwidths-mhz", type=float, nargs="+", default=[6, 8, 10, 12])
    p.add_argument("--task-sizes-mbits", type=float, nargs="+", default=[0.4, 0.8, 1.2, 1.6])
    # FIX: same as plot_fig7_devices_tasksize.py -- 100 episodes is far short
    # of Table 3's N_epi=2000 (undertrained policy), and 5 eval episodes plus
    # a single seed is too little data per grid cell.
    p.add_argument("--train-episodes", type=int, default=2000)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--out", type=str, default="outputs/fig8_bandwidth_tasksize.png")
    return p.parse_args()


def sweep_one_algo(algo, args):
    n_sizes = len(args.task_sizes_mbits)
    n_bw = len(args.bandwidths_mhz)
    energy_grid = np.zeros((n_sizes, n_bw))
    delay_grid = np.zeros((n_sizes, n_bw))
    energy_err_grid = np.zeros((n_sizes, n_bw))
    delay_err_grid = np.zeros((n_sizes, n_bw))

    for si, size_mbits in enumerate(args.task_sizes_mbits):
        for bi, bw_mhz in enumerate(args.bandwidths_mhz):
            print(f"=== Fig8 [{ALGO_LABELS[algo]}]: bandwidth={bw_mhz} MHz, "
                  f"task_size={size_mbits} Mbits, averaging over {len(args.seeds)} seeds ===")
            per_seed_energy, per_seed_delay = [], []
            for seed in args.seeds:
                cfg = Config(bandwidth_hz=bw_mhz * 1e6, task_bits_mean=size_mbits * 1e6, seed=seed)
                result = train_and_evaluate(cfg, args.train_episodes, args.eval_episodes, algo=algo)
                per_seed_energy.append(result["avg_energy"])
                per_seed_delay.append(result["avg_delay"])
            per_seed_energy = np.asarray(per_seed_energy)
            per_seed_delay = np.asarray(per_seed_delay)
            n_seeds = len(args.seeds)
            energy_grid[si, bi] = per_seed_energy.mean()
            delay_grid[si, bi] = per_seed_delay.mean()
            # FIX: SEM across seeds instead of raw per-slot std -- see the
            # identical fix/comment in plot_fig7_devices_tasksize.py.
            energy_err_grid[si, bi] = per_seed_energy.std() / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
            delay_err_grid[si, bi] = per_seed_delay.std() / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
            print(f"  -> mean avg_energy={energy_grid[si, bi]:.4f} J  "
                  f"mean avg_delay={delay_grid[si, bi]:.4f} s")

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
    ax.set_xlabel("Bandwidth (MHz)")
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
        plot_grouped_bars(axes[row][0], args.bandwidths_mhz, energy_grid, energy_err, task_labels,
                           "Energy consumption (J)",
                           f"(a) Energy consumption vs. bandwidth [{ALGO_LABELS[algo]}]")
        plot_grouped_bars(axes[row][1], args.bandwidths_mhz, delay_grid, delay_err, task_labels,
                           "Task completion time (s)",
                           f"(b) Task completion time vs. bandwidth [{ALGO_LABELS[algo]}]")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
