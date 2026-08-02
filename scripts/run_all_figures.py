#!/usr/bin/env python
"""
Runs every figure-reproduction script in sequence: convergence (Fig 3) and
Figs 4-8, comparing PPO-MEC-SC and DDPG-MEC-SC on all of them. Each script
writes its own PNG to outputs/.

This is a convenience wrapper -- for a first pass, keep --quick to use small
training-episode counts everywhere so you get a full set of (rough) plots
quickly; drop --quick and pass per-script overrides for higher-fidelity runs.
Pass --algos ppo to skip DDPG entirely and roughly halve the total runtime.

Usage:
    python scripts/run_all_figures.py --quick
    python scripts/run_all_figures.py --quick --algos ppo
    python scripts/run_all_figures.py            # uses each script's own defaults
"""

import argparse
import subprocess
import sys
import os

SCRIPTS_DIR = os.path.dirname(__file__)


def run(cmd):
    print(f"\n{'=' * 70}\nRunning: {' '.join(cmd)}\n{'=' * 70}")
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                    help="Use small episode counts / fewer sweep points for a fast first pass.")
    p.add_argument("--algos", type=str, nargs="+", default=["ppo", "ddpg"], choices=["ppo", "ddpg"])
    args = p.parse_args()

    py = sys.executable
    algo_flags = ["--algos"] + args.algos

    if args.quick:
        run([py, os.path.join(SCRIPTS_DIR, "plot_convergence.py"), *algo_flags,
             "--episodes", "100"])
        run([py, os.path.join(SCRIPTS_DIR, "plot_fig4_devices.py"), *algo_flags,
             "--devices", "5", "10", "15", "--train-episodes", "60"])
        run([py, os.path.join(SCRIPTS_DIR, "plot_fig5_uav_cpu.py"), *algo_flags,
             "--cpu-ghz", "1.4", "2.0", "2.9", "--train-episodes", "60"])
        run([py, os.path.join(SCRIPTS_DIR, "plot_fig6_jammer.py"), *algo_flags,
             "--jammer-dbm", "14", "20", "24", "--train-episodes", "60"])
        run([py, os.path.join(SCRIPTS_DIR, "plot_fig7_devices_tasksize.py"), *algo_flags,
             "--devices", "5", "15", "25", "--task-sizes-mbits", "0.4", "1.2",
             "--train-episodes", "40"])
        run([py, os.path.join(SCRIPTS_DIR, "plot_fig8_bandwidth_tasksize.py"), *algo_flags,
             "--bandwidths-mhz", "6", "10", "--task-sizes-mbits", "0.4", "1.2",
             "--train-episodes", "40"])
    else:
        run([py, os.path.join(SCRIPTS_DIR, "plot_convergence.py"), *algo_flags])
        run([py, os.path.join(SCRIPTS_DIR, "plot_fig4_devices.py"), *algo_flags])
        run([py, os.path.join(SCRIPTS_DIR, "plot_fig5_uav_cpu.py"), *algo_flags])
        run([py, os.path.join(SCRIPTS_DIR, "plot_fig6_jammer.py"), *algo_flags])
        run([py, os.path.join(SCRIPTS_DIR, "plot_fig7_devices_tasksize.py"), *algo_flags])
        run([py, os.path.join(SCRIPTS_DIR, "plot_fig8_bandwidth_tasksize.py"), *algo_flags])

    print("\nAll figures generated in outputs/")


if __name__ == "__main__":
    main()
