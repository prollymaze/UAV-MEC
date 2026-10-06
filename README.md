# uav-semcom-ppo

PPO-based joint UAV trajectory, device scheduling, task offloading, and
semantic symbol configuration for semantic-communication-enabled UAV-assisted
Mobile Edge Computing (MEC). Also includes a DDPG baseline agent
(DDPG-MEC-SC) for comparison against PPO in every figure, matching the
paper's practice of benchmarking PPO against baseline algorithms.

Implements:
- **Algorithm 1 — Action Mapping** (`src/uav_semcom_ppo/action_mapper.py`)
- **Algorithm 2 — PPO training loop with Lagrangian energy-budget enforcement**
  (`src/uav_semcom_ppo/agent.py`)
- **DDPG baseline** (`src/uav_semcom_ppo/ddpg_agent.py`) — off-policy
  comparison agent, not from the paper's Algorithm 2 but built to be
  directly comparable to it (same environment, same reward, same
  evaluation method).

## Project layout

```
uav_semcom_ppo/
├── README.md
├── pyproject.toml            # pip-installable package metadata
├── requirements.txt          # plain pip requirements (alternative to pyproject)
├── .gitignore
├── configs/
│   └── default.yaml          # reference copy of Config defaults (for experiment tracking)
├── src/
│   └── uav_semcom_ppo/
│       ├── __init__.py       # public API exports
│       ├── config.py         # Config dataclass: all constants & hyperparameters (PPO and DDPG)
│       ├── environment.py    # UAVSemanticEnv: state, channel model, dynamics, step()
│       ├── action_mapper.py  # ActionMapper: Algorithm 1 (shared by both PPO and DDPG)
│       ├── networks.py       # ActorCritic: hybrid discrete/continuous policy+value net (PPO)
│       ├── buffer.py         # RolloutBuffer: on-policy trajectory storage (PPO)
│       ├── agent.py          # PPOAgent: Algorithm 2 (GAE, clipped update, train loop)
│       ├── ddpg_networks.py  # DDPGActor (deterministic policy) + DDPGCritic (Q-value)
│       ├── replay_buffer.py  # ReplayBuffer: off-policy persistent replay storage (DDPG)
│       ├── ddpg_agent.py     # DDPGAgent: off-policy training loop, comparable to PPOAgent
│       └── utils.py          # set_seed() and other shared helpers
├── scripts/
│   ├── train.py                        # CLI entry point: basic training run (--algo ppo|ddpg)
│   ├── plot_convergence.py             # Fig. 3: reward vs. training steps, both algorithms
│   ├── sweep_utils.py                  # shared train_and_evaluate() helper, algorithm-agnostic
│   ├── plot_fig4_devices.py            # Fig. 4: energy & delay vs. number of IoT devices
│   ├── plot_fig5_uav_cpu.py            # Fig. 5: energy & delay vs. UAV CPU capacity
│   ├── plot_fig6_jammer.py             # Fig. 6: energy & delay vs. jammer power
│   ├── plot_fig7_devices_tasksize.py   # Fig. 7: bar chart, devices x task size, one row per algo
│   ├── plot_fig8_bandwidth_tasksize.py # Fig. 8: bar chart, bandwidth x task size, one row per algo
│   └── run_all_figures.py              # convenience: runs all of the above in sequence
├── outputs/                   # generated training_history.json, plots, checkpoints (gitignored)
└── tests/
    ├── test_environment.py    # sanity tests for env + action mapper shapes/invariants
    └── test_ddpg.py           # sanity tests for DDPGAgent: trains, evaluates, fills replay buffer
```

### What each piece is responsible for

| File | Responsibility |
|---|---|
| `config.py` | Every physical constant, comms parameter, energy budget, cost weight, PPO hyperparameter, and DDPG hyperparameter. Nothing else in the codebase should hardcode a number that belongs here. |
| `environment.py` | `UAVSemanticEnv` — episode state, air-to-ground channel gains, the semantic-similarity/compression proxy functions, UAV propulsion energy model, per-device staleness/fairness tracking, and the per-slot `step()` dynamics. Shared identically by PPO and DDPG. |
| `action_mapper.py` | `ActionMapper.map()` — takes raw actor outputs (from either algorithm's network) and projects them onto the feasible action set. PPO calls it with `sample=True/False`; DDPG always calls it with `sample=False` since its policy is deterministic (exploration comes from noise added to the raw outputs beforehand, not from sampling). |
| `networks.py` | `ActorCritic` — PPO's shared trunk plus 4 *distributional* policy heads (mean+std for continuous, logits for discrete) and a state-value head. |
| `buffer.py` | `RolloutBuffer` — plain lists that accumulate one episode of transitions before a single PPO update, then get discarded (on-policy). |
| `agent.py` | `PPOAgent` — action sampling (`act`), log-prob/value/entropy recomputation (`evaluate_actions`), GAE (`compute_gae`), the clipped-surrogate update (`update`), the outer training loop (`train`), and `evaluate()` for deterministic post-training metrics. |
| `ddpg_networks.py` | `DDPGActor` — same trunk shape as `ActorCritic`, but outputs one raw action vector directly (deterministic, no distribution). `DDPGCritic` — Q(s,a): takes state + the actor's raw action vector, outputs a scalar Q-value. |
| `replay_buffer.py` | `ReplayBuffer` — fixed-capacity circular buffer that persists across the whole training run (unlike `RolloutBuffer`), supporting random-batch sampling for DDPG's off-policy updates. |
| `ddpg_agent.py` | `DDPGAgent` — mirrors `PPOAgent`'s public interface exactly (`train()`, `evaluate()`) so every script can use either algorithm interchangeably. Implements the standard DDPG update (critic TD-target via target networks, actor gradient via `-Q(s, actor(s))`, Polyak soft target updates) plus a random-action warmup period and decaying Gaussian exploration noise. |
| `scripts/train.py` | The runnable entry point for a single training run. Builds `Config` → `UAVSemanticEnv` → `PPOAgent` or `DDPGAgent` (`--algo`), runs `.train()`, and saves the history JSON. |
| `scripts/sweep_utils.py` | `build_agent()`, `train_and_evaluate()`, `train_and_get_history()` — algorithm-agnostic helpers every Fig. 3-8 script calls, plus `ALGO_LABELS`/`ALGO_COLORS` dicts so every plot uses consistent naming ("PPO-MEC-SC"/"DDPG-MEC-SC") and colors (blue/orange). |
| `configs/default.yaml` | Human-readable mirror of `Config` defaults, useful once you want multiple named experiment configs (e.g. `configs/M10_devices.yaml`) — not wired into `train.py` automatically, see comment in the file for how to hook it up. |
| `tests/test_environment.py` | Fast, non-training sanity checks: state shapes, that `step()` returns sane types/signs, and that `ActionMapper` enforces the boundary and speed constraints. |
| `tests/test_ddpg.py` | Fast sanity checks for `DDPGAgent`: trains without error, `evaluate()` returns the expected keys with no NaNs, and the replay buffer actually fills up as expected. |

## Setup (Linux, venv)

```bash
cd uav_semcom_ppo
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Option A: editable install (recommended, gives you `import uav_semcom_ppo` anywhere)
pip install -e .

# Option B: just install requirements and rely on scripts/train.py's sys.path shim
pip install -r requirements.txt
```

## Run training

```bash
python scripts/train.py                                    # PPO by default
python scripts/train.py --algo ddpg --episodes 500 --seed 7
python scripts/train.py --devices 10 --output outputs/M10_history.json
```

This saves a JSON training history (per-episode reward, cost, energy used,
`mu`, and semantic shortfall) to `outputs/`.

## Run tests

```bash
pip install -e ".[dev]"   # or: pip install pytest
pytest tests/ -v
```

## Config values matched to the paper's Tables 2 & 3

`Config` defaults were updated to match the paper wherever it gives a fixed
value:

| Field | Paper value | Source |
|---|---|---|
| `M`, `L`, `W`, `H`, `v_max`, `N` | 15, 400 m, 400 m, 300 m, 15 m/s, 40 | Table 2 |
| `symbol_set` | 1, 2, ..., 20 | Table 2 (K) |
| `noise_power_w`, `p_max_w` | -140 dBW, 0.2 W | Table 2 |
| `f_loc_hz`, `f_uav_max_hz` | 1.05 GHz, 2.3 GHz | Table 2 |
| `E_b` | 80 kJ | Table 2 |
| `alpha`, `beta` | 2, 0.002 | Table 3 |
| `gamma` | 0.9 | Table 3 |
| `lr_actor`, `lr_critic` | 0.001 (paper uses one shared rate) | Table 3 |
| `train_epochs_per_update`, `minibatch_size` | 4, 256 | Table 3 |
| `hidden_layer_sizes` | [128, 64] | Table 3 |

`episodes` is left at the paper's 2000 as the *default*, but every plotting
script overrides it with a much smaller `--train-episodes` value so you can
get results without committing to the paper's full ~2x10^5-step runs -- bump
it up yourself if you want closer-to-paper convergence.

**Known remaining discrepancies** (not fixed, since they're bigger structural
changes than a config tweak):
- **Bandwidth `B` and task size `T_m`**: the paper sweeps these in Figs. 6-8
  rather than giving one fixed Table 2 value, so the current defaults
  (`bandwidth_hz`, `task_bits_mean`) are reasonable picks, not paper values.
- **Channel gain reference `rho0`**: Table 2 gives `rho0 = -50 dB` at a 1 m
  reference distance; `environment.py` instead uses a standard Friis
  free-space path-loss constant. Both are physically reasonable path-loss
  models, but they're not numerically identical.

**Fixed since the last round**: `environment.py::flight_energy()` now uses
the paper's actual Eq. 12 (`E_fly = 0.5 * M_UAV * v^2 * delta`) instead of a
detailed rotary-wing power model. One side effect worth knowing: Eq. 12
makes hovering (v=0) *free*, unlike the old model, which had a nonzero
hover-power floor. That means total energy per step is now much more
sensitive to how much the policy has learned to move -- so `Fig4a`'s energy
curve may look noisier at low training-episode counts than it did before,
since the trained policy hasn't fully converged on how much to hover vs.
move. Delay (`Fig4b`) converges to a clean trend much faster than energy
does; if the energy panel looks noisy, more `--train-episodes` per sweep
point is the first thing to try.

**Also fixed: device count M now causally affects delay/energy.**
Originally, `M` had almost no effect on any figure -- only one device is
served per slot, and nothing punished ignoring the rest. Two changes fixed
this:
1. `environment.py::step()` divides the available transmit bandwidth by
   `cfg.M`, approximating the paper's normalized long-term bandwidth
   allocation factor `bar_b_m[n]` (Algorithm 1, steps 8-10) -- more devices
   sharing the spectrum means less bandwidth per transmission.
2. `environment.py` now tracks `wait_time` per device (slots since it was
   last served) and returns `fairness_penalty` (the average wait time across
   *all* devices, served or not) in `step()`'s `info` dict. Both `PPOAgent`
   and `DDPGAgent` add `cfg.fairness_weight * fairness_penalty` as a reward
   penalty, and **`evaluate()`'s `avg_delay` now reports this fairness
   penalty, not raw per-service transmission delay `T`** -- deliberately,
   since `T` is selection-biased (a policy that ignores most devices would
   report a tiny average delay, since neglected devices never contribute a
   sample; `fairness_penalty` can't be gamed that way). More devices sharing
   a fixed `N`-slot mission now forces a higher average backlog if service
   is spread out, which is what makes `M` matter in Figs. 4, 7, and (via the
   symbol/bandwidth couplings) indirectly in 5, 6, 8 as well.

## Reproduce the paper's figures (PPO-MEC-SC only)

Every script trains a fresh PPO agent per sweep point, then evaluates the
greedy policy to get average energy/delay. Run from the project root:

```bash
# Fig. 3: reward convergence
python scripts/plot_convergence.py --episodes 300

# Fig. 4: energy & delay vs. number of IoT devices
python scripts/plot_fig4_devices.py

# Fig. 5: energy & delay vs. UAV CPU capacity
python scripts/plot_fig5_uav_cpu.py

# Fig. 6: energy & delay vs. jammer power
python scripts/plot_fig6_jammer.py

# Fig. 7: bar chart, devices x task size (slow: trains 5x4=20 agents by default)
python scripts/plot_fig7_devices_tasksize.py

# Fig. 8: bar chart, bandwidth x task size (slow: trains 4x4=16 agents by default)
python scripts/plot_fig8_bandwidth_tasksize.py

# ...or run everything at once with small settings for a fast first pass:
python scripts/run_all_figures.py --quick
```

All PNGs land in `outputs/`. Default `--train-episodes` values are kept low
(100-150) so the full set finishes in a reasonable time on a laptop CPU;
pass a larger `--train-episodes` (the paper uses 2000) for smoother, more
paper-faithful curves at the cost of much longer runtime.

**What this does and doesn't reproduce**: the trend directions (e.g. energy
rising with more devices, falling with more UAV CPU) should come out right
because they follow directly from the physics/cost model. The *absolute*
numbers won't match the paper's plots, for the same reason noted below --
the semantic-similarity curve isn't the paper's actual fitted DeepSC curve,
and each sweep point here trains far fewer episodes than the paper's 2000.
**PPO-MEC-SC vs. DDPG-MEC-SC is implemented and compared in Figs. 3-8**
(pass `--algos ppo` to any script to skip DDPG and roughly halve runtime).
SAC-MEC-SC and the non-SemCom PPO baseline (PPO-MEC-NSC) are not
implemented -- see "Extending to other baselines" below if you want to add
one.

## Extending to other baselines (SAC-MEC-SC, PPO-MEC-NSC)

`DDPGAgent` (`src/uav_semcom_ppo/ddpg_agent.py`) is written to be a template
for adding more comparison algorithms: it deliberately mirrors `PPOAgent`'s
public interface exactly -- `__init__(cfg, env)`, `train() -> history`,
`evaluate(episodes) -> {avg_delay, std_delay, avg_energy, std_energy}` -- and
reuses `UAVSemanticEnv` and `ActionMapper` unchanged. To add another
algorithm:
1. Implement a new `<Algo>Agent` class with that same interface (SAC would
   look a lot like `DDPGAgent` but with a stochastic actor + entropy
   temperature instead of deterministic + noise; PPO-MEC-NSC ("non-SemCom")
   would just be `PPOAgent` run against a modified environment where
   `semantic_compressed_bits()` always returns the raw, uncompressed size).
2. Register it in `scripts/sweep_utils.py`'s `AGENT_CLASSES`/`ALGO_LABELS`/
   `ALGO_COLORS` dicts.
3. Add its name to each script's `--algos` choices list.
No changes to the environment, action mapper, or any plotting script's core
logic should be needed beyond that.

## Notes carried over from earlier reproduction work

- The semantic-similarity function in `environment.py::semantic_similarity` is
  a calibrated sigmoid **placeholder** — the paper's empirical DeepSC-style
  rate-distortion curve-fit coefficients aren't published, so this proxy
  saturates with more symbols/word and higher SNR but isn't fit to real data.
- `Config.M` (number of IoT devices) is a plain int here, not padded to a
  fixed `max_devices`. If you want a single fixed-architecture policy that
  supports device-count sweeps (as in the Stable-Baselines3 version of this
  project), that padding logic would go in `environment.py::_get_state` and
  `networks.py` (fixed-size device/action heads with masking), not in
  `agent.py`.
- The jammer (Eq. 6-8 of the paper) is modeled in `environment.py::sinr()`:
  a fixed-position jammer with configurable transmit power interferes with
  the UAV-device link, plus a per-device NLoS obstruction penalty resampled
  each episode. `ActionMapper` and `UAVSemanticEnv.step()` both use this
  full SINR (not a plain SNR) for semantic-similarity and rate calculations.

### Most recent fix: Compute uses raw task bits (matches paper physics)

Semantic compression (DeepSC encoding) reduces the bits **transmitted over the
air**, but the device CPU and MEC CPU both operate on the **original task
information**. Previously `D_sem` (15–50% of raw bits) was used for all three
quantities — local compute, air-link transmission, and MEC compute — which cut
compute time/energy by 15–50% and made the per-slot cost 10–50× smaller than
the paper's scale.

After this fix:
- `T_loc = (1-R) × raw_bits × c / f_device`  (local compute uses raw bits)
- `T_tx  = R × D_sem / rate`                  (transmission uses compressed bits ✓)
- `T_uav = R × raw_bits × c / f_UAV`          (MEC compute uses raw bits)
- Same split for energy: `E_device`, `E_uav_comp` use raw bits; `E_trans` uses D_sem

This brings reported compute energy into the **0.3–2 J** range per slot
(E_fly dominates at ~4–100 J for a moving UAV), so total per-slot energy is
**5–100 J** depending on speed — comparable to the paper's Figs. 4a/5a
scale of 5–28 J. The episode reward now starts at roughly **−150 to −250**
and converges to **−20 to −30** for PPO-MEC-SC, matching Fig. 3's scale.


### Latest fix: Reduced state dimensionality

Removed per-device `wait_time` from the observable state vector. Previously, the policy 
could see wait times for all devices, making state_dim = 2M + 4 (54 dims at M=25). 
This made the state highly non-stationary (wait times change every step based on scheduling 
decisions) and difficult to learn from.

After fix: state_dim = 2M + 3 (53 dims at M=25), containing only UAV position, device 
task sizes, channel gains, energy used, and mission progress. Wait times are still tracked 
internally and used to compute the `fairness_penalty` in the reward (via the μ Lagrangian 
term), but they're not part of the observable state.

This matches the paper's approach more closely: fairness is enforced via the reward signal, 
not by making devices' "backlog" visible to the policy.

**Remaining issue:** Energy still shows noisy trends rather than monotonic increase with M.
This may be due to the aggressive semantic QoS penalty (eta_semantic = 5.0), which forces 
the policy to prioritize semantic reconstruction accuracy over transmission efficiency. 
If energy trends remain noisy, try reducing eta_semantic to 0.5-1.0 to relax the semantic 
constraint and see if transmission efficiency improves.


### DDPG Fix: TD3-Style Double Critics

DDPG was suffering from overestimation bias in Q-values, causing reward oscillations 
between -35 and -120 with no convergence. Implemented TD3 (Twin Delayed DDPG) 
improvements to stabilize it:

1. **Dual critics**: Two independent Q-networks, target computation uses `min(Q1, Q2)` 
   to prevent positive overestimation bias
2. **Target policy smoothing**: Add Gaussian noise (~0.2 std) to target actions during 
   TD target computation, reducing overfitting to sharp actions
3. **Delayed policy updates**: Actor updates only every N critic updates 
   (default N=2), allowing critics to stabilize before actor changes
4. **Gradient clipping**: Prevent exploding gradients (was already present)

**Result**: DDPG now converges to reward -70 to -75 (vs. before: -120 collapses). 
Still 20-25 units worse than PPO (-25 to -30), but much more stable and usable for 
comparison. The performance gap reflects DDPG's inherent difficulty with stochastic 
environments (random task sizes, channel conditions) without an explicit entropy bonus 
or exploration strategy like PPO's clipping.

**Remaining gap**: PPO is fundamentally better suited for this problem due to:
- On-policy learning (natural exploration through trust region)
- Clipped surrogate objective provides robustness
- GAE provides low-variance advantage estimates

DDPG requires tuning (noise decay, replay buffer size, update delays) which can 
be done in the Config class if needed for further improvement.

