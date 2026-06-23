# 1D Jump-Diffusion Flow-Map — Verification (accuracy + speed)

A minimal proofread of one idea: **a training-free diffusion flow map can learn
the large-Δt transition of a jump-diffusion SDE in a single step, matching the
classical numerical method on accuracy while being much faster.**

- **SDE (ground truth)**: 1D jump-diffusion. Default `merton` (linear drift +
  Brownian + compound-Poisson Gaussian jumps) — it has an **exact** transition
  density (Poisson-weighted Gaussian mixture), so accuracy is checked against a
  closed form. `double_well` (nonlinear, heavy-tailed jumps) is also available.
- **Classical numerical method**: fine-step jump-adapted Euler–Maruyama — both
  the ground-truth sampler and the speed baseline.
- **ML model**: one training-free diffusion flow map `G(x,z)` (your JCP method),
  producing `X_{t+Δt}` in a single forward pass.

## Three logic pieces

| Script | Role |
|---|---|
| `data_generation.py` | Make `(X_t, X_{t+Δt})` pairs from the SDE at large Δt → `data_pairs.npz`. |
| `training.py` | Train the flow map (training-free labels → `FN_Net` distillation) → `ckpt_flow.pt`. |
| `inference.py` | Compare ML vs numerical: **V1 accuracy**, **V2 speed**. |

`common.py` is the shared library: the SDE + exact Merton density + fine EM, and
the **training-free engine copied from your JCP code** (learn the increment,
KNN-once + GPU probability-flow ODE, `FN_Net` distillation).

## Run (workstation)

```bash
pip install numpy torch matplotlib      # + optional: faiss-gpu (faster KNN)
cd jump1d
python data_generation.py   # -> artifacts/data_pairs.npz, config.json
python training.py          # -> artifacts/ckpt_flow.pt
python inference.py         # -> artifacts/V1_accuracy.png, V2_speed.png, metrics.json
```

Pick the model and Δt in `Config` (`common.py`). Set `device="cpu"` if no GPU.
Speed knobs: `ode_steps` (1000; paper used 10000), `knn_k` (512),
`train_size_labels` (40k). For a smoke test, shrink `n_full`, `train_size_labels`,
`distill_iters`, `n_ref_mc`.

## What you get

- **V1 (accuracy)** — `V1_accuracy.png`: at several `X_t`, the ML transition
  histogram overlaid on the numerical truth and (for Merton) the exact pdf;
  `metrics.json` gives W1 / KL / Hellinger. This is the "is the idea correct?"
  check — the flow map should reproduce the (multi-modal, jump-induced) density.
- **V2 (speed)** — `V2_speed.png`: accuracy (W1 to ground truth) vs. wall-clock
  to generate the Δt-transition, ML (one step) vs. Euler–Maruyama at several
  sub-step counts; `metrics.json` reports the **speedup factor** vs. fine EM. The
  ML point should sit at low error *and* low time.
- **V3 (long-time rollout)** — `V3_terminal.png`: from several initial conditions,
  propagate to a terminal time `T = rollout_K·Δt` with the ML map (`rollout_K` big
  steps) vs. the numerical algorithm (`rollout_K·n_sub` small steps), and compare
  the **terminal-time distribution** (overlaid on the exact density at `T` for
  Merton). Tests whether the learned map composes over many steps to simulate a
  whole trajectory; `metrics.json` reports the per-init terminal W1 and the
  compounded rollout speedup.
  - *Caveat:* with Merton's positive drift the state leaves the training
    `x_range` over a long rollout (extrapolation). The increment is
    state-independent so it may still hold, but for a clean long-time test widen
    `Config.x_range` to cover the trajectory's reach and retrain.

## Bounded-domain variant (`model="bounded"`) — fixes long-rollout OOD, RE-aligned

The unbounded Merton rollout fails at long horizons because the runaway drift
pushes the state **out of the training range** (OOD): the terminal `W1` grows
with the initial condition. A bounded domain with **absorbing boundaries** fixes
this *structurally* — survivors never leave `[x_min, x_max]`, so the flow map is
never evaluated out-of-domain, and particles that would leave are removed by an
**exit head** (your `train_escape_binary` classifier). This is also exactly the
runaway-electron structure (the separatrix).

The SDE is `dx = drift_b·dt + sigma_b·dW + jumps` on `[x_min, x_max]`, absorbing
at both ends. Set in `Config`:

```python
model     = "bounded"
x_min, x_max = 0.0, 6.0
drift_b   = 0.0          # 0 = Brownian + jumps; can be mean-reverting
sigma_b   = 0.7
lam_b     = 1.0          # jump rate
jump_mean_b, jump_std_b = 0.0, 0.7
dt        = 1.0          # the ML big step
rollout_K = 8            # terminal time T = rollout_K * dt
data_dir  = "artifacts_bd"
```

Run the same three scripts (they branch on `model`):

```bash
python data_generation.py   # -> survivor pairs (data_pairs.npz) + exit labels (data_exit.npz)
python training.py          # -> ckpt_flow.pt (survivor flow map) + ckpt_exit.pt (exit head)
python inference.py         # -> B1_onestep.png, B2_rollout.png, metrics.json
```

Outputs:
- **B1** `B1_onestep.png`: the exit probability `P(exit|x)` (head vs. numerical)
  and the one-step survivor density (flow map vs. numerical) — verifies both heads.
- **B2** `B2_rollout.png`: the long-time rollout with exit — the **surviving
  fraction vs. step** and the **terminal surviving density**, ML vs. the numerical
  method; `metrics.json` reports terminal `W1` and the rollout speedup. This is
  the RE-style long-horizon prediction, and the terminal density should now match
  (no OOD), unlike the unbounded Merton rollout.

## Notes

- The larger Δt is, the more jumps per step (harder, more multi-modal target) and
  the bigger the speedup over fine EM — the regime where the method pays off.
- `merton` increments are state-independent (constant coefficients); `double_well`
  makes the conditioning non-trivial. Start with `merton`.
- This is intentionally one model and one SDE. A jump-aware (compound-structured)
  variant and the bounded-domain exit head are documented extensions, not part of
  this proofread.
```
