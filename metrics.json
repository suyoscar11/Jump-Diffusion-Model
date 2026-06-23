"""
inference.py  --  LOGIC PIECE 3 of 3
====================================
Proofread the idea: does the training-free diffusion flow map reproduce the
large-Delta-t transition of a jump-diffusion SDE, and is it faster than the
classical numerical method?

Compared quantities:
  - ML model        : one forward pass of the learned flow map  x -> x + G(x,z)/scale
  - classical method: fine-step jump-adapted Euler-Maruyama (the ground truth and
                      the speed baseline)
  - exact density   : for Merton only, the closed-form transition (gold truth)

Two outputs:
  V1 ACCURACY  -- learned vs ground-truth transition density at several X_t
                  (overlaid on the exact pdf for Merton); W1 / KL / Hellinger.
  V2 SPEED     -- wall-clock to generate the Dt-transition: ML (one step) vs
                  Euler-Maruyama at several sub-step counts, with accuracy for
                  each; reports the speedup factor.

Run:  python inference.py   ->  V1_accuracy.png, V2_speed.png, metrics.json
"""

import json
import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (Config, FN_Net, get_device, merton_logpdf,
                    merton_exact_sample, simulate_coarse_step,
                    reference_transition_samples, wasserstein1, kl_hist,
                    hellinger_hist, set_seed, simulate_bounded_step)


class FlowMap:
    """The learned ML model: one forward pass = one large-Dt step."""
    def __init__(self, cfg, dev):
        ck = torch.load(os.path.join(cfg.data_dir, "ckpt_flow.pt"), map_location=dev,
                        weights_only=False)
        self.net = FN_Net(2, 1, cfg.hid_size).to(dev); self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.xm, self.xs = ck["x_mean"], ck["x_std"]
        self.ym, self.ys = ck["y_mean"], ck["y_std"]
        self.scale = ck["diff_scale"]; self.dev = dev

    @torch.no_grad()
    def step(self, x, rng):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        z = rng.standard_normal((len(x), 1))
        inp = ((np.hstack([x, z]) - self.xm) / self.xs).astype(np.float32)
        out = self.net(torch.tensor(inp, device=self.dev)).cpu().numpy()
        incr = (out * self.ys + self.ym) / self.scale
        return (x + incr).ravel()


# ----------------------------------------------------------------------------
def v1_accuracy(fm, cfg, rng, metrics):
    test_x = [float(np.mean(cfg.x_range)) - 1.0,
              float(np.mean(cfg.x_range)),
              float(np.mean(cfg.x_range)) + 1.0]
    fig, ax = plt.subplots(1, len(test_x), figsize=(5.4 * len(test_x), 4.3))
    rows = []
    for k, x0 in enumerate(test_x):
        n = 80000
        truth = reference_transition_samples(np.full(n, x0), cfg, rng)  # classical/exact samples
        ml = fm.step(np.full(n, x0), rng)

        lo, hi = np.quantile(truth, 0.002), np.quantile(truth, 0.998)
        bins = np.linspace(lo, hi, 80)
        ax[k].hist(truth, bins=bins, density=True, alpha=0.35, color="gray",
                   label="numerical (truth)")
        ax[k].hist(ml, bins=bins, density=True, histtype="step", color="C3",
                   label="ML flow map")

        row = {"x0": x0, "W1_ml_vs_truth": wasserstein1(truth, ml)}
        if cfg.model == "merton":
            xx = np.linspace(lo, hi, 400)
            ax[k].plot(xx, np.exp(merton_logpdf(xx, x0, cfg)), "k-", lw=1.2, label="exact pdf")
            lpf = lambda g, x0=x0: merton_logpdf(g, x0, cfg)
            row["KL_ml_vs_exact"] = kl_hist(ml, lpf, lo, hi)
            row["Hellinger_ml_vs_exact"] = hellinger_hist(ml, lpf, lo, hi)
            # reference: how close the exact-sample truth itself is (sampling floor)
            row["W1_truth_vs_exactpdf_floor"] = wasserstein1(
                truth, merton_exact_sample(np.full(n, x0), cfg, rng)[0])
        rows.append(row)
        ax[k].set_title(f"V1: transition at X_t={x0:.2f}"); ax[k].set_xlabel("X_{t+dt}")
        ax[k].legend(fontsize=8)
    metrics["v1_accuracy"] = rows
    fig.tight_layout(); fig.savefig(os.path.join(cfg.data_dir, "V1_accuracy.png"), dpi=130)
    plt.close(fig)


def _truth_samples(x0, cfg, rng):
    """Gold ground truth for scoring accuracy: exact for Merton, very fine EM else."""
    if cfg.model == "merton":
        return merton_exact_sample(x0, cfg, rng)[0]
    return simulate_coarse_step(x0, cfg, rng, with_jumps=True, n_sub=cfg.n_sub)[0]


def v2_speed(fm, cfg, rng, metrics):
    M = cfg.n_ref_mc
    lo, hi = cfg.x_range
    x0 = rng.uniform(lo + 0.3, hi - 0.3, size=M)
    gold = _truth_samples(x0, cfg, rng)

    # ML: warm up once (GPU), then time one big step
    _ = fm.step(x0[:1000], rng)
    t0 = time.perf_counter(); ml = fm.step(x0, rng); t_ml = time.perf_counter() - t0
    ml_row = {"method": "ML flow map (1 step)", "time_s": t_ml,
              "W1_to_truth": wasserstein1(gold, ml)}

    # Classical numerical method (Euler-Maruyama) at several sub-step counts
    em_rows = []
    for n_sub in [5, 20, 50, 100, cfg.n_sub]:
        t0 = time.perf_counter()
        xe, _ = simulate_coarse_step(x0, cfg, rng, with_jumps=True, n_sub=n_sub)
        dt = time.perf_counter() - t0
        em_rows.append({"method": f"Euler-Maruyama n_sub={n_sub}", "n_sub": n_sub,
                        "time_s": dt, "W1_to_truth": wasserstein1(gold, xe)})

    fine = em_rows[-1]
    speedup = fine["time_s"] / max(t_ml, 1e-9)
    metrics["v2_speed"] = {"ml": ml_row, "euler_maruyama": em_rows,
                           "speedup_ML_vs_fineEM": speedup,
                           "note": "EM timed in numpy (CPU); ML on " + str(fm.dev)}

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot([r["time_s"] for r in em_rows], [r["W1_to_truth"] for r in em_rows],
            "ko-", label="Euler-Maruyama")
    for r in em_rows:
        ax.annotate(f"n_sub={r['n_sub']}", (r["time_s"], r["W1_to_truth"]), fontsize=7)
    ax.scatter([t_ml], [ml_row["W1_to_truth"]], c="C3", s=120, marker="*", zorder=5,
               label="ML flow map (1 step)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"wall-clock to generate {M} transitions [s]")
    ax.set_ylabel("W1 to ground truth (lower = better)")
    ax.set_title(f"V2: accuracy vs cost  (model={cfg.model}, dt={cfg.dt})\n"
                 f"ML speedup vs finest EM = {speedup:.1f}x")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(cfg.data_dir, "V2_speed.png"), dpi=130)
    plt.close(fig)


def v3_terminal(fm, cfg, rng, metrics):
    """Long-time test: from several initial conditions, propagate to a terminal
    time T = K*dt with the ML map (K big steps) and compare the terminal-time
    distribution against the ground truth (exact for Merton, numerical otherwise).
    Also reports the compounded speedup over the whole trajectory."""
    K = cfg.rollout_K
    T = K * cfg.dt
    lo, hi = cfg.x_range
    inits = [lo + 1.0, 0.5 * (lo + hi), hi - 1.0]
    N = 60000

    fig, ax = plt.subplots(1, len(inits), figsize=(5.4 * len(inits), 4.3))
    rows = []
    t_ml_tot, t_num_tot = 0.0, 0.0
    for k, x0 in enumerate(inits):
        # --- ML rollout: K big steps ---
        xl = np.full(N, x0, dtype=np.float64)
        t0 = time.perf_counter()
        for _ in range(K):
            xl = fm.step(xl, rng)
        t_ml_tot += time.perf_counter() - t0

        # --- ground truth at terminal time T ---
        if cfg.model == "merton":
            truth = merton_exact_sample(np.full(N, x0), cfg, rng, T=T)[0]
        else:
            truth = np.full(N, x0, dtype=np.float64)
        # numerical algorithm to T (small dt): K * n_sub fine steps -- also the speed baseline
        xt = np.full(N, x0, dtype=np.float64)
        t0 = time.perf_counter()
        for _ in range(K):
            xt, _ = simulate_coarse_step(xt, cfg, rng, with_jumps=True, n_sub=cfg.n_sub)
        t_num_tot += time.perf_counter() - t0
        if cfg.model != "merton":
            truth = xt

        w1 = wasserstein1(truth, xl)
        rows.append({"x0": x0, "W1_terminal": w1})

        lo_, hi_ = np.quantile(truth, 0.002), np.quantile(truth, 0.998)
        bins = np.linspace(lo_, hi_, 80)
        ax[k].hist(truth, bins=bins, density=True, alpha=0.35, color="gray",
                   label="ground truth")
        ax[k].hist(xl, bins=bins, density=True, histtype="step", color="C3",
                   label="ML rollout (%d steps)" % K)
        if cfg.model == "merton":
            xx = np.linspace(lo_, hi_, 400)
            ax[k].plot(xx, np.exp(merton_logpdf(xx, x0, cfg, T=T)), "k-", lw=1.0,
                       label="exact pdf @ T")
        ax[k].set_title(f"V3: terminal density, X_0={x0:.1f}, T={T:g}\n(W1={w1:.3f})")
        ax[k].set_xlabel("X_T"); ax[k].legend(fontsize=8)

    metrics["v3_terminal"] = {"K_steps": K, "T": T, "per_init": rows,
                              "ml_rollout_time_s": t_ml_tot,
                              "numerical_rollout_time_s": t_num_tot,
                              "speedup_rollout": t_num_tot / max(t_ml_tot, 1e-9),
                              "note": "numerical on CPU, ML on " + str(fm.dev)}
    fig.suptitle(f"Long-time rollout to T={T:g}: ML {K} big steps vs numerical "
                 f"{K*cfg.n_sub} small steps  "
                 f"(speedup {t_num_tot/max(t_ml_tot,1e-9):.0f}x)", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(cfg.data_dir, "V3_terminal.png"),
                                    dpi=130, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
#  Bounded-domain verification (model="bounded"): flow map + exit head
# ===========================================================================
class ExitHead:
    """Loads the trained P(exit during dt | x) classifier."""
    def __init__(self, cfg, dev):
        from common import ExitNet
        ck = torch.load(os.path.join(cfg.data_dir, "ckpt_exit.pt"),
                        map_location=dev, weights_only=False)
        self.net = ExitNet(cfg).to(dev); self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.xm, self.xs, self.dev = ck["x_mean"], ck["x_std"], dev

    @torch.no_grad()
    def prob(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        xs = torch.tensor((x - self.xm) / self.xs, dtype=torch.float32, device=self.dev)
        return self.net(xs).cpu().numpy()


def bounded_rollout_ml(fm, eh, cfg, x0, rng):
    """ML rollout with absorbing boundary: each big step, the exit head removes
    exiters, survivors propagate via the flow map (and any flow output that lands
    outside the domain is also treated as an exit)."""
    x = np.array(x0, dtype=np.float64).copy()
    alive = np.ones(x.shape, dtype=bool)
    survival = []
    for _ in range(cfg.rollout_K):
        idx = np.where(alive)[0]
        if len(idx):
            pe = eh.prob(x[idx])
            ex = rng.random(len(idx)) < pe
            alive[idx[ex]] = False
            keep = idx[~ex]
            if len(keep):
                xn = fm.step(x[keep], rng)
                oob = (xn < cfg.x_min) | (xn > cfg.x_max)
                alive[keep[oob]] = False
                x[keep[~oob]] = xn[~oob]
        survival.append(float(alive.mean()))
    return x, alive, survival


def bounded_rollout_num(cfg, x0, rng):
    """Numerical rollout with the same absorbing boundary (ground truth)."""
    x = np.array(x0, dtype=np.float64).copy()
    alive = np.ones(x.shape, dtype=bool)
    survival = []
    for _ in range(cfg.rollout_K):
        idx = np.where(alive)[0]
        if len(idx):
            xe, al = simulate_bounded_step(x[idx], cfg, rng)
            alive[idx[~al]] = False
            x[idx[al]] = xe[al]
        survival.append(float(alive.mean()))
    return x, alive, survival


def vb_onestep(fm, eh, cfg, rng, metrics):
    """B1: exit probability over the domain, and survivor transition density."""
    grid = np.linspace(cfg.x_min, cfg.x_max, 200)
    # exit prob: head vs MC truth
    pe_pred = eh.prob(grid)
    pe_true = np.empty_like(grid)
    for k, xv in enumerate(grid):
        _, al = simulate_bounded_step(np.full(4000, xv), cfg, rng)
        pe_true[k] = np.mean(~al)
    metrics["b1_exitprob_max_abs_err"] = float(np.max(np.abs(pe_pred - pe_true)))

    test_x = [cfg.x_min + 0.25 * (cfg.x_max - cfg.x_min),
              0.5 * (cfg.x_min + cfg.x_max)]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(grid, pe_true, "k-", label="numerical truth")
    ax[0].plot(grid, pe_pred, "r--", label="exit head")
    ax[0].set_title("B1: exit probability  P(exit | x)")
    ax[0].set_xlabel("x"); ax[0].legend()
    rows = []
    for j, x0 in enumerate(test_x):
        n = 60000
        xe, al = simulate_bounded_step(np.full(n, x0), cfg, rng)
        truth = xe[al]                     # survivor endpoints (numerical)
        ml = fm.step(np.full(n, x0), rng)  # flow map (survivor transition)
        ml = ml[(ml >= cfg.x_min) & (ml <= cfg.x_max)]
        w1 = wasserstein1(truth, ml); rows.append({"x0": x0, "W1": w1})
        bins = np.linspace(cfg.x_min, cfg.x_max, 70)
        ax[1 + j].hist(truth, bins=bins, density=True, alpha=0.35, color="gray",
                       label="numerical survivors")
        ax[1 + j].hist(ml, bins=bins, density=True, histtype="step", color="C3",
                       label="flow map")
        ax[1 + j].set_title(f"B1: survivor density | x={x0:.2f}  (W1={w1:.3f})")
        ax[1 + j].set_xlabel("x_next"); ax[1 + j].legend(fontsize=8)
    metrics["b1_survivor_W1"] = rows
    fig.tight_layout(); fig.savefig(os.path.join(cfg.data_dir, "B1_onestep.png"), dpi=130)
    plt.close(fig)


def vb_rollout(fm, eh, cfg, rng, metrics):
    """B2: long-time rollout with exit -- survival fraction over steps and the
    terminal surviving density, ML vs the numerical method.  Also the speedup."""
    N = cfg.n_ref_mc
    x0 = rng.uniform(cfg.x_min, cfg.x_max, size=N)

    t0 = time.perf_counter()
    xl, al_l, surv_l = bounded_rollout_ml(fm, eh, cfg, x0, rng)
    t_ml = time.perf_counter() - t0
    t0 = time.perf_counter()
    xn, al_n, surv_n = bounded_rollout_num(cfg, x0, rng)
    t_num = time.perf_counter() - t0

    K = cfg.rollout_K; T = K * cfg.dt
    term_truth, term_ml = xn[al_n], xl[al_l]
    w1_term = wasserstein1(term_truth, term_ml)
    metrics["b2_rollout"] = {
        "K_steps": K, "T": T,
        "survival_final_truth": surv_n[-1], "survival_final_ml": surv_l[-1],
        "terminal_density_W1": w1_term,
        "ml_time_s": t_ml, "numerical_time_s": t_num,
        "speedup": t_num / max(t_ml, 1e-9),
        "note": "numerical CPU, ML on " + str(fm.dev)}

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    steps = np.arange(1, K + 1)
    ax[0].plot(steps, surv_n, "k-o", label="numerical")
    ax[0].plot(steps, surv_l, "r--s", label="ML rollout")
    ax[0].set_title("B2: surviving fraction vs step"); ax[0].set_xlabel("big step")
    ax[0].set_ylabel("fraction still in domain"); ax[0].legend()
    bins = np.linspace(cfg.x_min, cfg.x_max, 70)
    ax[1].hist(term_truth, bins=bins, density=True, alpha=0.35, color="gray",
               label="numerical")
    ax[1].hist(term_ml, bins=bins, density=True, histtype="step", color="C3",
               label="ML rollout")
    ax[1].set_title(f"B2: terminal surviving density @ T={T:g}  (W1={w1_term:.3f})")
    ax[1].set_xlabel("x"); ax[1].legend()
    fig.suptitle(f"Bounded long rollout: ML {K} big steps vs numerical "
                 f"{K*cfg.n_sub} small steps (speedup {t_num/max(t_ml,1e-9):.0f}x)",
                 y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(cfg.data_dir, "B2_rollout.png"),
                                    dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = Config()
    set_seed(cfg.seed + 2)
    rng = np.random.default_rng(cfg.seed + 2)
    dev = get_device(cfg)
    print("Device:", dev, "| model:", cfg.model, "| dt:", cfg.dt)

    fm = FlowMap(cfg, dev)
    metrics = {"model": cfg.model, "dt": cfg.dt}

    if cfg.model == "bounded":
        eh = ExitHead(cfg, dev)
        print("B1: one-step survivor density + exit prob ..."); vb_onestep(fm, eh, cfg, rng, metrics)
        print("B2: long-time rollout with exit ...");           vb_rollout(fm, eh, cfg, rng, metrics)
        with open(os.path.join(cfg.data_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print("\n=== SUMMARY ==="); print(json.dumps(metrics, indent=2))
        print("\nWrote B1_onestep.png, B2_rollout.png, metrics.json to", cfg.data_dir)
        return

    print("V1: accuracy vs ground truth ...");  v1_accuracy(fm, cfg, rng, metrics)
    print("V2: speed vs numerical method ..."); v2_speed(fm, cfg, rng, metrics)
    print("V3: terminal-time long rollout ..."); v3_terminal(fm, cfg, rng, metrics)

    with open(os.path.join(cfg.data_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(metrics, indent=2))
    print("\nWrote V1_accuracy.png, V2_speed.png, metrics.json to", cfg.data_dir)


if __name__ == "__main__":
    main()
