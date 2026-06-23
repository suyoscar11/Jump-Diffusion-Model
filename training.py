"""
common.py
=========
Shared library for the 1D jump-diffusion flow-map verification (applied-math
version).

The training-free diffusion engine here MIRRORS the JCP reference code
(`Diffusion_Runaway-main/1D_git/label.gene.py` + `train_NN.py`):
  * learn the INCREMENT  (y - x) * diff_scale  (not the absolute next state),
  * find K nearest neighbours in x-space ONCE (FAISS if available, else torch),
  * run the probability-flow ODE on GPU in torch, reusing those neighbours,
    with the schedule  alpha(t)=1-t+dt,  sigma2(t)=t+dt,
  * distill a small FN_Net  G(x, z) -> increment  by supervised MSE.
This is the efficient path; my earlier per-step CPU-KNN sampler was the slow
part and has been removed.

On top of that engine we add the JUMP extension and compare two architectures:
  * MONOLITHIC      : learn p(X_{t+Dt}|X_t) directly (the existing method).
  * JUMP-STRUCTURED : continuous-flow head + Poisson jump-count + jump-size head,
                          X_{t+Dt} = G_flow(X_t,z) + sum_{i=1}^N G_jump(X_t,z_i),
                          N ~ Poisson(lambda(X_t) Dt).

Two pure-math test models (Config.model):
  * "merton"     : EXACT transition density (Poisson-weighted Gaussian mixture).
  * "double_well": heavy-tailed (Student-t) jumps; fine-MC reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ----------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------
@dataclass
class Config:
    model: str = "bounded"           # "merton" | "double_well" | "bounded"

    # ---- Merton parameters (constant coefficients) ------------------------
    mu: float = 0.5
    sigma_m: float = 0.20
    lam_m: float = 1.5
    jump_mean: float = 1.0
    jump_std: float = 0.20

    # ---- double-well parameters ------------------------------------------
    sigma_dw: float = 0.30
    lam_dw: float = 1.0
    jump_scale_dw: float = 0.5
    jump_df_dw: float = 3.0

    # ---- the LARGE coarse step the flow map learns -----------------------
    dt: float = 1.0
    x_range: tuple = (-2.5, 3.5)

    # ---- reference fine integrator (the "numerical method") --------------
    n_sub: int = 200
    seed: int = 0

    # ---- training-free diffusion engine (mirrors the JCP code) -----------
    diff_scale: float = 3.0          # scale on the increment (better conditioned)
    diff_scale_jump: float = 1.0     # scale on the jump-size target
    train_size_labels: int = 40000   # number of (x0,z)->increment labels to make
    knn_k: int = 512                 # neighbours per x0 (their short_size)
    ode_steps: int = 1000            # probability-flow ODE steps (their 10000; 1000 is plenty in 1D)
    label_chunk: int = 20000         # chunk over labels to bound GPU memory

    # ---- distillation net (their FN_Net) ---------------------------------
    hid_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-6
    distill_iters: int = 10000       # full-batch Adam, best-on-validation
    epochs_rate: int = 80
    batch_size_rate: int = 4096

    # ---- dataset sizes ----------------------------------------------------
    n_full: int = 400000             # big-step full-process pairs (mono + counts)
    n_continuous: int = 400000       # big-step continuous-only pairs (flow head)
    n_jump_samples: int = 300000     # single-jump samples (size head)
    n_ref_mc: int = 200000
    rollout_K: int = 8               # number of big ML steps for the terminal-time test

    # ---- bounded-domain model (model="bounded"): absorbing BCs, RE-aligned ---
    x_min: float = 0.0
    x_max: float = 6.0
    drift_b: float = 0.0             # drift inside the domain (0 = Brownian + jumps)
    sigma_b: float = 0.7
    lam_b: float = 1.0               # jump rate
    jump_mean_b: float = 0.0
    jump_std_b: float = 0.7
    exit_hidden: int = 256

    device: str = "cuda"
    data_dir: str = "artifacts_bd"   # bounded run -> separate folder (keeps Merton's artifacts/)

    def lam(self, x):
        x = np.asarray(x, dtype=np.float64)
        return (self.lam_m if self.model == "merton" else self.lam_dw) * np.ones_like(x)

    def to_dict(self):
        d = asdict(self); d["x_range"] = list(self.x_range); return d


# ----------------------------------------------------------------------------
# 2. The "true" SDE model
# ----------------------------------------------------------------------------
def drift(x, cfg: Config):
    x = np.asarray(x, dtype=np.float64)
    if cfg.model == "merton":
        return cfg.mu * np.ones_like(x)
    elif cfg.model == "double_well":
        return x - x ** 3
    raise ValueError(cfg.model)


def diffusion(x, cfg: Config):
    x = np.asarray(x, dtype=np.float64)
    s = cfg.sigma_m if cfg.model == "merton" else cfg.sigma_dw
    return np.full_like(x, s)


def sample_jump(x, cfg: Config, rng):
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if cfg.model == "merton":
        return rng.normal(cfg.jump_mean, cfg.jump_std, size=x.shape)
    return cfg.jump_scale_dw * rng.standard_t(cfg.jump_df_dw, size=x.shape)


def merton_logpdf(y, x0, cfg: Config, n_max=120, T=None):
    """Exact log transition density for Merton over a horizon T (Poisson-weighted
    Gaussian mixture). Merton is a Levy process, so the density over any horizon T
    is the same mixture with dt -> T.  T defaults to the one-step cfg.dt."""
    assert cfg.model == "merton"
    T = cfg.dt if T is None else T
    y = np.asarray(y, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64) * np.ones_like(y)
    lam_dt = cfg.lam_m * T
    comps = np.empty((n_max + 1, y.shape[0]))
    for n in range(n_max + 1):
        mean = x0 + cfg.mu * T + n * cfg.jump_mean
        var = cfg.sigma_m ** 2 * T + n * cfg.jump_std ** 2
        log_pois = -lam_dt + n * math.log(lam_dt) - math.lgamma(n + 1)
        comps[n] = log_pois - 0.5 * np.log(2 * math.pi * var) - 0.5 * (y - mean) ** 2 / var
    m = comps.max(axis=0)
    return m + np.log(np.sum(np.exp(comps - m), axis=0))


def merton_exact_sample(x0, cfg: Config, rng, T=None):
    T = cfg.dt if T is None else T
    x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))
    n = rng.poisson(cfg.lam_m * T, size=x0.shape)
    jump_sum = rng.normal(n * cfg.jump_mean, np.sqrt(n) * cfg.jump_std)
    cont = cfg.mu * T + cfg.sigma_m * math.sqrt(T) * rng.standard_normal(x0.shape)
    return x0 + cont + jump_sum, n


def simulate_coarse_step(x0, cfg: Config, rng, with_jumps=True, n_sub=None):
    """Fine jump-adapted Euler-Maruyama over one coarse step dt (the reference
    numerical method)."""
    n_sub = cfg.n_sub if n_sub is None else n_sub
    x = np.array(x0, dtype=np.float64).copy()
    dt_sub = cfg.dt / n_sub
    sqrt_dt = math.sqrt(dt_sub)
    n_jumps = np.zeros_like(x, dtype=np.int64)
    for _ in range(n_sub):
        if with_jumps:
            fired = rng.random(x.shape) < (1.0 - np.exp(-cfg.lam(x) * dt_sub))
            if fired.any():
                x[fired] += sample_jump(x[fired], cfg, rng)
                n_jumps[fired] += 1
        x = x + drift(x, cfg) * dt_sub + diffusion(x, cfg) * sqrt_dt * rng.standard_normal(x.shape)
    return x, n_jumps


def reference_transition_samples(x0, cfg: Config, rng):
    if cfg.model == "merton":
        return merton_exact_sample(x0, cfg, rng)[0]
    return simulate_coarse_step(x0, cfg, rng, with_jumps=True)[0]


# ---- bounded-domain jump-diffusion with absorbing boundaries (RE-aligned) ---
def simulate_bounded_step(x0, cfg: Config, rng, n_sub=None):
    """One coarse step dt of  dx = drift_b dt + sigma_b dW + jumps  on the domain
    [x_min, x_max] with ABSORBING boundaries.  Returns (x_end, alive):
      alive[i] == False  if particle i crossed a boundary during the step (exited).
    Surviving particles stay inside the domain by construction, so a flow map
    trained on survivors is never evaluated out-of-domain -- this is what removes
    the OOD problem over long rollouts."""
    n_sub = cfg.n_sub if n_sub is None else n_sub
    x = np.array(x0, dtype=np.float64).copy()
    alive = np.ones(x.shape, dtype=bool)
    dt_sub = cfg.dt / n_sub
    sqrt_dt = math.sqrt(dt_sub)
    for _ in range(n_sub):
        a = alive
        if not a.any():
            break
        # jump sub-step
        fired = a & (rng.random(x.shape) < (1.0 - math.exp(-cfg.lam_b * dt_sub)))
        if fired.any():
            x[fired] += rng.normal(cfg.jump_mean_b, cfg.jump_std_b, size=int(fired.sum()))
        # diffusion sub-step (only living particles move)
        x[a] += cfg.drift_b * dt_sub + cfg.sigma_b * sqrt_dt * rng.standard_normal(int(a.sum()))
        # absorb whatever left the domain this sub-step
        out = a & ((x < cfg.x_min) | (x > cfg.x_max))
        alive[out] = False
    return x, alive


# ----------------------------------------------------------------------------
# 3. Utilities
# ----------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)


def get_device(cfg):
    if _HAS_TORCH and cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def wasserstein1(x, y):
    x = np.sort(np.asarray(x, dtype=np.float64)); y = np.sort(np.asarray(y, dtype=np.float64))
    n = max(len(x), len(y)); qs = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(x, qs) - np.quantile(y, qs))))


def kl_hist(samples, logpdf_fn, lo, hi, nbins=200):
    grid = np.linspace(lo, hi, nbins + 1); centers = 0.5 * (grid[:-1] + grid[1:])
    width = grid[1] - grid[0]
    p = np.exp(logpdf_fn(centers)); p /= p.sum() * width
    hist, _ = np.histogram(samples, bins=grid, density=True)
    eps = 1e-8
    return float(np.sum(p * (np.log(p + eps) - np.log(hist + eps))) * width)


def hellinger_hist(samples, logpdf_fn, lo, hi, nbins=200):
    grid = np.linspace(lo, hi, nbins + 1); centers = 0.5 * (grid[:-1] + grid[1:])
    width = grid[1] - grid[0]
    p = np.exp(logpdf_fn(centers)); p /= p.sum() * width
    hist, _ = np.histogram(samples, bins=grid, density=True)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(hist)) ** 2) * width))


# ----------------------------------------------------------------------------
# 4. Training-free diffusion engine  (mirrors label.gene.py)
# ----------------------------------------------------------------------------
# Forward-process schedule:  alpha(0)=1, sigma2(0)=0 (data) ;  ~noise at t=1.
def cond_alpha(t, dt):  return 1.0 - t + dt
def cond_sigma2(t, dt): return t + dt
def drift_f(t, dt):     return -1.0 / cond_alpha(t, dt)
def diff_g2(t, dt):     return 1.0 - 2.0 * drift_f(t, dt) * cond_sigma2(t, dt)


def knn_neighbors(c_sample, c0, k, device):
    """Indices (len(c0), k) of nearest neighbours of c0 within c_sample (L2 in
    conditioning space). FAISS-GPU if available, else a chunked torch fallback."""
    c_sample = np.ascontiguousarray(c_sample, dtype=np.float32)
    c0 = np.ascontiguousarray(c0, dtype=np.float32)
    d = c_sample.shape[1]
    try:
        import faiss
        if device.type == "cuda" and faiss.get_num_gpus() > 0:
            index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d)
        else:
            index = faiss.IndexFlatL2(d)
        index.add(c_sample)
        _, idx = index.search(c0, k)
        return idx
    except Exception:
        if not _HAS_TORCH:
            raise
        cs = torch.tensor(c_sample, device=device)
        out = np.empty((len(c0), k), dtype=np.int64)
        bs = 1024
        for i in range(0, len(c0), bs):
            cb = torch.tensor(c0[i:i + bs], device=device)
            dist = torch.cdist(cb, cs)
            out[i:i + bs] = torch.topk(dist, k, largest=False).indices.cpu().numpy()
        return out


def ode_solve(zt, neigh, ode_steps):
    """Probability-flow ODE (their ODE_solver), batched and reusing neighbours.
    zt:(B,d) latent;  neigh:(B,k,d) neighbour TARGETS (scaled increments)."""
    device = zt.device
    t_vec = torch.linspace(1.0, 0.0, ode_steps + 1, device=device)
    for j in range(ode_steps):
        t = t_vec[j + 1]; dt = t_vec[j] - t_vec[j + 1]
        a = cond_alpha(t, dt); s2 = cond_sigma2(t, dt)
        diff = zt[:, None, :] - a * neigh                 # (B,k,d)
        logw = -0.5 * torch.sum(diff ** 2, dim=2) / s2    # (B,k)
        w = torch.softmax(logw, dim=1)                    # (B,k)
        score = torch.sum((-diff / s2) * w[:, :, None], dim=1)  # (B,d)
        zt = zt - (drift_f(t, dt) * zt - 0.5 * diff_g2(t, dt) * score) * dt
    return zt


def generate_labels(c_sample, target_sample, cfg, rng, device):
    """Make (c0, zT) -> target labels via the training-free probability-flow ODE.
    c_sample:(N,d) conditioning;  target_sample:(N,d) the quantity to learn
    (already scaled, e.g. (y-x)*diff_scale or eps*diff_scale_jump).
    Returns c0:(B,d), zT:(B,d), y_gen:(B,d)."""
    N, d = c_sample.shape
    B = min(cfg.train_size_labels, N)
    sel = rng.permutation(N)[:B]
    c0 = c_sample[sel]
    idx = knn_neighbors(c_sample, c0, cfg.knn_k, device)        # (B,k)
    neigh_all = target_sample[idx]                              # (B,k,d)

    zT = rng.standard_normal((B, d)).astype(np.float32)
    y_gen = np.empty((B, d), dtype=np.float32)
    for s in range(0, B, cfg.label_chunk):
        e = min(s + cfg.label_chunk, B)
        zt = torch.tensor(zT[s:e], device=device)
        neigh = torch.tensor(neigh_all[s:e], dtype=torch.float32, device=device)
        y_gen[s:e] = ode_solve(zt, neigh, cfg.ode_steps).cpu().numpy()
    return c0.astype(np.float32), zT, y_gen


# ----------------------------------------------------------------------------
# 5. Neural networks  (FN_Net distillation net + Poisson rate net)
# ----------------------------------------------------------------------------
if _HAS_TORCH:

    class FN_Net(nn.Module):
        """Distilled generator G(x, z) -> target.  Same architecture as the JCP
        train_NN.py: two hidden tanh layers."""
        def __init__(self, input_dim, output_dim, hid_size):
            super().__init__()
            self.input = nn.Linear(input_dim, hid_size)
            self.fc1 = nn.Linear(hid_size, hid_size)
            self.output = nn.Linear(hid_size, output_dim)

        def forward(self, x):
            x = torch.tanh(self.input(x))
            x = torch.tanh(self.fc1(x))
            return self.output(x)

    class RateNet(nn.Module):
        """r(x) = E[#jumps in dt | x]  (Poisson mean, softplus output)."""
        def __init__(self, cfg: Config):
            super().__init__()
            h = cfg.hid_size
            self.net = nn.Sequential(
                nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                nn.Linear(h, 1), nn.Softplus())

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class ExitNet(nn.Module):
        """P(exit during dt | x): the absorbing-boundary indicator.  Same shape as
        the JCP train_escape_binary EscapeModel (3 hidden LeakyReLU + dropout)."""
        def __init__(self, cfg: Config):
            super().__init__()
            h = cfg.exit_hidden
            self.net = nn.Sequential(
                nn.Linear(1, h), nn.LeakyReLU(0.01), nn.Dropout(0.2),
                nn.Linear(h, h), nn.LeakyReLU(0.01), nn.Dropout(0.2),
                nn.Linear(h, h), nn.LeakyReLU(0.01), nn.Dropout(0.2),
                nn.Linear(h, 1), nn.Sigmoid())

        def forward(self, x):
            return self.net(x).squeeze(-1)
