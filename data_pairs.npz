"""
training.py  --  LOGIC PIECE 2 of 3
===================================
Train ONE training-free diffusion flow map for the large-Delta-t transition,
following the JCP recipe (label.gene.py + train_NN.py):

  1. learn the INCREMENT  (y - x) * diff_scale,
  2. generate (x0, z) -> increment labels with the probability-flow ODE
     (KNN once + GPU ODE; the training-free score, no score network trained),
  3. distill FN_Net  G(x, z) -> increment  by MSE (standardized, best-on-valid).

At inference one forward pass gives X_{t+Dt} = x + G(x, z)/diff_scale.

Run:  python training.py
Outputs:  ckpt_flow.pt
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import Config, FN_Net, generate_labels, get_device, set_seed


def train_exit_head(cfg, dev):
    """Binary classifier P(exit during dt | x), trained with BCE -- the
    absorbing-boundary indicator (mirrors the JCP train_escape_binary)."""
    from common import ExitNet
    d = np.load(os.path.join(cfg.data_dir, "data_exit.npz"))
    x = d["x"].astype(np.float32).reshape(-1, 1)
    y = d["exited"].astype(np.float32)
    xm, xs = float(x.mean()), float(x.std() + 1e-8)
    net = ExitNet(cfg).to(dev)
    Xb = torch.tensor((x - xm) / xs, dtype=torch.float32)
    Yb = torch.tensor(y, dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xb, Yb), batch_size=2048, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    lossfn = nn.BCELoss()
    for ep in range(50):
        net.train(); tot = 0.0
        for xi, yi in dl:
            xi, yi = xi.to(dev), yi.to(dev)
            opt.zero_grad(); loss = lossfn(net(xi), yi)
            loss.backward(); opt.step(); tot += loss.item() * len(yi)
        if (ep + 1) % 10 == 0:
            print(f"    [exit] epoch {ep+1}/50  bce={tot/len(Xb):.5f}")
    net.eval()
    torch.save({"state_dict": net.state_dict(), "x_mean": xm, "x_std": xs,
                "cfg": cfg.to_dict()}, os.path.join(cfg.data_dir, "ckpt_exit.pt"))
    print("    [exit] saved ckpt_exit.pt")


def main():
    cfg = Config()
    set_seed(cfg.seed + 1)
    rng = np.random.default_rng(cfg.seed + 1)
    dev = get_device(cfg)
    print("Device:", dev, "| model:", cfg.model)

    d = np.load(os.path.join(cfg.data_dir, "data_pairs.npz"))
    x = d["x_sample"].astype(np.float64).reshape(-1, 1)
    y = d["y_sample"].astype(np.float64).reshape(-1, 1)
    target = (y - x) * cfg.diff_scale                 # learn the scaled increment

    # ---- (1)+(2) training-free labels ------------------------------------
    print(f"Generating training-free labels "
          f"({cfg.train_size_labels} pts x {cfg.ode_steps} ODE steps) ...")
    c0, zT, y_gen = generate_labels(x.astype(np.float32), target.astype(np.float32),
                                    cfg, rng, dev)

    xTrain = np.hstack([c0, zT]).astype(np.float32)   # (B, 2) = (x0, z)
    yTrain = y_gen.astype(np.float32)                 # (B, 1) = increment
    good = np.isfinite(xTrain).all(1) & np.isfinite(yTrain).all(1)
    xTrain, yTrain = xTrain[good], yTrain[good]
    print(f"Usable labels: {len(xTrain)}")

    # ---- (3) distill FN_Net ----------------------------------------------
    xm, xs = xTrain.mean(0, keepdims=True), xTrain.std(0, keepdims=True) + 1e-8
    ym, ys = yTrain.mean(0, keepdims=True), yTrain.std(0, keepdims=True) + 1e-8
    Xn = torch.tensor((xTrain - xm) / xs, dtype=torch.float32, device=dev)
    Yn = torch.tensor((yTrain - ym) / ys, dtype=torch.float32, device=dev)

    n = len(Xn); perm = torch.randperm(n); Xn, Yn = Xn[perm], Yn[perm]
    ntr = int(0.9 * n)
    Xtr, Ytr, Xva, Yva = Xn[:ntr], Yn[:ntr], Xn[ntr:], Yn[ntr:]

    net = FN_Net(2, 1, cfg.hid_size).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lossfn = nn.MSELoss()
    best, best_state = 1e9, None
    for j in range(cfg.distill_iters):
        opt.zero_grad()
        loss = lossfn(net(Xtr), Ytr); loss.backward(); opt.step()
        with torch.no_grad():
            v = lossfn(net(Xva), Yva).item()
        if v < best:
            best = v; best_state = {k: val.clone() for k, val in net.state_dict().items()}
        if j % (cfg.distill_iters // 5) == 0:
            print(f"  iter {j} train={loss.item():.6f} valid={v:.6f}")
    net.load_state_dict(best_state)

    torch.save({"state_dict": net.state_dict(),
                "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys,
                "diff_scale": cfg.diff_scale, "cfg": cfg.to_dict()},
               os.path.join(cfg.data_dir, "ckpt_flow.pt"))
    print(f"Saved ckpt_flow.pt  (best valid mse = {best:.6f})")

    if cfg.model == "bounded":
        print("Training exit head (absorbing-boundary indicator) ...")
        train_exit_head(cfg, dev)


if __name__ == "__main__":
    main()
