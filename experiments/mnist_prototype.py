"""Reproduce the prototype MNIST experiment (paper Fig. 4): 7x7 MNIST, 49 inputs,
10 camera bins, ~10^4 programmable macropixels, trained end-to-end through the
full physical model (Mitsuba projection -> photoconductor/Pockels -> BPM).

Then evaluate the trained DMD pattern on a perturbed "hardware" instance with
shot noise, camera read noise, DMD quantisation and a different focus.

    python experiments/mnist_prototype.py --iters 300
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import drjit as dr  # noqa: E402
import mitsuba as mi  # noqa: E402
from drjit.cuda.ad import Float, UInt32  # noqa: E402

from pwg.device import PrototypeConfig, PrototypeDevice  # noqa: E402
from pwg.noise import Detector  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_mnist(n_train=20000, n_test=5000):
    d = np.load(os.path.join(ROOT, "data", "mnist.npz"), allow_pickle=False)

    def prep(x):
        x = x.astype(np.float32) / 255.0
        x = x.reshape(-1, 14, 2, 14, 2).mean((2, 4))           # 28 -> 14
        x = x[:, :14, :14].reshape(-1, 7, 2, 7, 2).mean((2, 4))  # 14 -> 7 (paper)
        x = x.reshape(-1, 49)
        return x / np.linalg.norm(x, axis=1, keepdims=True)    # unit input power
    return (prep(d["x_train"][:n_train]), d["y_train"][:n_train].astype(np.int64),
            prep(d["x_test"][:n_test]), d["y_test"][:n_test].astype(np.int64))


def ce_loss(P, y, n_out, scale):
    """Softmax cross-entropy on normalised bin powers; returns (loss, logits)."""
    S = len(y)
    tot = dr.block_sum(P, n_out)
    idx = dr.arange(UInt32, S * n_out)
    tot_b = dr.gather(Float, tot, idx // n_out)
    logits = scale * P / (tot_b + 1e-12)
    lse = dr.log(dr.block_sum(dr.exp(logits), n_out))
    picked = dr.gather(Float, logits, UInt32(np.arange(S) * n_out + y))
    return dr.mean(lse - picked), logits


def accuracy(P_np, y):
    return float((P_np.reshape(len(y), -1).argmax(1) == y).mean())


def upsample(theta, cfg, mz):
    """Macropixels [rows, cols/mz] -> DMD duty cycles [rows, cols] in [0,1]."""
    R, C = cfg.dmd_rows, cfg.dmd_cols
    idx = dr.arange(UInt32, R * C)
    r, c = idx // C, idx % C
    return dr.gather(Float, 1.0 / (1.0 + dr.exp(-theta)), r * (C // mz) + c // mz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--macro_z", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "mnist_prototype"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    Xtr, ytr, Xte, yte = load_mnist()
    cfg = PrototypeConfig()
    dev = PrototypeDevice(cfg)
    mz = args.macro_z
    n_par = cfg.dmd_rows * (cfg.dmd_cols // mz)
    print(f"programmable macropixels: {n_par}")
    rng = np.random.default_rng(0)

    opt = mi.ad.Adam(lr=args.lr)
    opt["theta"] = Float(rng.normal(0, 0.5, n_par).astype(np.float32))
    opt["scale"] = Float(10.0)
    hist = []
    t0 = time.time()
    for it in range(args.iters):
        sel = rng.choice(len(ytr), args.batch, replace=False)
        pat = upsample(opt["theta"], cfg, mz)
        dn = dev.index_map(pat, seed=it)
        re, im = dev.transfer(dn)
        Ur, Ui = dev.output_matrix(re, im)
        P = dev.bin_powers(Ur, Ui, Xtr[sel])
        loss, _ = ce_loss(P, ytr[sel], cfg.n_out, opt["scale"])
        dr.backward(loss)
        opt.step()
        if it % 10 == 0 or it == args.iters - 1:
            acc = accuracy(P.numpy(), ytr[sel])
            tr = float(P.numpy().reshape(len(sel), -1).sum(1).mean())
            hist.append(dict(it=it, loss=float(loss.numpy().ravel()[0]), acc=acc, bin_power=tr,
                             t=time.time() - t0))
            print(f"it {it:4d} loss {hist[-1]['loss']:.3f} batch-acc {acc:.3f} "
                  f"power-in-bins {tr:.3f} ({time.time()-t0:.0f}s)", flush=True)

    theta = opt["theta"]
    pat = upsample(theta, cfg, mz)
    dr.eval(pat)
    np.save(os.path.join(args.out, "dmd_pattern.npy"), pat.numpy().reshape(cfg.dmd_rows, cfg.dmd_cols))

    # ---------------------------------------------------------- evaluation
    def evaluate(device, pattern, detector=None, spp=64, seed=12345):
        with dr.suspend_grad():
            dn = device.index_map(pattern, spp=spp, seed=seed)
            re, im = device.transfer(dn)
            Ur, Ui = device.output_matrix(re, im)
            P = device.bin_powers(Ur, Ui, Xte).numpy().reshape(len(yte), -1)
        if detector is not None:
            P = detector.measure(P, rng)
        return float((P.argmax(1) == yte).mean()), dn

    res = {}
    res["test_acc_ideal"], dn = evaluate(dev, pat)
    np.save(os.path.join(args.out, "dn.npy"), dn.numpy().reshape(dev.grid.nz, dev.grid.nx)[:, ::4])
    # shot-noise + read-noise limited camera at several optical energies per inference
    for E_fJ in [1, 10, 100, 1000]:
        det = Detector(energy_per_inference_J=E_fJ * 1e-15)
        res[f"test_acc_ideal_shot_{E_fJ}fJ"], _ = evaluate(dev, pat, det)
    # perturbed hardware: refocus error, 8-bit DMD PWM, 5% thicker photoconductor,
    # different voltage calibration (2%), Monte Carlo seed.
    from pwg.stack import Stack
    hw_cfg = PrototypeConfig(defocus_um=cfg.defocus_um + 15.0, voltage=cfg.voltage * 0.98)
    hw = PrototypeDevice(hw_cfg, stack=Stack.wright2024(t_srn=4.2), verbose=True,
                         sigma=(dev.resp.sigma_d, dev.resp.sigma_ph))
    pat_q = Float(np.round(pat.numpy() * 255) / 255)
    res["test_acc_hardware"], _ = evaluate(hw, pat_q)
    res["test_acc_hardware_shot_100fJ"], _ = evaluate(hw, pat_q, Detector(energy_per_inference_J=100e-15))

    # digital baseline: linear layer + softmax on |x|^2-free features (paper: ~90%)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    res["digital_linear_acc"] = float(lr.score(Xte, yte))
    res["hist"] = hist
    res["n_params"] = n_par
    print(json.dumps({k: v for k, v in res.items() if k != "hist"}, indent=1))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
