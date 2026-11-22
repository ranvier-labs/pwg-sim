"""Length scaling of a programmable multimode waveguide implementing Haar-random
N x N unitaries (Onodera et al., Fig. 5 / Supp. Sec. 8: L ~ sqrt(N) vs. L ~ N).

For each N we inverse-design dn(x, z) in [0, dn_max] for a sweep of lengths
L = c * sqrt(N) * lambda0 / dn_max and record the achieved fidelity.

    python experiments/unitary_scaling.py --N 8 16 32 64 128 --c 0.25 0.5 1 2
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
from drjit.cuda.ad import Float  # noqa: E402

from pwg.mmwg import MMWGConfig, MultimodeWaveguide, fidelity, haar_unitary  # noqa: E402

mi.set_variant("cuda_ad_mono")
ROOT = os.path.join(os.path.dirname(__file__), "..")


def phase_invariant_loss(re, im, tr, ti, tgt_pow, dx, N):
    """min over a global phase of ||out - e^{i phi} tgt||^2 / N.

    The global phase is physically irrelevant and would otherwise waste index budget."""
    ov_r = dr.sum(tr * re + ti * im)
    ov_i = dr.sum(tr * im - ti * re)
    return (dr.sum(re * re + im * im) + tgt_pow - 2 * dr.sqrt(ov_r * ov_r + ov_i * ov_i)) * (dx / N)


def design(N, L, iters, lr, seed=0, dn_max=5e-3, log_every=50, verbose=True, **cfg):
    rng = np.random.default_rng(seed)
    wg = MultimodeWaveguide(MMWGConfig(n_modes=N, length_um=L, dn_max=dn_max, **cfg))
    U = haar_unitary(N, rng)
    tgt, Tt = wg.target_fields(U)
    tr, ti = Float(tgt.real.ravel()), Float(tgt.imag.ravel())
    tgt_pow = float(np.sum(np.abs(tgt) ** 2))
    opt = mi.ad.Adam(lr=lr)
    opt["theta"] = Float(rng.normal(0, 0.1, wg.n_par).astype(np.float32))
    t0 = time.time()
    best = (-1, None)
    for it in range(iters):
        dn = wg.index_map(opt["theta"])
        re, im = wg.transfer(dn)
        loss = phase_invariant_loss(re, im, tr, ti, tgt_pow, wg.grid.dx, N)
        dr.backward(loss)
        if it % log_every == 0 or it == iters - 1:
            T = wg.matrix(re.numpy(), im.numpy())
            F, Fn, err = fidelity(T, Tt)
            if F > best[0]:
                best = (F, dict(F=F, Fn=Fn, err=err, loss=float(loss.numpy().ravel()[0]), it=it))
            if verbose:
                print(f"  N={N} L={L:.0f}um it {it:4d} loss {float(loss.numpy().ravel()[0]):.4f} "
                      f"F={F:.4f} Fn={Fn:.4f} err={err:.3f} ({time.time()-t0:.0f}s)", flush=True)
        opt.step()
    out = dict(N=N, L_um=L, W_um=wg.W, nx=wg.grid.nx, nz=wg.grid.nz, n_par=wg.n_par,
               confinement_min=float(wg.confinement.min()), time_s=time.time() - t0, **best[1])
    return out, wg, opt["theta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--c", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--dn_max", type=float, default=5e-3)
    ap.add_argument("--res_um", type=float, nargs="+", default=[1.0])
    ap.add_argument("--res_z_um", type=float, nargs="+", default=[0.0])
    ap.add_argument("--na", type=float, default=0.3)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "unitary_scaling.jsonl"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for N in args.N:
      for rx in args.res_um:
       for rz in args.res_z_um:
        for c in args.c:
            L = c * math.sqrt(N) * 1.55 / args.dn_max
            res, _, _ = design(N, L, args.iters, args.lr, dn_max=args.dn_max,
                               res_um=rx, res_z_um=rz, na=args.na)
            res.update(c=c, dn_max=args.dn_max, res_um=rx, res_z_um=rz, na=args.na)
            print(json.dumps(res), flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
