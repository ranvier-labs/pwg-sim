"""Explain the measured L(N) with the linear-controllability model (pwg.controllability).

For every simulated (N, c, dn_max, res) point predict 1 - F_norm with the
budgeted minimum-error programming, using the *same* Haar targets as the
inverse-design runs.  One constant kappa (achievable rms(dn)/dn_max) is fitted
on the dn_max = 5e-3 sweep; everything else (other dn_max, resolutions, the
required-length law) is then a prediction.

    python experiments/explain_scaling.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pwg.controllability import Budgeted  # noqa: E402
from pwg.mmwg import MMWGConfig, MultimodeWaveguide, haar_unitary  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results")
_CACHE = {}


def model(N, res_um=1.0, res_z_um=0.0, na=0.3):
    key = (N, res_um, res_z_um, na)
    if key not in _CACHE:
        wg = MultimodeWaveguide(MMWGConfig(n_modes=N, length_um=100.0, res_um=res_um,
                                           res_z_um=res_z_um, na=na))
        _CACHE[key] = Budgeted(wg)
    return _CACHE[key]


def target(N):
    return haar_unitary(N, np.random.default_rng(0))     # same draw as design(seed=0)


def predict_points(points, kappas):
    """points: list of dicts with N, L_um, dn_max, res_um, res_z_um -> 1-Fn per kappa."""
    out = []
    for p in points:
        t = time.time()
        b = model(p["N"], p.get("res_um", 1.0), p.get("res_z_um", 0.0), p.get("na", 0.3))
        e = b.eig(p["L_um"], (p.get("res_z_um") or 1.0))
        U = target(p["N"])
        out.append([b.infidelity(p["L_um"], U, k * p["dn_max"], eigs=e) for k in kappas])
        print(f"  N={p['N']} L={p['L_um']:.0f} dn={p['dn_max']:g} rx={p.get('res_um',1)} "
              f"rz={p.get('res_z_um',0)}: sim {1-p['Fn']:.2e}  model(kappa) "
              + " ".join(f"{v:.1e}" for v in out[-1]) + f"  ({time.time()-t:.0f}s)", flush=True)
    return np.array(out)


def load(path, **default):
    if not os.path.exists(path):
        return []
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        for k, v in default.items():
            r.setdefault(k, v)
    return rows


def main():
    kappas = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
    base = load(os.path.join(R, "unitary_scaling.jsonl"), dn_max=5e-3, res_um=1.0, res_z_um=0.0)
    base = [r for r in base if r["N"] <= int(os.environ.get("NMAX", 128))]
    print("[fit] dn_max = 5e-3 sweep")
    pred = predict_points(base, kappas)
    sim = np.array([max(1 - r["Fn"], 1e-6) for r in base])
    # fit kappa on log-infidelity where the simulation is above its floor
    err = [np.mean((np.log(np.clip(pred[:, i], 1e-6, 1)) - np.log(sim)) ** 2) for i in range(len(kappas))]
    ib = int(np.argmin(err))
    kap = float(kappas[ib])
    print("kappa fit:", dict(zip(kappas.tolist(), np.round(err, 3))), "->", kap)
    res = dict(kappa=kap, kappas=kappas.tolist(), fit_err=err,
               base=[dict(N=r["N"], c=r["c"], L_um=r["L_um"], sim=1 - r["Fn"], model=pred[i, ib])
                     for i, r in enumerate(base)])
    # blind predictions for the other experiments
    for name, path, dflt in [("dn1e-2", "scaling_dn1e-2.jsonl", {}),
                             ("dn2.5e-3", "scaling_dn2.5e-3.jsonl", {}),
                             ("res_x", "resolution_x.jsonl", {}),
                             ("res_z", "resolution_z.jsonl", {})]:
        rows = load(os.path.join(R, path), res_um=1.0, res_z_um=0.0)
        if not rows:
            continue
        print(f"[predict] {name}")
        pr = predict_points(rows, [kap])
        res[name] = [dict(N=r["N"], c=r["c"], L_um=r["L_um"], dn_max=r["dn_max"], res_um=r["res_um"],
                          res_z_um=r["res_z_um"], sim=1 - r["Fn"], model=float(pr[i, 0]))
                     for i, r in enumerate(rows)]
    with open(os.path.join(R, "explain_scaling.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
