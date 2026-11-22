"""Sidewall roughness of the programmable multimode waveguide.

A. Validation: explicit-roughness BPM Monte Carlo (fine dz resolves the full
   forward spectrum) vs. the coupled-mode / Fermi-golden-rule rates.
B. Realistic roughness: per-mode radiation loss, intermodal scattering and
   backscatter vs N (golden rule), and the total roughness loss of a chip
   whose length follows the measured L ~ c sqrt(N) lambda/dn_max.
C. Device impact: a unitary designed on the smooth guide evaluated on rough
   chips, and re-optimised in situ on the rough chip (roughness is static, so
   its intermodal part is a fixed perturbation that programming can absorb;
   radiation loss cannot be undone).

    python experiments/roughness.py --part A B C
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
from pwg.roughness import golden_rule  # noqa: E402

mi.set_variant("cuda_ad_mono")
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results")


def out_of_basis_loss(wg, theta_val=-20.0):
    th = Float(np.full(wg.n_par, theta_val, np.float32))
    with dr.suspend_grad():
        re, im = wg.transfer(wg.index_map(th))
    T = wg.matrix(re.numpy(), im.numpy())
    return T


def part_a(N=16, sigma_nm=100.0, lc=0.3, L=1000.0, dz=0.1, seeds=8):
    """Monte Carlo BPM vs golden rule at exaggerated roughness (measurable over 300 um)."""
    print(f"[A] N={N} sigma={sigma_nm} nm Lc={lc} um L={L} um dz={dz} um, {seeds} chips", flush=True)
    res = {}
    smooth = MultimodeWaveguide(MMWGConfig(n_modes=N, length_um=L, dz=dz))
    T0 = out_of_basis_loss(smooth)
    P0 = (np.abs(T0) ** 2).sum(0)
    Ps, Xs = [], []
    for s in range(seeds):
        wg = MultimodeWaveguide(MMWGConfig(n_modes=N, length_um=L, dz=dz, rough_sigma_nm=sigma_nm,
                                           rough_lc_um=lc, rough_seed=s))
        T = out_of_basis_loss(wg)
        Ps.append((np.abs(T) ** 2).sum(0) / P0)                 # relative to smooth guide
        D = np.abs(np.diag(T)) ** 2 / P0
        Xs.append(1 - D / ((np.abs(T) ** 2).sum(0) / P0))        # fraction moved to other modes
        print(f"  chip {s}: mean out-of-basis loss {1 - Ps[-1].mean():.3e}", flush=True)
    bpm_loss = 1 - np.mean(Ps, 0)
    bpm_x = np.mean(Xs, 0)
    gr = golden_rule(smooth.guide, smooth.grid.n0, smooth.grid.x(), smooth.grid.dx, N,
                     sigma_nm * 1e-3, lc)
    gr_loss = 1 - np.exp(-gr["rad_db_cm"] * L * 1e-4 * math.log(10) / 10)
    gr_x = 1 - np.exp(-gr["intermodal_per_cm"] * L * 1e-4)
    for m in range(N):
        print(f"  mode {m:2d}: radiation BPM {bpm_loss[m]:.3e}  GR {gr_loss[m]:.3e} | "
              f"intermodal BPM {bpm_x[m]:.3e} GR {gr_x[m]:.3e}")
    res = dict(N=N, sigma_nm=sigma_nm, lc=lc, L=L, dz=dz, seeds=seeds,
               bpm_rad=bpm_loss.tolist(), gr_rad=gr_loss.tolist(),
               bpm_intermodal=bpm_x.tolist(), gr_intermodal=gr_x.tolist(),
               gr_back_db_cm=gr["back_db_cm"].tolist())
    with open(os.path.join(OUT, "roughness_A.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def part_b(Ns=(16, 32, 64, 128, 256), sigmas=(2.0, 5.0, 10.0), lcs=(0.05, 0.2, 1.0), c=1.0,
           dn_max=5e-3):
    rows = []
    for N in Ns:
        wg = MultimodeWaveguide(MMWGConfig(n_modes=N, length_um=100.0))
        x, dx = wg.grid.x(), wg.grid.dx
        for lc in lcs:
            for s in sigmas:
                gr = golden_rule(wg.guide, wg.grid.n0, x, dx, N, s * 1e-3, lc)
                L_cm = c * math.sqrt(N) * 1.55e-4 / dn_max
                rad = gr["rad_db_cm"]
                row = dict(N=N, sigma_nm=s, lc=lc, L_cm=L_cm,
                           rad_mean_db_cm=float(rad.mean()), rad_max_db_cm=float(rad.max()),
                           back_mean_db_cm=float(gr["back_db_cm"].mean()),
                           intermodal_mean_per_cm=float(gr["intermodal_per_cm"].mean()),
                           chip_rad_mean_db=float(rad.mean() * L_cm),
                           chip_rad_max_db=float(rad.max() * L_cm),
                           chip_intermodal_mean=float(gr["intermodal_per_cm"].mean() * L_cm))
                rows.append(row)
                print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, "roughness_B.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows


def part_c(N=32, c=1.0, sigmas=(5.0, 20.0, 50.0), lc=0.2, iters=300, retrain=150):
    sys.path.insert(0, os.path.dirname(__file__))
    from unitary_scaling import design, phase_invariant_loss
    L = c * math.sqrt(N) * 1.55 / 5e-3
    res0, wg, theta = design(N, L, iters, 0.1, verbose=False)
    U = haar_unitary(N, np.random.default_rng(0))
    tgt, Tt = wg.target_fields(U)
    out = dict(N=N, L_um=L, smooth=res0, rough=[])
    print(f"[C] smooth design: Fn={res0['Fn']:.4f} F={res0['F']:.4f}", flush=True)
    for s in sigmas:
        rw = MultimodeWaveguide(MMWGConfig(n_modes=N, length_um=L, rough_sigma_nm=s, rough_lc_um=lc,
                                           rough_seed=1))
        with dr.suspend_grad():
            re, im = rw.transfer(rw.index_map(Float(theta)))
        F, Fn, err = fidelity(rw.matrix(re.numpy(), im.numpy()), Tt)
        # in-situ re-optimisation on this specific rough chip (warm start)
        opt = mi.ad.Adam(lr=0.05)
        opt["theta"] = Float(theta)
        tr, ti = Float(tgt.real.ravel()), Float(tgt.imag.ravel())
        tp = float(np.sum(np.abs(tgt) ** 2))
        for it in range(retrain):
            re, im = rw.transfer(rw.index_map(opt["theta"]))
            loss = phase_invariant_loss(re, im, tr, ti, tp, rw.grid.dx, N)
            dr.backward(loss)
            opt.step()
        with dr.suspend_grad():
            re, im = rw.transfer(rw.index_map(opt["theta"]))
        F2, Fn2, err2 = fidelity(rw.matrix(re.numpy(), im.numpy()), Tt)
        row = dict(sigma_nm=s, lc=lc, F=F, Fn=Fn, err=err, F_retrained=F2, Fn_retrained=Fn2,
                   err_retrained=err2)
        out["rough"].append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, "roughness_C.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    return out


class _LinearTable:
    """Idealised local-index table n_eff(d) = n1 - (dn/d0) d (for contrast comparisons)."""

    def __init__(self, n1, dn, d0=0.1):
        self.n1, self.dn, self.d0 = n1, dn, d0

    def n_re(self, d):
        return self.n1 - self.dn * np.asarray(d) / self.d0

    def n_im(self, d):
        return 0.0 * np.asarray(d)


def part_d(sigma_nm=5.0, lc=0.2, dx=0.02):
    """Same physics, same golden rule: roughness loss of a *single-mode* guide vs lateral
    contrast.  Small-contrast multimode PMW walls vs the high-contrast single-mode
    wires that MZI meshes / crossbars need for tight bends and compact couplers."""
    from pwg.lateral import EtchedGuide
    n1 = 1.95
    rows = []
    for dn in [0.023, 0.1, 0.3, 0.6]:
        # single-mode width: V = k0 w NA / 2 ~ 0.9 * pi/2
        na = math.sqrt(n1**2 - (n1 - dn) ** 2)
        w = 0.9 * 1.55 / (2 * na)
        g = EtchedGuide(width=w, etch_depth=0.1, sidewall_deg=89.9, table=_LinearTable(n1, dn))
        win = max(40.0, 30 * w)
        nx = int(2 ** math.ceil(math.log2(win / dx)))
        x = (np.arange(nx) - nx / 2) * dx
        gr = golden_rule(g, n1, x, dx, 1, sigma_nm * 1e-3, lc)
        rows.append(dict(dn_lateral=dn, na=na, width_um=w, rad_db_cm=float(gr["rad_db_cm"][0]),
                         back_db_cm=float(gr["back_db_cm"][0])))
        print(json.dumps(rows[-1]), flush=True)
    with open(os.path.join(OUT, "roughness_D.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["A", "B", "C"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if "A" in a.part:
        part_a()
    if "B" in a.part:
        part_b()
    if "C" in a.part:
        part_c()
    if "D" in a.part:
        part_d()
