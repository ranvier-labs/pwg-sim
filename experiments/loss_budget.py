"""Loss-limited vector size: programmable multimode waveguide vs MZI mesh vs crossbar.

1. Stack design trade-off (physics from pwg.stack / pwg.electro): thicker top
   SiO2 cladding suppresses leakage of the TM0 mode into the high-index SRN
   photoconductor (and the metal electrode), but also reduces the voltage that
   drops across the LN core, i.e. dn_max.  For each top-cladding thickness we
   compute loss [dB/cm], dn_prog,max at the breakdown-limited voltage, the
   length needed for N = 1000 (L = c sqrt(N) lambda/dn_max, c from the BPM
   scaling study) and the resulting chip loss.

2. Shot-noise-limited optical energy per MAC vs N.  To read each of the N
   outputs with SNR s (n_ph = s^2 detected photons per output), the source
   must supply E = N n_ph h nu / eta(N); per MAC (N^2 of them):
        E_MAC = n_ph h nu / (N eta(N)).
   This is the "1/N regime" -- it persists only while eta(N) decays slower
   than 1/N.  Architectures:
        MZI mesh (Clements)   eta = 10^(-N * IL_mzi / 10)
        crossbar              eta = 10^(-N * IL_cell / 10) / N   (1/N fan-out)
        PMW (this work)       eta = 10^(-(alpha * c sqrt(N) lambda/dn + IL_io)/10)
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pwg.electro import LocalResponse  # noqa: E402
from pwg.stack import Stack, core_mode  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
H, C = 6.62607015e-34, 299792458.0


def scaling_constant(default=1.0):
    """c in L = c sqrt(N) lambda/dn_max at which F_norm >= 0.99, from the BPM study."""
    p = os.path.join(ROOT, "results", "unitary_scaling.jsonl")
    if not os.path.exists(p):
        return default, None
    rows = [json.loads(l) for l in open(p)]
    cs = {}
    for r in rows:
        cs.setdefault(r["N"], []).append((r["c"], r["Fn"]))
    need = {}
    for N, v in cs.items():
        v.sort()
        ok = [c for c, f in v if f >= 0.99]
        if ok:
            need[N] = min(ok)
    if not need:
        return default, None
    return max(need.values()), need


def length_law(dn_max=5e-3):
    """Measured L(N) [um] at dn_max = 5e-3 (1 - F_norm = 1%), power-law fit a N^p.

    Uses results/required_c.json written by make_figures.py.  Rescaled to other
    dn_max by L ~ 1/dn_max (exact for the coupled-mode generator argument)."""
    p = os.path.join(ROOT, "results", "required_c.json")
    if not os.path.exists(p):
        return None
    need = {int(k): v for k, v in json.load(open(p)).items()}
    n = np.array(sorted(need), float)
    L = np.array([need[int(k)] * math.sqrt(k) * 1.55 / 5e-3 for k in n])
    pf = np.polyfit(np.log(n), np.log(L), 1)
    return dict(p=float(pf[0]), a=float(math.exp(pf[1])), N=n.tolist(), L_um=L.tolist())


def stack_tradeoff(V_max_field=50.0):
    rows = []
    for t_top in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        st = Stack.wright2024(t_top=t_top)
        m = core_mode(st)
        resp = LocalResponse(st)
        resp.calibrate()  # same photoconductor; calibration at the paper's point (t_top=1)
        if t_top != 1.0:
            ref = LocalResponse(Stack.wright2024()).calibrate()
            resp.sigma_d, resp.sigma_ph = ref.sigma_d, ref.sigma_ph
        # largest voltage keeping the LN field below breakdown margin (fully bright)
        e_per_volt = abs(resp.ln_field(1e6, voltage=1.0)) / 1e6  # V/um per V
        V = V_max_field / e_per_volt
        resp.voltage = V
        dn = float(resp.dn_eff(57.0) - resp.dn_eff(0.0))
        rows.append(dict(t_top=t_top, n_eff=m.n_eff.real, loss_db_cm=m.loss_db_per_cm,
                         V=V, dn_prog=dn, Gamma=resp.Gamma))
        print(rows[-1], flush=True)
    return rows


def energy_per_mac(N, eta, snr_bits=6, wavelength=1.55e-6, qe=0.8):
    s = 2 ** snr_bits
    n_ph = s**2 / qe                       # shot-noise-limited: SNR = sqrt(n)
    return n_ph * H * C / wavelength / (N * eta)


def main():
    out = os.path.join(ROOT, "results")
    figd = os.path.join(ROOT, "figures")
    os.makedirs(figd, exist_ok=True)
    c_fit, need = scaling_constant()
    law = length_law()
    print("scaling constant c =", c_fit, need, "measured law", law)

    se = os.path.join(ROOT, "results", "scaling_explained.json")
    if os.path.exists(se):
        sys.path.insert(0, os.path.dirname(__file__))
        import scaling_explained as SE
        par = json.load(open(se))

        def L_cm(Nv, dn, na=0.3):
            Nv = np.atleast_1d(np.asarray(Nv, float))
            out = np.array([SE.model(int(n), dn, na, par["a"], par["b"])[0] * 1e-4 for n in Nv])
            return out if out.size > 1 else float(out[0])
        law = dict(model="a sqrt(1.5 lnN) sqrt(N) lambda/dn + b pi N/dbeta", a=par["a"], b=par["b"])
    else:
        L_cm = None

    def _L_cm_powerlaw(Nv, dn):
        if law is None:
            return c_fit * np.sqrt(Nv) * 1.55e-4 / dn
        return law["a"] * np.asarray(Nv, float) ** law["p"] * 1e-4 * (5e-3 / dn)
    if L_cm is None:
        L_cm = _L_cm_powerlaw
    rows = stack_tradeoff()

    lam = 1.55e-6
    N = np.logspace(1, 4.5, 200)
    curves = {}
    # MZI mesh: 0.1 - 0.3 dB per MZI column (state of the art incl. phase shifters)
    for il in [0.1, 0.3]:
        curves[f"MZI mesh ({il} dB/column)"] = 10 ** (-N * il / 10)
    for il in [0.02]:
        curves[f"crossbar ({il} dB/cell, 1/N fan-out)"] = 10 ** (-N * il / 10) / N
    # PMW with the prototype stack (alpha from mode solver) and an optimised stack
    best = min(rows, key=lambda r: r["loss_db_cm"] * math.sqrt(1000) * c_fit * 1.55e-4 / r["dn_prog"])
    proto = next(r for r in rows if r["t_top"] == 1.0)
    pmw = {}
    for lab, alpha, dn, na in [("PMW, prototype stack, NA 0.3", proto["loss_db_cm"], 5e-3, 0.3),
                               ("PMW, prototype stack, NA 0.6", proto["loss_db_cm"], 5e-3, 0.6),
                               ("PMW, prototype stack, dn=1e-3, NA 0.3", proto["loss_db_cm"], 1e-3, 0.3),
                               ("PMW, 0.05 dB/cm, NA 0.3", 0.05, 5e-3, 0.3)]:
        Lc = L_cm(N, dn, na) if "model" in (law or {}) else L_cm(N, dn)
        loss = alpha * Lc + 1.0       # + 1 dB total I/O coupling
        curves[lab] = 10 ** (-loss / 10)
        pmw[lab] = dict(alpha=alpha, dn=dn,
                        L_1k_cm=float(Lc[np.argmin(abs(N - 1000))]), loss_1k_db=float(alpha * Lc[np.argmin(abs(N - 1000))]),
                        L_10k_cm=float(Lc[np.argmin(abs(N - 1e4))]), loss_10k_db=float(alpha * Lc[np.argmin(abs(N - 1e4))]),
                        L_1k_cm_sqrtlaw=float(math.sqrt(1000) * lam / dn * 100))
    summary = dict(c=c_fit, c_per_N=need, length_law=law, stack=rows, pmw=pmw)
    for lab, eta in curves.items():
        e = energy_per_mac(N, eta)
        i = int(np.argmin(e))
        summary.setdefault("optimum", {})[lab] = dict(N_opt=float(N[i]), E_mac_min_J=float(e[i]))
    with open(os.path.join(out, "loss_budget.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["optimum"], indent=1))
    print(json.dumps(pmw, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    t = [r["t_top"] for r in rows]
    ax[0].semilogy(t, [r["loss_db_cm"] for r in rows], "o-", label="TM0 loss [dB/cm]")
    ax[0].semilogy(t, [r["dn_prog"] * 1e3 for r in rows], "s-", label="dn_prog,max x1e3 (at 50 V/um)")
    ax[0].set_xlabel("top SiO2 cladding thickness [um]")
    ax[0].set_title("Stack trade-off (mode solver + EO circuit)")
    ax[0].legend(); ax[0].grid(alpha=.3)
    for lab, eta in curves.items():
        ax[1].loglog(N, energy_per_mac(N, eta), label=lab)
    ax[1].loglog(N, energy_per_mac(N, np.ones_like(N)), "k:", label="lossless (1/N)")
    ax[1].set_ylim(1e-21, 1e-14)
    ax[1].set_xlabel("vector size N"); ax[1].set_ylabel("optical energy per MAC [J] (6-bit, shot-noise)")
    ttl = "two-limit length law" if law and "model" in law else (f"L ~ N^{law['p']:.2f}" if law else f"c = {c_fit:g}")
    ax[1].set_title(f"Loss-limited 1/N regime, {ttl}")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(figd, "loss_budget.png"), dpi=130)


if __name__ == "__main__":
    main()
