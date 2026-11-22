"""Two propagation-length bounds explain the measured L(N).

    L_amp  = a sqrt(N) lambda / dn_max          (generator-norm / Mandelstam-Tamm-type bound;
                                                 the paper's sqrt(N) argument)
    L_info = b pi N / dbeta(N)                  (Lloyd-Montangero counting: N^2 parameters from
                                                 ~2N transverse profiles x dbeta L / pi
                                                 resolvable beat frequencies)
    L_req  = (L_amp^p + L_info^p)^(1/p)

dbeta(N) = spread of propagation constants of the N used modes, taken from the
mode solver (fit-free; tends to k0 NA^2 / 2n for large N).  a and p are fitted
on the NA = 0.3 runs (dn_max = 2.5e-3, 5e-3, 1e-2); the NA = 0.2 / 0.45 runs are
predictions (NA enters only through the mode-solver dbeta).

Also produces the resolution figure (transverse PSF and z-pixel length).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import scipy.optimize as so

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results")
F = os.path.join(ROOT, "figures")
LAM, N_EFF = 1.55, 1.95
_DB = {}


def dbeta(N, na):
    """Spread of beta over the N used modes (from the mode solver), cached on disk."""
    path = os.path.join(R, "dbeta_cache.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    key = f"{N}_{na}"
    if key not in cache and N > 256:
        # step-index extrapolation: dbeta(N) = dbeta_inf * (N / (N + spare))^2,
        # dbeta_inf fitted to the solved points of this NA (-> k0 NA^2 / 2n)
        pts = [(int(k.split("_")[0]), v) for k, v in cache.items()
               if k.endswith(f"_{na}") and int(k.split("_")[0]) >= 32]
        if pts:
            dinf = np.mean([v / (n / (n + 4)) ** 2 for n, v in pts])
        else:   # no solved geometry for this NA: paraxial band edge k0 NA^2 / 2n
            dinf = (2 * math.pi / LAM) * na**2 / (2 * N_EFF)
        return float(dinf * (N / (N + 4)) ** 2)
    if key not in cache:
        from pwg.mmwg import MMWGConfig, MultimodeWaveguide
        try:
            wg = MultimodeWaveguide(MMWGConfig(n_modes=N, length_um=10.0, na=na))
        except ValueError:
            # NA beyond the tabulated shallow-etch geometry: paraxial step-index band
            return float((2 * math.pi / LAM) * na**2 / (2 * N_EFF) * (N / (N + 4)) ** 2)
        cache[key] = float(wg.beta.max() - wg.beta.min())
        json.dump(cache, open(path, "w"), indent=1)
    return cache[key]


def load(path, **dflt):
    p = os.path.join(R, path)
    if not os.path.exists(p):
        return []
    rows = [json.loads(l) for l in open(p)]
    for r in rows:
        for k, v in dflt.items():
            r.setdefault(k, v)
    return rows


def required(rows, thr=0.01):
    """Group by (N, dn_max, na, res) and interpolate L at 1 - Fn = thr (log-log)."""
    groups = {}
    for r in rows:
        k = (r["N"], r["dn_max"], r.get("na", 0.3), r.get("res_um", 1.0), r.get("res_z_um", 0.0))
        groups.setdefault(k, []).append(r)
    out = []
    for k, rr in groups.items():
        rr.sort(key=lambda r: r["L_um"])
        L = np.array([r["L_um"] for r in rr])
        e = np.array([max(1 - r["Fn"], 1e-7) for r in rr])
        Lr = None
        for i in range(len(L) - 1):
            if e[i] > thr >= e[i + 1]:
                t = (math.log(e[i]) - math.log(thr)) / (math.log(e[i]) - math.log(e[i + 1]))
                Lr = L[i] * (L[i + 1] / L[i]) ** t
                break
        if Lr is None and e[0] <= thr:
            Lr = float("nan")   # below the sampled range
        if Lr is not None and not math.isnan(Lr):
            out.append(dict(N=k[0], dn_max=k[1], na=k[2], res_um=k[3], res_z_um=k[4], L_req=Lr))
    return out


LOGF = True   # amplitude prefactor ~ sqrt(ln N^{3/2}) (peak-to-rms of the programmed pattern)


def model(N, dn, na, a, b):
    """Additive two-limit law: L = a f(N) sqrt(N) lambda/dn_max + b pi N / dbeta(N),
    f(N) = sqrt(1.5 ln N) if LOGF else 1."""
    f = math.sqrt(1.5 * math.log(N)) if LOGF else 1.0
    La = a * f * math.sqrt(N) * LAM / dn
    Li = b * math.pi * N / dbeta(N, na)
    return La + Li, La, Li


def main():
    base = load("unitary_scaling.jsonl", dn_max=5e-3, na=0.3, res_um=1.0, res_z_um=0.0)
    extra = (load("scaling_dn1e-2.jsonl") + load("scaling_dn2.5e-3.jsonl")
             + load("scaling_na0.2.jsonl") + load("scaling_na0.45.jsonl"))
    allr = required(base) + required(extra)
    # fit on every NA = 0.3 point (three dn_max values pin a vs b); NA != 0.3 are predictions
    # points pre-registered as predictions (results/prediction_*.json) stay out of the fit
    held = set()
    for fn in ("prediction_N64_dn.json", "prediction_na0.2.json"):
        pth = os.path.join(R, fn)
        if os.path.exists(pth):
            d = json.load(open(pth))
            for q in (d if isinstance(d, list) else [d]):
                held.add((q["N"], q["dn_max"], q["na"]))
    is_test = lambda r: abs(r["na"] - 0.3) >= 1e-9 or (r["N"], r["dn_max"], r["na"]) in held
    rb = [r for r in allr if not is_test(r)]
    rx = [r for r in allr if is_test(r)]
    print("required L (fit set):", [(r["N"], round(r["L_req"])) for r in rb])
    print("required L (test set):", [(r["N"], r["dn_max"], r["na"], round(r["L_req"])) for r in rx])

    def resid(q):
        a, b = q
        return [math.log(model(r["N"], r["dn_max"], r["na"], a, b)[0] / r["L_req"]) for r in rb]
    global LOGF
    fits = {}
    for lf in (False, True):
        LOGF = lf
        sol = so.least_squares(resid, x0=[0.5, 0.3], bounds=([0.01, 0.0], [5, 5]))
        fits[lf] = (sol.x, float(np.sqrt(np.mean(np.square(sol.fun)))))
        print(f"fit (log factor={lf}): a = {sol.x[0]:.3f}, b = {sol.x[1]:.3f}, rms log-resid = {fits[lf][1]:.3f}")
    LOGF = True
    a, p = fits[True][0]          # p holds b (kept name for the json/plots)
    rows = []
    for r in rb + rx:
        Lm, La, Li = model(r["N"], r["dn_max"], r["na"], a, p)
        rows.append(dict(r, L_model=Lm, L_amp=La, L_info=Li, fit=r in rb))
        print(f"  N={r['N']:4d} dn={r['dn_max']:.1e} NA={r['na']}  L_sim={r['L_req']:7.0f}  "
              f"L_model={Lm:7.0f}  (amp {La:6.0f}, info {Li:6.0f})  {'fit' if r in rb else 'PREDICTION'}")
    extrap = []
    for na in (0.3, 0.6):
        for N in (1000, 10000):
            Lm, La, Li = model(N, 5e-3, na, a, p)
            extrap.append(dict(N=N, na=na, dn_max=5e-3, L_mm=Lm / 1e3, amp_mm=La / 1e3, info_mm=Li / 1e3))
            print(f"  extrapolation N={N} NA={na}: L = {Lm/1e3:.1f} mm (amp {La/1e3:.1f}, info {Li/1e3:.1f})")
    json.dump(dict(extrapolation=extrap, a=a, b=p, log_factor=True, fits={str(k): [list(v[0]), v[1]] for k, v in fits.items()}, rows=rows), open(os.path.join(R, "scaling_explained.json"), "w"), indent=1)

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    # (a) control-rank collapse
    cr = os.path.join(R, "control_rank.json")
    if not os.path.exists(cr):
        cr_rows = [json.loads(l) for l in open(os.path.join(R, "control_rank.log")) if l.startswith("{")]
    else:
        cr_rows = json.load(open(cr))
    mk = {0.2: "s", 0.3: "o", 0.45: "^"}
    for (N, na) in sorted({(r["N"], r["na"]) for r in cr_rows}):
        rr = [r for r in cr_rows if r["N"] == N and r["na"] == na]
        ax[0].semilogx([r["L_um"] / r["L_info"] for r in rr], [r["rank"] / r["R"] for r in rr],
                       mk.get(na, "o") + "-", ms=4, lw=0.8,
                       color=plt.cm.viridis(math.log2(N) / 7), label=f"N={N}, NA={na}")
    ax[0].axvline(1, color="k", ls=":")
    ax[0].set_xlabel(r"$L / L_{\rm info}$,   $L_{\rm info} = \pi N/\Delta\beta$")
    ax[0].set_ylabel("controllable generator directions / $N^2$")
    ax[0].set_title("(a) first-order control rank (fit-free)")
    ax[0].legend(fontsize=6, ncol=2)
    # (b) required L vs N: data + model for each dn_max
    Ns = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
    for dn, col in [(2.5e-3, "C2"), (5e-3, "C0"), (1e-2, "C3")]:
        pts = [r for r in rows if r["dn_max"] == dn and r["na"] == 0.3]
        ax[1].loglog([r["N"] for r in pts], [r["L_req"] / 1e3 for r in pts], "o", color=col,
                     label=f"$\\Delta n_{{max}}$={dn:g}")
        nn = Ns[Ns <= 1024]
        try:
            ax[1].loglog(nn, [model(n, dn, 0.3, a, p)[0] / 1e3 for n in nn], "-", color=col, lw=1)
        except Exception as ex:  # dbeta for large N may be expensive
            print("skip model curve", ex)
    nn = Ns
    ax[1].loglog(nn, [model(n, 5e-3, 0.3, a, p)[1] / 1e3 for n in nn], "k--", lw=0.8,
                 label="amplitude term only (5e-3)")
    ax[1].set_xlabel("N"); ax[1].set_ylabel("required length [mm]  (1 - F = 1 %)")
    ax[1].set_title("(b) required length, NA 0.3: simulation (dots) vs model (lines)")
    ax[1].legend(fontsize=7)
    # (c) predicted vs simulated
    cols = {2.5e-3: "C2", 5e-3: "C0", 1e-2: "C3"}
    for r in rows:
        pred = not r["fit"]
        ax[2].loglog(r["L_req"] / 1e3, r["L_model"] / 1e3, "D" if pred else "o",
                     color="k" if pred else cols.get(r["dn_max"], "C7"), ms=6 if pred else 5,
                     mfc="none" if pred else None)
        if pred:
            ax[2].annotate(f"N={r['N']}, NA {r['na']}, $\\Delta n$={r['dn_max']:g}", (r["L_req"] / 1e3, r["L_model"] / 1e3),
                           fontsize=7, xytext=(6, -3), textcoords="offset points")
    ax[2].plot([], [], "o", color="C7", label="fit (NA 0.3, colour = $\\Delta n_{max}$)")
    ax[2].plot([], [], "D", color="k", mfc="none", label="held-out prediction")
    ax[2].legend(fontsize=7, loc="upper left")
    lim = [0.3, 10]
    ax[2].loglog(lim, lim, "k:", lw=0.8)
    ax[2].set_xlabel("simulated required L [mm]"); ax[2].set_ylabel("model L [mm]")
    ax[2].set_title(f"(c) model vs simulation (a={a:.2f}, b={p:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "scaling_explained.png"), dpi=140)
    resolution_figure()


def resolution_figure():
    rx = load("resolution_x.jsonl")
    rz = load("resolution_z.jsonl")
    base = [r for r in load("unitary_scaling.jsonl") if r["N"] == 32 and abs(r["c"] - 1.4) < 1e-6]
    if not (rx or rz):
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    b0 = base[0] if base else None
    xs = ([dict(res_um=1.0, Fn=b0["Fn"])] if b0 else []) + rx
    xs.sort(key=lambda r: r["res_um"])
    ax[0].semilogy([r["res_um"] for r in xs], [max(1 - r["Fn"], 1e-7) for r in xs], "o-")
    na = 0.3
    rc = LAM / (2 * na)     # period of the highest transverse coupling frequency 2 k0 NA
    ax[0].axvline(rc, color="k", ls=":", label=fr"$\lambda/2\mathrm{{NA}}$ = {rc:.1f} µm")
    ax[0].axhline(0.01, color="0.5", lw=0.6)
    ax[0].axvspan(6, 9, color="0.85", label="prototype (~6-9 µm)")
    ax[0].set_xlabel("transverse programming resolution (PSF FWHM) [µm]")
    ax[0].set_ylabel("1 - F_norm"); ax[0].legend(fontsize=7)
    ax[0].set_title("N=32, NA 0.3, L = 2.46 mm")
    zs = ([dict(res_z_um=1.0, Fn=b0["Fn"])] if b0 else []) + rz
    zs.sort(key=lambda r: r["res_z_um"])
    ax[1].loglog([r["res_z_um"] for r in zs], [max(1 - r["Fn"], 1e-7) for r in zs], "o-")
    db = dbeta(32, 0.3)
    ax[1].axvline(math.pi / db, color="k", ls=":", label=fr"$\pi/\Delta\beta$ = {math.pi/db:.0f} µm")
    ax[1].axvline(2 * math.pi / db, color="k", ls="--", label=fr"$2\pi/\Delta\beta$ = {2*math.pi/db:.0f} µm")
    ax[1].axhline(0.01, color="0.5", lw=0.6)
    ax[1].set_xlabel("programming pixel length along z [µm]")
    ax[1].set_ylabel("1 - F_norm"); ax[1].legend(fontsize=7)
    ax[1].set_title("same device, z-pixelation")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "resolution.png"), dpi=140)


if __name__ == "__main__":
    main()
