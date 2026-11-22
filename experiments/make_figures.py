"""Generate figures from results/ into figures/."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results")
F = os.path.join(ROOT, "figures")
os.makedirs(F, exist_ok=True)


def mnist():
    d = os.path.join(R, "mnist_prototype")
    if not os.path.exists(os.path.join(d, "results.json")):
        return
    res = json.load(open(os.path.join(d, "results.json")))
    pat = np.load(os.path.join(d, "dmd_pattern.npy"))
    dn = np.load(os.path.join(d, "dn.npy"))
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(3, 1, 1)
    ax.imshow(pat, aspect="auto", cmap="gray", extent=[0, pat.shape[1] * 9.133e-3, -0.507, 0.507])
    ax.set_title("trained DMD duty cycles (111 x 1000 mirrors, 9.13 um on chip)")
    ax.set_ylabel("x [mm]")
    ax = fig.add_subplot(3, 1, 2)
    nz, nx = dn.shape
    im = ax.imshow(dn.T * 1e3, aspect="auto", cmap="magma",
                   extent=[0, nz * 4.567e-3, -nx * 3e-3 / 2, nx * 3e-3 / 2])
    ax.set_ylim(-0.55, 0.55)
    plt.colorbar(im, ax=ax, label="dn_eff x1e3")
    ax.set_title("resulting programmed effective index (Mitsuba projection -> photoconductor -> Pockels)")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("x [mm]")
    ax = fig.add_subplot(3, 2, 5)
    h = res["hist"]
    ax.plot([e["it"] for e in h], [e["acc"] for e in h], label="batch accuracy")
    ax.plot([e["it"] for e in h], [e["bin_power"] for e in h], label="power in output bins")
    ax.axhline(res["test_acc_ideal"], color="k", ls=":", label=f"test {res['test_acc_ideal']:.3f}")
    ax.legend(); ax.set_xlabel("iteration")
    ax = fig.add_subplot(3, 2, 6)
    labels = ["ideal", "1 fJ", "10 fJ", "100 fJ", "hardware\n(perturbed)", "hw+100 fJ", "digital\nlinear"]
    keys = ["test_acc_ideal", "test_acc_ideal_shot_1fJ", "test_acc_ideal_shot_10fJ",
            "test_acc_ideal_shot_100fJ", "test_acc_hardware", "test_acc_hardware_shot_100fJ",
            "digital_linear_acc"]
    v = [res[k] for k in keys]
    ax.bar(range(len(v)), v, color=["C0"] * 4 + ["C1"] * 2 + ["C2"])
    ax.set_xticks(range(len(v))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0.8, 0.92); ax.axhline(0.862, color="r", ls="--", lw=1, label="paper (expt.)")
    ax.legend(fontsize=7); ax.set_title("7x7 MNIST test accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "mnist_prototype.png"), dpi=120)


def scaling():
    p = os.path.join(R, "unitary_scaling.jsonl")
    if not os.path.exists(p):
        return
    rows = [json.loads(l) for l in open(p)]
    Ns = sorted({r["N"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    need = {}
    for N in Ns:
        rr = sorted([r for r in rows if r["N"] == N], key=lambda r: r["c"])
        ax[0].plot([r["c"] for r in rr], [1 - r["Fn"] for r in rr], "o-", label=f"N={N}")
        # interpolate c at which 1-Fn = 0.01 (log-linear)
        cs = np.array([r["c"] for r in rr]); e = np.array([max(1 - r["Fn"], 1e-6) for r in rr])
        for i in range(len(cs) - 1):
            if e[i] > 0.01 >= e[i + 1]:
                t = (math.log(e[i]) - math.log(0.01)) / (math.log(e[i]) - math.log(e[i + 1]))
                need[N] = cs[i] * (cs[i + 1] / cs[i]) ** t
                break
        else:
            if e[0] <= 0.01:
                need[N] = cs[0]
    pl = os.path.join(R, "unitary_scaling_long.jsonl")
    if os.path.exists(pl):
        for i, r in enumerate(json.loads(l) for l in open(pl)):
            ax[0].plot(r["c"], 1 - r["Fn"], "x", color=f"C{Ns.index(r['N'])}", ms=10, mew=2,
                       label="1000 iterations" if i == 0 else None)
    ax[0].set_yscale("log"); ax[0].axhline(0.01, color="k", ls=":")
    ax[0].set_xlabel("c   (L = c sqrt(N) lambda0 / dn_max)"); ax[0].set_ylabel("1 - normalised fidelity")
    ax[0].legend(); ax[0].set_title("Haar-random unitaries, dn_max = 5e-3")
    if need:
        n = np.array(sorted(need)); L = np.array([need[k] * math.sqrt(k) * 1.55 / 5e-3 for k in n])
        ax[1].loglog(n, L / 1e3, "o", ms=8, label="required L (1-Fn = 1%)")
        if len(n) >= 2:
            p_ = np.polyfit(np.log(n), np.log(L), 1)
            nn = np.logspace(np.log10(n.min()), np.log10(1024), 50)
            ax[1].loglog(nn, np.exp(np.polyval(p_, np.log(nn))) / 1e3, "-",
                         label=f"fit: L ~ N^{p_[0]:.2f}")
            ref = L[0] / 1e3
            ax[1].loglog(nn, ref * (nn / n[0]), "k:", label="L ~ N (mesh/crossbar)")
            ax[1].loglog(nn, ref * np.sqrt(nn / n[0]), "k--", label="L ~ sqrt(N)")
        ax[1].set_xlabel("N"); ax[1].set_ylabel("length [mm]"); ax[1].legend()
        ax[1].set_title("length needed vs vector size")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "unitary_scaling.png"), dpi=130)
    json.dump({str(k): v for k, v in need.items()}, open(os.path.join(R, "required_c.json"), "w"))


def roughness():
    pa = os.path.join(R, "roughness_A.json")
    pb = os.path.join(R, "roughness_B.json")
    pc = os.path.join(R, "roughness_C.json")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if os.path.exists(pa):
        a = json.load(open(pa))
        m = np.arange(a["N"])
        ax[0].semilogy(m, a["bpm_rad"], "o", label="BPM Monte Carlo (dz=0.1 um)")
        ax[0].semilogy(m, a["gr_rad"], "-", label="coupled-mode golden rule")
        ax[0].semilogy(m, np.maximum(a["bpm_intermodal"], 1e-6), "s", label="BPM intermodal")
        ax[0].semilogy(m, a["gr_intermodal"], "--", label="GR intermodal")
        ax[0].set_xlabel("mode index"); ax[0].set_ylabel(f"fraction over {a['L']:.0f} um")
        ax[0].set_title(f"validation: sigma={a['sigma_nm']:.0f} nm, Lc={a['lc']} um")
        ax[0].legend(fontsize=7)
    if os.path.exists(pb):
        b = json.load(open(pb))
        for lc in sorted({r["lc"] for r in b}):
            for s in sorted({r["sigma_nm"] for r in b}):
                rr = sorted([r for r in b if r["lc"] == lc and r["sigma_nm"] == s], key=lambda r: r["N"])
                if s == 5.0 or lc == 0.2:
                    ax[1].loglog([r["N"] for r in rr], [r["chip_rad_max_db"] for r in rr], "o-",
                                 label=f"sigma={s:g} nm, Lc={lc:g} um")
        ax[1].set_xlabel("N"); ax[1].set_ylabel("worst-mode roughness loss, whole chip [dB]")
        ax[1].set_title("radiation loss, L = sqrt(N) lambda/dn_max")
        ax[1].legend(fontsize=6)
    if os.path.exists(pc):
        c = json.load(open(pc))
        s = [r["sigma_nm"] for r in c["rough"]]
        ax[2].semilogy(s, [1 - r["Fn"] for r in c["rough"]], "o-", label="smooth design on rough chip")
        ax[2].semilogy(s, [1 - r["Fn_retrained"] for r in c["rough"]], "s-", label="re-optimised in situ")
        ax[2].axhline(1 - c["smooth"]["Fn"], color="k", ls=":", label="smooth chip")
        ax[2].set_xlabel("sigma [nm]"); ax[2].set_ylabel("1 - Fn")
        ax[2].set_title(f"N={c['N']} unitary on rough chips"); ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "roughness.png"), dpi=130)


if __name__ == "__main__":
    mnist()
    scaling()
    roughness()
