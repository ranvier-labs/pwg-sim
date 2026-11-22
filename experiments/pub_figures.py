"""Publication-quality figures (vector PDF + 300 dpi PNG) into figures/pub/.

    PYTHONPATH=. python experiments/pub_figures.py [--only fig_stack_modes ...]

Figures
-------
fig_stack_modes        vertical stack + TM0 mode; etched multimode guide lateral modes
fig_projection_optics  Mitsuba thin-lens DMD -> chip projection and index-programming chain
fig_light_propagation  BPM intensity: MNIST prototype; programmable multimode unitary
fig_device_render      Mitsuba 3 beauty render of the chip (cut-away stack, DMD light, guided light)
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, PowerNorm  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402

from pwg import viz  # noqa: E402
from pwg.viz import C, COL1, COL2, panel, save  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "figures", "pub")
CACHE = os.path.join(OUT, "_cache")
os.makedirs(CACHE, exist_ok=True)
viz.setup()


def cbar(fig, ax, im, label, **kw):
    cax = inset_axes(ax, width="3%", height="100%", loc="lower left",
                     bbox_to_anchor=(1.02, 0.0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cb = fig.colorbar(im, cax=cax, **kw)
    cb.set_label(label)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)
    return cb


# --------------------------------------------------------------------------- 4
def fig_stack_modes():
    from pwg.stack import Stack, core_mode
    from pwg.mmwg import MMWGConfig, MultimodeWaveguide

    st = Stack.wright2024()
    m = core_mode(st)
    edges = np.cumsum([0] + [l.thickness for l in st.layers])
    y0 = edges[1]                                   # top of the simulated Si = 0 (BOX bottom)
    y = m.y - y0
    YL = (-0.5, 8.7)
    keep = (y > YL[0]) & (y < YL[1])
    n_prof = np.zeros_like(m.y)
    for i, l in enumerate(st.layers):
        n_prof[(m.y >= edges[i]) & (m.y < edges[i + 1])] = np.real(l.n_y)

    wg = MultimodeWaveguide(MMWGConfig(n_modes=16, length_um=100.0))
    x = wg.grid.x()
    xs = np.abs(x) <= wg.W / 2 + 12

    fig = plt.figure(figsize=(COL2, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.1], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    cols = {"Si": "#d9d6cf", "BOX": "#eef3f8", "LN": "#f6dfcf", "TOX": "#eef3f8",
            "SRN": "#e3ecdc", "Ti": "#c9c6bd", "Au": "#e9cf7c", "air": "#ffffff"}
    names = {"Si": "p$^{++}$Si", "BOX": "SiO$_2$", "LN": "LN (Z-cut)", "TOX": "SiO$_2$",
             "SRN": "SRN photocond."}
    ypos = {"BOX": 0.45, "SRN": 4.35}
    for i, l in enumerate(st.layers):
        a, b = edges[i] - y0, edges[i + 1] - y0
        lo, hi = max(a, YL[0]), min(b, YL[1])
        if hi <= lo:
            continue
        ax.axhspan(lo, hi, color=cols[l.name], lw=0, zorder=0)
        if l.name in names and (hi - lo) > 0.25:
            ax.text(3.57, ypos.get(l.name, 0.5 * (lo + hi)), names[l.name], fontsize=6,
                    color=C["ink2"], ha="right", va="center")
    ax.axhline(edges[5] - y0, color="#b8962e", lw=1.6, zorder=1)
    ax.text(3.57, edges[5] - y0 + 0.12, "Ti/Au electrode (air above)", fontsize=6,
            color=C["ink2"], ha="right", va="bottom")
    ax.plot(n_prof[keep], y[keep], color=C["ink"], lw=0.8, label="refractive index $n$")
    I = np.abs(m.Hx) ** 2
    I /= I.max()
    ax.fill_betweenx(y[keep], 1.0, 1.0 + 1.3 * I[keep], color=C["orange"], alpha=0.35, lw=0)
    ax.plot(1.0 + 1.3 * I[keep], y[keep], color=C["orange"], lw=0.9,
            label="TM$_0$ $|H_x|^2$ (norm.)")
    ax.set_xlim(0.95, 3.6)
    ax.set_ylim(*YL)
    ax.set_xlabel("refractive index $n$ / mode intensity")
    ax.set_ylabel("height above Si, $y$ (µm)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.87), handlelength=1.2)
    ax.text(1.52, 1.25, f"$n_{{\\rm eff}}$ = {m.n_eff.real:.3f}, {m.loss_db_per_cm:.2f} dB cm$^{{-1}}$",
            va="bottom", fontsize=6, color=C["orange"])
    panel(ax, "a", x=-0.18)

    ax = fig.add_subplot(gs[1])
    core = np.abs(x) <= wg.W / 2
    ax.axvspan(-wg.W / 2, wg.W / 2, color=C["grid"], alpha=0.6, lw=0, zorder=0)
    ax.text(0, -0.85, "unetched LN core (0.700 µm)", ha="center", va="center", fontsize=6,
            color=C["ink2"])
    for sgn in (-1, 1):
        ax.text(sgn * (wg.W / 2 + 6), -0.85, f"etched\n(−{wg.etch*1e3:.1f} nm)", ha="center",
                va="center", fontsize=6, color=C["ink2"])
    sel = [0, 1, 2, 7, 15]
    for j, k in enumerate(sel):
        e = np.real(wg.modes[k] * np.exp(-1j * np.angle(wg.modes[k][np.argmax(np.abs(wg.modes[k]))])))
        e = e / np.abs(e).max()
        ax.plot(x[xs], e[xs] * 0.4 + 1.05 * j, color=viz.CAT[j % 8], lw=0.8)
        ax.text(wg.W / 2 + 12.5, 1.05 * j, f"$m$ = {k}", fontsize=6, va="center",
                color=C["ink2"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-1.25, 1.05 * (len(sel) - 1) + 0.55)
    ax.set_xlim(x[xs][0], x[xs][-1] + 6)
    ax.set_xlabel("lateral position $x$ (µm)")
    ax.set_title(f"Guided modes $E_m(x)$ of the etched multimode guide (NA 0.3, $W$ = {wg.W:.1f} µm, "
                 f"70° sidewall)", loc="left")
    panel(ax, "b", x=-0.03)
    save(fig, os.path.join(OUT, "fig_stack_modes"))
    plt.close(fig)


# --------------------------------------------------------------------------- 2
def _render(pattern, defocus, ss=4, spp=512, na=0.10, seed=0):
    import mitsuba as mi
    from pwg.projector import ProjectorOptics
    R, Cc = pattern.shape
    p = ProjectorOptics(n_rows=R, n_cols=Cc, supersample=ss, defocus_um=defocus, na_chip=na).build()
    img = p.render(mi.Float(pattern.astype(np.float32).ravel()), spp=spp, seed=seed).numpy()
    return img, p


def _disk_esf(x, R):
    """Geometric edge-spread function of a uniform disk blur of radius R."""
    if R <= 0:
        return (x > 0).astype(float)
    u = np.clip(x / R, -1, 1)
    return (np.arccos(-u) + u * np.sqrt(1 - u * u)) / np.pi


def fig_projection_optics():
    from scipy.ndimage import convolve1d, gaussian_filter1d
    from pwg.electro import LocalResponse, lateral_lsf
    from pwg.stack import Stack

    pat_full = np.load(os.path.join(ROOT, "results", "mnist_prototype", "dmd_pattern.npy"))
    r0, c0, R, Cc = 40, 296, 12, 20
    pat = pat_full[r0:r0 + R, c0:c0 + Cc]
    cache = os.path.join(CACHE, "proj2.npz")
    if os.path.exists(cache):
        d = dict(np.load(cache))
    else:
        d = {}
        for df in (0, 25, 50):
            d[f"img{df}"], pobj = _render(pat, df, ss=8, spp=1024)
        old = os.path.join(CACHE, "proj.npz")
        if os.path.exists(old):
            o = np.load(old)
            for k in o.files:
                if k.startswith("edge"):
                    d[k] = o[k]
        else:
            edge = np.zeros((8, 40), np.float32)
            edge[:, 20:] = 1
            for df in (0, 10, 25, 50):
                im, pe = _render(edge, df, ss=16, spp=1024)
                d[f"edge{df}"] = im[16:-16].mean(0)
            d["edge_px"] = np.array([pe.render_pixel_um])
        d["px"] = np.array([pobj.render_pixel_um, pobj.chip_pixel_um])
        np.savez(cache, **d)
    rp, cp = d["px"]
    na = 0.10

    st = Stack.wright2024()
    resp = LocalResponse(st).calibrate()
    xl, lsf, fwhm = lateral_lsf(resp)
    I_max = 57.0
    sig_d = 0.51 * 0.525 / na / 2.3548 / rp          # Airy-equivalent diffraction blur [px]
    img = d["img25"]
    img = gaussian_filter1d(gaussian_filter1d(img, sig_d, 0, mode="nearest"), sig_d, 1, mode="nearest")
    I = I_max * img
    dn_loc = resp.dn_eff(I) - resp.dn_eff(0.0)
    k = np.interp(np.arange(-int(12 / rp), int(12 / rp) + 1) * rp, xl, lsf)
    k /= k.sum()
    dn = convolve1d(convolve1d(dn_loc, k, axis=0, mode="nearest"), k, axis=1, mode="nearest")

    fig = plt.figure(figsize=(COL2, 3.75))
    gs = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.95, left=0.07, right=0.985, top=0.90, bottom=0.11)
    ext = [0, Cc * cp, R * cp, 0]                     # chip coordinates (µm)
    zc = int(0.55 * img.shape[1])                     # line-cut column
    xs = (np.arange(img.shape[0]) + 0.5) * rp

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(pat, cmap="gray", vmin=0, vmax=1, extent=[0, Cc * 13.7, R * 13.7, 0], aspect="equal")
    ax.set_title("DMD duty cycle", loc="left")
    ax.set_xlabel("$z_{\\rm DMD}$ (µm)"); ax.set_ylabel("$x_{\\rm DMD}$ (µm)")
    panel(ax, "a", x=-0.32, y=1.05)
    for j, df in enumerate((0, 50)):
        ax = fig.add_subplot(gs[0, 1 + j])
        im = ax.imshow(d[f"img{df}"], cmap="cividis", vmin=0, vmax=1, extent=ext, aspect="equal")
        ax.set_title(f"chip, defocus {df} µm", loc="left")
        ax.set_xlabel("$z$ (µm)")
        ax.axvline(zc * rp, color="w", lw=0.5, ls=(0, (2, 2)))
        if j == 0:
            ax.set_ylabel("$x$ (µm)")
            panel(ax, "b", x=-0.32, y=1.05)
        else:
            ax.set_yticklabels([])
            cbar(fig, ax, im, "$I/I_{\\rm max}$")
    ax = fig.add_subplot(gs[0, 3])
    ax.step(np.arange(R + 1) * cp, np.r_[pat[:, int(zc * rp / cp)], pat[-1, int(zc * rp / cp)]],
            where="post", color=C["muted"], lw=0.8, label="mirror")
    for j, df in enumerate((0, 25, 50)):
        ax.plot(xs, d[f"img{df}"][:, zc], color=viz.CAT[j], lw=0.9, label=f"{df} µm")
    ax.set_xlabel("$x$ (µm)")
    ax.set_ylabel("$I/I_{\\rm max}$ along dashed line")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.6, columnspacing=0.8,
              handlelength=1.0, borderaxespad=0.1)
    panel(ax, "c", x=-0.38, y=1.12)

    # edge response
    ax = fig.add_subplot(gs[1, 0])
    ep = float(d["edge_px"][0])
    for j, df in enumerate((0, 10, 25, 50)):
        e = d[f"edge{df}"]
        zz = (np.arange(e.size) + 0.5) * ep - 20 * 13.7 / 1.5
        sel = np.abs(zz) < 9
        ax.plot(zz[sel][::2], e[sel][::2], "o", ms=1.8, mew=0, color=viz.CAT[j], label=f"{df} µm")
        zf = np.linspace(-9, 9, 400)
        ax.plot(zf, _disk_esf(zf, na * df), "-", lw=0.7, color=viz.CAT[j])
    ax.set_xlabel("distance from mirror edge (µm)")
    ax.set_ylabel("$I/I_{\\rm max}$")
    ax.set_title("edge response", loc="left")
    ax.legend(title="defocus", loc="upper left", fontsize=5.6, title_fontsize=5.6,
              handletextpad=0.1, labelspacing=0.2, borderaxespad=0.1)
    ax.text(0.98, 0.04, "dots: Mitsuba\nlines: disk\n$\\varnothing = 2\\,{\\rm NA}\\,\\delta$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.4, color=C["ink2"],
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    panel(ax, "d", x=-0.32, y=1.05)

    for j, (t, arr, cm, vmax, lab) in enumerate([
            ("irradiance", I, "cividis", I_max, "mW cm$^{-2}$"),
            ("$\\Delta n_{\\rm eff}$ after LSF", dn * 1e3, "magma", dn_loc.max() * 1e3, "$10^{-3}$")]):
        ax = fig.add_subplot(gs[1, 1 + j])
        im = ax.imshow(arr, cmap=cm, extent=ext, aspect="equal", vmin=0, vmax=vmax)
        ax.set_title(t, loc="left")
        ax.set_xlabel("$z$ (µm)")
        ax.axvline(zc * rp, color="w", lw=0.5, ls=(0, (2, 2)))
        if j == 0:
            ax.set_ylabel("$x$ (µm)")
            panel(ax, "e", x=-0.32, y=1.05)
        else:
            ax.set_yticklabels([])
        cbar(fig, ax, im, lab)
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(xs, I[:, zc] / I_max, color=C["blue"], lw=0.9, label="$I/I_{\\rm max}$")
    ax.plot(xs, dn_loc[:, zc] / dn_loc.max(), color=C["orange"], lw=0.9, label="local $\\Delta n$")
    ax.plot(xs, dn[:, zc] / dn_loc.max(), color=C["ink"], lw=0.9, ls=(0, (3, 1.5)),
            label="after LSF")
    ax.set_xlabel("$x$ (µm)"); ax.set_ylabel("normalised, along dashed line")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.6, columnspacing=0.8,
              handlelength=1.4, borderaxespad=0.1)
    panel(ax, "f", x=-0.38, y=1.12)
    print(f"  NA {na}, chip px {cp:.2f} µm, LSF FWHM {fwhm:.2f} µm")
    save(fig, os.path.join(OUT, "fig_projection_optics"))
    plt.close(fig)


# --------------------------------------------------------------------------- 3
def _record(prop, dn_np, e, every):
    """Propagate one complex input field, return |E|^2 every ``every`` steps [nrec, nx]."""
    from drjit.cuda import Float
    rec = {k: None for k in range(0, prop.g.nz, every)}
    re, im, _ = prop.forward(Float(dn_np.astype(np.float32).ravel()),
                             Float(e.real.astype(np.float32)), Float(e.imag.astype(np.float32)),
                             record=rec)
    out = np.stack([rec[k] for k in sorted(rec)] + [re.numpy() ** 2 + im.numpy() ** 2])
    return out


def _mnist_maps():
    cache = os.path.join(CACHE, "mnist_maps.npz")
    if os.path.exists(cache):
        return dict(np.load(cache, allow_pickle=False))
    import drjit as dr
    from drjit.cuda.ad import Float as FAD
    from pwg.device import PrototypeConfig, PrototypeDevice
    sys.path.insert(0, os.path.dirname(__file__))
    from mnist_prototype import load_mnist
    _, _, Xte, yte = load_mnist(n_train=10, n_test=400)
    dev = PrototypeDevice(PrototypeConfig(), verbose=False)
    pat = np.load(os.path.join(ROOT, "results", "mnist_prototype", "dmd_pattern.npy"))
    with dr.suspend_grad():
        dn = dev.index_map(FAD(pat.astype(np.float32).ravel()), spp=64, seed=1).numpy()
        re, im = dev.transfer(FAD(dn))
        Ur, Ui = dev.output_matrix(re, im)
        P = dev.bin_powers(Ur, Ui, Xte).numpy().reshape(len(yte), -1)
    pred = P.argmax(1)
    picks = []
    for digit in (0, 3, 7):
        i = int(np.nonzero((yte == digit) & (pred == digit))[0][0])
        picks.append(i)
    maps = []
    for i in picks:
        e = (Xte[i][:, None] * dev.ein).sum(0).astype(np.complex64)
        maps.append(_record(dev.prop, dn, e, 8))
    d = dict(maps=np.stack(maps), digits=np.stack([Xte[i].reshape(7, 7) for i in picks]),
             labels=yte[picks], P=P[picks], x=dev.x, dz=np.array([dev.grid.dz * 8]),
             L=np.array([dev.grid.length]), masks=dev.out_masks,
             dn=dn.reshape(dev.grid.nz, dev.grid.nx)[::8, ::2], acc=np.array([(pred == yte).mean()]))
    np.savez(cache, **d)
    return d


def _mmwg_maps():
    cache = os.path.join(CACHE, "mmwg_maps.npz")
    if os.path.exists(cache):
        return dict(np.load(cache, allow_pickle=False))
    import drjit as dr
    from drjit.cuda.ad import Float as FAD
    sys.path.insert(0, os.path.dirname(__file__))
    from unitary_scaling import design
    from pwg.mmwg import haar_unitary, fidelity
    N, L = 16, 1240.0
    res, wg, theta = design(N, L, 300, 0.1, verbose=False)
    U = haar_unitary(N, np.random.default_rng(0))
    _, Tt = wg.target_fields(U)
    with dr.suspend_grad():
        dn_p = wg.index_map(FAD(theta)).numpy()
        dn_0 = wg.index_map(FAD(np.full(wg.n_par, -30.0, np.float32))).numpy()
        re, im = wg.transfer(FAD(dn_p))
    T = wg.matrix(re.numpy(), im.numpy())
    F, Fn, err = fidelity(T, Tt)
    e = wg.modes[0].astype(np.complex64)
    m0 = _record(wg.prop, dn_0, e, 4)
    m1 = _record(wg.prop, dn_p, e, 4)
    nz, nx = wg.grid.nz, wg.grid.nx
    d = dict(m0=m0, m1=m1, dn=(dn_p - dn_0).reshape(nz, nx), x=wg.grid.x(), dz=np.array([wg.grid.dz * 4]),
             L=np.array([wg.grid.length]), W=np.array([wg.W]), T=T, Tt=Tt, beta=wg.beta,
             Fn=np.array([Fn]), F=np.array([F]), modes=wg.modes)
    np.savez(cache, **d)
    return d


def fig_light_propagation():
    a = _mnist_maps()
    b = _mmwg_maps()
    fig = plt.figure(figsize=(COL2, 6.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.32)
    # ---------------- (a) MNIST prototype
    ga = outer[0].subgridspec(3, 3, width_ratios=[0.07, 1, 0.12], wspace=0.30, hspace=0.18)
    x = a["x"] * 1e-3
    L = float(a["L"][0]) * 1e-3
    xs = np.abs(x) < 0.52
    cents = [x[m.astype(bool)].mean() for m in a["masks"]]
    for r in range(3):
        I = a["maps"][r][:, xs]
        I = I / np.percentile(I, 99.7)
        ax0 = fig.add_subplot(ga[r, 0])
        ax0.imshow(a["digits"][r], cmap="gray_r", aspect="equal")
        ax0.set_xticks([]); ax0.set_yticks([])
        for sp in ax0.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color(C["muted"])
        if r == 0:
            ax0.set_title("input", fontsize=6.5)
            panel(ax0, "a", x=-0.15, y=1.15)
        ax = fig.add_subplot(ga[r, 1])
        im = ax.imshow(I.T, cmap="inferno", norm=PowerNorm(0.5, vmin=0, vmax=1), aspect="auto",
                       extent=[0, L, x[xs][-1], x[xs][0]], interpolation="bilinear")
        ax.set_yticks([-0.4, 0, 0.4])
        if r == 1:
            ax.set_ylabel("$x$ (mm)", labelpad=1)
        if r < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("propagation distance $z$ (mm)")
        if r == 0:
            ax.set_title("guided intensity $|E|^2$ through the trained prototype (Mitsuba-projected index)",
                         loc="left")
        # output bins
        axb = fig.add_subplot(ga[r, 2])
        axb.set_ylim(x[xs][-1], x[xs][0])
        Pn = a["P"][r] / a["P"][r].sum()
        lab = int(a["labels"][r])
        for k, (cx, p) in enumerate(zip(cents, Pn)):
            axb.barh(cx, p, height=0.045, color=C["orange"] if k == lab else C["muted"], lw=0)
        axb.text(Pn[lab] + 0.02, cents[lab], f"class {lab}", fontsize=5.6, va="center",
                 color=C["ink"])
        for k in (0, 9):
            axb.text(-0.015, cents[k], str(k), fontsize=5, ha="right", va="center", color=C["ink2"])
        axb.set_xlim(0, 0.62)
        axb.set_yticks([])
        axb.spines["left"].set_visible(False)
        axb.tick_params(axis="x", labelsize=5.5)
        if r == 2:
            axb.set_xlabel("bin power", fontsize=6)
        else:
            axb.set_xticklabels([])
    # ---------------- (b, c) programmable multimode waveguide
    gb = outer[1].subgridspec(2, 3, width_ratios=[1, 1, 0.58], hspace=0.5, wspace=0.78)
    xb = b["x"]
    W = float(b["W"][0])
    sx = np.abs(xb) < W / 2 + 8
    Lb = float(b["L"][0]) * 1e-3
    for j, (key, ttl) in enumerate([("m0", "unprogrammed: input mode $m$ = 0"),
                                    ("m1", "programmed: Haar-random 16×16 $U$")]):
        ax = fig.add_subplot(gb[0, j])
        I = b[key][:, sx]
        I = I / np.percentile(b["m1"][:, sx], 99.7)
        im = ax.imshow(I.T, cmap="inferno", norm=PowerNorm(0.5, vmin=0, vmax=1), aspect="auto",
                       extent=[0, Lb, xb[sx][-1], xb[sx][0]], interpolation="bilinear")
        for sgn in (-1, 1):
            ax.axhline(sgn * W / 2, color="w", lw=0.4, ls=(0, (3, 3)), alpha=0.6)
        ax.set_title(ttl, loc="left")
        ax.set_xlabel("$z$ (mm)")
        if j == 0:
            ax.set_ylabel("$x$ (µm)")
            panel(ax, "b", x=-0.2)
        else:
            ax.set_yticklabels([])
            cbar(fig, ax, im, "$|E|^2$ (norm.)")
    ax = fig.add_subplot(gb[1, 0])
    dn = b["dn"][:, sx]
    im = ax.imshow(dn.T * 1e3, cmap="magma", vmin=0, vmax=5, aspect="auto",
                   extent=[0, Lb, xb[sx][-1], xb[sx][0]], interpolation="nearest")
    ax.set_title("designed programmed index $\\Delta n_{\\rm prog}(x,z)$", loc="left")
    ax_c = ax
    ax.set_xlabel("$z$ (mm)"); ax.set_ylabel("$x$ (µm)")
    cbar(fig, ax, im, "$\\Delta n$ ($10^{-3}$)")
    panel(ax, "c", x=-0.2)
    # zoom of the index
    ax = fig.add_subplot(gb[1, 1])
    z0 = int(0.45 * dn.shape[0]); nzz = int(120 / 1.0)
    from matplotlib.patches import Rectangle
    ax_c.add_patch(Rectangle((z0 * 1e-3, -15), nzz * 1e-3, 30, fill=False, ec="w", lw=0.6))
    xz = np.abs(xb[sx]) < 15
    ax.imshow(dn[z0:z0 + nzz, xz].T * 1e3, cmap="magma", vmin=0, vmax=5, aspect="auto",
              extent=[z0 * 1e-3, (z0 + nzz) * 1e-3, xb[sx][xz][-1], xb[sx][xz][0]])
    ax.set_title("zoom (box in c; 1 µm resolution)", loc="left")
    ax.set_xlabel("$z$ (mm)"); ax.set_yticklabels([])
    ax.set_ylabel("")
    # (d) matrices
    gm = gb[:, 2].subgridspec(2, 1, hspace=0.45)
    T = b["T"] / np.exp(1j * b["beta"] * float(b["L"][0]))[:, None]
    Tt = b["Tt"] / np.exp(1j * b["beta"] * float(b["L"][0]))[:, None]
    vmax = max(np.abs(T).max(), np.abs(Tt).max())
    for j, (M, ttl) in enumerate([(Tt, "target $|U_{ij}|$"),
                                  (T, f"realised $|T_{{ij}}|$  ($F_n$ = {float(b['Fn'][0]):.4f})")]):
        ax = fig.add_subplot(gm[j])
        im = ax.imshow(np.abs(M), cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(ttl, loc="left", fontsize=6.5)
        ax.set_xticks([0, 15]); ax.set_yticks([0, 15])
        ax.set_xlabel("input mode $j$", fontsize=6) if j == 1 else ax.set_xticklabels([])
        ax.set_ylabel("output $i$", fontsize=6)
        if j == 0:
            panel(ax, "d", x=-0.1, y=1.0)
        cbar(fig, ax, im, "")
    save(fig, os.path.join(OUT, "fig_light_propagation"))
    plt.close(fig)


# --------------------------------------------------------------------------- 1
EXAG = 60.0      # vertical exaggeration of the thin-film stack in the render


def _srgb_to_lin(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _chip_scene(light_img, dmd_img, L=9.13, Wd=1.014, half_w=1.25, x_cut=5.0,
                res=(2400, 1000), spp=256):
    """Mitsuba 3 scene dict of the cut-away chip.  World: X = propagation z [mm],
    Z = lateral x [mm], Y = up; film thicknesses scaled by EXAG."""
    import mitsuba as mi
    T = mi.ScalarTransform4f
    um = 1e-3 * EXAG
    t_si, t_box, t_ln, t_tox, t_srn, t_au = 0.32, 2.0 * um, 0.7 * um, 1.0 * um, 4.0 * um, 0.15 * um
    y = 0.0
    lay = {}
    for name, t in [("si", t_si), ("box", t_box), ("ln", t_ln), ("tox", t_tox), ("srn", t_srn), ("au", t_au)]:
        lay[name] = (y, y + t)
        y += t

    def box(name, x0, x1, z0, z1, bsdf):
        y0, y1 = lay[name]
        return {"type": "cube", "bsdf": bsdf,
                "to_world": T().translate([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
                @ T().scale([(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2])}

    def plastic(rgb, rough=0.25):
        return {"type": "roughplastic", "distribution": "ggx", "alpha": rough,
                "diffuse_reflectance": {"type": "rgb", "value": rgb}, "int_ior": 1.5}

    def rect(x0, x1, z0, z1, yy, tex, scale, base):
        return {"type": "rectangle",
                "to_world": T().translate([(x0 + x1) / 2, yy, (z0 + z1) / 2])
                @ T().rotate([1, 0, 0], -90) @ T().scale([(x1 - x0) / 2, (z1 - z0) / 2, 1]),
                "bsdf": base,
                "emitter": {"type": "area", "radiance": {
                    "type": "bitmap", "raw": True, "filter_type": "bilinear",
                    "data": mi.TensorXf(np.ascontiguousarray(tex * scale, dtype=np.float32))}}}

    x0, x1 = -0.35, L + 0.35
    sc = {"type": "scene",
          "integrator": {"type": "path", "max_depth": 8},
          "sensor": {"type": "perspective", "fov": 47, "fov_axis": "x",
                     "to_world": T().look_at(origin=[L * 0.42, 7.6, 11.0], target=[L * 0.5, 0.2, -0.3],
                                             up=[0, 1, 0]),
                     "film": {"type": "hdrfilm", "width": res[0], "height": res[1],
                              "rfilter": {"type": "gaussian"}},
                     "sampler": {"type": "independent", "sample_count": spp}},
          "env": {"type": "constant", "radiance": {"type": "rgb", "value": [0.55, 0.56, 0.58]}},
          "key": {"type": "rectangle", "to_world": T().look_at(origin=[2, 9, 6], target=[4.5, 0, 0], up=[0, 0, 1])
                  @ T().scale([4, 4, 1]), "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [2.2, 2.15, 2.05]}}},
          "fill": {"type": "rectangle", "to_world": T().look_at(origin=[14, 4, -3], target=[4.5, 0, 0], up=[0, 1, 0])
                   @ T().scale([3, 3, 1]), "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [0.8, 0.82, 0.9]}}},
          "floor": {"type": "rectangle", "to_world": T().translate([4.5, -0.001, 0]) @ T().rotate([1, 0, 0], -90)
                    @ T().scale([40, 40, 1]), "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.82, 0.82, 0.8]}}},
          }
    # lower stack, full chip
    sc["si"] = box("si", x0, x1, -half_w, half_w, plastic([0.16, 0.16, 0.18], 0.15))
    sc["box"] = box("box", x0, x1, -half_w, half_w, plastic([0.80, 0.86, 0.92], 0.3))
    sc["ln"] = box("ln", x0, x1, -half_w, half_w, plastic([0.42, 0.22, 0.16], 0.2))
    # upper stack only over the input part (cut-away beyond x_cut)
    sc["tox"] = box("tox", x0, x_cut, -half_w, half_w, plastic([0.80, 0.86, 0.92], 0.3))
    sc["srn"] = box("srn", x0, x_cut, -half_w, half_w, plastic([0.47, 0.55, 0.42], 0.3))
    sc["au"] = box("au", x0, x_cut, -half_w, half_w,
                   {"type": "roughconductor", "material": "Au", "alpha": 0.25})
    # guided light on the exposed LN core (z from x_cut to L) and DMD light on the electrode
    nl = light_img.shape[1]
    c_cut = int(round(nl * x_cut / L))
    eps = 2e-4
    sc["light"] = rect(x_cut, L, -Wd / 2, Wd / 2, lay["ln"][1] + eps, light_img[:, c_cut:], 9.0,
                       plastic([0.42, 0.22, 0.16], 0.2))
    nd = dmd_img.shape[1]
    d_cut = int(round(nd * x_cut / L))
    sc["dmd"] = rect(0.0, x_cut, -Wd / 2, Wd / 2, lay["au"][1] + eps, dmd_img[:, :d_cut], 1.6,
                     {"type": "roughconductor", "material": "Au", "alpha": 0.25})
    return sc, lay


def fig_device_render():
    import matplotlib.cm as cm
    a = _mnist_maps()                                    # computed with the cuda_ad_mono variant
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb")
    I = a["maps"][0]                                      # [nrec, nx] for digit 0
    x = a["x"] * 1e-3
    xs = np.abs(x) <= 0.507
    I = I[:, xs].T                                        # rows: lateral x, cols: z
    I = I / np.percentile(I, 99.5)
    a_ = np.clip(np.sqrt(np.clip(I, 0, None)), 0, 1)[..., None]
    light = a_ * np.array([1.0, 0.16, 0.03])[None, None, :] + a_ ** 3 * np.array([0.0, 0.55, 0.35])[None, None, :]
    light = light[::-1]                                   # rect v axis maps to -Z
    pat = np.load(os.path.join(ROOT, "results", "mnist_prototype", "dmd_pattern.npy"))
    green = np.array([0.03, 0.55, 0.06])
    dmd = (pat[..., None] * green[None, None, :])[::-1]
    out_exr = os.path.join(CACHE, "chip_render.npy")
    if not os.path.exists(out_exr):
        sc, lay = _chip_scene(light, dmd)
        scene = mi.load_dict(sc)
        img = mi.render(scene).numpy()
        np.save(out_exr, img)
    img = np.load(out_exr)
    # filmic-ish tone map + sRGB
    v = img / (1.0 + img * 0.15)
    v = np.clip(v, 0, None)
    srgb = np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1 / 2.4) - 0.055)
    srgb = np.clip(srgb, 0, 1)
    plt.imsave(os.path.join(OUT, "fig_device_render_raw.png"), srgb)
    y0c, y1c = 170, 900                                   # crop empty floor/background
    H, Wp = srgb.shape[:2]
    fig = plt.figure(figsize=(COL2, COL2 * (y1c - y0c) / Wp))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(srgb, interpolation="lanczos", extent=[0, Wp, H, 0])
    ax.set_xlim(0, Wp); ax.set_ylim(y1c, y0c)
    ax.axis("off")
    kw = dict(fontsize=6.5, color=C["ink"], arrowprops=dict(arrowstyle="-", lw=0.5, color=C["ink2"],
                                                            shrinkA=1, shrinkB=0))
    # top call-outs (pixel coordinates of the 2400 x 1000 render)
    ax.annotate("DMD-projected light pattern (525 nm)\n→ photoconductor sets $\\Delta n(x,z)$",
                xy=(760, 440), xytext=(690, 235), ha="left", va="center", **kw)
    ax.annotate("semi-transparent Ti/Au\nelectrode (1.1 kV, 10 Hz)", xy=(540, 560), xytext=(60, 235),
                ha="left", va="center", **kw)
    ax.annotate("guided 1550 nm light $|E|^2$ in the LN core\n(cut-away; MNIST digit routed to its output bin)",
                xy=(1800, 470), xytext=(1430, 262), ha="left", va="center", **kw)
    # layer call-outs along the front face
    layers = [("SRN photoconductor 4 µm", (450, 628)), ("SiO$_2$ 1 µm", (650, 650)),
              ("Z-cut LN 0.7 µm", (850, 652)), ("SiO$_2$ 2 µm", (1050, 671)),
              ("p$^{++}$Si (ground)", (1250, 712))]
    for k, (t, xy) in enumerate(layers):
        ax.annotate(t, xy=xy, xytext=(250 + 310 * k, 845), ha="center", va="center", **kw)
    ax.text(2385, 180, f"Mitsuba 3 path tracing · film thicknesses ×{EXAG:.0f} · 9.1 mm × 2.5 mm chip",
            ha="right", va="top", fontsize=5.8, color=C["ink2"])
    save(fig, os.path.join(OUT, "fig_device_render"))
    plt.close(fig)


FIGS = {"fig_stack_modes": fig_stack_modes, "fig_projection_optics": fig_projection_optics,
        "fig_light_propagation": fig_light_propagation, "fig_device_render": fig_device_render}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    for name, f in FIGS.items():
        if a.only and name not in a.only:
            continue
        print("[pub]", name, flush=True)
        f()


if __name__ == "__main__":
    main()
