"""Vertical layer stack of the 2D-programmable waveguide (Onodera, Stein, ..., Wright,
McMahon, arXiv:2402.17750 / Nature Physics 2026) and its optical slab modes.

Stack (bottom -> top), Methods 0.1 / Supp. 1.1 of the paper:

    p++ Si substrate (ground electrode)   n = 3.48 + 0.027i
    SiO2 bottom cladding     2.0 um       n = 1.444,  eps_r = 3.9
    Z-cut MgO:LN core        0.7 um       n_e = 2.1376 (c-axis normal to slab)
                                          n_o = 2.2111, eps_x = 85.2, eps_y = 27.8
    SiO2 top cladding        1.0 um
    Si-rich SiN (SRN) photoconductor 4.0 um  n = 2.03 + 1.4e-5 i, eps_r = 7.2
    Ti 10 nm / Au 5 nm semi-transparent top electrode
    air

The optical mode is TM0 (E mostly along the c-axis, y), so it sees n_e in the
normal direction and n_o along propagation.  The mode solver below is a
finite-difference solver for the anisotropic TM slab equation

    eps_y d/dy ( 1/eps_z dHx/dy ) + k0^2 eps_y Hx = beta^2 Hx

with complex-coordinate-stretching PMLs on both sides, so that leakage into the
high-index Si substrate, into the SRN (n=2.03 > n_eff) and absorption in the
metal electrode all appear as Im(n_eff).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class Layer:
    name: str
    thickness: float          # um (substrate/superstrate: thickness of simulated part)
    n_y: complex              # optical index for E along y (slab normal)
    n_z: complex = None       # optical index for E along z (in-plane); default isotropic
    eps_dc_y: float = 1.0     # low-frequency permittivity along y (electrostatics)
    eps_dc_x: float = None    # low-frequency permittivity in-plane
    sigma: float = 0.0        # DC conductivity [S/m]

    def __post_init__(self):
        if self.n_z is None:
            self.n_z = self.n_y
        if self.eps_dc_x is None:
            self.eps_dc_x = self.eps_dc_y


@dataclass
class Stack:
    layers: list
    wavelength: float = 1.55
    core: str = "LN"

    @staticmethod
    def wright2024(**overrides) -> "Stack":
        """The fabricated prototype stack (paper values)."""
        p = dict(t_box=2.0, t_ln=0.7, t_top=1.0, t_srn=4.0,
                 n_e=2.1376, n_o=2.2111, n_sio2=1.444, n_srn=2.03 + 1.4e-5j,
                 n_si=3.48 + 0.027j, n_ti=3.69 + 4.61j, n_au=0.52 + 10.7j)
        p.update(overrides)
        L = [
            Layer("Si", 1.5, p["n_si"], eps_dc_y=11.7, sigma=1e4),
            Layer("BOX", p["t_box"], p["n_sio2"], eps_dc_y=3.9),
            Layer("LN", p["t_ln"], p["n_e"], p["n_o"], eps_dc_y=27.8, eps_dc_x=85.2),
            Layer("TOX", p["t_top"], p["n_sio2"], eps_dc_y=3.9),
            Layer("SRN", p["t_srn"], p["n_srn"], eps_dc_y=7.2),
            Layer("Ti", 0.010, p["n_ti"]),
            Layer("Au", 0.005, p["n_au"]),
            Layer("air", 1.5, 1.0),
        ]
        return Stack(L)

    def layer(self, name) -> Layer:
        return next(l for l in self.layers if l.name == name)

    def with_layer(self, name, **kw) -> "Stack":
        return Stack([replace(l, **kw) if l.name == name else l for l in self.layers],
                     self.wavelength, self.core)


@dataclass
class Mode:
    n_eff: complex
    y: np.ndarray
    Hx: np.ndarray
    Ey: np.ndarray

    @property
    def loss_db_per_cm(self) -> float:
        k0 = 2 * math.pi / 1.55
        return 20 / math.log(10) * k0 * self.n_eff.imag * 1e4


def solve_modes(stack: Stack, n_modes=4, dy=0.0025, pml=1.0, target=None, pol="TM"):
    """Return the guided/leaky modes nearest ``target`` (default: core index)."""
    k0 = 2 * math.pi / stack.wavelength
    edges = np.cumsum([0] + [l.thickness for l in stack.layers])
    y0, y1 = -pml, edges[-1] + pml
    ny = int(round((y1 - y0) / dy))
    y = y0 + (np.arange(ny) + 0.5) * dy            # nodes
    yh = y0 + np.arange(ny + 1) * dy               # half nodes (cell faces)

    def prop(yy, attr):
        out = np.empty(yy.shape, complex)
        for i, l in enumerate(stack.layers):
            m = (yy >= edges[i]) & (yy < edges[i + 1])
            out[m] = getattr(l, attr)
        out[yy < edges[0]] = getattr(stack.layers[0], attr)
        out[yy >= edges[-1]] = getattr(stack.layers[-1], attr)
        return out

    def stretch(yy):
        d = np.maximum(0, np.maximum(edges[0] - yy, yy - edges[-1])) / pml
        return 1 + 1j * 6.0 * d**3

    if pol == "TM":
        ey = prop(y, "n_y") ** 2
        ez_h = prop(yh, "n_z") ** 2
        a = 1.0 / (stretch(yh) * ez_h)              # on faces
        pre = ey / stretch(y)
    else:  # TE: d2Ex/dy2 + k0^2 eps Ex = beta^2 Ex  (Ex sees n_o for z-cut LN)
        ey = prop(y, "n_z") ** 2
        a = 1.0 / stretch(yh)
        pre = 1.0 / stretch(y)
    main = -(a[:-1] + a[1:]) / dy**2
    D = sp.diags([a[1:-1] / dy**2, main, a[1:-1] / dy**2], [-1, 0, 1], format="csc")
    A = sp.diags(pre) @ D + sp.diags(k0**2 * ey)
    if target is None:
        target = stack.layer(stack.core).n_y.real
    vals, vecs = spla.eigs(A.tocsc(), k=n_modes, sigma=(k0 * target) ** 2)
    neff = np.sqrt(vals.astype(complex)) / k0
    modes = []
    for i in np.argsort(-neff.real):
        h = vecs[:, i]
        h = h / h[np.argmax(np.abs(h))]
        e = h / prop(y, "n_y") ** 2 if pol == "TM" else h
        modes.append(Mode(complex(neff[i]), y, h, e))
    return modes


def core_mode(stack: Stack, pol="TM", **kw) -> Mode:
    """Fundamental mode most strongly confined to the core."""
    edges = np.cumsum([0] + [l.thickness for l in stack.layers])
    idx = [l.name for l in stack.layers].index(stack.core)
    lo, hi = edges[idx], edges[idx + 1]
    best, score = None, -1
    for m in solve_modes(stack, pol=pol, **kw):
        if m.n_eff.real < 1.444 or m.n_eff.real > stack.layer(stack.core).n_z.real + 0.1:
            continue
        w = np.abs(m.Hx) ** 2
        frac = w[(m.y >= lo) & (m.y < hi)].sum() / w.sum()
        # prefer TM0: high confinement, no node in the core
        if frac > score:
            best, score = m, frac
    return best


def eo_overlap(stack: Stack, dn=1e-4) -> float:
    """Gamma = d n_eff / d n_e(LN): fraction of a core index change seen by the mode."""
    ln = stack.layer("LN")
    up = core_mode(stack.with_layer("LN", n_y=ln.n_y + dn)).n_eff.real
    dw = core_mode(stack.with_layer("LN", n_y=ln.n_y - dn)).n_eff.real
    return (up - dw) / (2 * dn)
