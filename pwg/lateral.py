"""Physically derived lateral (in-plane) index profile of an etched multimode waveguide.

Lateral confinement is made by a shallow partial etch of the LN film (the top
SiO2 is assumed planarised, so it is thicker by the etch depth over the
etched region).  The in-plane model needs n_eff(x) and the local loss alpha(x):

1. **Local effective index.**  At each lateral position the vertical stack is
   a slab with local LN thickness t(x); the TM0 mode solver of :mod:`pwg.stack`
   gives n_eff(t) and Im n_eff(t) (leakage into Si / SRN / metal).  We tabulate
   it over the etch range and interpolate.  This is the (local) effective index
   method; its validity is checked against a full 2D cross-section solve in
   :func:`check_against_2d`.

2. **Geometry.**  The sidewall is a straight slope at angle ``sidewall_deg``
   (Ar-milled LN: typically 60-75 deg), so t(x) ramps over a run of
   d_etch / tan(theta) -- tens of nm, i.e. *sharper* than the BPM grid.

3. **Discretisation by sub-pixel averaging.**  Each grid cell gets the
   cell-average of eps = n_eff^2 (computed on a fine sub-grid).  For the
   dominant TM0 field (E_y, parallel to the sidewall) the averaged eps is the
   first-order-correct effective permittivity of a cell cut by an interface
   (Farjadpour et al., Opt. Lett. 31, 2972 (2006)).  Because the cell value is
   a continuous function of the edge position, nanometre edge displacements
   (sidewall roughness) are represented exactly on a 0.5 um grid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from .stack import Stack, core_mode

K0_1550 = 2 * math.pi / 1.55


class LocalIndexTable:
    """n_eff(t_LN) and loss(t_LN) for the planarised partially-etched stack."""

    def __init__(self, t0=0.7, t_top0=1.0, d_max=0.12, n=13, **stack_kw):
        self.t0 = t0
        ds = np.linspace(0, d_max, n)
        ne, ni = [], []
        for d in ds:
            m = core_mode(Stack.wright2024(t_ln=t0 - d, t_top=t_top0 + d, **stack_kw))
            ne.append(m.n_eff.real)
            ni.append(m.n_eff.imag)
        self.ds = ds
        self.n_re = CubicSpline(ds, ne)
        self.n_im = CubicSpline(ds, np.maximum(ni, 0))

    def etch_for_na(self, na: float) -> float:
        n1 = float(self.n_re(0.0))
        n2 = math.sqrt(n1**2 - na**2)
        return brentq(lambda d: float(self.n_re(d)) - n2, 0, self.ds[-1])


@dataclass
class EtchedGuide:
    width: float                 # top width of the unetched core [um]
    etch_depth: float            # [um]
    sidewall_deg: float = 70.0
    table: LocalIndexTable = None

    @property
    def run(self):
        return self.etch_depth / math.tan(math.radians(self.sidewall_deg))

    def depth_at(self, x, shift_l=0.0, shift_r=0.0):
        """Local etch depth d(x) for edge displacements (positive = wider core)."""
        xl = -self.width / 2 - shift_l
        xr = self.width / 2 + shift_r
        r = max(self.run, 1e-6)
        dl = np.clip((xl - x) / r, 0, 1)
        dr = np.clip((x - xr) / r, 0, 1)
        return self.etch_depth * np.maximum(dl, dr)

    def cell_profile(self, x_cells, dx, shift_l=0.0, shift_r=0.0, sub=64):
        """Cell-averaged (eps-averaged) effective index and loss on a grid.

        ``shift_l`` / ``shift_r`` may be arrays (one per z-step) -> returns [nz, nx].
        Returns (dn, alpha_np) with dn = n_eff - n_core and alpha the amplitude
        attenuation coefficient [1/um] (from Im n_eff).
        """
        sl = np.atleast_1d(shift_l)
        sr = np.atleast_1d(shift_r)
        nz = max(len(sl), len(sr))
        sl = np.broadcast_to(sl, nz)
        sr = np.broadcast_to(sr, nz)
        n1 = float(self.table.n_re(0.0))
        # Only cells near the edges need sub-sampling
        u = (np.arange(sub) + 0.5) / sub - 0.5
        xs = x_cells[:, None] + u[None, :] * dx                   # [nx, sub]
        dn = np.empty((nz, len(x_cells)), np.float64)
        al = np.empty((nz, len(x_cells)), np.float64)
        for k in range(nz):
            d = self.depth_at(xs, sl[k], sr[k])
            ne = self.table.n_re(d)
            dn[k] = np.sqrt((ne**2).mean(1)) - n1
            al[k] = (K0_1550 * self.table.n_im(d)).mean(1)
        return dn.squeeze(0) if nz == 1 else dn, al.squeeze(0) if nz == 1 else al


def check_against_2d(guide: EtchedGuide, n_modes=8, dx=0.04, dy=0.01, margin=4.0):
    """Validate the local-effective-index reduction with a 2D (x, y) mode solve.

    Both sides use the same scalar (E_y-dominant) model of the real etched
    cross-section (BOX / LN / planarised SiO2 / SRN, truncated 1 um into the
    SRN): (a) a full 2D finite-difference eigen-solve, and (b) the reduction
    used by the BPM -- local 1D slab solves n_eff(d(x)), sub-pixel eps-averaged,
    then a 1D lateral eigen-solve.  Returns the lateral-mode offsets
    n_eff,0 - n_eff,m for both; their agreement measures the EIM error in the
    intermodal beat lengths, which is what the programmed coupling relies on.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    k0 = K0_1550
    st = Stack.wright2024()
    t_box, t_ln, t_top = 2.0, 0.7, 1.0
    y1 = t_box + t_ln + t_top + 1.0
    W = guide.width + 2 * guide.run + 2 * margin
    nx, ny = int(W / dx), int(y1 / dy)
    x = (np.arange(nx) + 0.5) * dx - W / 2
    y = (np.arange(ny) + 0.5) * dy
    n_e, n_ox, n_srn = st.layer("LN").n_y.real, 1.444, 2.03

    def column_eps(depth):
        e = np.full(ny, n_ox**2)
        e[(y >= t_box) & (y < t_box + t_ln - depth)] = n_e**2
        e[y >= t_box + t_ln + t_top] = n_srn**2
        return e

    def lap(n, h):
        return sp.diags([1, -2, 1], [-1, 0, 1], (n, n)) / h**2

    # (a) 2D with sub-pixel averaged columns (fine sampling of the slope)
    sub = 16
    u = (np.arange(sub) + 0.5) / sub - 0.5
    eps = np.stack([np.mean([column_eps(dd) for dd in guide.depth_at(xc + u * dx)], 0)
                    for xc in x], 1)
    L = sp.kron(sp.eye(ny), lap(nx, dx)) + sp.kron(lap(ny, dy), sp.eye(nx)) \
        + sp.diags(k0**2 * eps.ravel())
    vals = spla.eigsh(L.tocsc(), k=n_modes, sigma=(k0 * n_e) ** 2, which="LM",
                      return_eigenvectors=False)
    b2 = np.sort(np.sqrt(vals) / k0)[::-1]

    # (b) local 1D slab n_eff(d), then lateral 1D with eps-averaging
    def slab(depth):
        v = spla.eigsh((lap(ny, dy) + sp.diags(k0**2 * column_eps(depth))).tocsc(), k=1,
                       sigma=(k0 * n_e) ** 2, which="LM", return_eigenvectors=False)
        return math.sqrt(v[0]) / k0
    dd = np.linspace(0, guide.etch_depth, 9)
    tab = CubicSpline(dd, [slab(v) for v in dd])
    neff_cells = np.array([np.sqrt(np.mean(tab(guide.depth_at(xc + u * dx)) ** 2)) for xc in x])
    L1 = lap(nx, dx) + sp.diags(k0**2 * neff_cells**2)
    v1 = spla.eigsh(L1.tocsc(), k=n_modes, sigma=(k0 * n_e) ** 2, which="LM",
                    return_eigenvectors=False)
    b1 = np.sort(np.sqrt(v1) / k0)[::-1]
    return b2[0] - b2, b1[0] - b1
