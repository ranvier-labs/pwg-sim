"""Programmable multimode waveguide (PMW) for large unitary / matrix operations.

This is the scaled-up multimode architecture of Onodera et al.: a laterally multimode,
low-loss waveguide whose refractive index is programmed in 2D.  The N guided
modes are the input and output vector basis; programming dn(x, z) within
[0, dn_max] couples them.  Coupled-mode theory (paper Supp. Sec. 8) predicts
that a Haar-random N x N unitary needs only

    L_z ~ sqrt(N) * lambda0 / dn_max

because the rms element of the Hermitian generator -i logm(U) is <= pi/sqrt(N),
versus L ~ N for MZI meshes / crossbars.  :class:`MultimodeWaveguide` sets up
the guided modes (computed with the *same* spectral operator as the BPM so
that static propagation is exactly diagonal) and the differentiable programmed
propagation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import drjit as dr
import numpy as np
from drjit.cuda import Float as FloatNP
from drjit.cuda.ad import Float, UInt32

from . import ops
from .bpm import Grid, Propagator, propagate
from .lateral import EtchedGuide, LocalIndexTable

_TABLE = None


def _table():
    global _TABLE
    if _TABLE is None:
        _TABLE = LocalIndexTable()
    return _TABLE


def next_pow2(n):
    return 1 << int(math.ceil(math.log2(n)))


@dataclass
class MMWGConfig:
    n_modes: int = 32
    length_um: float = 1000.0
    dn_max: float = 5e-3          # paper Fig. 5
    na: float = 0.3               # lateral NA of the multimode waveguide
    dx: float = 0.5
    dz: float = 1.0               # converged for the sharp etched edge (dz=2 leaks ~0.5 dB/cm)
    res_z_um: float = 0.0         # programming pixel length along z (0 -> one pixel per step)
    res_um: float = 1.0           # programming resolution (paper: ~0.5 um; 1 um resolves
                                  # all coupling spatial frequencies < 2 k0 NA)
    wavelength: float = 1.55
    sidewall_deg: float = 70.0    # etched sidewall angle (lateral geometry, pwg.lateral)
    rough_sigma_nm: float = 0.0   # rms sidewall roughness (0 = smooth)
    rough_lc_um: float = 0.5      # roughness correlation length
    rough_seed: int = 0
    spare_modes: int = 4          # guided modes beyond N (keeps the N used well guided)


class MultimodeWaveguide:
    def __init__(self, cfg: MMWGConfig):
        self.cfg = c = cfg
        tab = _table()
        self.table = tab
        n_core = float(tab.n_re(0.0))
        self.etch = tab.etch_for_na(c.na)
        self.dn_clad = float(tab.n_re(self.etch)) - n_core
        n_guided = c.n_modes + c.spare_modes
        self.W = n_guided * c.wavelength / (2 * c.na)       # V-number -> mode count
        self.guide = EtchedGuide(width=self.W, etch_depth=self.etch, sidewall_deg=c.sidewall_deg,
                                 table=tab)
        window = 1.5 * self.W + 60.0
        nx = next_pow2(int(math.ceil(window / c.dx)))
        nz = int(round(c.length_um / c.dz))
        # material/leakage loss enters through the local loss profile alpha(x)
        self.grid = Grid(nx=nx, dx=c.dx, nz=nz, dz=c.dz, wavelength=c.wavelength, n0=n_core,
                         loss_db_per_cm=0.0, absorber_width=0.1)
        x = self.grid.x()
        self.core = np.abs(x) <= self.W / 2
        dn, alpha = self.guide.cell_profile(x, c.dx)
        self.static = dn.astype(np.float32)
        self.amp_x = np.exp(-alpha * c.dz)                  # local loss per step
        self._modes()
        self.prop = Propagator(self.grid, amp_x=self.amp_x)
        self.rough = None
        if c.rough_sigma_nm > 0:
            self.set_roughness(c.rough_sigma_nm, c.rough_lc_um, c.rough_seed)
        # programmable pixels: inside the core only
        self.ix = np.nonzero(self.core)[0].astype(np.uint32)
        self.n_px = len(self.ix)
        self.zrep = max(1, int(round(c.res_z_um / c.dz))) if c.res_z_um > 0 else 1
        self.nz_par = -(-nz // self.zrep)
        self.n_par = self.n_px * self.nz_par

    def set_roughness(self, sigma_nm, lc_um, seed=0):
        """Re-discretise the guide with random sidewall displacements (one chip instance)."""
        from .roughness import edge_process_fast
        g = self.grid
        rng = np.random.default_rng(seed)
        sl = edge_process_fast(g.nz, g.dz, sigma_nm * 1e-3, lc_um, rng)
        sr = edge_process_fast(g.nz, g.dz, sigma_nm * 1e-3, lc_um, rng)
        x = g.x()
        # only cells near the walls change: recompute those, copy the rest
        reach = self.guide.run + 6 * sigma_nm * 1e-3 + 2 * g.dx
        near = np.abs(np.abs(x) - self.W / 2) < reach + self.guide.run
        full = np.tile(self.static, (g.nz, 1))
        dn, _ = self.guide.cell_profile(x[near], g.dx, shift_l=sl, shift_r=sr)
        full[:, near] = dn
        self.rough = (sl, sr)
        self.static_rows = Float(full.astype(np.float32).ravel())
        return self

    def _modes(self):
        """Guided modes as eigenvectors of the *discrete* Strang step operator.

        The BPM computes E_out = prod_k (P_h M_k P_h) E_in, so the eigenvectors of
        S = P_h M_static P_h are propagated exactly invariantly (no splitting
        crosstalk) when nothing is programmed.  They converge to the continuous
        waveguide modes as dz -> 0.
        """
        g, c = self.grid, self.cfg
        nx = g.nx
        F = np.fft.fft(np.eye(nx), axis=0) / math.sqrt(nx)
        Ph = F.conj().T @ (g.transfer(g.dz / 2)[:, None] * F)
        amp = g.absorber() * self.amp_x
        M = amp * np.exp(1j * g.k0 * g.dz * self.static)
        S = Ph @ (M[:, None] * Ph)
        lam, v = np.linalg.eig(S)
        conf = (np.abs(v[self.core]) ** 2).sum(0) / (np.abs(v) ** 2).sum(0)
        ok = (np.abs(lam) > amp.max() * (1 - 1e-4)) & (conf > 0.5) & (np.angle(lam) > g.k0 * g.dz * self.dn_clad)
        cand = np.nonzero(ok)[0]
        cand = cand[np.argsort(-np.angle(lam[cand]))]
        if len(cand) < c.n_modes:
            raise ValueError(f"only {len(cand)} guided modes, need {c.n_modes}")
        order = cand[: c.n_modes]
        self.beta = np.angle(lam[order]) / g.dz          # beta - k n0 per um
        self.mode_decay = np.abs(lam[order])
        modes = v[:, order].T
        # S is not exactly normal (absorber), so orthonormalise for a clean basis
        q, _ = np.linalg.qr(modes.T * math.sqrt(g.dx))
        modes = q.T / math.sqrt(g.dx)
        # Grid-independent phase convention: <ref|psi> real positive with an
        # off-centre reference that overlaps every low-order mode (the "largest
        # peak" convention is ambiguous for antisymmetric modes).
        x = g.x()
        ref = np.exp(-(((x - 0.17 * self.W) / (0.3 * self.W)) ** 2)) * (1 + x / self.W)
        ov = modes @ ref
        modes = modes * np.exp(-1j * np.angle(ov))[:, None]
        self.modes = modes
        self.confinement = (np.abs(modes[:, self.core]) ** 2).sum(1) * g.dx

    # ------------------------------------------------------------------
    def index_map(self, theta: Float) -> Float:
        """theta: [nz_par, n_px] logits -> full dn on the BPM grid (static + programmed).

        Each z-pixel spans ``zrep`` BPM steps (block-constant); in x the pattern is
        blurred by a Gaussian PSF of FWHM ``res_um``."""
        g, c = self.grid, self.cfg
        dn_p = c.dn_max / (1.0 + dr.exp(-theta))
        if self.zrep > 1:
            idx = dr.arange(UInt32, g.nz * self.n_px)
            dn_p = dr.gather(Float, dn_p, (idx // self.n_px // self.zrep) * self.n_px + idx % self.n_px)
        sig = (c.res_um / 2.3548) / g.dx
        dn_p = ops.conv_rows(dn_p, g.nz, self.n_px, ops.gaussian_taps(sig), axis=1)
        # scatter into the full-width grid
        idx = dr.arange(UInt32, g.nz * self.n_px)
        r, j = idx // self.n_px, idx % self.n_px
        full = dr.zeros(Float, g.nz * g.nx)
        dst = r * g.nx + dr.gather(UInt32, UInt32(self.ix), j)
        dr.scatter(full, dn_p, dst)
        if self.rough is not None:
            return full + self.static_rows
        stat = dr.gather(Float, Float(self.static), dr.arange(UInt32, g.nz * g.nx) % g.nx)
        return full + stat

    def transfer(self, dn: Float):
        """Propagate all N input modes; returns output fields (re, im) [N*nx]."""
        e = self.modes.astype(np.complex64)
        return propagate(self.prop, dn, FloatNP(e.real.ravel()), FloatNP(e.imag.ravel()),
                         self.cfg.n_modes)

    def target_fields(self, U: np.ndarray):
        """Target output field for input mode j: sum_i (D U)_ij psi_i, D = free phases."""
        D = np.exp(1j * self.beta * self.grid.length)
        T = D[:, None] * U
        tgt = T.T @ self.modes                      # [N(inputs), nx]
        return tgt.astype(np.complex64), T

    def matrix(self, re: np.ndarray, im: np.ndarray):
        """Realised mode-basis matrix T_ij = <psi_i | out_j>."""
        out = (re + 1j * im).reshape(self.cfg.n_modes, -1)
        return (self.modes.conj() @ out.T) * self.grid.dx


def haar_unitary(n, rng):
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def fidelity(T, Tt):
    n = T.shape[0]
    ov = np.trace(Tt.conj().T @ T)
    F = abs(ov) / n
    Fn = abs(ov) / (math.sqrt(n) * np.linalg.norm(T))
    # mean element error after removing global phase and global transmission
    Tc = T * np.exp(-1j * np.angle(ov)) * math.sqrt(n) / np.linalg.norm(T)
    err = np.abs(Tc - Tt).sum() / np.abs(Tt).sum()
    return F, Fn, err
