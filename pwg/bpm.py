"""Differentiable split-step Fourier beam propagation (2D: x transverse, z propagation).

Physics
-------
We propagate the scalar envelope of the fundamental (TM0) slab mode in the
plane of the slab.  After the effective-index reduction of the vertical stack
the field obeys the one-way Helmholtz equation

    dE/dz = i (sqrt(k0^2 n0^2 + d^2/dx^2) - k0 n0) E + i k0 dn(x, z) E - alpha/2 E

with ``n0`` the slab-mode effective index and ``dn`` the (programmed)
effective-index perturbation.  Diffraction is applied *exactly* in k-space
(no paraxial approximation; this is the "extended Fresnel" propagator used by
Onodera et al.), the index term in real space, Strang-split:

    E_out = P_half  prod_k [ M_k P ]  ...  with  M_k = A exp(i k0 dz dn_k)

``A`` is a super-Gaussian absorbing boundary (transparent, reflection-free
edges) times the uniform propagation loss.  Evanescent spatial frequencies
(|kx| > k0 n0) decay as they physically would.

Gradients
---------
Storing the AD graph of thousands of FFT stages is prohibitively expensive,
so the propagation is wrapped in a ``dr.CustomOp`` whose backward pass runs
the exact discrete adjoint (P^H = F^-1 conj(H) F, M^H = conj(M)) with
sqrt-checkpointing: forward fields are stored every ``K`` steps and recomputed
segment-by-segment during the reverse sweep.  The resulting gradient is exact
for the discretised system (verified against finite differences in tests).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import drjit as dr
import numpy as np
from drjit.cuda import Float, UInt32, TensorXf
from drjit.cuda.ad import Float as FloatAD

from .fft import fft


@dataclass
class Grid:
    """Simulation grid for the in-plane propagation problem.  Lengths in um."""

    nx: int                    # transverse samples (power of two)
    dx: float                  # transverse pitch [um]
    nz: int                    # number of propagation steps
    dz: float                  # step length [um]
    wavelength: float = 1.55   # vacuum wavelength [um]
    n0: float = 1.94           # background effective index (TM0 of the LN slab)
    loss_db_per_cm: float = 0.0
    absorber_width: float = 0.08   # fraction of window on each side used as absorber
    absorber_order: int = 8
    paraxial: bool = False

    @property
    def k0(self) -> float:
        return 2 * math.pi / self.wavelength

    @property
    def width(self) -> float:
        return self.nx * self.dx

    @property
    def length(self) -> float:
        return self.nz * self.dz

    def x(self) -> np.ndarray:
        return (np.arange(self.nx) - self.nx / 2) * self.dx

    def z(self) -> np.ndarray:
        """z position of each index slice (centre of step)."""
        return (np.arange(self.nz) + 0.5) * self.dz

    def kx(self) -> np.ndarray:
        return 2 * math.pi * np.fft.fftfreq(self.nx, d=self.dx)

    def transfer(self, dz: float) -> np.ndarray:
        """Exact (or paraxial) free-propagation transfer function in k-space."""
        k = self.k0 * self.n0
        kx = self.kx()
        if self.paraxial:
            return np.exp(-1j * kx**2 / (2 * k) * dz)
        kz = np.sqrt((k**2 - kx**2).astype(complex))
        return np.exp(1j * (kz - k) * dz)

    def absorber(self) -> np.ndarray:
        x = self.x()
        half = self.width / 2
        w = self.absorber_width * self.width
        if w <= 0:
            return np.ones(self.nx)
        # Smooth super-Gaussian roll-off in the outer ``w`` of each side; the
        # per-step amplitude factor is chosen so that the cumulative damping
        # over ~w/tan(theta) of propagation is strong but locally gentle.
        u = np.clip((np.abs(x) - (half - w)) / w, 0, None)
        return np.exp(-(u**self.absorber_order) * 4.0 * min(1.0, self.dz / 10.0))


def _rowsum(v: Float, batch: int, nx: int) -> Float:
    if batch == 1:
        return v
    return dr.sum(TensorXf(v, shape=(batch, nx)), axis=0).array


class Propagator:
    """Precomputed operators for one grid; runs forward / adjoint sweeps."""

    def __init__(self, grid: Grid, checkpoint_every: int | None = None, amp_x=None):
        """``amp_x``: optional per-x amplitude factor per step (local loss profile)."""
        self.g = grid
        self.K = checkpoint_every or max(1, int(math.sqrt(grid.nz)))
        hf = grid.transfer(grid.dz)
        hh = grid.transfer(grid.dz / 2)
        self.Hf = (Float(hf.real.astype(np.float32)), Float(hf.imag.astype(np.float32)))
        self.Hh = (Float(hh.real.astype(np.float32)), Float(hh.imag.astype(np.float32)))
        amp = grid.absorber() * (1.0 if amp_x is None else np.asarray(amp_x)) * math.exp(-grid.loss_db_per_cm / (20 * math.log10(math.e)) * grid.dz * 1e-4)
        self.A = Float(amp.astype(np.float32))
        dr.eval(self.Hf, self.Hh, self.A)

    # -- elementary operators ------------------------------------------------
    def _kspace(self, re, im, H, conj=False):
        nx = self.g.nx
        re, im = fft(re, im, nx)
        idx = dr.arange(UInt32, dr.width(re)) & UInt32(nx - 1)
        hr = dr.gather(Float, H[0], idx)
        hi = dr.gather(Float, H[1], idx)
        if conj:
            hi = -hi
        re, im = re * hr - im * hi, re * hi + im * hr
        return fft(re, im, nx, inverse=True)

    def _modulate(self, re, im, dn: Float, k: int, conj=False):
        nx = self.g.nx
        xi = dr.arange(UInt32, dr.width(re)) & UInt32(nx - 1)
        a = dr.gather(Float, self.A, xi)
        phi = dr.gather(Float, dn, xi + UInt32(k * nx)) * (self.g.k0 * self.g.dz)
        s, c = dr.sincos(phi)
        if conj:
            s = -s
        c, s = c * a, s * a
        out = (re * c - im * s, re * s + im * c)
        dr.eval(out)
        return out

    def _step(self, re, im, dn, k):
        re, im = self._modulate(re, im, dn, k)
        H = self.Hh if k == self.g.nz - 1 else self.Hf
        return self._kspace(re, im, H)

    # -- sweeps ----------------------------------------------------------------
    def forward(self, dn: Float, re: Float, im: Float, keep_checkpoints=False, record=None):
        """Propagate.  ``record`` (list of z indices) stores |E|^2 snapshots."""
        re, im = self._kspace(re, im, self.Hh)
        ckpt = {}
        for k in range(self.g.nz):
            if keep_checkpoints and k % self.K == 0:
                ckpt[k] = (Float(re), Float(im))
            if record is not None and k in record:
                record[k] = (re * re + im * im).numpy()
            re, im = self._step(re, im, dn, k)
        return re, im, ckpt

    def adjoint(self, dn: Float, ckpt: dict, g_re: Float, g_im: Float, batch: int) -> Float:
        """Return dL/d(dn) (shape nz*nx) given dL/dRe(E_out), dL/dIm(E_out)."""
        nx, nz = self.g.nx, self.g.nz
        scale = self.g.k0 * self.g.dz
        grad = dr.zeros(Float, nz * nx)
        lr, li = Float(g_re), Float(g_im)
        starts = sorted(ckpt.keys())
        for c in reversed(starts):
            end = min(c + self.K, nz)
            re, im = ckpt[c]
            fields = []
            for k in range(c, end):
                re, im = self._modulate(re, im, dn, k)
                fields.append((re, im))
                H = self.Hh if k == nz - 1 else self.Hf
                if k < end - 1:
                    re, im = self._kspace(re, im, H)
            for k in range(end - 1, c - 1, -1):
                H = self.Hh if k == nz - 1 else self.Hf
                lr, li = self._kspace(lr, li, H, conj=True)
                fr, fi = fields[k - c]
                row = _rowsum(li * fr - lr * fi, batch, nx) * scale
                dr.scatter(grad, row, dr.arange(UInt32, nx) + UInt32(k * nx))
                lr, li = self._modulate(lr, li, dn, k, conj=True)
            dr.eval(grad)
        return grad


class _BPMOp(dr.CustomOp):
    def eval(self, dn, prop, ein_re, ein_im, batch):
        self.prop = prop
        self.batch = batch
        self.dn = dr.detach(dn)
        re, im, self.ckpt = prop.forward(Float(self.dn), ein_re, ein_im, keep_checkpoints=True)
        return FloatAD(re), FloatAD(im)

    def forward(self):
        raise NotImplementedError("forward-mode AD not supported; use backward")

    def backward(self):
        g_re, g_im = self.grad_out()
        grad = self.prop.adjoint(Float(self.dn), self.ckpt, Float(dr.detach(g_re)),
                                 Float(dr.detach(g_im)), self.batch)
        self.set_grad_in("dn", FloatAD(grad))

    def name(self):
        return "BPM"


def propagate(prop: Propagator, dn: FloatAD, ein_re, ein_im, batch: int):
    """Differentiable propagation of ``batch`` input fields through index map ``dn``.

    ``dn``: flat ``nz*nx`` effective-index perturbation (AD-enabled Float).
    ``ein_re/ein_im``: flat ``batch*nx`` input field (plain Floats).
    Returns AD Floats (re, im) of the output field.
    """
    if not dr.grad_enabled(dn):
        re, im, _ = prop.forward(Float(dr.detach(dn)), Float(ein_re), Float(ein_im))
        return FloatAD(re), FloatAD(im)
    return dr.custom(_BPMOp, dn, prop, Float(ein_re), Float(ein_im), batch)
