"""End-to-end model of the 2D-programmable waveguide prototype.

    DMD duty cycles  --(Mitsuba thin-lens projection + Airy blur)-->  irradiance I(x, z)
    I(x, z)          --(SRN photoconductor / series-impedance / Pockels)-->  dn_eff local
    dn_eff local     --(electrostatic fringing LSF)-->  dn_eff(x, z)
    dn_eff(x, z)     --(split-step BPM of the TM0 slab mode)-->  output field
    output field     --(4f relay + camera bins, shot/read noise)-->  measured outputs

Every stage is differentiable in Dr.Jit, so the DMD pattern can be trained
end-to-end (in-silico training), and the same model with perturbed physical
parameters serves as the "hardware" for robustness tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import drjit as dr
import numpy as np
from drjit.cuda import Float as FloatNP
from drjit.cuda.ad import Float, UInt32

from . import ops
from .bpm import Grid, Propagator, propagate
from .electro import LocalResponse, lateral_lsf
from .projector import ProjectorOptics
from .stack import Stack, core_mode


@dataclass
class PrototypeConfig:
    # programmable region (paper: 9.1 mm x 1 mm used out of the projected field)
    dmd_rows: int = 111            # across x  (111 * 9.13 um = 1.01 mm)
    dmd_cols: int = 1000           # along z   (1000 * 9.13 um = 9.13 mm)
    supersample: int = 2           # render pixels per DMD pixel
    z_pool: int = 1                # render columns per BPM step
    defocus_um: float = 5.0
    na_proj: float = 0.10
    spp: int = 16
    # electrical operating point
    voltage: float = 1100.0
    freq: float = 10.0
    I_max: float = 57.0            # mW/cm^2 for a fully-on mirror
    # optical grid
    nx: int = 2048
    dx: float = 0.75
    loss_db_per_cm: float | None = None   # None -> from mode solver
    # inputs (paper MNIST: 49 Gaussian beams, w0 = 6 um, pitch 8.2 um)
    n_in: int = 49
    w0_in: float = 6.0
    pitch_in: float = 8.2
    # outputs: camera bins at the output facet
    n_out: int = 10
    out_span: float = 600.0        # um covered by the bins
    out_fill: float = 0.8          # fraction of each bin pitch that is integrated
    use_mitsuba: bool = True


class PrototypeDevice:
    def __init__(self, cfg: PrototypeConfig = PrototypeConfig(), stack: Stack | None = None,
                 verbose=True, sigma: tuple | None = None):
        """``sigma=(sigma_d, sigma_ph)`` fixes the SRN conductivities (e.g. to model a
        fabricated device that differs from the one the photoconductor was calibrated on);
        by default they are fitted to the paper's measured dark/bright index changes."""
        self.cfg = c = cfg
        self.stack = stack or Stack.wright2024()
        mode = core_mode(self.stack)
        self.n_eff = mode.n_eff.real
        self.mode_loss = mode.loss_db_per_cm
        self.resp = LocalResponse(self.stack, voltage=c.voltage, freq=c.freq)
        if sigma is None:
            # calibrate at the paper's characterisation point, then operate at cfg values
            self.resp.calibrate()
        else:
            self.resp.sigma_d, self.resp.sigma_ph = sigma
        _, _, self.es_fwhm = lateral_lsf(self.resp)
        self.proj = ProjectorOptics(n_rows=c.dmd_rows, n_cols=c.dmd_cols, na_chip=c.na_proj,
                                    defocus_um=c.defocus_um, supersample=c.supersample)
        if c.use_mitsuba:
            self.proj.build()
        rp = self.proj.render_pixel_um
        dz = rp * c.z_pool
        nz = c.dmd_cols * c.supersample // c.z_pool
        loss = self.mode_loss if c.loss_db_per_cm is None else c.loss_db_per_cm
        self.grid = Grid(nx=c.nx, dx=c.dx, nz=nz, dz=dz, n0=self.n_eff, loss_db_per_cm=loss)
        self.prop = Propagator(self.grid)
        # in-plane blur kernels on the BPM grid
        self.sig_opt = self.proj.diffraction_sigma_um()
        self.sig_es = self.es_fwhm / 2.3548
        self.x = self.grid.x()
        H = c.dmd_rows * self.proj.chip_pixel_um
        self.x_render = -H / 2 + (np.arange(c.dmd_rows * c.supersample) + 0.5) * rp
        self._inputs()
        self._outputs()
        if verbose:
            print(f"[device] n_eff={self.n_eff:.4f}  loss={loss:.3f} dB/cm  Gamma={self.resp.Gamma:.3f}"
                  f"  sigma_d={self.resp.sigma_d:.2e} sigma_ph={self.resp.sigma_ph:.2e} S/m"
                  f"  ES-LSF={self.es_fwhm:.2f} um  grid {c.nx}x{nz} (dz={dz:.2f} um, L={self.grid.length/1e3:.2f} mm)"
                  f"  dn_prog,max={self.dn_prog_max():.2e}")

    # ------------------------------------------------------------------ I/O
    def _inputs(self):
        c = self.cfg
        xs = (np.arange(c.n_in) - (c.n_in - 1) / 2) * c.pitch_in
        self.x_in = xs
        e = np.exp(-((self.x[None, :] - xs[:, None]) / c.w0_in) ** 2)
        e /= np.sqrt((e**2).sum(1, keepdims=True) * self.grid.dx)   # unit power each
        self.ein = e.astype(np.float32)

    def _outputs(self):
        c = self.cfg
        pitch = c.out_span / c.n_out
        centres = (np.arange(c.n_out) - (c.n_out - 1) / 2) * pitch
        half = pitch * c.out_fill / 2
        self.out_masks = np.stack([(np.abs(self.x - x0) <= half) for x0 in centres]).astype(np.float32)
        npx = int(self.out_masks.sum(1).min())
        # Pixel indices per bin, equal count per bin (for block reductions)
        self.out_idx = np.concatenate([np.nonzero(m)[0][:npx] for m in self.out_masks]).astype(np.uint32)
        self.out_px = npx

    def dn_prog_max(self):
        return float(self.resp.dn_eff(self.cfg.I_max) - self.resp.dn_eff(0.0))

    # ---------------------------------------------------------- index map
    def irradiance(self, pattern: Float, spp=None, seed=0) -> Float:
        """Relative chip irradiance on the [render z, render x] grid (flat)."""
        c = self.cfg
        R, C = c.dmd_rows * c.supersample, c.dmd_cols * c.supersample
        if c.use_mitsuba:
            img = self.proj.render(pattern, spp=spp or c.spp, seed=seed).array   # [R, C] x-major
        else:  # ideal nearest-neighbour projection (fast digital model)
            idx = dr.arange(UInt32, R * C)
            r, col = idx // C, idx % C
            img = dr.gather(Float, pattern, (r // c.supersample) * c.dmd_cols + col // c.supersample)
        return ops.transpose(img, R, C)                                           # [C, R] z-major

    def index_map(self, pattern: Float, spp=None, seed=0) -> Float:
        """Programmed effective-index perturbation on the BPM grid ([nz, nx] flat)."""
        c, g = self.cfg, self.grid
        img = self.irradiance(pattern, spp, seed)
        C, R = c.dmd_cols * c.supersample, c.dmd_rows * c.supersample
        img = ops.pool_rows(img, C, R, c.z_pool)
        I = ops.resample_rows(img, g.nz, self.x_render, self.x, fill=0.0)
        I = ops.conv_rows(I, g.nz, g.nx, ops.gaussian_taps(self.sig_opt / g.dx), axis=1)
        dn = self.resp.dn_eff_dr(I * c.I_max)
        dn = ops.conv_rows(dn, g.nz, g.nx, ops.gaussian_taps(self.sig_es / g.dx), axis=1)
        dn = ops.conv_rows(dn, g.nz, g.nx, ops.gaussian_taps(self.sig_es / g.dz), axis=0)
        return dn

    # ----------------------------------------------------------- forward
    def transfer(self, dn: Float):
        """Output fields for each input basis beam: (re, im) AD arrays [n_in * nx]."""
        e = self.ein.ravel()
        return propagate(self.prop, dn, FloatNP(e), FloatNP(np.zeros_like(e)), self.cfg.n_in)

    def output_matrix(self, re: Float, im: Float):
        """Restrict to camera-bin pixels: [n_in, n_out*px] (re, im)."""
        nb = self.cfg.n_out * self.out_px
        idx = dr.arange(UInt32, self.cfg.n_in * nb)
        i, p = idx // nb, idx % nb
        src = i * self.grid.nx + dr.gather(UInt32, UInt32(self.out_idx), p)
        return dr.gather(Float, re, src), dr.gather(Float, im, src)

    def bin_powers(self, Ur: Float, Ui: Float, X: np.ndarray) -> Float:
        """Detected power per output bin for a batch of input vectors ``X`` [S, n_in].

        Returns flat [S, n_out] (units: input power fraction).
        """
        S, n_in = X.shape
        nb = self.cfg.n_out * self.out_px
        idx = dr.arange(UInt32, S * nb)
        s, p = idx // nb, idx % nb
        Xd = FloatNP(X.astype(np.float32).ravel())
        er = dr.zeros(Float, S * nb)
        ei = dr.zeros(Float, S * nb)
        for i in range(n_in):
            xi = Float(dr.gather(FloatNP, Xd, s * n_in + i))
            er += xi * dr.gather(Float, Ur, i * nb + p)
            ei += xi * dr.gather(Float, Ui, i * nb + p)
        I = (er * er + ei * ei) * self.grid.dx
        return dr.block_sum(I, self.out_px)
