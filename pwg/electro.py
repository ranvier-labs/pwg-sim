"""Photoconductive electro-optic index programming.

A sinusoidal bias V cos(wt) (~1 kV, 10-26 Hz) is applied between the top Au
electrode and the grounded Si substrate.  The four dielectric layers form a
series impedance; illuminating the Si-rich-nitride (SRN) photoconductor
raises its conductivity, shunting its capacitance so that more of the voltage
drops across the thin LN core.  The Pockels effect (r33, field along the
c-axis) turns the LN field into an index change

    dn_LN = 1/2 n_e^3 r33 |E_LN|,   dn_eff = Gamma * dn_LN

where Gamma = d n_eff / d n_e is the optical overlap from :mod:`pwg.stack`.

* :class:`LocalResponse`  -- 1D (laterally uniform) AC circuit, numpy + Dr.Jit.
  The SRN photoconductivity follows sigma(I) = sigma_d + sigma_ph (I/I_ref)^gamma;
  sigma_d and sigma_ph are calibrated to the paper's measured dark / bright
  index changes (4.0e-3 / 5.2e-3 at 1100 V, 10 Hz, 50 mW/cm^2).
* :func:`lateral_lsf`  -- 2D (x, y) electro-quasistatic finite-difference solve
  of div((sigma + i w eps0 eps) grad phi) = 0 with anisotropic LN permittivity,
  giving the line-spread function that limits the in-plane resolution of the
  programmed index (paper: ~3.5 um FWHM electrostatic limit).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.optimize as so
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .stack import Stack, eo_overlap

EPS0 = 8.8541878128e-12


@dataclass
class LocalResponse:
    stack: Stack
    voltage: float = 1100.0     # amplitude [V]
    freq: float = 10.0          # [Hz]
    r33: float = 27e-12         # [m/V]  value used in the paper (Supp. 1.3)
    sigma_d: float = 0.0        # SRN dark conductivity [S/m]
    sigma_ph: float = 0.0       # SRN photoconductivity at I_ref [S/m]
    I_ref: float = 50.0         # [mW/cm^2]
    gamma: float = 0.8          # photoconductive exponent (a-SiN:H typically 0.5-1)
    Gamma: float = None         # optical EO overlap; computed if None

    def __post_init__(self):
        if self.Gamma is None:
            self.Gamma = eo_overlap(self.stack)
        names = ["BOX", "LN", "TOX", "SRN"]
        self._layers = [self.stack.layer(n) for n in names]

    # --- numpy reference ------------------------------------------------------
    def sigma_srn(self, I):
        I = np.asarray(I, float)
        return self.sigma_d + self.sigma_ph * (np.maximum(I, 0) / self.I_ref) ** self.gamma

    def ln_field(self, I, voltage=None, freq=None):
        """Complex amplitude of the field in the LN core [V/m]."""
        V = self.voltage if voltage is None else voltage
        w = 2 * math.pi * (self.freq if freq is None else freq)
        z = 0
        for l in self._layers:
            sig = self.sigma_srn(I) if l.name == "SRN" else l.sigma
            z = z + (l.thickness * 1e-6) / (sig + 1j * w * EPS0 * l.eps_dc_y)
        ln = self._layers[1]
        z_ln = (ln.thickness * 1e-6) / (1j * w * EPS0 * ln.eps_dc_y)
        return V * (z_ln / z) / (ln.thickness * 1e-6)

    def dn_ln(self, I, **kw):
        ne = self.stack.layer("LN").n_y.real
        return 0.5 * ne**3 * self.r33 * np.abs(self.ln_field(I, **kw))

    def dn_eff(self, I, **kw):
        return self.Gamma * self.dn_ln(I, **kw)

    def calibrate(self, dn_dark=4.0e-3, dn_bright=5.2e-3, I_bright=50.0, V=1100.0, f=10.0):
        """Fit sigma_d, sigma_ph to measured *material* LN index changes (paper Supp. 1.3)."""
        def res(p):
            self.sigma_d, self.sigma_ph = float(10 ** p[0]), float(10 ** p[1])
            return [self.dn_ln(0.0, voltage=V, freq=f) / dn_dark - 1,
                    self.dn_ln(I_bright, voltage=V, freq=f) / dn_bright - 1]
        sol = so.least_squares(res, x0=[-9.5, -8.0], bounds=([-14, -12], [-4, -2]))
        self.sigma_d, self.sigma_ph = float(10 ** sol.x[0]), float(10 ** sol.x[1])
        return self

    # --- Dr.Jit (differentiable) version -------------------------------------
    def dn_eff_dr(self, I, offset_dark=True):
        """Dr.Jit version of :meth:`dn_eff` for an AD array of irradiance [mW/cm^2].

        If ``offset_dark`` the (uniform) dark-state index change is subtracted, so
        the result is the *programmable* index dn_bright - dn_dark >= 0.
        """
        import drjit as dr
        w = 2 * math.pi * self.freq
        I = dr.maximum(I, 0.0)
        sig = self.sigma_d + self.sigma_ph * dr.power(I * (1.0 / self.I_ref) + 1e-9, self.gamma)
        zr, zi = 0.0, 0.0
        for l in self._layers:
            t = l.thickness * 1e-6
            b = w * EPS0 * l.eps_dc_y
            s = sig if l.name == "SRN" else l.sigma
            den = s * s + b * b
            zr = zr + t * s / den
            zi = zi - t * b / den
        ln = self._layers[1]
        t_ln = ln.thickness * 1e-6
        zln_i = -t_ln / (w * EPS0 * ln.eps_dc_y)       # purely capacitive
        mag = abs(zln_i) / dr.sqrt(zr * zr + zi * zi)  # |V_LN/V|
        ne = self.stack.layer("LN").n_y.real
        dn = self.Gamma * 0.5 * ne**3 * self.r33 * self.voltage * mag / t_ln
        if offset_dark:
            dn = dn - float(self.dn_eff(0.0))
        return dn


def lateral_lsf(resp: LocalResponse, I_bright=50.0, half_width=40.0, dx=0.1, dy=0.05):
    """Edge response of E_y in the LN core to a dark/bright step in illumination.

    Returns ``(x, lsf, fwhm)`` where ``lsf`` is the normalised line-spread
    function (derivative of the edge response).
    """
    st = resp.stack
    names = ["BOX", "LN", "TOX", "SRN"]
    layers = [st.layer(n) for n in names]
    edges = np.cumsum([0] + [l.thickness for l in layers])
    ny = int(round(edges[-1] / dy))
    nx = int(round(2 * half_width / dx))
    x = -half_width + (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    w = 2 * math.pi * resp.freq
    kx = np.zeros((ny, nx), complex)
    ky = np.zeros((ny, nx), complex)
    for i, l in enumerate(layers):
        m = (y >= edges[i]) & (y < edges[i + 1])
        if l.name == "SRN":
            sig = np.where(x > 0, resp.sigma_srn(I_bright), resp.sigma_srn(0.0))[None, :]
        else:
            sig = l.sigma
        kx[m] = sig + 1j * w * EPS0 * l.eps_dc_x
        ky[m] = sig + 1j * w * EPS0 * l.eps_dc_y
    hm = lambda a, b: 2 * a * b / (a + b)
    idx = np.arange(nx * ny).reshape(ny, nx)
    rows, cols, vals = [], [], []
    rhs = np.zeros(nx * ny, complex)
    diag = np.zeros(nx * ny, complex)
    # x-faces (Neumann at the lateral edges)
    cx = hm(kx[:, :-1], kx[:, 1:]) / dx**2
    a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    rows += [a, b]; cols += [b, a]; vals += [cx.ravel(), cx.ravel()]
    np.add.at(diag, a, -cx.ravel()); np.add.at(diag, b, -cx.ravel())
    # y-faces
    cy = hm(ky[:-1], ky[1:]) / dy**2
    a, b = idx[:-1].ravel(), idx[1:].ravel()
    rows += [a, b]; cols += [b, a]; vals += [cy.ravel(), cy.ravel()]
    np.add.at(diag, a, -cy.ravel()); np.add.at(diag, b, -cy.ravel())
    # Dirichlet: phi=0 below (Si ground, half a cell), phi=V on top electrode
    cb = ky[0] / (dy**2 / 2)
    np.add.at(diag, idx[0], -cb)
    ct = ky[-1] / (dy**2 / 2)
    np.add.at(diag, idx[-1], -ct)
    rhs[idx[-1]] -= ct * resp.voltage
    rows.append(np.arange(nx * ny)); cols.append(np.arange(nx * ny)); vals.append(diag)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nx * ny, nx * ny))
    phi = spla.spsolve(A.tocsc(), rhs).reshape(ny, nx)
    ln = (y >= edges[1]) & (y < edges[2])
    ey = -np.gradient(phi, dy, axis=0)[ln].mean(axis=0)
    edge = np.abs(ey)
    edge = (edge - edge[:20].mean()) / (edge[-20:].mean() - edge[:20].mean())
    lsf = np.gradient(edge, dx)
    lsf = np.clip(lsf, 0, None)
    lsf /= lsf.sum() * dx
    above = x[lsf >= lsf.max() / 2]
    fwhm = above.max() - above.min() + dx
    return x, lsf, fwhm
