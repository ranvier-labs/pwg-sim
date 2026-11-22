"""Sidewall roughness of the etched multimode waveguide.

Model
-----
Each sidewall is displaced by an independent stationary Gaussian random
process delta(z) with exponential autocorrelation

    R(u) = sigma^2 exp(-|u| / Lc),     S(f) = 2 sigma^2 Lc / (1 + f^2 Lc^2)

(S is the two-sided spectral density in angular spatial frequency f; this is
the standard Payne-Lacey model fitted to AFM/SEM sidewall measurements).

Two complementary treatments
----------------------------
* **BPM (explicit).**  The displaced geometry is re-discretised every step
  with sub-pixel eps-averaging (:mod:`pwg.lateral`), so nm-scale displacements
  enter exactly.  Because the BPM step integrates the index over dz, only
  spatial frequencies |f| < pi/dz are represented.  With dz << Lc (or a fine
  validation run, dz = 0.1 um) the full forward-scattering spectrum is captured.
* **Coupled-mode / Fermi golden rule (analytic).**  To first order in delta,
  power in mode m scatters into mode r at rate

      Gamma_rm = |kappa_rm|^2 S(beta_m - beta_r),
      kappa_rm = k0 * sum_x psi_r*(x) dn(x)/d(delta) psi_m(x) dx,

  summed over both walls.  Summing r over the radiation continuum (box modes
  of a wide window without absorber) gives the radiation loss of each guided
  mode; summing over guided r gives intermodal crosstalk.  Backscattering
  (f ~ 2 beta) is outside a one-way model and is estimated from the same
  formula with a counter-propagating final state.
"""
from __future__ import annotations

import math

import numpy as np


def exponential_psd(f, sigma, lc):
    return 2 * sigma**2 * lc / (1 + (f * lc) ** 2)


def edge_process(n, dz, sigma, lc, rng, fine=8):
    """Sample delta(z) [um] and return its average over each of ``n`` steps of length dz.

    Generated exactly as an AR(1) (Ornstein-Uhlenbeck) process on a fine grid
    (spacing min(dz, lc/fine)), which has precisely the exponential ACF.
    """
    if sigma <= 0:
        return np.zeros(n)
    h = min(dz, lc / fine)
    m = int(math.ceil(dz / h))
    h = dz / m
    a = math.exp(-h / lc)
    w = rng.normal(0, sigma * math.sqrt(1 - a * a), n * m)
    x = np.empty(n * m)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n * m):         # fast enough for ~1e6 samples; vectorise if needed
        x[i] = a * x[i - 1] + w[i]
    return x.reshape(n, m).mean(1)


def edge_process_fast(n, dz, sigma, lc, rng, fine=8):
    """Vectorised equivalent of :func:`edge_process` via scipy.signal.lfilter."""
    from scipy.signal import lfilter
    if sigma <= 0:
        return np.zeros(n)
    h = min(dz, lc / fine)
    m = int(math.ceil(dz / h))
    h = dz / m
    a = math.exp(-h / lc)
    w = rng.normal(0, sigma * math.sqrt(1 - a * a), n * m)
    w[0] = rng.normal(0, sigma)
    x = lfilter([1.0], [1.0, -a], w)
    return x.reshape(n, m).mean(1)


def golden_rule(guide, n_core, x, dx, n_guided, sigma, lc, wavelength=1.55, eps=0.02, sub=2048):
    """Coupled-mode roughness scattering rates for an etched guide on grid ``x``.

    Uses the Hermitian one-way Hamiltonian H = sqrt(k^2 + d_x^2) - k + k0 dn(x)
    (no absorber: radiation is a discretised continuum of box modes; the window
    must be wide enough that S(f) is sampled finely).

    Returns dict with per-guided-mode radiation loss [dB/cm], intermodal
    scattering rate [1/cm], backscatter estimate [dB/cm], and the guided betas.
    """
    k0 = 2 * math.pi / wavelength
    k = k0 * n_core
    nx = len(x)
    kx = 2 * math.pi * np.fft.fftfreq(nx, d=dx)
    kz = np.real(np.sqrt((k**2 - kx**2).astype(complex))) - k
    F = np.fft.fft(np.eye(nx), axis=0) / math.sqrt(nx)
    dn0, _ = guide.cell_profile(x, dx)
    Hm = F.conj().T @ (kz[:, None] * F) + np.diag(k0 * dn0)
    Hm = 0.5 * (Hm + Hm.conj().T)
    w, v = np.linalg.eigh(Hm)
    order = np.argsort(-w)
    w, v = w[order], v[:, order] / math.sqrt(dx)     # normalised: sum |v|^2 dx = 1
    # perturbation from displacing each wall outward by eps (numerical derivative)
    # central difference on a fine sub-grid (cell averages are piecewise linear in
    # the edge position, so this is exact up to the sub-sampling)
    near = np.abs(np.abs(x) - guide.width / 2) < guide.run + eps + 2 * dx
    def prof(sl, sr):
        out = dn0.copy()
        out[near] = guide.cell_profile(x[near], dx, shift_l=sl, shift_r=sr, sub=sub)[0]
        return out
    dl_p, dl_m = prof(eps, 0.0), prof(-eps, 0.0)
    dr_p, dr_m = prof(0.0, eps), prof(0.0, -eps)
    rates = {}
    G = v[:, :n_guided]
    tot_rad = np.zeros(n_guided)
    tot_mm = np.zeros(n_guided)
    back = np.zeros(n_guided)
    beta_abs = w + k                                  # absolute propagation constants
    for dpert in ((dl_p - dl_m) / (2 * eps), (dr_p - dr_m) / (2 * eps)):
        kap = k0 * (v.conj().T @ (dpert[:, None] * G)) * dx      # [all, guided]
        dbeta = w[None, :n_guided] - w[:, None]                    # beta_m - beta_r
        rate = np.abs(kap) ** 2 * exponential_psd(dbeta, sigma, lc)   # [1/um]
        guided_r = np.zeros(len(w), bool)
        guided_r[:n_guided] = True
        rate[np.arange(n_guided), np.arange(n_guided)] = 0
        tot_rad += rate[~guided_r].sum(0)
        tot_mm += rate[guided_r].sum(0)
        # backscatter into the counter-propagating copy of each mode r:
        # phase mismatch beta_m + beta_r (guided only; radiation backward similar order)
        fb = beta_abs[None, :n_guided] + beta_abs[:n_guided, None]
        back += (np.abs(kap[:n_guided]) ** 2 * exponential_psd(fb, sigma, lc)).sum(0)
    to_db_cm = 10 / math.log(10) * 1e4
    return dict(beta=w[:n_guided], rad_db_cm=tot_rad * to_db_cm,
                intermodal_per_cm=tot_mm * 1e4, back_db_cm=back * to_db_cm)
