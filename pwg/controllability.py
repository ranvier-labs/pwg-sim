"""Linear controllability of a programmable multimode waveguide.

Why does the required length deviate from the sqrt(N) law?  To first order
(coupled-mode theory, interaction picture, first Magnus term) the programmed
perturbation dn(x, z) produces the Hermitian generator

    A_kl = k0 \\int dz \\int dx  E_k(x) E_l(x) e^{i (beta_k - beta_l) z} dn(x, z),
    U    = exp(i A)                     (in the frame co-rotating with the modes).

This is a *linear* map J from the programmable pixels theta (dn = B theta, B the
optical/electrostatic blur, block-constant z pixels of length p_z) to the N^2
real coordinates of A.  The minimum-norm pattern that realises the generator
A = -i logm(U) of a target unitary is theta* = J^T (J J^T)^{-1} a, so

    rms(dn*)^2 = a^T G^{-1} a / (#pixels),     G = J J^T  (N^2 x N^2 Gram matrix).

G factorises exactly into a transverse and a longitudinal part,

    G_{rr'} = k0^2 dx^2 [F B B^T F^T]_{pp'}  *  [h h^T]_{rr'}

with F_p(x) = E_k E_l (pair p = (k, l)) and h_r(z_b) the cos / sin of the beat
phase integrated over z-pixel b.  Two regimes follow:

* **Resolved (G well conditioned):** all pair beat notes are separated by more
  than 2 pi / L, G ~ k0^2 L / W * I, and rms(dn*) = ||A||_F / (k0 L) =
  pi sqrt(N/3) / (k0 L)  -> the paper's L ~ sqrt(N) lambda / dn_max.
* **Unresolved:** the number of independent controls is only about
  (2N transverse profiles) x (Delta beta_max L / pi resolvable beat
  frequencies), where Delta beta_max ~ k0 NA^2 / 2n is the spread of mode
  propagation constants.  Until that reaches N^2 the Gram matrix has small
  eigenvalues and the required index blows up; this sets
  L_dof ~ pi N / Delta beta_max ~ N n lambda / NA^2, linear in N.

The crossover N* = (NA^2 / (n dn_max))^2 (~85 for NA 0.3, dn_max 5e-3) lies
inside the simulated range, which is why the measured exponent is ~0.66.

The model has one free constant kappa: the achievable rms(dn) / dn_max of a
pattern confined to [0, dn_max] (uniform: 0.29, bang-bang: 0.5).
"""
from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sla


def _real_modes(wg):
    m = wg.modes
    ph = np.exp(-1j * np.angle(m[np.arange(len(m)), np.argmax(np.abs(m), 1)]))
    r = np.real(m * ph[:, None])
    return r / np.sqrt((r**2).sum(1, keepdims=True) * wg.grid.dx)


def _blur_matrix(n, sigma_px):
    """Gaussian blur with clamped edges, as in pwg.ops.conv_rows (rows: output)."""
    from .ops import gaussian_taps
    taps = gaussian_taps(sigma_px)
    r = len(taps) // 2
    B = np.zeros((n, n))
    for t, w in zip(range(-r, r + 1), taps):
        j = np.clip(np.arange(n) + t, 0, n - 1)
        np.add.at(B, (np.arange(n), j), w)
    return B


class Controllability:
    """Precompute the transverse part for a waveguide; evaluate rms(dn*) vs L."""

    def __init__(self, wg, res_um=None):
        self.wg = wg
        g = wg.grid
        self.k0 = g.k0
        self.dx = g.dx
        N = wg.cfg.n_modes
        self.N = N
        E = _real_modes(wg)[:, wg.ix]                         # core pixels only
        res = wg.cfg.res_um if res_um is None else res_um
        B = _blur_matrix(len(wg.ix), (res / 2.3548) / g.dx)
        kk, ll = np.triu_indices(N)
        self.pairs = np.stack([kk, ll], 1)
        F = E[kk] * E[ll]                                    # [P, n_px]
        FB = F @ B                                            # dn = B theta -> J ~ F B
        self.X = (FB @ FB.T) * (self.k0 * self.dx) ** 2       # [P, P]
        self.dbeta = wg.beta[kk] - wg.beta[ll]
        # real DOF rows: (pair, cos) for all pairs, (pair, sin) for k < l
        off = kk != ll
        self.row_pair = np.concatenate([np.arange(len(kk)), np.nonzero(off)[0]])
        self.row_sin = np.concatenate([np.zeros(len(kk), bool), np.ones(off.sum(), bool)])
        self.n_px = len(wg.ix)

    def h(self, L, pz):
        """[R, nb] longitudinal factors: block integrals of cos / sin(dbeta z)."""
        nb = max(1, int(round(L / pz)))
        zb = (np.arange(nb) + 0.5) * (L / nb)
        d = self.dbeta[self.row_pair][:, None]
        w = (L / nb) * np.sinc(d * (L / nb) / (2 * np.pi))  # np.sinc(x) = sin(pi x)/(pi x)
        ph = d * zb[None, :]
        return np.where(self.row_sin[:, None], np.sin(ph), np.cos(ph)) * w, nb

    def target(self, U):
        """Real coordinates of the minimal generator of U (global phase removed)."""
        w, V = np.linalg.eig(U)
        ph = np.angle(w)
        ph = ph - np.angle(np.mean(np.exp(1j * ph)))          # optimal global phase
        ph = (ph + np.pi) % (2 * np.pi) - np.pi
        A = (V * ph) @ np.linalg.inv(V)
        A = 0.5 * (A + A.conj().T)
        kk, ll = self.pairs[:, 0], self.pairs[:, 1]
        a = A[kk, ll]
        return np.concatenate([a.real, a[kk != ll].imag])

    def rms_dn(self, L, Us, pz=None, ridge=1e-12):
        """Minimum-norm rms(dn) needed for each target in ``Us`` at length L."""
        pz = pz or (self.wg.cfg.res_z_um or self.wg.grid.dz)
        H, nb = self.h(L, pz)
        G = self.X[np.ix_(self.row_pair, self.row_pair)] * (H @ H.T)
        G[np.diag_indices_from(G)] += ridge * np.trace(G) / len(G)
        cf = sla.cho_factor(G, lower=True, check_finite=False)
        out = []
        for U in Us:
            a = self.target(U)
            out.append(math.sqrt(a @ sla.cho_solve(cf, a, check_finite=False) / (self.n_px * nb)))
        return np.array(out)

    def spectrum(self, L, pz=None):
        """Eigenvalues of G normalised by the resolved-regime value k0^2 L/W-ish mean."""
        pz = pz or (self.wg.cfg.res_z_um or self.wg.grid.dz)
        H, _ = self.h(L, pz)
        G = self.X[np.ix_(self.row_pair, self.row_pair)] * (H @ H.T)
        ev = np.linalg.eigvalsh(G)
        return ev[::-1] / np.mean(np.diag(G))

    def required_length(self, dn_max, kappa, Us, L_grid, pz=None):
        """Smallest L with rms(dn*) <= kappa * dn_max (log-interpolated on L_grid)."""
        r = np.array([self.rms_dn(L, Us, pz).mean() for L in L_grid])
        tgt = kappa * dn_max
        for i in range(len(L_grid) - 1):
            if r[i] > tgt >= r[i + 1]:
                t = (math.log(r[i]) - math.log(tgt)) / (math.log(r[i]) - math.log(r[i + 1]))
                return L_grid[i] * (L_grid[i + 1] / L_grid[i]) ** t, r
        return (L_grid[0] if r[0] <= tgt else np.nan), r


def _bisect_log(f, lo, hi, it=80):
    """Root of a monotone decreasing f on [lo, hi] (log-space bisection)."""
    for _ in range(it):
        mid = math.sqrt(lo * hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


class Budgeted(Controllability):
    """Best achievable generator error under an rms(dn) budget (Tikhonov / water-filling).

    minimise ||W^1/2 (J theta - a)||^2  s.t.  rms(theta) <= budget,
    W = 1 on diagonal rows and 2 on off-diagonal rows so that the error equals
    ||dA||_F^2.  For a unitary error exp(i dA):  1 - F_norm ~ ||dA||_F^2 / (2N).
    """

    def eig(self, L, pz=None):
        pz = pz or (self.wg.cfg.res_z_um or self.wg.grid.dz)
        H, nb = self.h(L, pz)
        w = np.where(self.pairs[self.row_pair, 0] == self.pairs[self.row_pair, 1], 1.0, 2.0)
        sw = np.sqrt(w)
        G = self.X[np.ix_(self.row_pair, self.row_pair)] * (H @ H.T)
        G = sw[:, None] * G * sw[None, :]
        g, V = np.linalg.eigh(G)
        return np.clip(g, 0, None), V, sw, nb

    def infidelity(self, L, U, budget, pz=None, eigs=None):
        g, V, sw, nb = eigs if eigs is not None else self.eig(L, pz)
        alpha2 = (V.T @ (sw * self.target(U))) ** 2
        P = self.n_px * nb

        def excess(lam):     # theta^2/P - budget^2, decreasing in lam
            return (alpha2 * g / (g + lam) ** 2).sum() / P - budget**2
        if excess(1e-30 * g.max()) <= 0:
            resid2 = float(alpha2[g <= 1e-14 * g.max()].sum())
        else:
            lam = _bisect_log(excess, 1e-30 * g.max(), 1e6 * g.max())
            resid2 = float((alpha2 * lam**2 / (g + lam) ** 2).sum())
        return min(1.0, resid2 / (2 * self.N))
