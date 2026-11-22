"""Shot-noise-driven photonic denoising diffusion in programmable multimode waveguides.

Theory
------
Consider the variance-preserving (VP) forward diffusion SDE on R^d:

    dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW_t,      t in [0, 1],

with stationary Gaussian prior gamma = N(0, I_d) and marginal kernel
q(x_t | x_0) = N(alpha_t x_0, sigma_t^2 I_d), alpha_t^2 + sigma_t^2 = 1.

Writing the reverse-time SDE in terms of the *excess score*
u(x, t) = grad_x log(p_t(x) / gamma(x)) = grad_x log p_t(x) + x:

    dx = [-0.5 * beta(t) * x + beta(t) * u(x, t)] dtau + sqrt(beta(t)) dW_tau,

where tau = 1 - t runs forward from 0 to 1.  In unit-variance v-prediction
coordinates tilde_u(x, t) = (sigma_t / alpha_t) * u(x, t) = E[sigma_t x_0 - alpha_t eps | x_t],
the exact finite-step reverse update (with passive attenuation sqrt(alpha_step) < 1) is

    x_{t_{m-1}} = sqrt(alpha_step) * x_{t_m} + kappa_m * tilde_u(x_{t_m}, t_m) + sigma_rev * xi_m.

When the final programmable multimode waveguide (PMW) stage outputs a coherent
optical field with real quadrature Re(a_out,k) = sqrt(eta_chip) * c_m * tilde_u_k
and is read out by balanced homodyne detection (quantum efficiency eta_QE, local
oscillator amplitude alpha_LO), the balanced photocurrent difference
D_k = n_{+,k} - n_{-,k} (two independent Poisson detectors) yields an unbiased
estimate of tilde_u_k with isotropic vacuum shot-noise variance 1 / (4 * eta_tot * c_m^2)
where eta_tot = eta_chip * eta_QE.

Choosing the purely deterministic laser-amplitude schedule

    c_m = kappa_m / (2 * sqrt(eta_tot) * sigma_rev)  ->  (alpha_t / sigma_t) * sqrt(beta(t) dt / (4 eta_tot))

calibrates the intrinsic quantum shot noise to equal sigma_rev * xi_m *identically*,
requiring zero external random number generation.  Summing the launched useful
signal photons over all steps m = 1..T yields a finite, T-independent continuous-time
limit fixed by the de Bruijn / Fokker-Planck relative-entropy identity:

    E[N_phot,useful] -> (1 / (2 * eta_tot)) * [KL(p_eps || gamma) - KL(p_1 || gamma)].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0


# ---------------------------------------------------------------------------
# 1. VP-SDE schedule and exact Gaussian-mixture analytics
# ---------------------------------------------------------------------------

@dataclass
class VPSchedule:
    """Linear-beta VP-SDE schedule on t in [0, 1]."""

    beta_min: float = 0.1
    beta_max: float = 20.0
    eps: float = 1e-3

    def beta(self, t: np.ndarray | float) -> np.ndarray | float:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def T_int(self, t: np.ndarray | float) -> np.ndarray | float:
        """Integral int_0^t beta(s) ds."""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2)

    def alpha_sigma(self, t: np.ndarray | float):
        """Return (alpha_t, sigma_t) with alpha_t^2 + sigma_t^2 = 1."""
        T = self.T_int(t)
        alpha_sq = np.exp(-T)
        alpha = np.sqrt(alpha_sq)
        sigma = np.sqrt(np.maximum(1.0 - alpha_sq, 1e-12))
        return alpha, sigma

    def time_grid(self, n_steps: int) -> np.ndarray:
        """Descending time grid t_T = 1.0 > ... > t_0 = eps (length n_steps + 1)."""
        return np.linspace(1.0, self.eps, n_steps + 1)

    def step_coeffs(self, t_curr: float, t_next: float, variance_mode: str = "beta"):
        """Return (sqrt_alpha_step, kappa, sigma_rev, alpha_curr, sigma_curr).

        Exact finite-step reverse transition in v-prediction coordinates
        tilde_u = (sigma_t / alpha_t) * u:

            x_{t_next} = sqrt_alpha_step * x_{t_curr} + kappa * tilde_u + sigma_rev * xi.

        ``variance_mode``:
          - ``"beta"``: sigma_rev^2 = beta_step (exact stationary variance for N(0, I)).
          - ``"posterior"``: sigma_rev^2 = beta_step * (1 - alpha_bar_next) / (1 - alpha_bar_curr).
        """
        a_c, s_c = self.alpha_sigma(t_curr)
        a_n, s_n = self.alpha_sigma(t_next)
        ab_c = float(a_c ** 2)
        ab_n = float(a_n ** 2)
        alpha_step = ab_c / ab_n
        beta_step = max(1.0 - alpha_step, 1e-12)
        sqrt_alpha_step = math.sqrt(alpha_step)
        kappa = math.sqrt(ab_n) * beta_step / float(s_c)
        if variance_mode == "beta":
            sigma_rev = math.sqrt(beta_step)
        elif variance_mode == "posterior":
            sigma_rev = math.sqrt(beta_step * max(1.0 - ab_n, 1e-12) / max(1.0 - ab_c, 1e-12))
        else:
            raise ValueError(f"unknown variance_mode: {variance_mode}")
        return sqrt_alpha_step, kappa, sigma_rev, float(a_c), float(s_c)


class GaussianMixtureDistribution:
    """Exact isotropic-component Gaussian mixture p_0(x) = sum_k w_k N(mu_k, s0^2 I)
    with analytical score, excess score, and Monte Carlo KL(p_t || gamma)."""

    def __init__(self, means: np.ndarray, std: float = 0.15, weights: np.ndarray | None = None):
        self.means = np.asarray(means, dtype=np.float64)   # [K, d]
        self.K, self.d = self.means.shape
        self.std = float(std)
        self.weights = np.ones(self.K) / self.K if weights is None else np.asarray(weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()

    @classmethod
    def ring_2d(cls, n_modes: int = 8, radius: float = 1.35, std: float = 0.16) -> "GaussianMixtureDistribution":
        ang = np.linspace(0, 2 * math.pi, n_modes, endpoint=False)
        means = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
        return cls(means, std=std)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(self.K, size=n, p=self.weights)
        return self.means[idx] + rng.normal(0.0, self.std, size=(n, self.d))

    def log_prob_t(self, x: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
        """Exact log p_t(x) for x of shape [B, d]."""
        var_t = (alpha * self.std) ** 2 + sigma ** 2
        diff = x[:, None, :] - alpha * self.means[None, :, :]          # [B, K, d]
        sq = np.sum(diff ** 2, axis=-1)                                # [B, K]
        log_comp = np.log(self.weights)[None, :] - 0.5 * self.d * math.log(2 * math.pi * var_t) - 0.5 * sq / var_t
        m = np.max(log_comp, axis=1, keepdims=True)
        return (m + np.log(np.sum(np.exp(log_comp - m), axis=1, keepdims=True))).squeeze(-1)

    def score_t(self, x: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
        """Exact grad_x log p_t(x) for x of shape [B, d]."""
        var_t = (alpha * self.std) ** 2 + sigma ** 2
        diff = x[:, None, :] - alpha * self.means[None, :, :]          # [B, K, d]
        sq = np.sum(diff ** 2, axis=-1)                                # [B, K]
        logits = np.log(self.weights)[None, :] - 0.5 * sq / var_t
        logits -= np.max(logits, axis=1, keepdims=True)
        resp = np.exp(logits)
        resp /= np.sum(resp, axis=1, keepdims=True)                    # [B, K]
        mean_post = resp @ (alpha * self.means)                        # [B, d]
        return -(x - mean_post) / var_t

    def tilde_u_exact(self, x: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
        """Exact v-prediction target tilde_u(x, t) = (sigma/alpha) * (score_t(x) + x)."""
        u = self.score_t(x, alpha, sigma) + x
        return (sigma / max(alpha, 1e-12)) * u

    def kl_to_gamma(self, alpha: float, sigma: float, n_mc: int = 100_000, seed: int = 0) -> float:
        """Monte Carlo estimate of KL(p_t || N(0, I_d)) in nats."""
        rng = np.random.default_rng(seed)
        x0 = self.sample(n_mc, rng)
        xt = alpha * x0 + sigma * rng.normal(size=x0.shape)
        lp = self.log_prob_t(xt, alpha, sigma)
        lg = -0.5 * self.d * math.log(2 * math.pi) - 0.5 * np.sum(xt ** 2, axis=-1)
        return float(np.mean(lp - lg))


# ---------------------------------------------------------------------------
# 2. Exact quantum balanced-homodyne detector (Poisson dual photodiodes)
# ---------------------------------------------------------------------------

@dataclass
class HomodyneDetector:
    """Physical balanced homodyne detector for coherent optical mode outputs.

    Interferes signal field a_out (in units of sqrt(photons)) with a coherent
    local oscillator alpha_LO (real, phase-locked to the Re quadrature) on a
    50:50 beamsplitter, then samples independent Poisson photoelectron counts
    on the + and - photodiodes plus additive electronic read noise.
    """

    eta_qe: float = 0.8                     # photodiode quantum efficiency
    eta_chip: float = 0.90                  # waveguide chip power transmission
    lo_photons_per_mode: float = 1e5        # |alpha_LO|^2 per mode per step
    read_noise_e: float = 30.0              # rms electronic read noise per photodiode
    wavelength_m: float = 1.55e-6
    exact_poisson: bool = True              # True: exact Pois(+)/Pois(-); False: ideal Gaussian

    @property
    def eta_tot(self) -> float:
        return self.eta_qe * self.eta_chip

    @property
    def alpha_lo(self) -> float:
        return math.sqrt(self.lo_photons_per_mode)

    @property
    def photon_energy_J(self) -> float:
        return H_PLANCK * C_LIGHT / self.wavelength_m

    def measure_real_quadrature(
        self,
        a_out: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict]:
        """Measure Re(a_out) (in sqrt(photons)) via exact balanced homodyne detection.

        Parameters
        ----------
        a_out : complex or real ndarray of shape [..., d]
            Optical output field *after* chip transmission (so |a_out|^2 are photons
            arriving at the beamsplitter per mode).

        Returns
        -------
        x_hat : real ndarray of shape [..., d]
            Unbiased estimate of Re(a_out) with vacuum noise variance
            (1 + |a_out|^2 / alpha_LO^2 + 2 * read_noise^2 / (eta_qe * alpha_LO^2)) / (4 * eta_qe).
        info : dict
            Diagnostics: launched/detected signal photons, LO photons, excess noise factor.
        """
        a_out = np.asarray(a_out)
        a_re = np.real(a_out).astype(np.float64)
        a_im = np.imag(a_out).astype(np.float64) if np.iscomplexobj(a_out) else np.zeros_like(a_re)
        alo = self.alpha_lo

        if self.exact_poisson:
            lam_p = 0.5 * self.eta_qe * ((alo + a_re) ** 2 + a_im ** 2)
            lam_m = 0.5 * self.eta_qe * ((alo - a_re) ** 2 + a_im ** 2)
            # Safe Poisson sampling (handles both small and large mean counts cleanly)
            n_p = rng.poisson(np.clip(lam_p, 0.0, None)).astype(np.float64)
            n_m = rng.poisson(np.clip(lam_m, 0.0, None)).astype(np.float64)
            if self.read_noise_e > 0.0:
                n_p += rng.normal(0.0, self.read_noise_e, size=n_p.shape)
                n_m += rng.normal(0.0, self.read_noise_e, size=n_m.shape)
            diff = n_p - n_m
            x_hat = diff / (2.0 * self.eta_qe * alo)
        else:
            # Ideal strong-LO quantum-limited Gaussian homodyne
            var = 1.0 / (4.0 * self.eta_qe)
            x_hat = a_re + rng.normal(0.0, math.sqrt(var), size=a_re.shape)

        sig_phot_arr = float(np.mean(np.sum(a_re ** 2 + a_im ** 2, axis=-1)))
        useful_phot_arr = float(np.mean(np.sum(a_re ** 2, axis=-1)))
        mean_mode_phot = float(np.mean(a_re ** 2 + a_im ** 2))
        excess_lo = mean_mode_phot / max(self.lo_photons_per_mode, 1e-12)
        excess_read = 2.0 * (self.read_noise_e ** 2) / max(self.eta_qe * self.lo_photons_per_mode, 1e-12)
        info = dict(
            arriving_photons=sig_phot_arr,
            launched_photons=sig_phot_arr / self.eta_chip,
            useful_launched_photons=useful_phot_arr / self.eta_chip,
            lo_photons=float(self.lo_photons_per_mode * a_re.shape[-1]),
            excess_lo_var_frac=excess_lo,
            excess_read_var_frac=excess_read,
        )
        return x_hat, info

    def calibrated_amplitude(self, kappa: float, sigma_rev: float) -> float:
        """Exact deterministic pulse amplitude c_m so homodyne shot noise equals sigma_rev."""
        return kappa / (2.0 * math.sqrt(self.eta_tot) * max(sigma_rev, 1e-12))


# ---------------------------------------------------------------------------
# 3. Unitary-constrained complex photonic score network (pure NumPy + exact AD)
# ---------------------------------------------------------------------------

def sinusoidal_time_embedding(t: np.ndarray, dim: int) -> np.ndarray:
    """Standard sinusoidal time embedding of shape [B, dim] for t in [0, 1]."""
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    half = dim // 2
    freqs = np.exp(np.linspace(0.0, math.log(1000.0), max(half, 1)))[None, :]
    ang = 2.0 * math.pi * t * freqs
    emb = np.concatenate([np.sin(ang), np.cos(ang)], axis=1)
    if dim % 2 == 1:
        emb = np.concatenate([emb, t], axis=1)
    return emb[:, :dim]


def cayley_unitary(A: np.ndarray, phi: np.ndarray):
    """Map unconstrained complex matrix A [N, N] and phases phi [N] to exact U in U(N).

    K = 0.5 * (A - A^H) is skew-Hermitian; U_0 = (I - K)(I + K)^{-1} is unitary;
    U = diag(exp(i phi)) @ U_0 covers all of U(N) smoothly.
    Returns (U, cache).
    """
    N = A.shape[0]
    eye = np.eye(N, dtype=np.complex128)
    K = 0.5 * (A - A.conj().T)
    IpK = eye + K
    ImK = eye - K
    # Solve (I + K)^T X^T = (I - K)^T  <=>  U_0 = (I - K) @ inv(I + K)
    U0 = np.linalg.solve(IpK.T, ImK.T).T
    d_phase = np.exp(1j * phi)
    U = d_phase[:, None] * U0
    return U, (K, U0, d_phase)


def cayley_unitary_backward(G_U: np.ndarray, cache):
    """Exact backward pass for cayley_unitary: dL = Re Tr(G_U^H dU).

    Returns (G_A, G_phi) matching central finite differences to machine precision.
    """
    K, U0, d_phase = cache
    N = K.shape[0]
    eye = np.eye(N, dtype=np.complex128)
    # U = diag(d_phase) @ U0
    # dL = Re sum_{ij} conj(G_U[i,j]) * (i * d_phase[i] * U0[i,j] * dphi[i] + d_phase[i] * dU0[i,j])
    G_U0 = d_phase.conj()[:, None] * G_U
    G_phi = np.real(np.sum(G_U.conj() * (1j * d_phase[:, None] * U0), axis=1))
    # dU0 = -(I + U0) dK (I + K)^{-1}
    # P = -(I + U0^H) G_U0 (I - K)^{-1}
    # Solve (I - K)^T P^T = (-(I + U0^H) G_U0)^T
    rhs = -(eye + U0.conj().T) @ G_U0
    P = np.linalg.solve((eye - K).T, rhs.T).T
    G_A = 0.5 * (P - P.conj().T)
    return G_A, G_phi


def _silu_fwd_bwd(x: np.ndarray):
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))
    y = x * sig
    dy = sig * (1.0 + x * (1.0 - sig))
    return y, dy


def _tanh_fwd_bwd(x: np.ndarray):
    y = np.tanh(x)
    return y, 1.0 - y * y


def _square_fwd_bwd(z_re: np.ndarray, z_im: np.ndarray):
    """Pure optical intensity (photodiode) nonlinearity with soft saturation:
    y = |z|^2 / (1 + 0.25 |z|^2) - 0.5  (real output modulating next stage)."""
    r2 = z_re * z_re + z_im * z_im
    denom = 1.0 + 0.25 * r2
    y = r2 / denom - 0.5
    d_r2 = 1.0 / (denom * denom)
    return y, 2.0 * z_re * d_r2, 2.0 * z_im * d_r2


@dataclass
class PhotonicNetConfig:
    data_dim: int = 64              # dimension d of diffusion state x_t
    n_modes: int = 64               # waveguide guided modes N >= d
    n_layers: int = 4               # number of physical PMW stages
    time_dim: int = 32              # sinusoidal time-embedding dimension
    n_classes: int = 0              # 0 = unconditional; >0 = class-conditional
    nonlinearity: str = "silu"      # "silu", "tanh", or "square" (pure optical |z|^2)
    eta_chip: float = 0.90          # single-pass PMW power transmission
    ancilla_reg: float = 0.0        # optional penalty on unmeasured output quadrature power
    seed: int = 42


class UnitaryScoreNet:
    """Multi-stage complex unitary photonic score network modeling cascaded N-mode PMWs.

    Every linear mixing stage is an exact complex unitary matrix U_l in U(N)
    scaled by sqrt(eta_chip) (passive physical waveguide transmission).  After
    training, each U_l is directly compiled into a physical refractive-index map
    dn_l(x, z) in [0, dn_max] on a ``pwg.mmwg.MultimodeWaveguide``.
    """

    def __init__(self, cfg: PhotonicNetConfig):
        self.cfg = cfg
        assert cfg.n_modes >= cfg.data_dim
        rng = np.random.default_rng(cfg.seed)
        N, L, td = cfg.n_modes, cfg.n_layers, cfg.time_dim
        cond_dim = td + cfg.n_classes
        self.cond_dim = cond_dim

        # Unconstrained Cayley parameters A_l (complex N x N) and phases phi_l (real N)
        self.A = [
            (rng.normal(0.0, 0.35 / math.sqrt(N), size=(N, N))
             + 1j * rng.normal(0.0, 0.35 / math.sqrt(N), size=(N, N)))
            for _ in range(L)
        ]
        self.phi = [rng.uniform(-0.2, 0.2, size=(N,)) for _ in range(L)]

        # Input conditioning projection onto the imaginary quadrature (and any unused real modes)
        self.W_in_im = rng.normal(0.0, 0.5 / math.sqrt(cond_dim), size=(cond_dim, N))
        self.b_in_im = np.zeros(N)
        self.in_scale = np.full(1, math.sqrt(N / max(cfg.data_dim, 1)), dtype=np.float64)

        # Hidden-layer coherent time-conditioned bias, O-E-O channel gain, and residual gain
        self.W_t_re = [rng.normal(0.0, 0.3 / math.sqrt(cond_dim), size=(cond_dim, N)) for _ in range(L - 1)]
        self.W_t_im = [rng.normal(0.0, 0.3 / math.sqrt(cond_dim), size=(cond_dim, N)) for _ in range(L - 1)]
        self.b_re = [np.zeros(N) for _ in range(L - 1)]
        self.b_im = [np.zeros(N) for _ in range(L - 1)]
        self.act_gain = [np.full(N, 1.25) for _ in range(L - 1)]
        self.res_gain = [np.full(N, 0.5) for _ in range(L - 1)]

        # Final scalar readout scale (absorbed physically into calibrated pulse energy c_m)
        self.out_gain = np.array([0.25], dtype=np.float64)
        self.out_bias = np.zeros(cfg.data_dim)

        # Optional override: compiled physical BPM transfer matrices T_l [N, N]
        self.compiled_T: list[np.ndarray] | None = None
        self.compiled_eta: list[float] | None = None

    def _cond_vec(self, t: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        e = sinusoidal_time_embedding(t, self.cfg.time_dim)
        if self.cfg.n_classes > 0:
            oh = np.zeros((e.shape[0], self.cfg.n_classes), dtype=np.float64)
            if y is not None:
                oh[np.arange(e.shape[0]), np.asarray(y, dtype=int)] = 1.0
            e = np.concatenate([e, oh], axis=1)
        return e

    def get_unitaries(self) -> list[np.ndarray]:
        return [cayley_unitary(A, ph)[0] for A, ph in zip(self.A, self.phi)]

    def forward(
        self,
        x: np.ndarray,
        t: np.ndarray,
        y: np.ndarray | None = None,
        return_cache: bool = False,
        override_T: list[np.ndarray] | None = None,
        hidden_noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        """Forward pass returning tilde_u [B, d] (and optional cache/diagnostics)."""
        cfg = self.cfg
        B = x.shape[0]
        N, d, L = cfg.n_modes, cfg.data_dim, cfg.n_layers
        sqrt_eta = math.sqrt(cfg.eta_chip)
        cond = self._cond_vec(t, y)

        # Encode x_t in Re(a_in[:d]) and conditioning in Im(a_in)
        h_re = np.zeros((B, N), dtype=np.float64)
        h_re[:, :d] = self.in_scale[0] * x
        h_im = cond @ self.W_in_im + self.b_in_im[None, :]
        h = h_re + 1j * h_im

        Ts = override_T if override_T is not None else self.compiled_T
        unitary_caches = []
        layer_caches = []

        for l in range(L):
            if Ts is not None:
                T_l = Ts[l]
                unitary_caches.append(None)
                # Normalize chip loss out of hidden propagation via optical gain;
                # keep exact physical T_l on final output stage.
                z = h @ T_l.T
                z_norm = z / sqrt_eta
            else:
                U_l, ucache = cayley_unitary(self.A[l], self.phi[l])
                unitary_caches.append(ucache)
                z_norm = h @ U_l.T
                z = sqrt_eta * z_norm

            if l < L - 1:
                if hidden_noise_std > 0.0 and rng is not None:
                    z_norm = z_norm + (
                        rng.normal(0.0, hidden_noise_std, size=z_norm.shape)
                        + 1j * rng.normal(0.0, hidden_noise_std, size=z_norm.shape)
                    )
                bre = cond @ self.W_t_re[l] + self.b_re[l][None, :]
                bim = cond @ self.W_t_im[l] + self.b_im[l][None, :]
                u_re = self.act_gain[l][None, :] * z_norm.real + bre
                u_im = self.act_gain[l][None, :] * z_norm.imag + bim

                if cfg.nonlinearity == "silu":
                    a_re, d_re = _silu_fwd_bwd(u_re)
                    a_im, d_im = _silu_fwd_bwd(u_im)
                    act = a_re + 1j * a_im
                    act_cache = (d_re, d_im)
                elif cfg.nonlinearity == "tanh":
                    a_re, d_re = _tanh_fwd_bwd(u_re)
                    a_im, d_im = _tanh_fwd_bwd(u_im)
                    act = a_re + 1j * a_im
                    act_cache = (d_re, d_im)
                elif cfg.nonlinearity == "square":
                    # Pure optical intensity nonlinearity: photodiode reads |u|^2,
                    # drives real amplitude of next stage; phase bias stays in Im.
                    a_re, dr_re, dr_im = _square_fwd_bwd(u_re, u_im)
                    act = a_re + 1j * u_im * 0.25
                    act_cache = (dr_re, dr_im)
                else:
                    raise ValueError(f"unknown nonlinearity: {cfg.nonlinearity}")

                h_next = h + self.res_gain[l][None, :] * act
                layer_caches.append((h, z_norm, act, act_cache))
                h = h_next
            else:
                # Final physical PMW output field (pre-pulse-energy scaling)
                a_raw = z                                      # [B, N], includes sqrt(eta_chip)
                out = (self.out_gain[0] / sqrt_eta) * a_raw[:, :d].real + self.out_bias[None, :]
                layer_caches.append((h, z_norm, a_raw, None))

        if not return_cache:
            return out
        return out, dict(x=x, cond=cond, unitary_caches=unitary_caches, layer_caches=layer_caches)

    def loss_and_grads(
        self,
        x: np.ndarray,
        t: np.ndarray,
        target: np.ndarray,
        y: np.ndarray | None = None,
    ) -> tuple[float, dict]:
        """Compute MSE v-prediction loss (+ optional ancilla power reg) and exact gradients."""
        cfg = self.cfg
        B = x.shape[0]
        N, d, L = cfg.n_modes, cfg.data_dim, cfg.n_layers
        sqrt_eta = math.sqrt(cfg.eta_chip)

        out, cache = self.forward(x, t, y, return_cache=True)
        diff = out - target
        mse = float(np.mean(np.sum(diff ** 2, axis=1)) / d)

        # Optional regularization on unmeasured output power (Im of [:d] and all of [d:])
        h_last, z_last, _, _ = cache["layer_caches"][-1]
        anc_loss = 0.0
        g_zlast = np.zeros_like(z_last)
        # Upstream gradient from MSE: d(mse)/d(out) = 2 * diff / (B * d)
        g_out = (2.0 / (B * d)) * diff
        g_out_bias = np.sum(g_out, axis=0)
        g_out_gain = np.array([np.sum(g_out * z_last[:, :d].real)], dtype=np.float64)
        g_zlast[:, :d] += self.out_gain[0] * g_out

        if cfg.ancilla_reg > 0.0:
            im_part = z_last[:, :d].imag
            extra = z_last[:, d:]
            anc_pow = (np.sum(im_part ** 2) + np.sum(np.abs(extra) ** 2)) / (B * N)
            anc_loss = cfg.ancilla_reg * float(anc_pow)
            scale_a = (2.0 * cfg.ancilla_reg) / (B * N)
            g_zlast[:, :d] += 1j * (scale_a * im_part)
            if N > d:
                g_zlast[:, d:] += scale_a * extra

        total_loss = mse + anc_loss

        grads = dict(
            A=[None] * L,
            phi=[None] * L,
            W_t_re=[None] * (L - 1),
            W_t_im=[None] * (L - 1),
            b_re=[None] * (L - 1),
            b_im=[None] * (L - 1),
            act_gain=[None] * (L - 1),
            res_gain=[None] * (L - 1),
            out_gain=g_out_gain,
            out_bias=g_out_bias,
        )

        # Backward through final layer: z_last = h_last @ U_{L-1}^T
        U_last = cayley_unitary(self.A[L - 1], self.phi[L - 1])[0]
        G_U = g_zlast.T @ h_last.conj()
        grads["A"][L - 1], grads["phi"][L - 1] = cayley_unitary_backward(G_U, cache["unitary_caches"][L - 1])
        g_h = g_zlast @ U_last.conj()

        cond = cache["cond"]
        for l in range(L - 2, -1, -1):
            h_in, z_norm, act, act_cache = cache["layer_caches"][l]
            # h_next = h_in + res_gain[l] * act
            grads["res_gain"][l] = np.real(np.sum(g_h.conj() * act, axis=0))
            g_act = g_h * self.res_gain[l][None, :]

            if cfg.nonlinearity in ("silu", "tanh"):
                d_re, d_im = act_cache
                g_u_re = g_act.real * d_re
                g_u_im = g_act.imag * d_im
            elif cfg.nonlinearity == "square":
                dr_re, dr_im = act_cache
                g_u_re = g_act.real * dr_re
                g_u_im = g_act.real * dr_im + g_act.imag * 0.25
            else:
                raise ValueError(cfg.nonlinearity)

            grads["W_t_re"][l] = cond.T @ g_u_re
            grads["W_t_im"][l] = cond.T @ g_u_im
            grads["b_re"][l] = np.sum(g_u_re, axis=0)
            grads["b_im"][l] = np.sum(g_u_im, axis=0)
            grads["act_gain"][l] = np.sum(g_u_re * z_norm.real + g_u_im * z_norm.imag, axis=0)

            g_z = self.act_gain[l][None, :] * (g_u_re + 1j * g_u_im)
            U_l = cayley_unitary(self.A[l], self.phi[l])[0]
            G_U = g_z.T @ h_in.conj()
            grads["A"][l], grads["phi"][l] = cayley_unitary_backward(G_U, cache["unitary_caches"][l])
            # Residual skip gradient + layer gradient
            g_h = g_h + g_z @ U_l.conj()

        # Input encoding gradients
        grads["W_in_im"] = cond.T @ g_h.imag
        grads["b_in_im"] = np.sum(g_h.imag, axis=0)
        grads["in_scale"] = np.array([np.sum(g_h[:, :d].real * x)], dtype=np.float64)
        return total_loss, grads

    def param_items(self):
        """Yield (key, array) pairs for generic optimizers and serialization."""
        L = self.cfg.n_layers
        for l in range(L):
            yield f"A_{l}", self.A[l]
            yield f"phi_{l}", self.phi[l]
        for l in range(L - 1):
            yield f"W_t_re_{l}", self.W_t_re[l]
            yield f"W_t_im_{l}", self.W_t_im[l]
            yield f"b_re_{l}", self.b_re[l]
            yield f"b_im_{l}", self.b_im[l]
            yield f"act_gain_{l}", self.act_gain[l]
            yield f"res_gain_{l}", self.res_gain[l]
        yield "W_in_im", self.W_in_im
        yield "b_in_im", self.b_in_im
        yield "in_scale", self.in_scale
        yield "out_gain", self.out_gain
        yield "out_bias", self.out_bias

    def grad_items(self, grads: dict):
        L = self.cfg.n_layers
        for l in range(L):
            yield f"A_{l}", grads["A"][l]
            yield f"phi_{l}", grads["phi"][l]
        for l in range(L - 1):
            yield f"W_t_re_{l}", grads["W_t_re"][l]
            yield f"W_t_im_{l}", grads["W_t_im"][l]
            yield f"b_re_{l}", grads["b_re"][l]
            yield f"b_im_{l}", grads["b_im"][l]
            yield f"act_gain_{l}", grads["act_gain"][l]
            yield f"res_gain_{l}", grads["res_gain"][l]
        yield "W_in_im", grads["W_in_im"]
        yield "b_in_im", grads["b_in_im"]
        yield "in_scale", grads["in_scale"]
        yield "out_gain", grads["out_gain"]
        yield "out_bias", grads["out_bias"]

    def save(self, path: str):
        data = {k: v for k, v in self.param_items()}
        for k, v in self.cfg.__dict__.items():
            data[f"cfg_{k}"] = np.array(v)
        if self.compiled_T is not None:
            for l, T in enumerate(self.compiled_T):
                data[f"compiled_T_{l}"] = T
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> "UnitaryScoreNet":
        z = np.load(path, allow_pickle=False)
        cfg = PhotonicNetConfig(
            data_dim=int(z["cfg_data_dim"]),
            n_modes=int(z["cfg_n_modes"]),
            n_layers=int(z["cfg_n_layers"]),
            time_dim=int(z["cfg_time_dim"]),
            n_classes=int(z["cfg_n_classes"]),
            nonlinearity=str(z["cfg_nonlinearity"]),
            eta_chip=float(z["cfg_eta_chip"]),
            ancilla_reg=float(z["cfg_ancilla_reg"]),
            seed=int(z["cfg_seed"]),
        )
        net = cls(cfg)
        for k, arr in net.param_items():
            arr[...] = z[k]
        if "compiled_T_0" in z:
            net.compiled_T = [z[f"compiled_T_{l}"] for l in range(cfg.n_layers)]
        return net

    def forward_optical_field(
        self,
        x: np.ndarray,
        t: np.ndarray,
        y: np.ndarray | None = None,
        override_T: list[np.ndarray] | None = None,
        hidden_noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (a_final_unit [B, N], out_bias [d]) where a_final_unit is the
        unscaled physical output field of the final PMW stage (including sqrt(eta_chip))
        such that tilde_u = (out_gain / sqrt(eta_chip)) * Re(a_final_unit[:, :d]) + out_bias.

        When the optical pulse amplitude into the final PMW is scaled by c_m * out_gain,
        the physical output field arriving at the balanced homodyne receiver is
        a_out = c_m * out_gain * a_final_unit, with
        Re(a_out[:, :d]) = sqrt(eta_chip) * c_m * (tilde_u - out_bias).
        """
        _, cache = self.forward(
            x, t, y, return_cache=True, override_T=override_T,
            hidden_noise_std=hidden_noise_std, rng=rng,
        )
        _, _, a_raw, _ = cache["layer_caches"][-1]
        return a_raw, self.out_bias


class ComplexAdam:
    """Standard Adam optimizer supporting both real and complex NumPy parameter arrays."""

    def __init__(self, net: UnitaryScoreNet, lr: float = 2e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.net = net
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_idx = 0
        self.m = {k: np.zeros_like(v) for k, v in net.param_items()}
        self.v = {k: np.zeros(v.shape, dtype=np.float64) for k, v in net.param_items()}

    def step(self, grads: dict, lr: float | None = None, grad_clip: float = 5.0):
        self.step_idx += 1
        lr_now = self.lr if lr is None else lr
        # Global norm clip for training stability
        glist = list(self.net.grad_items(grads))
        gnorm = math.sqrt(sum(float(np.sum(np.abs(g) ** 2)) for _, g in glist))
        scale = min(1.0, grad_clip / max(gnorm, 1e-12))

        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** self.step_idx
        bc2 = 1.0 - b2 ** self.step_idx

        params = dict(self.net.param_items())
        for k, g in glist:
            g = g * scale
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * (np.abs(g) ** 2)
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            params[k] -= lr_now * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# 4. Shot-noise-driven reverse diffusion samplers (Exact Poisson + Digital ref)
# ---------------------------------------------------------------------------

def sample_photonic_diffusion(
    score_fn: Callable,
    schedule: VPSchedule,
    detector: HomodyneDetector,
    shape: tuple[int, int],
    n_steps: int = 50,
    rng: np.random.Generator | None = None,
    mode: str = "homodyne_residual",
    variance_mode: str = "beta",
    drift_std_per_step: float = 0.0,
    drift_corr: float = 0.95,
    record_steps: list[int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Run reverse-time denoising diffusion driven purely by quantum shot noise.

    Parameters
    ----------
    score_fn : callable
        ``score_fn(x, t_vec)`` returning:
          - For ``mode="digital"`` or ``mode="homodyne_fullstate"``: ``tilde_u`` [B, d].
          - For ``mode="homodyne_residual"``: either ``(a_final_unit [B, N], out_bias [d], out_gain)``
            or plain ``tilde_u`` [B, d] (ideal unitary limit with zero ancilla overhead).
    mode : str
        - ``"homodyne_residual"``: Architecture B — excess-score residual update with
          deterministic pulse schedule c_m = kappa_m / (2 sqrt(eta_tot) sigma_rev).
          Zero external Gaussian RNG; all diffusion noise comes from Poisson balanced
          homodyne photocounts!
        - ``"homodyne_fullstate"``: Architecture A — full-state predictor mu_theta read
          out directly through balanced homodyne detection (photon budget scales O(T^2)).
        - ``"digital"``: Exact digital PRNG reference using identical score_fn.
    drift_std_per_step : float
        Optional slow correlated analog drift (AR(1) process with correlation ``drift_corr``)
        perturbing the score output across reverse steps.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    B, d = shape
    ts = schedule.time_grid(n_steps)
    # Initial prior x_T ~ N(0, I_d) (can itself be obtained from a dark/vacuum homodyne read!)
    if mode.startswith("homodyne"):
        # Vacuum homodyne read (a_out = 0) produces N(0, 1/(4 eta_qe)) -> rescale by 2 sqrt(eta_qe)
        v_raw, _ = detector.measure_real_quadrature(np.zeros((B, d), dtype=np.complex128), rng)
        x = 2.0 * math.sqrt(detector.eta_qe) * v_raw
    else:
        x = rng.normal(0.0, 1.0, size=(B, d))

    total_launched_photons = 0.0
    total_useful_photons = 0.0
    total_lo_photons = 0.0
    per_step_photons = []
    per_step_useful = []
    excess_lo_list = []
    excess_read_list = []
    traj = {}
    if record_steps is not None and 0 in record_steps:
        traj[0] = x.copy()

    drift = np.zeros((B, d), dtype=np.float64)
    sqrt_eta = math.sqrt(detector.eta_chip)

    for m in range(n_steps):
        t_c = float(ts[m])
        t_n = float(ts[m + 1])
        t_vec = np.full(B, t_c, dtype=np.float64)
        sqrt_a, kappa, sigma_rev, a_c, s_c = schedule.step_coeffs(t_c, t_n, variance_mode=variance_mode)

        res = score_fn(x, t_vec)
        if isinstance(res, tuple):
            a_unit, out_bias, out_gain = res
            tilde_u_clean = (out_gain / sqrt_eta) * a_unit[:, :d].real + out_bias[None, :]
        else:
            tilde_u_clean = np.asarray(res, dtype=np.float64)
            a_unit = sqrt_eta * tilde_u_clean.astype(np.complex128)
            out_bias = np.zeros(d)
            out_gain = 1.0

        # Optional slow correlated temporal drift
        if drift_std_per_step > 0.0:
            innov_std = drift_std_per_step * math.sqrt(max(1.0 - drift_corr ** 2, 1e-12))
            drift = drift_corr * drift + rng.normal(0.0, innov_std, size=drift.shape)
            tilde_u_clean = tilde_u_clean + drift
            a_unit = a_unit.copy()
            a_unit[:, :d] += (sqrt_eta / out_gain) * drift

        if mode == "digital":
            x = sqrt_a * x + kappa * tilde_u_clean + sigma_rev * rng.normal(size=(B, d))
            per_step_photons.append(0.0)
            per_step_useful.append(0.0)
        elif mode == "homodyne_residual":
            c_m = detector.calibrated_amplitude(kappa, sigma_rev)
            # Physical optical field arriving at the balanced homodyne array
            a_out = c_m * out_gain * a_unit                   # [B, N], in sqrt(photons)
            # Measure the first d modes via exact Poisson balanced homodyne
            meas_re, info = detector.measure_real_quadrature(a_out[:, :d], rng)
            # Include any power in unused modes [d:] in total launched photons
            extra_phot = float(np.mean(np.sum(np.abs(a_out[:, d:]) ** 2, axis=-1))) / detector.eta_chip if a_out.shape[1] > d else 0.0
            tilde_u_hat = meas_re / (sqrt_eta * c_m) + out_bias[None, :]
            x = sqrt_a * x + kappa * tilde_u_hat
            step_launched = info["launched_photons"] + extra_phot
            total_launched_photons += step_launched
            total_useful_photons += info["useful_launched_photons"]
            total_lo_photons += info["lo_photons"]
            per_step_photons.append(step_launched)
            per_step_useful.append(info["useful_launched_photons"])
            excess_lo_list.append(info["excess_lo_var_frac"])
            excess_read_list.append(info["excess_read_var_frac"])
        elif mode == "homodyne_fullstate":
            # Architecture A: waveguide directly outputs full state mu = sqrt_a * x + kappa * tilde_u
            mu = sqrt_a * x + kappa * tilde_u_clean
            c_full = 1.0 / (2.0 * math.sqrt(detector.eta_tot) * max(sigma_rev, 1e-12))
            a_out = sqrt_eta * c_full * mu.astype(np.complex128)
            meas_re, info = detector.measure_real_quadrature(a_out, rng)
            x = meas_re / (sqrt_eta * c_full)
            total_launched_photons += info["launched_photons"]
            total_useful_photons += info["useful_launched_photons"]
            total_lo_photons += info["lo_photons"]
            per_step_photons.append(info["launched_photons"])
            per_step_useful.append(info["useful_launched_photons"])
            excess_lo_list.append(info["excess_lo_var_frac"])
            excess_read_list.append(info["excess_read_var_frac"])
        else:
            raise ValueError(f"unknown sampling mode: {mode}")

        if record_steps is not None and (m + 1) in record_steps:
            traj[m + 1] = x.copy()

    # Final deterministic Tweedie cleanup at t = eps -> 0 (negligible energy, high SNR)
    a_e, s_e = schedule.alpha_sigma(schedule.eps)
    res_f = score_fn(x, np.full(B, schedule.eps))
    if isinstance(res_f, tuple):
        a_u, ob, og = res_f
        tu_f = (og / sqrt_eta) * a_u[:, :d].real + ob[None, :]
    else:
        tu_f = np.asarray(res_f, dtype=np.float64)
    x_final = float(a_e) * x + float(s_e) * tu_f

    stats = dict(
        total_launched_photons=total_launched_photons,
        total_useful_photons=total_useful_photons,
        total_lo_photons=total_lo_photons,
        signal_energy_J=total_launched_photons * detector.photon_energy_J,
        useful_energy_J=total_useful_photons * detector.photon_energy_J,
        lo_energy_J=total_lo_photons * detector.photon_energy_J,
        dilation_overhead=total_launched_photons / max(total_useful_photons, 1e-12),
        per_step_photons=per_step_photons,
        per_step_useful=per_step_useful,
        max_excess_lo_var_frac=max(excess_lo_list) if excess_lo_list else 0.0,
        mean_excess_read_var_frac=float(np.mean(excess_read_list)) if excess_read_list else 0.0,
        traj=traj,
    )
    return x_final, stats


# ---------------------------------------------------------------------------
# 5. Physical BPM compilation of trained unitaries onto MultimodeWaveguide
# ---------------------------------------------------------------------------

def compile_unitary_to_bpm(
    U_target: np.ndarray,
    na: float = 0.45,
    length_um: float | None = None,
    dn_max: float = 5e-3,
    res_um: float = 1.0,
    iters: int = 300,
    lr: float = 0.15,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Compile a complex N x N unitary matrix U_target into a physical
    refractive-index map dn(x, z) in [0, dn_max] on a shallow-etched
    MgO:LN/SRN ``MultimodeWaveguide`` using exact split-step BPM adjoints.

    Requires CUDA Dr.Jit / Mitsuba (imported lazily so CPU tests stay portable).
    """
    import time
    import drjit as dr
    import mitsuba as mi
    from drjit.cuda.ad import Float
    from .mmwg import MMWGConfig, MultimodeWaveguide, fidelity

    if mi.variant() is None:
        mi.set_variant("cuda_ad_mono")

    N = U_target.shape[0]
    if length_um is None:
        # Use our calibrated two-limit law with ~15% safety margin for F_n >= 0.99
        lam, n_eff = 1.55, 1.95
        db_approx = (2 * math.pi / lam) * na ** 2 / (2 * n_eff) * (N / (N + 4)) ** 2
        L_est = 0.29 * math.sqrt(1.5 * math.log(max(N, 2))) * math.sqrt(N) * lam / dn_max + 0.33 * math.pi * N / db_approx
        length_um = float(round(1.18 * L_est / 10.0) * 10.0)

    cfg = MMWGConfig(n_modes=N, length_um=length_um, dn_max=dn_max, na=na, res_um=res_um)
    wg = MultimodeWaveguide(cfg)
    tgt, Tt = wg.target_fields(U_target)
    tgt_re = Float(tgt.real.ravel())
    tgt_im = Float(tgt.imag.ravel())

    rng = np.random.default_rng(seed)
    opt = dr.opt.Adam(lr=lr)
    opt["theta"] = Float(rng.normal(0.0, 0.2, size=wg.n_par).astype(np.float32))

    t0 = time.time()
    best_Fn = -1.0
    best_theta = None
    for it in range(iters):
        dn = wg.index_map(opt["theta"])
        re, im = wg.transfer(dn)
        # Global-phase-invariant field overlap loss (same as unitary_scaling.py)
        ov_re = dr.sum(tgt_re * re + tgt_im * im)
        ov_im = dr.sum(tgt_re * im - tgt_im * re)
        ov2 = ov_re * ov_re + ov_im * ov_im
        norm_out = dr.sum(re * re + im * im)
        norm_tgt = float(np.sum(np.abs(tgt) ** 2))
        loss = 1.0 - ov2 / (norm_out * norm_tgt + 1e-12)
        dr.backward(loss)
        opt.step()
        if (it + 1) % 50 == 0 or it == iters - 1:
            T_cur = wg.matrix(re.numpy(), im.numpy())
            F_cur, Fn_cur, err_cur = fidelity(T_cur, Tt)
            if Fn_cur > best_Fn:
                best_Fn = float(Fn_cur)
                best_theta = opt["theta"].numpy().copy()
            if verbose:
                print(f"    [BPM compile N={N} L={length_um:.0f}um] it {it+1:3d}/{iters}: "
                      f"F={F_cur:.4f}, F_n={Fn_cur:.5f}, err={err_cur:.4f} ({time.time()-t0:.1f}s)", flush=True)

    # Evaluate final compiled matrix with best theta and remove the static
    # waveguide free-propagation phase D = diag(exp(i beta L)) and global phase
    opt["theta"] = Float(best_theta)
    dn_final = wg.index_map(opt["theta"])
    re, im = wg.transfer(dn_final)
    T_raw = wg.matrix(re.numpy(), im.numpy())
    F_f, Fn_f, err_f = fidelity(T_raw, Tt)
    # Remove known static output phases D = exp(i beta L) and scalar global phase
    D = np.exp(1j * wg.beta * wg.grid.length)
    ov = np.trace(Tt.conj().T @ T_raw)
    gphase = np.exp(-1j * np.angle(ov))
    T_eff = (D.conj()[:, None] * T_raw) * gphase
    eta_actual = float(np.linalg.norm(T_eff, "fro") ** 2 / N)

    return dict(
        theta=best_theta,
        dn=dn_final.numpy().reshape(wg.grid.nz, wg.grid.nx),
        T_raw=T_raw,
        T_eff=T_eff,
        D_phase=D,
        global_phase=complex(gphase),
        F=float(F_f),
        Fn=float(Fn_f),
        err=float(err_f),
        eta_actual=eta_actual,
        cfg=cfg.__dict__,
        W_um=float(wg.W),
        L_um=float(length_um),
    )


def evaluate_compiled_bpm(
    theta: np.ndarray,
    cfg_dict: dict,
    D_phase: np.ndarray,
    global_phase: complex,
    rough_sigma_nm: float = 0.0,
    rough_lc_um: float = 0.2,
    rough_seed: int = 0,
    res_um_override: float | None = None,
    extra_loss_db_per_cm: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Re-evaluate a compiled physical waveguide under hardware perturbations
    (sidewall roughness, finite programming resolution, or excess SRN cavity loss)."""
    import drjit as dr
    import mitsuba as mi
    from drjit.cuda.ad import Float
    from .mmwg import MMWGConfig, MultimodeWaveguide

    if mi.variant() is None:
        mi.set_variant("cuda_ad_mono")

    c = dict(cfg_dict)
    c["rough_sigma_nm"] = rough_sigma_nm
    c["rough_lc_um"] = rough_lc_um
    c["rough_seed"] = rough_seed
    if res_um_override is not None:
        c["res_um"] = res_um_override
    wg = MultimodeWaveguide(MMWGConfig(**c))
    dn = wg.index_map(Float(theta.astype(np.float32)))
    re, im = wg.transfer(dn)
    T_raw = wg.matrix(re.numpy(), im.numpy())
    if extra_loss_db_per_cm > 0.0:
        att = 10.0 ** (-(extra_loss_db_per_cm * (wg.grid.length * 1e-4)) / 20.0)
        T_raw = T_raw * att
    T_eff = (D_phase.conj()[:, None] * T_raw) * global_phase
    eta_actual = float(np.linalg.norm(T_eff, "fro") ** 2 / wg.cfg.n_modes)
    return T_eff, eta_actual
