"""All-optical nonlinear cavity denoising diffusion in programmable multimode waveguides.

Physics
-------
This module implements TWO progressively more integrated all-optical photonic
diffusion architectures that eliminate intermediate electronic (O-E-O) conversions:

1. **All-Optical Hidden-Nonlinearity Cavity (1 balanced-homodyne readout per step)**
   Replaces all (L - 1) intermediate O-E-O hidden activations with passive all-optical
   nonlinearities---chi^(3) Kerr self-phase modulation and/or saturable absorption---
   inside a low-loss optical feedback cavity built around 1 shared (or M_rt cascaded)
   programmable multimode waveguide(s):

       a^(0) = 0
       for k = 1 .. M_rt:
           a_cav^(k) = r * exp(i * phi_loop) * a_nl^(k-1) + t_in * a_in(x_t, t, y)
           z^(k)     = T_PMW^(k) @ a_cav^(k)
           a_nl^(k)  = Phi_opt(z^(k) + 0.35 * b_pump(t, y))

   with strictly passive, BIBO-stable optical transfer function (|Phi_opt(z)| <= |z|):

       Phi_opt(z) = exp( i * gamma * |z|^2  -  (alpha_0 / 2) / (1 + |z|^2 / I_sat) ) * z.

   Inside the cavity, the pulse circulates at its nonlinear working energy E_cav ~ N E_NL;
   at the output, a scalar variable tap tau_m << 1 extracts the calibrated sub-photon field
   into ONE balanced-homodyne receiver per reverse step, suppressing intracavity Kerr
   squeezing/excess noise by tau_m^2 while injecting pristine vacuum shot noise.

2. **Folded Single-Process All-Optical Cavity (ZERO intermediate O-E-O across ALL T steps;
   ONLY ONE final readout at t = 0)**
   Packs the d real diffusion coordinates into d/2 complex optical modes
   a_x = (x_even + i * x_odd) / sqrt(2) (eliminating quadrature projection!) circulating
   uninterruptedly for all T reverse round trips inside a passive optical storage ring.
   In log-SNR time s(t) = int_0^t beta(u) du discretized uniformly into T steps
   Delta_s = (s_max - s_min) / T, EVERY round trip has the exact SAME constant passive
   power transmission alpha_step = exp(-Delta_s) and loss beta_step = 1 - alpha_step.
   When beta_step exceeds the ring waveguide loss (e.g. beta_step = 5.8% = 0.26 dB vs.
   3.4% = 0.15 dB chip loss), the storage ring is 100% PASSIVE and STATIC: its fixed
   passive loss IS the VP-SDE contraction sqrt(1 - beta_step), and its open coupler port
   injects calibrated optical ASE (at macroscopic n_op = E_NL / hbar*omega) or pure
   vacuum (at single-photon n_op = 1/2) providing the exact reverse Brownian kick
   sqrt(beta_step) * zeta_m.  A passive tap feeds a copy of a_x into the all-optical
   Kerr/SA PMW score branch (with modest branch gain G_br compensating passive tap and
   recombiner loss) and coherently recombines kappa_m * tilde_u_C into the storage ring.
   Only ONE dual-quadrature (8-port) balanced-homodyne readout occurs at the very end (t = 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .diffusion import (
    H_PLANCK,
    C_LIGHT,
    HomodyneDetector,
    VPSchedule,
    cayley_unitary,
    cayley_unitary_backward,
    sinusoidal_time_embedding,
)


# ---------------------------------------------------------------------------
# 1. Passive all-optical nonlinearities (exact forward + Wirtinger backward)
# ---------------------------------------------------------------------------

def optical_nonlinearity_fwd(
    z: np.ndarray,
    gamma: float,
    alpha0: float,
    i_sat: float = 1.0,
) -> tuple[np.ndarray, tuple]:
    """Passive Kerr + saturable-absorber optical transfer function:

        w = exp(i * gamma * |z|^2 - (alpha0 / 2) / (1 + |z|^2 / i_sat)) * z.

    Returns (w, cache).
    """
    r2 = z.real * z.real + z.imag * z.imag
    denom = 1.0 + r2 / i_sat
    phi = gamma * r2
    lam = (0.5 * alpha0) / denom
    exp_f = np.exp(-lam) * (np.cos(phi) + 1j * np.sin(phi))
    w = exp_f * z
    # f'(r^2) = i * gamma + (alpha0 / (2 * i_sat)) / (1 + r^2 / i_sat)^2
    df_re = (0.5 * alpha0 / i_sat) / (denom * denom)
    df_im = gamma
    return w, (z, w, exp_f, r2, denom, df_re, df_im)


def optical_nonlinearity_bwd(
    g_w: np.ndarray,
    cache: tuple,
) -> tuple[np.ndarray, float, float]:
    """Exact complex backward pass for optical_nonlinearity_fwd:
        dL = Re Tr(g_w^H dw).

    Returns (g_z, g_gamma, g_alpha0).
    """
    z, w, exp_f, r2, denom, df_re, df_im = cache
    gw_w = g_w.conj() * w
    re_term = gw_w.real * df_re - gw_w.imag * df_im
    g_z = exp_f.conj() * g_w + (2.0 * re_term) * z
    g_gamma = float(np.sum(-r2 * gw_w.imag))
    g_alpha0 = float(np.sum((-0.5 / denom) * gw_w.real))
    return g_z, g_gamma, g_alpha0


# ---------------------------------------------------------------------------
# 2. Helper: real <-> complex mode packing & uniform-s VP time grid
# ---------------------------------------------------------------------------

def pack_real_to_complex(x: np.ndarray) -> np.ndarray:
    """Pack x [..., d] (d even) into d/2 circularly symmetric complex modes
    a_x = (x_even + i * x_odd) / sqrt(2), so x ~ N(0, I_d) <=> a_x ~ CN(0, I_{d/2})."""
    return (x[..., 0::2] + 1j * x[..., 1::2]) / math.sqrt(2.0)


def unpack_complex_to_real(a: np.ndarray) -> np.ndarray:
    """Inverse of pack_real_to_complex: a [..., d/2] -> x [..., d]."""
    out = np.empty(a.shape[:-1] + (2 * a.shape[-1],), dtype=np.float64)
    s2 = math.sqrt(2.0)
    out[..., 0::2] = s2 * a.real
    out[..., 1::2] = s2 * a.imag
    return out


def uniform_s_time_grid(
    schedule: VPSchedule,
    n_steps: int,
    s_min: float | None = None,
    s_max: float | None = None,
) -> np.ndarray:
    """Return descending t-grid of length n_steps + 1 with strictly uniform spacing
    in integrated noise s(t) = int_0^t beta(u) du.

    Every step has IDENTICAL passive round-trip contraction alpha_step = exp(-Delta_s)
    and loss beta_step = 1 - exp(-Delta_s), enabling a 100% static passive storage ring.
    """
    b0, b1 = schedule.beta_min, schedule.beta_max
    db = b1 - b0
    s0 = float(schedule.T_int(schedule.eps)) if s_min is None else float(s_min)
    s1 = float(schedule.T_int(1.0)) if s_max is None else float(s_max)
    s_grid = np.linspace(s1, s0, n_steps + 1)
    if abs(db) < 1e-12:
        t_grid = s_grid / b0
    else:
        t_grid = (-b0 + np.sqrt(np.maximum(b0 * b0 + 2.0 * db * s_grid, 0.0))) / db
    return np.clip(t_grid, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3. All-optical nonlinear cavity score network
# ---------------------------------------------------------------------------

@dataclass
class OpticalCavityConfig:
    data_dim: int = 64
    n_modes: int = 64
    n_roundtrips: int = 6             # number of optical cavity round trips M_rt
    time_dim: int = 32
    n_classes: int = 10
    nonlinearity: str = "kerr_sa"     # "kerr" (lossless chi^3), "sa" (saturable absorber), "kerr_sa"
    shared_waveguide: bool = True     # True: 1 shared physical PMW in ring cavity; False: M_rt cascaded PMWs
    complex_state: bool = False       # False: real quadrature I/O (1 read/step); True: d/2 complex modes (folded loop!)
    cavity_feedback_r: float = 0.82   # passive feedback coupler amplitude reflectivity r < 1
    gamma_kerr: float = 0.65          # Kerr nonlinear phase coefficient (rad / unit intensity)
    alpha0_sa: float = 1.6            # unsaturated power absorption of SA (alpha_0; 0 for pure Kerr)
    i_sat: float = 1.0                # SA saturation intensity (in dimensionless cavity units)
    eta_chip: float = 0.966           # single-pass PMW power transmission (2.5 mm LN/SRN @ 0.3 dB/cm)
    ancilla_reg: float = 0.002        # mild penalty on unmeasured output mode power
    seed: int = 42


class OpticalCavityScoreNet:
    """All-optical nonlinear cavity photonic score network (zero intermediate O-E-O).

    Supports two physical operating modes:
      - ``complex_state=False``: real-quadrature score network for 1-readout-per-step
        all-optical hidden cavity diffusion (Request A).
      - ``complex_state=True``: complex-mode score branch (d/2 complex state modes +
        N - d/2 optical conditioning modes, zero electronic bias) for folding ALL T
        reverse steps into ONE uninterrupted all-optical cavity process (Request B).
    """

    def __init__(self, cfg: OpticalCavityConfig):
        self.cfg = cfg
        if cfg.complex_state:
            assert cfg.data_dim % 2 == 0, "complex_state requires even data_dim"
            assert cfg.n_modes >= cfg.data_dim // 2
            self.d_c = cfg.data_dim // 2
        else:
            assert cfg.n_modes >= cfg.data_dim
            self.d_c = cfg.data_dim
        assert 0.0 < cfg.cavity_feedback_r < 1.0

        rng = np.random.default_rng(cfg.seed)
        N, M, td = cfg.n_modes, cfg.n_roundtrips, cfg.time_dim
        cond_dim = td + cfg.n_classes
        self.cond_dim = cond_dim

        # Number of distinct physical cavity PMWs: 1 (recirculating ring) or M (cascaded)
        self.n_wg = 1 if cfg.shared_waveguide else M

        # Cayley parameters for each physical PMW unitary U_w in U(N)
        self.A = [
            (rng.normal(0.0, 0.35 / math.sqrt(N), size=(N, N))
             + 1j * rng.normal(0.0, 0.35 / math.sqrt(N), size=(N, N)))
            for _ in range(self.n_wg)
        ]
        self.phi = [rng.uniform(-0.2, 0.2, size=(N,)) for _ in range(self.n_wg)]

        # Output passive linear interferometer U_out (fixed PMW rotating cavity eigenbasis to readout modes)
        self.A_out = (rng.normal(0.0, 0.35 / math.sqrt(N), size=(N, N))
                      + 1j * rng.normal(0.0, 0.35 / math.sqrt(N), size=(N, N)))
        self.phi_out = rng.uniform(-0.2, 0.2, size=(N,))

        # Input optical conditioning ports
        if cfg.complex_state:
            n_cond_modes = max(N - self.d_c, 0)
            self.W_in_re = rng.normal(0.0, 0.5 / math.sqrt(cond_dim), size=(cond_dim, n_cond_modes))
            self.b_in_re = np.zeros(n_cond_modes)
            self.W_in_im = rng.normal(0.0, 0.5 / math.sqrt(cond_dim), size=(cond_dim, n_cond_modes))
            self.b_in_im = np.zeros(n_cond_modes)
            self.in_scale = np.full(1, 0.85, dtype=np.float64)
        else:
            self.W_in_re = np.zeros((cond_dim, 0), dtype=np.float64)
            self.b_in_re = np.zeros(0, dtype=np.float64)
            self.W_in_im = rng.normal(0.0, 0.5 / math.sqrt(cond_dim), size=(cond_dim, N))
            self.b_in_im = np.zeros(N)
            self.in_scale = np.full(1, 0.75 * math.sqrt(N / max(cfg.data_dim, 1)), dtype=np.float64)

        # Steady coherent optical bias/pump wave injected passively inside the cavity
        self.W_pump_re = rng.normal(0.0, 0.35 / math.sqrt(cond_dim), size=(cond_dim, N))
        self.W_pump_im = rng.normal(0.0, 0.35 / math.sqrt(cond_dim), size=(cond_dim, N))
        self.b_pump_re = np.zeros(N)
        self.b_pump_im = np.zeros(N)

        # Static per-mode passive cavity feedback phase shift
        self.loop_phase = rng.uniform(-0.3, 0.3, size=(N,))

        # Passive all-optical nonlinearity parameters
        if cfg.nonlinearity == "kerr":
            self._gamma = float(cfg.gamma_kerr)
            self._alpha0 = 0.0
        elif cfg.nonlinearity == "sa":
            self._gamma = 0.0
            self._alpha0 = float(cfg.alpha0_sa)
        elif cfg.nonlinearity == "kerr_sa":
            self._gamma = float(cfg.gamma_kerr)
            self._alpha0 = float(cfg.alpha0_sa)
        else:
            raise ValueError(f"unknown nonlinearity: {cfg.nonlinearity}")

        # Output scalar gain (absorbed physically into variable tap tau_m or branch VOA/OPA)
        self.out_gain = np.array([0.4], dtype=np.float64)
        # Electronic bias ONLY for real-quadrature mode; strictly ZERO in folded complex_state mode!
        self.out_bias = np.zeros(0 if cfg.complex_state else cfg.data_dim, dtype=np.float64)

        # Optional compiled physical BPM transfer matrices: (T_wg_list, T_out)
        self.compiled_T_wg: list[np.ndarray] | None = None
        self.compiled_T_out: np.ndarray | None = None

    def _cond_vec(self, t: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        e = sinusoidal_time_embedding(t, self.cfg.time_dim)
        if self.cfg.n_classes > 0:
            oh = np.zeros((e.shape[0], self.cfg.n_classes), dtype=np.float64)
            if y is not None:
                oh[np.arange(e.shape[0]), np.asarray(y, dtype=int)] = 1.0
            e = np.concatenate([e, oh], axis=1)
        return e

    def get_unitaries(self) -> tuple[list[np.ndarray], np.ndarray]:
        Us = [cayley_unitary(A, ph)[0] for A, ph in zip(self.A, self.phi)]
        U_out = cayley_unitary(self.A_out, self.phi_out)[0]
        return Us, U_out

    def forward_complex(
        self,
        a_x: np.ndarray,
        t: np.ndarray,
        y: np.ndarray | None = None,
        override_T: tuple[list[np.ndarray], np.ndarray] | None = None,
        cav_photons_per_mode: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, dict]:
        """All-optical complex-in -> complex-out score branch (for folded cavity!).

        Parameters
        ----------
        a_x : complex ndarray [B, d/2]
            Circulating complex optical state modes directly tapped from storage ring.

        Returns
        -------
        u_complex : complex ndarray [B, d/2]
            Complex optical score field tilde_u_C ready for coherent optical recombination.
        info : dict
            Intracavity intensity, Kerr phase, SA transmission diagnostics.
        """
        assert self.cfg.complex_state, "forward_complex requires complex_state=True"
        x_real = unpack_complex_to_real(a_x)
        _, cache = self.forward(
            x_real, t, y=y, return_cache=True, override_T=override_T,
            cav_photons_per_mode=cav_photons_per_mode, rng=rng,
        )
        sqrt_eta = math.sqrt(self.cfg.eta_chip)
        scale_out = self.out_gain[0] / (cache["t_out"] * sqrt_eta)
        u_c = scale_out * cache["a_out_raw"][:, :self.d_c]
        return u_c, dict(
            mean_cav_intens=cache["mean_cav_intens"],
            mean_kerr_phase=cache["mean_kerr_phase"],
            mean_sa_trans=cache["mean_sa_trans"],
        )

    def forward(
        self,
        x: np.ndarray,
        t: np.ndarray,
        y: np.ndarray | None = None,
        return_cache: bool = False,
        override_T: tuple[list[np.ndarray], np.ndarray] | None = None,
        cav_photons_per_mode: float | None = None,
        rng: np.random.Generator | None = None,
    ):
        """Run M_rt passive all-optical cavity round trips + passive output rotation."""
        cfg = self.cfg
        B = x.shape[0]
        N, d, M = cfg.n_modes, cfg.data_dim, cfg.n_roundtrips
        sqrt_eta = math.sqrt(cfg.eta_chip)
        r_fb = cfg.cavity_feedback_r
        t_in = math.sqrt(1.0 - r_fb * r_fb)
        t_out = t_in

        cond = self._cond_vec(t, y)

        if cfg.complex_state:
            d_c = self.d_c
            a_x = pack_real_to_complex(x)  # [B, d/2]
            a_in = np.zeros((B, N), dtype=np.complex128)
            a_in[:, :d_c] = self.in_scale[0] * a_x
            if N > d_c:
                a_in[:, d_c:] = (
                    (cond @ self.W_in_re + self.b_in_re[None, :])
                    + 1j * (cond @ self.W_in_im + self.b_in_im[None, :])
                )
        else:
            a_x = None
            a_in_re = np.zeros((B, N), dtype=np.float64)
            a_in_re[:, :d] = self.in_scale[0] * x
            a_in_im = cond @ self.W_in_im + self.b_in_im[None, :]
            a_in = a_in_re + 1j * a_in_im

        b_pump = (
            (cond @ self.W_pump_re + self.b_pump_re[None, :])
            + 1j * (cond @ self.W_pump_im + self.b_pump_im[None, :])
        )
        loop_rot = np.exp(1j * self.loop_phase)

        # Resolve physical waveguide matrices
        if override_T is not None:
            T_wgs, T_out = override_T
            u_caches = [None] * self.n_wg
            u_out_cache = None
        elif self.compiled_T_wg is not None and self.compiled_T_out is not None:
            T_wgs, T_out = self.compiled_T_wg, self.compiled_T_out
            u_caches = [None] * self.n_wg
            u_out_cache = None
        else:
            T_wgs = []
            u_caches = []
            for w in range(self.n_wg):
                Uw, uc = cayley_unitary(self.A[w], self.phi[w])
                T_wgs.append(sqrt_eta * Uw)
                u_caches.append(uc)
            U_out, u_out_cache = cayley_unitary(self.A_out, self.phi_out)
            T_out = sqrt_eta * U_out

        # Circulate M_rt all-optical round trips inside the passive nonlinear cavity
        a_fb = np.zeros((B, N), dtype=np.complex128)
        rt_caches = []
        mean_cav_intens = 0.0
        mean_sa_trans = 0.0
        mean_kerr_phase = 0.0

        for k in range(M):
            w_idx = 0 if cfg.shared_waveguide else k
            T_k = T_wgs[w_idx]

            a_cav = (r_fb * loop_rot[None, :]) * a_fb + t_in * a_in
            z_k = a_cav @ T_k.T

            # Exact quantum fluctuation-dissipation vacuum noise from waveguide loss (1 - eta_chip)
            if cav_photons_per_mode is not None and rng is not None:
                eta_k = min(float(np.linalg.norm(T_k, "fro") ** 2 / N), 1.0)
                vac_std = math.sqrt(max(1.0 - eta_k, 0.0)) / (2.0 * math.sqrt(cav_photons_per_mode))
                if vac_std > 0.0:
                    z_k = z_k + (rng.normal(0.0, vac_std, size=z_k.shape)
                                 + 1j * rng.normal(0.0, vac_std, size=z_k.shape))

            u_k = z_k + 0.35 * b_pump
            a_nl, nl_cache = optical_nonlinearity_fwd(u_k, self._gamma, self._alpha0, cfg.i_sat)

            # Exact quantum vacuum noise from saturable-absorber absorption (1 - T_SA)
            if cav_photons_per_mode is not None and rng is not None and self._alpha0 > 0.0:
                r2_k = u_k.real ** 2 + u_k.imag ** 2
                t_sa_pow = np.exp(-self._alpha0 / (1.0 + r2_k / cfg.i_sat))
                vac_sa = np.sqrt(np.clip(1.0 - t_sa_pow, 0.0, 1.0)) / (2.0 * math.sqrt(cav_photons_per_mode))
                a_nl = a_nl + (rng.normal(0.0, vac_sa) + 1j * rng.normal(0.0, vac_sa))

            r2_now = u_k.real ** 2 + u_k.imag ** 2
            mean_cav_intens += float(np.mean(r2_now)) / M
            mean_kerr_phase += float(np.mean(abs(self._gamma) * r2_now)) / M
            mean_sa_trans += float(np.mean(np.exp(-self._alpha0 / (1.0 + r2_now / cfg.i_sat)))) / M

            rt_caches.append((w_idx, a_fb, a_cav, z_k, u_k, a_nl, nl_cache))
            a_fb = a_nl

        a_out_raw = (t_out * a_fb) @ T_out.T
        scale_out = self.out_gain[0] / (t_out * sqrt_eta)
        if cfg.complex_state:
            out = unpack_complex_to_real(scale_out * a_out_raw[:, :self.d_c])
        else:
            out = scale_out * a_out_raw[:, :d].real + self.out_bias[None, :]

        if not return_cache:
            return out
        return out, dict(
            x=x, a_x=a_x, cond=cond, a_in=a_in, b_pump=b_pump, loop_rot=loop_rot,
            T_wgs=T_wgs, T_out=T_out, u_caches=u_caches, u_out_cache=u_out_cache,
            rt_caches=rt_caches, a_fb_final=a_fb, a_out_raw=a_out_raw,
            t_in=t_in, t_out=t_out,
            mean_cav_intens=mean_cav_intens,
            mean_kerr_phase=mean_kerr_phase,
            mean_sa_trans=mean_sa_trans,
        )

    def loss_and_grads(
        self,
        x: np.ndarray,
        t: np.ndarray,
        target: np.ndarray,
        y: np.ndarray | None = None,
    ) -> tuple[float, dict]:
        """Exact backpropagation through time (BPTT) across all M_rt optical round trips."""
        cfg = self.cfg
        B = x.shape[0]
        N, d, M = cfg.n_modes, cfg.data_dim, cfg.n_roundtrips
        sqrt_eta = math.sqrt(cfg.eta_chip)
        r_fb = cfg.cavity_feedback_r

        out, cache = self.forward(x, t, y, return_cache=True)
        diff = out - target
        mse = float(np.mean(np.sum(diff ** 2, axis=1)) / d)

        a_out_raw = cache["a_out_raw"]
        t_out = cache["t_out"]
        t_in = cache["t_in"]
        norm_f = 1.0 / (t_out * sqrt_eta)
        scale_out = self.out_gain[0] * norm_f

        g_out = (2.0 / (B * d)) * diff
        g_aout = np.zeros_like(a_out_raw)

        if cfg.complex_state:
            d_c = self.d_c
            s2 = math.sqrt(2.0)
            g_out_bias = np.zeros(0, dtype=np.float64)
            # out_even = s2 * scale_out * Re(a_out_raw[:, :d_c]), out_odd = s2 * scale_out * Im(...)
            g_uc = s2 * (g_out[:, 0::2] + 1j * g_out[:, 1::2])  # dL = Re Tr(g_uc^H d(scale_out * a_out_raw))
            g_out_gain = np.array([
                float(np.real(np.sum(g_uc.conj() * (a_out_raw[:, :d_c] * norm_f))))
            ], dtype=np.float64)
            g_aout[:, :d_c] += scale_out * g_uc
        else:
            g_out_bias = np.sum(g_out, axis=0)
            g_out_gain = np.array([
                float(np.sum(g_out * (a_out_raw[:, :d].real * norm_f)))
            ], dtype=np.float64)
            g_aout[:, :d] += scale_out * g_out

        anc_loss = 0.0
        if cfg.ancilla_reg > 0.0:
            z_scaled = a_out_raw * norm_f
            if cfg.complex_state:
                d_c = self.d_c
                if N > d_c:
                    extra = z_scaled[:, d_c:]
                    anc_pow = float(np.sum(np.abs(extra) ** 2)) / (B * N)
                    anc_loss = cfg.ancilla_reg * anc_pow
                    sc = (2.0 * cfg.ancilla_reg * norm_f) / (B * N)
                    g_aout[:, d_c:] += sc * extra
            else:
                im_part = z_scaled[:, :d].imag
                extra = z_scaled[:, d:]
                anc_pow = (np.sum(im_part ** 2) + np.sum(np.abs(extra) ** 2)) / (B * N)
                anc_loss = cfg.ancilla_reg * float(anc_pow)
                sc = (2.0 * cfg.ancilla_reg * norm_f) / (B * N)
                g_aout[:, :d] += 1j * (sc * im_part)
                if N > d:
                    g_aout[:, d:] += sc * extra

        total_loss = mse + anc_loss

        a_fb_final = cache["a_fb_final"]
        T_out = cache["T_out"]
        G_Uout = (sqrt_eta * g_aout).T @ (t_out * a_fb_final).conj()
        g_A_out, g_phi_out = cayley_unitary_backward(G_Uout, cache["u_out_cache"])
        g_afb = (g_aout @ T_out.conj()) * t_out

        G_U_wgs = [np.zeros((N, N), dtype=np.complex128) for _ in range(self.n_wg)]
        g_a_in = np.zeros((B, N), dtype=np.complex128)
        g_b_pump = np.zeros((B, N), dtype=np.complex128)
        g_loop_phase = np.zeros(N, dtype=np.float64)
        loop_rot = cache["loop_rot"]

        for k in range(M - 1, -1, -1):
            w_idx, a_fb_prev, a_cav, z_k, u_k, a_nl, nl_cache = cache["rt_caches"][k]
            T_k = cache["T_wgs"][w_idx]

            g_u, _, _ = optical_nonlinearity_bwd(g_afb, nl_cache)
            g_b_pump += 0.35 * g_u
            g_z = g_u

            G_U_wgs[w_idx] += (sqrt_eta * g_z).T @ a_cav.conj()
            g_acav = g_z @ T_k.conj()

            g_a_in += t_in * g_acav
            g_loop_phase += np.real(np.sum(g_acav.conj() * (1j * r_fb * loop_rot[None, :] * a_fb_prev), axis=0))
            g_afb = g_acav * (r_fb * loop_rot.conj()[None, :])

        g_A = []
        g_phi = []
        for w in range(self.n_wg):
            ga, gp = cayley_unitary_backward(G_U_wgs[w], cache["u_caches"][w])
            g_A.append(ga)
            g_phi.append(gp)

        cond = cache["cond"]
        if cfg.complex_state:
            d_c = self.d_c
            a_x = cache["a_x"]
            g_in_scale = np.array([float(np.real(np.sum(g_a_in[:, :d_c].conj() * a_x)))], dtype=np.float64)
            if N > d_c:
                g_W_in_re = cond.T @ g_a_in[:, d_c:].real
                g_b_in_re = np.sum(g_a_in[:, d_c:].real, axis=0)
                g_W_in_im = cond.T @ g_a_in[:, d_c:].imag
                g_b_in_im = np.sum(g_a_in[:, d_c:].imag, axis=0)
            else:
                g_W_in_re = np.zeros_like(self.W_in_re)
                g_b_in_re = np.zeros_like(self.b_in_re)
                g_W_in_im = np.zeros_like(self.W_in_im)
                g_b_in_im = np.zeros_like(self.b_in_im)
        else:
            g_in_scale = np.array([float(np.sum(g_a_in[:, :d].real * x))], dtype=np.float64)
            g_W_in_re = np.zeros_like(self.W_in_re)
            g_b_in_re = np.zeros_like(self.b_in_re)
            g_W_in_im = cond.T @ g_a_in.imag
            g_b_in_im = np.sum(g_a_in.imag, axis=0)

        grads = dict(
            A=g_A,
            phi=g_phi,
            A_out=g_A_out,
            phi_out=g_phi_out,
            W_in_re=g_W_in_re,
            b_in_re=g_b_in_re,
            W_in_im=g_W_in_im,
            b_in_im=g_b_in_im,
            in_scale=g_in_scale,
            W_pump_re=cond.T @ g_b_pump.real,
            W_pump_im=cond.T @ g_b_pump.imag,
            b_pump_re=np.sum(g_b_pump.real, axis=0),
            b_pump_im=np.sum(g_b_pump.imag, axis=0),
            loop_phase=g_loop_phase,
            out_gain=g_out_gain,
            out_bias=g_out_bias,
        )
        return total_loss, grads

    def param_items(self):
        for w in range(self.n_wg):
            yield f"A_{w}", self.A[w]
            yield f"phi_{w}", self.phi[w]
        yield "A_out", self.A_out
        yield "phi_out", self.phi_out
        if self.cfg.complex_state:
            yield "W_in_re", self.W_in_re
            yield "b_in_re", self.b_in_re
        yield "W_in_im", self.W_in_im
        yield "b_in_im", self.b_in_im
        yield "in_scale", self.in_scale
        yield "W_pump_re", self.W_pump_re
        yield "W_pump_im", self.W_pump_im
        yield "b_pump_re", self.b_pump_re
        yield "b_pump_im", self.b_pump_im
        yield "loop_phase", self.loop_phase
        yield "out_gain", self.out_gain
        if not self.cfg.complex_state:
            yield "out_bias", self.out_bias

    def grad_items(self, grads: dict):
        for w in range(self.n_wg):
            yield f"A_{w}", grads["A"][w]
            yield f"phi_{w}", grads["phi"][w]
        yield "A_out", grads["A_out"]
        yield "phi_out", grads["phi_out"]
        if self.cfg.complex_state:
            yield "W_in_re", grads["W_in_re"]
            yield "b_in_re", grads["b_in_re"]
        yield "W_in_im", grads["W_in_im"]
        yield "b_in_im", grads["b_in_im"]
        yield "in_scale", grads["in_scale"]
        yield "W_pump_re", grads["W_pump_re"]
        yield "W_pump_im", grads["W_pump_im"]
        yield "b_pump_re", grads["b_pump_re"]
        yield "b_pump_im", grads["b_pump_im"]
        yield "loop_phase", grads["loop_phase"]
        yield "out_gain", grads["out_gain"]
        if not self.cfg.complex_state:
            yield "out_bias", grads["out_bias"]

    def forward_optical_field(
        self,
        x: np.ndarray,
        t: np.ndarray,
        y: np.ndarray | None = None,
        override_T: tuple[list[np.ndarray], np.ndarray] | None = None,
        cav_photons_per_mode: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (a_final_unit [B, N], out_bias [d], effective_out_gain) compatible with
        ``pwg.diffusion.sample_photonic_diffusion`` (for complex_state=False mode)."""
        assert not self.cfg.complex_state
        _, cache = self.forward(
            x, t, y, return_cache=True, override_T=override_T,
            cav_photons_per_mode=cav_photons_per_mode, rng=rng,
        )
        a_unit = cache["a_out_raw"] / cache["t_out"]
        return a_unit, self.out_bias, float(self.out_gain[0])

    def save(self, path: str):
        data = {k: v for k, v in self.param_items()}
        for k, v in self.cfg.__dict__.items():
            data[f"cfg_{k}"] = np.array(v)
        if self.compiled_T_wg is not None:
            for w, T in enumerate(self.compiled_T_wg):
                data[f"compiled_T_wg_{w}"] = T
            data["compiled_T_out"] = self.compiled_T_out
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> "OpticalCavityScoreNet":
        z = np.load(path, allow_pickle=False)
        cfg = OpticalCavityConfig(
            data_dim=int(z["cfg_data_dim"]),
            n_modes=int(z["cfg_n_modes"]),
            n_roundtrips=int(z["cfg_n_roundtrips"]),
            time_dim=int(z["cfg_time_dim"]),
            n_classes=int(z["cfg_n_classes"]),
            nonlinearity=str(z["cfg_nonlinearity"]),
            shared_waveguide=bool(z["cfg_shared_waveguide"]),
            complex_state=bool(z["cfg_complex_state"]) if "cfg_complex_state" in z else False,
            cavity_feedback_r=float(z["cfg_cavity_feedback_r"]),
            gamma_kerr=float(z["cfg_gamma_kerr"]),
            alpha0_sa=float(z["cfg_alpha0_sa"]),
            i_sat=float(z["cfg_i_sat"]),
            eta_chip=float(z["cfg_eta_chip"]),
            ancilla_reg=float(z["cfg_ancilla_reg"]),
            seed=int(z["cfg_seed"]),
        )
        net = cls(cfg)
        for k, arr in net.param_items():
            arr[...] = z[k]
        if "compiled_T_wg_0" in z:
            net.compiled_T_wg = [z[f"compiled_T_wg_{w}"] for w in range(net.n_wg)]
            net.compiled_T_out = z["compiled_T_out"]
        return net


# ---------------------------------------------------------------------------
# 4. Folded Single-Process All-Optical Diffusion Cavity Simulator (Request B)
# ---------------------------------------------------------------------------

@dataclass
class FoldedCavityHardware:
    """Physical parameters of the folded single-process all-optical diffusion cavity.

    All T reverse diffusion steps execute inside ONE uninterrupted optical ring
    evolution with ZERO intermediate electronic conversion and ONLY ONE dual-quadrature
    (8-port) balanced-homodyne readout at t = 0.
    """

    n_op_photons_per_mode: float = 1e4  # operating mean photon number per mode (E_NL / hbar*omega; 0.5 = pure vacuum!)
    ring_chip_trans: float = 0.966      # passive storage-ring waveguide transmission per round trip
    rho_tap: float = 0.05               # passive power tap into nonlinear score branch
    rho_rec: float = 0.05               # passive power coupling recombining score branch into ring
    amp_nsp: float = 1.5                # branch optical amplifier (OPA/SOA) spontaneous-emission factor (>= 1)
    phase_jitter_rad: float = 0.0       # rms round-trip cavity phase-locking error (rad)
    wavelength_m: float = 1.55e-6
    wpe_laser: float = 0.05             # wall-plug efficiency of laser / optical pump
    e_io_per_mode_J: float = 3.0e-12    # E-O + O-E electronic I/O energy per mode per conversion (for comparison)
    roundtrip_time_ps: float = 50.0     # optical round-trip time of storage ring (ps)

    @property
    def passive_ring_trans(self) -> float:
        """Total passive power transmission of the storage ring per round trip."""
        return self.ring_chip_trans * (1.0 - self.rho_tap) * (1.0 - self.rho_rec)

    @property
    def branch_preamp_gain(self) -> float:
        """Pre-amplifier power gain G_in = 1 / rho_tap restoring tapped state to working point n_op."""
        return 1.0 / max(self.rho_tap, 1e-6)

    def step_amp_gain_and_ase_photons(self, kappa: float) -> tuple[float, float]:
        """Return (total_branch_gain_G, amp_ase_photons_entering_ring) for step weight kappa.

        The score branch uses:
          - Pre-amp of gain G_in = 1 / rho_tap to restore the tapped state amplitude
            sqrt(rho_tap) * a_x back to working point n_op (adding n_sp*(G_in - 1) ASE
            photons at the score-branch scale, which couple into the ring with weight kappa^2).
          - Optional post-amp of gain G_out = max(kappa^2 / rho_rec, 1.0) when kappa exceeds
            the passive recombining coupler amplitude sqrt(rho_rec) (otherwise a passive VOA
            attenuates to kappa), contributing rho_rec * n_sp * (G_out - 1) ASE photons into
            the ring.
        """
        g_in = self.branch_preamp_gain
        g_out = max((kappa * kappa) / max(self.rho_rec, 1e-6), 1.0)
        ase_ring_photons = (
            (kappa * kappa) * self.amp_nsp * max(g_in - 1.0, 0.0)
            + self.rho_rec * self.amp_nsp * max(g_out - 1.0, 0.0)
        )
        return g_in * g_out, ase_ring_photons

    @property
    def photon_energy_J(self) -> float:
        return H_PLANCK * C_LIGHT / self.wavelength_m


def sample_folded_optical_diffusion(
    net: OpticalCavityScoreNet,
    schedule: VPSchedule,
    hw: FoldedCavityHardware,
    detector: HomodyneDetector,
    n_samples: int,
    n_steps: int = 100,
    y: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    grid_mode: str = "uniform_s",
    s_max: float = 8.0,
    s_min: float = 0.002,
    override_T: tuple[list[np.ndarray], np.ndarray] | None = None,
    noiseless_final_step: bool = True,
    record_steps: list[int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Fold ALL T reverse diffusion steps into ONE uninterrupted all-optical cavity process.

    Physics per round trip m = 0 .. T - 1 (ZERO intermediate O-E-O!):
      1. Passive storage-ring attenuation sqrt(1 - beta_step) + open-port optical ASE/vacuum
         noise sqrt(beta_step) * zeta_m (zero digital PRNG for Brownian noise!).
      2. Passive tap rho_tap feeds a_x directly into the all-optical Kerr/SA PMW score branch
         (with exact intracavity loss & SA vacuum noise at n_op photons/mode).
      3. Branch optical amplifier (gain G_br, noise factor n_sp) restores optical power for
         coherent recombination through rho_rec, injecting realistic amplifier ASE noise of
         complex variance rho_rec * n_sp * (G_br - 1) / n_op into the ring!
         Note: when branch ASE variance < beta_step, the open-port ASE source is simply
         trimmed to beta_step - var_amp_ring so total Brownian variance remains exact!
         When var_amp_ring > beta_step (excessive gain / too low n_op), the excess noise
         faithfully degrades sample quality.
      4. Optional round-trip optical phase jitter exp(i * delta_phi_m) models finite cavity
         phase-locking stability.
      5. At m = T (t = 0), ONE single 8-port (dual-quadrature) balanced-homodyne detector
         reads out all d = 2 * d_c pixel coordinates via exact Poisson dual photodiodes!
    """
    assert net.cfg.complex_state, "Folded single-process cavity requires complex_state=True"
    if rng is None:
        rng = np.random.default_rng(0)

    B = n_samples
    d = net.cfg.data_dim
    d_c = net.d_c
    n_op = hw.n_op_photons_per_mode

    if grid_mode == "uniform_s":
        ts = uniform_s_time_grid(schedule, n_steps, s_min=s_min, s_max=s_max)
    elif grid_mode == "uniform_t":
        ts = schedule.time_grid(n_steps)
    else:
        raise ValueError(f"unknown grid_mode: {grid_mode}")

    # Step 0: initialize storage ring directly from optical ASE/vacuum noise CN(0, I_{d_c})
    a_x = (rng.normal(0.0, 1.0 / math.sqrt(2.0), size=(B, d_c))
           + 1j * rng.normal(0.0, 1.0 / math.sqrt(2.0), size=(B, d_c)))

    traj = {}
    if record_steps is not None and 0 in record_steps:
        traj[0] = unpack_complex_to_real(a_x)

    beta_steps = []
    amp_gains = []
    var_amp_list = []
    ring_passive_ok = True
    mean_cav_intens = 0.0
    mean_kerr_phase = 0.0
    mean_sa_trans = 0.0
    total_cav_photons = 0.0

    for m in range(n_steps):
        t_c = float(ts[m])
        t_n = float(ts[m + 1])
        t_vec = np.full(B, t_c, dtype=np.float64)
        sqrt_a, kappa, sigma_rev, _, _ = schedule.step_coeffs(t_c, t_n, variance_mode="beta")
        beta_step = 1.0 - sqrt_a * sqrt_a
        beta_steps.append(beta_step)

        if beta_step < (1.0 - hw.passive_ring_trans) - 1e-6:
            ring_passive_ok = False

        # Step-exact branch optical amplifier gain and spontaneous-emission noise into ring
        g_step, ase_phot_ring = hw.step_amp_gain_and_ase_photons(kappa)
        amp_gains.append(g_step)
        var_amp_ring = ase_phot_ring / max(n_op, 0.5)
        var_amp_list.append(var_amp_ring)

        # All-optical nonlinear Kerr/SA PMW score branch (zero O-E-O!)
        u_c, cinfo = net.forward_complex(
            a_x, t_vec, y=y, override_T=override_T,
            cav_photons_per_mode=n_op, rng=rng,
        )
        mean_cav_intens += cinfo["mean_cav_intens"] / n_steps
        mean_kerr_phase += cinfo["mean_kerr_phase"] / n_steps
        mean_sa_trans += cinfo["mean_sa_trans"] / n_steps

        # Track intracavity optical photon energy per step (storage ring + score branch)
        ring_phot = float(np.mean(np.sum(np.abs(a_x) ** 2, axis=1))) * n_op
        branch_phot = cinfo["mean_cav_intens"] * net.cfg.n_modes * n_op
        total_cav_photons += ring_phot + branch_phot

        # Optional cavity phase-locking error on coherent recombination
        if hw.phase_jitter_rad > 0.0:
            dphi = rng.normal(0.0, hw.phase_jitter_rad, size=(B, 1))
            rot = np.exp(1j * dphi)
        else:
            rot = 1.0

        # Open-port optical ASE / vacuum Brownian noise + branch OPA/SOA ASE noise
        is_last = (m == n_steps - 1) and noiseless_final_step
        if is_last:
            # Close ASE port shutter on final round trip (exact noiseless posterior readout);
            # only unavoidable branch-amplifier spontaneous emission enters!
            var_total_noise = var_amp_ring
        else:
            var_open_port = max(beta_step - var_amp_ring, 0.0)
            var_total_noise = var_open_port + var_amp_ring

        if var_total_noise > 0.0:
            std_q = math.sqrt(0.5 * var_total_noise)
            noise_c = rng.normal(0.0, std_q, size=(B, d_c)) + 1j * rng.normal(0.0, std_q, size=(B, d_c))
        else:
            noise_c = 0.0

        # Coherent all-optical round-trip update inside the storage ring!
        a_x = sqrt_a * a_x + (kappa * rot) * u_c + noise_c

        if record_steps is not None and (m + 1) in record_steps:
            traj[m + 1] = unpack_complex_to_real(a_x)

    # ONLY ONE electronic readout across the ENTIRE T-step generation (at t = 0):
    # Dual-quadrature (8-port) balanced homodyne detector splits each mode 50:50 onto Re & Im pairs.
    scale_read = math.sqrt(0.5 * n_op * detector.eta_chip)
    re_hat, info_re = detector.measure_real_quadrature(scale_read * a_x.real, rng)
    im_hat, info_im = detector.measure_real_quadrature(scale_read * a_x.imag, rng)
    a_read = (re_hat + 1j * im_hat) / max(scale_read, 1e-12)
    x_final = unpack_complex_to_real(a_read)

    # Comprehensive energy & latency accounting
    e_ph = hw.photon_energy_J
    optical_energy_intracavity_J = total_cav_photons * e_ph
    optical_wallplug_J = optical_energy_intracavity_J / hw.wpe_laser
    state_oeo_energy_J = 0.5 * d * hw.e_io_per_mode_J  # 1 final readout only (O-E half of E-O+O-E)
    total_wallplug_folded_J = optical_wallplug_J + state_oeo_energy_J

    info = dict(
        n_steps=n_steps,
        grid_mode=grid_mode,
        beta_step_min=float(min(beta_steps)),
        beta_step_max=float(max(beta_steps)),
        passive_ring_loss=float(1.0 - hw.passive_ring_trans),
        ring_strictly_passive=bool(ring_passive_ok),
        n_op_photons_per_mode=float(n_op),
        branch_amp_gain_mean_dB=float(10.0 * math.log10(max(float(np.mean(amp_gains)), 1.0))),
        branch_amp_gain_max_dB=float(10.0 * math.log10(max(float(np.max(amp_gains)), 1.0))),
        var_amp_ring_mean=float(np.mean(var_amp_list)),
        amp_noise_fraction=float(np.mean(var_amp_list) / max(min(beta_steps), 1e-12)),
        phase_jitter_rad=float(hw.phase_jitter_rad),
        mean_cav_intens=float(mean_cav_intens),
        mean_kerr_phase=float(mean_kerr_phase),
        mean_sa_trans=float(mean_sa_trans),
        total_intracavity_photons=float(total_cav_photons),
        optical_wallplug_nJ=float(optical_wallplug_J * 1e9),
        state_oeo_energy_nJ=float(state_oeo_energy_J * 1e9),
        total_wallplug_nJ=float(total_wallplug_folded_J * 1e9),
        total_latency_ns=float(n_steps * hw.roundtrip_time_ps * 1e-3),
        n_state_oeo_conversions=1,
        final_readout_photons=float(info_re["arriving_photons"] + info_im["arriving_photons"]),
        trajectory=traj,
    )
    return x_final, info
