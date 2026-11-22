"""Unit and physics tests for all-optical nonlinear cavity photonic diffusion."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pwg.cavity_diffusion import (
    FoldedCavityHardware,
    OpticalCavityConfig,
    OpticalCavityScoreNet,
    optical_nonlinearity_bwd,
    optical_nonlinearity_fwd,
    pack_real_to_complex,
    sample_folded_optical_diffusion,
    uniform_s_time_grid,
    unpack_complex_to_real,
)
from pwg.diffusion import HomodyneDetector, VPSchedule


@pytest.mark.parametrize("gamma,alpha0", [(0.8, 0.0), (0.0, 1.8), (0.65, 1.4)])
def test_optical_nonlinearity_passivity_and_gradients(gamma, alpha0):
    rng = np.random.default_rng(0)
    z = rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
    w, cache = optical_nonlinearity_fwd(z, gamma=gamma, alpha0=alpha0, i_sat=1.2)
    # Strict passivity: |Phi_opt(z)| <= |z| everywhere; exact norm conservation for pure Kerr
    if alpha0 == 0.0:
        assert np.allclose(np.abs(w), np.abs(z), atol=1e-14)
    else:
        assert np.all(np.abs(w) <= np.abs(z) + 1e-14)

    W = rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
    g_z, g_gam, g_a0 = optical_nonlinearity_bwd(W, cache)

    eps = 1e-6
    for i in range(4):
        for j in range(5):
            zp, zm = z.copy(), z.copy()
            zp[i, j] += eps; zm[i, j] -= eps
            fd_re = (np.real(np.sum(W.conj() * optical_nonlinearity_fwd(zp, gamma, alpha0, 1.2)[0]))
                     - np.real(np.sum(W.conj() * optical_nonlinearity_fwd(zm, gamma, alpha0, 1.2)[0]))) / (2 * eps)
            assert abs(fd_re - g_z[i, j].real) < 1e-6

            zp, zm = z.copy(), z.copy()
            zp[i, j] += 1j * eps; zm[i, j] -= 1j * eps
            fd_im = (np.real(np.sum(W.conj() * optical_nonlinearity_fwd(zp, gamma, alpha0, 1.2)[0]))
                     - np.real(np.sum(W.conj() * optical_nonlinearity_fwd(zm, gamma, alpha0, 1.2)[0]))) / (2 * eps)
            assert abs(fd_im - g_z[i, j].imag) < 1e-6

    fd_gam = (np.real(np.sum(W.conj() * optical_nonlinearity_fwd(z, gamma + eps, alpha0, 1.2)[0]))
              - np.real(np.sum(W.conj() * optical_nonlinearity_fwd(z, gamma - eps, alpha0, 1.2)[0]))) / (2 * eps)
    fd_a0 = (np.real(np.sum(W.conj() * optical_nonlinearity_fwd(z, gamma, alpha0 + eps, 1.2)[0]))
             - np.real(np.sum(W.conj() * optical_nonlinearity_fwd(z, gamma, alpha0 - eps, 1.2)[0]))) / (2 * eps)
    assert abs(fd_gam - g_gam) < 1e-6
    assert abs(fd_a0 - g_a0) < 1e-6


@pytest.mark.parametrize("nonlinearity,shared,complex_state", [
    ("kerr", True, False),
    ("sa", True, False),
    ("kerr_sa", True, False),
    ("kerr_sa", False, False),
    ("kerr_sa", True, True),
    ("kerr", False, True),
])
def test_optical_cavity_bptt_gradients(nonlinearity, shared, complex_state):
    rng = np.random.default_rng(3)
    cfg = OpticalCavityConfig(
        data_dim=4, n_modes=6, n_roundtrips=4, time_dim=4, n_classes=3,
        nonlinearity=nonlinearity, shared_waveguide=shared,
        complex_state=complex_state, ancilla_reg=0.05, seed=9,
    )
    net = OpticalCavityScoreNet(cfg)
    x = rng.normal(size=(5, 4))
    t = rng.uniform(0.05, 0.95, size=(5,))
    y = rng.integers(0, 3, size=(5,))
    tgt = rng.normal(size=(5, 4))

    loss0, grads = net.loss_and_grads(x, t, tgt, y=y)
    gdict = dict(net.grad_items(grads))
    eps = 1e-6
    for name, param in net.param_items():
        g = gdict[name]
        flat_p = param.ravel()
        flat_g = g.ravel()
        idxs = rng.choice(len(flat_p), size=min(3, len(flat_p)), replace=False)
        for idx in idxs:
            orig = flat_p[idx]
            flat_p[idx] = orig + eps
            lp, _ = net.loss_and_grads(x, t, tgt, y=y)
            flat_p[idx] = orig - eps
            lm, _ = net.loss_and_grads(x, t, tgt, y=y)
            flat_p[idx] = orig
            fd_re = (lp - lm) / (2 * eps)
            assert abs(fd_re - np.real(flat_g[idx])) < 1e-5 * (1.0 + abs(fd_re)), (
                f"{nonlinearity}/{shared}/{complex_state} {name}[{idx}] re: fd={fd_re} vs ad={np.real(flat_g[idx])}"
            )
            if np.iscomplexobj(param):
                flat_p[idx] = orig + 1j * eps
                lp, _ = net.loss_and_grads(x, t, tgt, y=y)
                flat_p[idx] = orig - 1j * eps
                lm, _ = net.loss_and_grads(x, t, tgt, y=y)
                flat_p[idx] = orig
                fd_im = (lp - lm) / (2 * eps)
                assert abs(fd_im - np.imag(flat_g[idx])) < 1e-5 * (1.0 + abs(fd_im)), (
                    f"{nonlinearity}/{shared}/{complex_state} {name}[{idx}] im: fd={fd_im} vs ad={np.imag(flat_g[idx])}"
                )


def test_uniform_s_grid_constant_passive_loss_and_folded_sampler():
    sched = VPSchedule()
    ts = uniform_s_time_grid(sched, n_steps=60, s_min=0.002, s_max=8.0)
    betas = []
    for m in range(60):
        sq_a, _, _, _, _ = sched.step_coeffs(float(ts[m]), float(ts[m + 1]), variance_mode="beta")
        betas.append(1.0 - sq_a * sq_a)
    # Every round trip must have strictly identical passive loss beta_step!
    assert np.allclose(betas, betas[0], atol=1e-10)
    expected_beta = 1.0 - math.exp(-(8.0 - 0.002) / 60.0)
    assert abs(betas[0] - expected_beta) < 1e-10

    # Complex packing round-trip preserves Euclidean norm
    rng = np.random.default_rng(7)
    x = rng.normal(size=(8, 6))
    a = pack_real_to_complex(x)
    assert np.allclose(unpack_complex_to_real(a), x, atol=1e-14)
    assert np.allclose(np.sum(np.abs(a) ** 2, axis=1), 0.5 * np.sum(x ** 2, axis=1), atol=1e-14)

    # Smoke-test folded single-process cavity sampler (check strictly passive storage ring!)
    cfg = OpticalCavityConfig(
        data_dim=4, n_modes=8, n_roundtrips=3, time_dim=4, n_classes=0,
        nonlinearity="kerr_sa", shared_waveguide=True, complex_state=True, seed=11,
    )
    net = OpticalCavityScoreNet(cfg)
    hw = FoldedCavityHardware(
        n_op_photons_per_mode=1e4, ring_chip_trans=0.966, rho_tap=0.04, rho_rec=0.04,
    )
    det = HomodyneDetector(eta_qe=0.8, eta_chip=0.966, lo_photons_per_mode=1e5, read_noise_e=30.0)
    # With 60 uniform-s steps, beta_step = 12.48% > passive ring loss (10.97%) -> strictly passive!
    x_out, info = sample_folded_optical_diffusion(
        net, sched, hw, det, n_samples=16, n_steps=60, rng=rng, s_max=8.0, s_min=0.002,
    )
    assert x_out.shape == (16, 4)
    assert np.all(np.isfinite(x_out))
    assert info["ring_strictly_passive"] is True
    assert info["n_state_oeo_conversions"] == 1
    assert info["amp_noise_fraction"] < 0.05
