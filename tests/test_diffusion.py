"""Unit and physics tests for shot-noise-driven photonic denoising diffusion."""
from __future__ import annotations

import math
import numpy as np
import pytest

from pwg.diffusion import (
    GaussianMixtureDistribution,
    HomodyneDetector,
    PhotonicNetConfig,
    UnitaryScoreNet,
    VPSchedule,
    cayley_unitary,
    cayley_unitary_backward,
    sample_photonic_diffusion,
)


def test_cayley_unitarity_and_gradient():
    rng = np.random.default_rng(0)
    N = 6
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    phi = rng.uniform(-1.0, 1.0, size=(N,))
    U, cache = cayley_unitary(A, phi)
    # Exact unitarity
    assert np.max(np.abs(U.conj().T @ U - np.eye(N))) < 1e-13

    # Random linear scalar loss L = Re Tr(W^H U)
    W = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    G_A, G_phi = cayley_unitary_backward(W, cache)

    eps = 1e-6
    for i in range(N):
        for j in range(N):
            # Real part of A
            Ap, Am = A.copy(), A.copy()
            Ap[i, j] += eps; Am[i, j] -= eps
            fd_re = (np.real(np.sum(W.conj() * cayley_unitary(Ap, phi)[0]))
                     - np.real(np.sum(W.conj() * cayley_unitary(Am, phi)[0]))) / (2 * eps)
            assert abs(fd_re - G_A[i, j].real) < 1e-6
            # Imag part of A
            Ap, Am = A.copy(), A.copy()
            Ap[i, j] += 1j * eps; Am[i, j] -= 1j * eps
            fd_im = (np.real(np.sum(W.conj() * cayley_unitary(Ap, phi)[0]))
                     - np.real(np.sum(W.conj() * cayley_unitary(Am, phi)[0]))) / (2 * eps)
            assert abs(fd_im - G_A[i, j].imag) < 1e-6
    for i in range(N):
        pp, pm = phi.copy(), phi.copy()
        pp[i] += eps; pm[i] -= eps
        fd = (np.real(np.sum(W.conj() * cayley_unitary(A, pp)[0]))
              - np.real(np.sum(W.conj() * cayley_unitary(A, pm)[0]))) / (2 * eps)
        assert abs(fd - G_phi[i]) < 1e-6


@pytest.mark.parametrize("nonlinearity", ["silu", "tanh", "square"])
def test_unitary_score_net_gradients(nonlinearity):
    rng = np.random.default_rng(1)
    cfg = PhotonicNetConfig(
        data_dim=4, n_modes=6, n_layers=3, time_dim=4,
        nonlinearity=nonlinearity, ancilla_reg=0.05, seed=7,
    )
    net = UnitaryScoreNet(cfg)
    x = rng.normal(size=(5, 4))
    t = rng.uniform(0.05, 0.95, size=(5,))
    tgt = rng.normal(size=(5, 4))

    loss0, grads = net.loss_and_grads(x, t, tgt)
    gdict = dict(net.grad_items(grads))
    eps = 1e-6
    for name, param in net.param_items():
        g = gdict[name]
        # Spot-check 3 entries per parameter tensor
        flat_p = param.ravel()
        flat_g = g.ravel()
        idxs = rng.choice(len(flat_p), size=min(3, len(flat_p)), replace=False)
        for idx in idxs:
            orig = flat_p[idx]
            flat_p[idx] = orig + eps
            lp, _ = net.loss_and_grads(x, t, tgt)
            flat_p[idx] = orig - eps
            lm, _ = net.loss_and_grads(x, t, tgt)
            flat_p[idx] = orig
            fd_re = (lp - lm) / (2 * eps)
            assert abs(fd_re - np.real(flat_g[idx])) < 1e-5 * (1.0 + abs(fd_re)), (
                f"{nonlinearity} {name}[{idx}] re: fd={fd_re} vs ad={np.real(flat_g[idx])}"
            )
            if np.iscomplexobj(param):
                flat_p[idx] = orig + 1j * eps
                lp, _ = net.loss_and_grads(x, t, tgt)
                flat_p[idx] = orig - 1j * eps
                lm, _ = net.loss_and_grads(x, t, tgt)
                flat_p[idx] = orig
                fd_im = (lp - lm) / (2 * eps)
                assert abs(fd_im - np.imag(flat_g[idx])) < 1e-5 * (1.0 + abs(fd_im)), (
                    f"{nonlinearity} {name}[{idx}] im: fd={fd_im} vs ad={np.imag(flat_g[idx])}"
                )


def test_homodyne_poisson_statistics():
    """Verify exact mean, variance, and state-independence of balanced Poisson homodyne."""
    rng = np.random.default_rng(42)
    det = HomodyneDetector(eta_qe=0.8, eta_chip=0.9, lo_photons_per_mode=1e6, read_noise_e=30.0)
    # Complex signal with both real and imaginary components
    a_true = np.array([0.0, 2.5 - 3.0j, -1.7 + 4.2j, 0.3 + 0.8j])
    a_batch = np.tile(a_true, (40_000, 1))
    meas, info = det.measure_real_quadrature(a_batch, rng)
    mean_est = meas.mean(axis=0)
    var_est = meas.var(axis=0)
    expected_var = (1.0 + 2 * 30.0 ** 2 / (0.8 * 1e6)) / (4.0 * 0.8)
    assert np.allclose(mean_est, a_true.real, atol=0.015)
    assert np.allclose(var_est, expected_var, rtol=0.03)


def test_de_bruijn_kl_photon_theorem_exact():
    """Verify that exact Poisson balanced-homodyne residual sampling matches the
    de Bruijn / KL photon theorem and reproduces the 2D 8-mode Gaussian mixture."""
    gmm = GaussianMixtureDistribution.ring_2d(n_modes=8, radius=1.35, std=0.16)
    sched = VPSchedule(beta_min=0.1, beta_max=20.0, eps=2e-3)
    det = HomodyneDetector(eta_qe=0.8, eta_chip=0.9, lo_photons_per_mode=1e6, read_noise_e=0.0)

    a_eps, s_eps = sched.alpha_sigma(sched.eps)
    a_1, s_1 = sched.alpha_sigma(1.0)
    kl_eps = gmm.kl_to_gamma(float(a_eps), float(s_eps), n_mc=60_000, seed=1)
    kl_1 = gmm.kl_to_gamma(float(a_1), float(s_1), n_mc=60_000, seed=2)
    n_kl_theory = (kl_eps - kl_1) / (2.0 * det.eta_tot)

    def exact_score_fn(x, t_vec):
        a, s = sched.alpha_sigma(float(t_vec[0]))
        return gmm.tilde_u_exact(x, float(a), float(s))

    rng = np.random.default_rng(123)
    samples, stats = sample_photonic_diffusion(
        exact_score_fn, sched, det, shape=(2000, 2), n_steps=200,
        rng=rng, mode="homodyne_residual", variance_mode="beta",
    )
    # 1. Total launched signal photons must match the exact KL bound within ~6%
    rel_err = abs(stats["total_launched_photons"] - n_kl_theory) / n_kl_theory
    assert rel_err < 0.06, f"photons={stats['total_launched_photons']:.3f} vs KL theory={n_kl_theory:.3f}"

    # 2. Sampled distribution must closely match the 8-mode ring radial & angular moments
    radii = np.linalg.norm(samples, axis=1)
    assert abs(radii.mean() - 1.35) < 0.06
    assert abs(radii.std() - 0.16) < 0.05
