"""Experiment 1: First-principles verification of the Quantum Shot-Noise--KL Theorem
and physical N=16 BPM wave-optics photonic diffusion on a 2D 8-mode Gaussian mixture.

Measures:
  1A. Step-count sweep T in {10, 20, 40, 80, 160, 320, 640}:
      - Architecture B (excess-score residual + exact Poisson balanced homodyne):
        total signal photons converge to the finite T-independent de Bruijn / KL limit
        N_KL = [KL(p_eps || gamma) - KL(p_1 || gamma)] / (2 * eta_tot).
      - Architecture A (full-state predictor + exact Poisson balanced homodyne):
        total signal photons grow as O(T^2).
  1B. Finite local-oscillator and read-noise phase diagram (|alpha_LO|^2 in [1e2, 1e6],
      sigma_read = 30 e-).
  1C. End-to-end physical N=16 shallow-etched MgO:LN/SRN BPM compilation of a 3-layer
      UnitaryScoreNet on the 2D 8-mode ring, verified with direct field-by-field BPM
      propagation and sidewall roughness (sigma = 10, 50 nm).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pwg.diffusion import (
    ComplexAdam,
    GaussianMixtureDistribution,
    HomodyneDetector,
    PhotonicNetConfig,
    UnitaryScoreNet,
    VPSchedule,
    compile_unitary_to_bpm,
    evaluate_compiled_bpm,
    sample_photonic_diffusion,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "results", "diffusion_kl")
os.makedirs(OUT_DIR, exist_ok=True)


def rbf_mmd2(x: np.ndarray, y: np.ndarray, sigmas=(0.15, 0.35, 0.8)) -> float:
    """Unbiased multi-scale RBF MMD^2 between 2D point clouds x [N,2] and y [M,2]."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xx = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    yy = np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    xy = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    n, m = len(x), len(y)
    tot = 0.0
    for s in sigmas:
        inv = 1.0 / (2.0 * s * s)
        kxx = np.exp(-xx * inv)
        kyy = np.exp(-yy * inv)
        kxy = np.exp(-xy * inv)
        np.fill_diagonal(kxx, 0.0)
        np.fill_diagonal(kyy, 0.0)
        tot += kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean()
    return float(max(tot / len(sigmas), 0.0))


def ring_metrics(samples: np.ndarray, ref: np.ndarray, gmm: GaussianMixtureDistribution) -> dict:
    r = np.linalg.norm(samples, axis=1)
    # Assign each sample to nearest ring mode
    d2 = np.sum((samples[:, None, :] - gmm.means[None, :, :]) ** 2, axis=-1)
    nearest = np.argmin(d2, axis=1)
    min_dist = np.sqrt(np.min(d2, axis=1))
    counts = np.bincount(nearest, minlength=gmm.K).astype(float)
    probs = counts / counts.sum()
    ent = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    mmd2 = rbf_mmd2(samples[:800], ref[:800])
    return dict(
        r_mean=float(r.mean()),
        r_std=float(r.std()),
        mean_mode_dist=float(min_dist.mean()),
        mode_entropy_nats=ent,
        max_entropy_nats=math.log(gmm.K),
        mmd2=mmd2,
    )


def main():
    t0 = time.time()
    gmm = GaussianMixtureDistribution.ring_2d(n_modes=8, radius=1.35, std=0.16)
    sched = VPSchedule(beta_min=0.1, beta_max=20.0, eps=2e-3)
    det_ideal_lo = HomodyneDetector(eta_qe=0.8, eta_chip=0.90, lo_photons_per_mode=1e6, read_noise_e=0.0)

    rng_ref = np.random.default_rng(999)
    ref_samples = gmm.sample(2000, rng_ref)

    # Exact KL(p_eps || gamma) - KL(p_1 || gamma)
    a_eps, s_eps = sched.alpha_sigma(sched.eps)
    a_1, s_1 = sched.alpha_sigma(1.0)
    kl_eps = gmm.kl_to_gamma(float(a_eps), float(s_eps), n_mc=200_000, seed=1)
    kl_1 = gmm.kl_to_gamma(float(a_1), float(s_1), n_mc=200_000, seed=2)
    delta_kl = kl_eps - kl_1
    n_kl_theory = delta_kl / (2.0 * det_ideal_lo.eta_tot)
    print(f"2D 8-mode ring: KL(p_eps||gamma)={kl_eps:.4f} nats, KL(p_1||gamma)={kl_1:.6f} nats")
    print(f"Exact theoretical quantum photon bound N_KL = {n_kl_theory:.4f} photons/sample "
          f"({n_kl_theory / (delta_kl / math.log(2)):.4f} photons/bit)", flush=True)

    def exact_score(x, t_vec):
        a, s = sched.alpha_sigma(float(t_vec[0]))
        return gmm.tilde_u_exact(x, float(a), float(s))

    # ------------------------------------------------------------------
    # Part 1A: Step-count sweep T (Architecture B vs Architecture A vs Digital)
    # ------------------------------------------------------------------
    T_list = [10, 20, 40, 80, 160, 320, 640]
    sweep_T = []
    for T in T_list:
        s_res, st_res = sample_photonic_diffusion(
            exact_score, sched, det_ideal_lo, shape=(2000, 2), n_steps=T,
            rng=np.random.default_rng(100 + T), mode="homodyne_residual",
        )
        s_full, st_full = sample_photonic_diffusion(
            exact_score, sched, det_ideal_lo, shape=(2000, 2), n_steps=T,
            rng=np.random.default_rng(200 + T), mode="homodyne_fullstate",
        )
        s_dig, _ = sample_photonic_diffusion(
            exact_score, sched, det_ideal_lo, shape=(2000, 2), n_steps=T,
            rng=np.random.default_rng(300 + T), mode="digital",
        )
        m_res = ring_metrics(s_res, ref_samples, gmm)
        m_full = ring_metrics(s_full, ref_samples, gmm)
        m_dig = ring_metrics(s_dig, ref_samples, gmm)
        row = dict(
            T=T,
            residual_photons=st_res["total_launched_photons"],
            residual_photons_per_mode_step=st_res["total_launched_photons"] / (2 * T),
            fullstate_photons=st_full["total_launched_photons"],
            fullstate_photons_per_mode_step=st_full["total_launched_photons"] / (2 * T),
            m_residual=m_res,
            m_fullstate=m_full,
            m_digital=m_dig,
        )
        sweep_T.append(row)
        print(f"  T={T:3d}: Res={row['residual_photons']:6.2f} ph ({row['residual_photons_per_mode_step']:.4f}/mode/step, "
              f"MMD2={m_res['mmd2']:.5f}) | Full={row['fullstate_photons']:8.1f} ph (MMD2={m_full['mmd2']:.5f}) | "
              f"Dig MMD2={m_dig['mmd2']:.5f}", flush=True)

    # ------------------------------------------------------------------
    # Part 1B: Finite LO power & read noise phase diagram (sigma_read = 30 e-)
    # Average over 5 independent seeds (2000 samples each) for low-variance MMD^2!
    # ------------------------------------------------------------------
    lo_grid = [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]
    lo_sweep = []
    for n_lo in lo_grid:
        det_lo = HomodyneDetector(eta_qe=0.8, eta_chip=0.90, lo_photons_per_mode=n_lo, read_noise_e=30.0)
        mmds, rstds, rmeans, mdists = [], [], [], []
        ex_read_vals, ex_lo_vals = [], []
        for s_idx in range(5):
            s_lo, st_lo = sample_photonic_diffusion(
                exact_score, sched, det_lo, shape=(2500, 2), n_steps=120,
                rng=np.random.default_rng(500 + 17 * s_idx), mode="homodyne_residual",
            )
            m_i = ring_metrics(s_lo, ref_samples, gmm)
            mmds.append(m_i["mmd2"])
            rstds.append(m_i["r_std"])
            rmeans.append(m_i["r_mean"])
            mdists.append(m_i["mean_mode_dist"])
            ex_read_vals.append(st_lo["mean_excess_read_var_frac"])
            ex_lo_vals.append(st_lo["max_excess_lo_var_frac"])
        m_lo = dict(
            r_mean=float(np.mean(rmeans)),
            r_std=float(np.mean(rstds)),
            mean_mode_dist=float(np.mean(mdists)),
            mmd2=float(np.mean(mmds)),
            mmd2_std=float(np.std(mmds) / math.sqrt(len(mmds))),
        )
        lo_sweep.append(dict(
            lo_photons_per_mode=n_lo,
            excess_read_var_frac=float(np.mean(ex_read_vals)),
            max_excess_lo_var_frac=float(np.mean(ex_lo_vals)),
            metrics=m_lo,
        ))
        print(f"  LO={n_lo:8.0f} ph/mode (read_noise=30e): excess_read={np.mean(ex_read_vals):.4f}, "
              f"r_std={m_lo['r_std']:.3f}, MMD2={m_lo['mmd2']:.5f}", flush=True)

    # ------------------------------------------------------------------
    # Part 1C: Train 4-layer N=16 UnitaryScoreNet on 2D ring & compile to physical BPM
    # ------------------------------------------------------------------
    print("\nTraining 4-layer N=16 UnitaryScoreNet on 2D 8-mode ring...", flush=True)
    cfg = PhotonicNetConfig(
        data_dim=2, n_modes=16, n_layers=4, time_dim=16,
        nonlinearity="silu", eta_chip=0.99, ancilla_reg=0.0, seed=11,
    )
    net = UnitaryScoreNet(cfg)
    opt = ComplexAdam(net, lr=6e-3)
    ema = {k: v.copy() for k, v in net.param_items()}
    rng_tr = np.random.default_rng(7)
    n_iters = 22000
    for it in range(1, n_iters + 1):
        x0 = gmm.sample(512, rng_tr)
        u = rng_tr.uniform(0.0, 1.0, size=(512,))
        t_b = sched.eps + (1.0 - sched.eps) * (u ** 2.3)
        a_b, s_b = sched.alpha_sigma(t_b)
        eps_b = rng_tr.normal(size=x0.shape)
        xt = a_b[:, None] * x0 + s_b[:, None] * eps_b
        tgt = s_b[:, None] * x0 - a_b[:, None] * eps_b
        lr_now = 2e-4 + (6e-3 - 2e-4) * 0.5 * (1.0 + math.cos(math.pi * it / n_iters))
        loss, grads = net.loss_and_grads(xt, t_b, tgt)
        opt.step(grads, lr=lr_now)
        for k, v in net.param_items():
            ema[k] = 0.997 * ema[k] + 0.003 * v if it > 500 else v.copy()
        if it % 5500 == 0:
            print(f"  it {it:5d}/{n_iters}: loss={loss:.4f}, out_gain={net.out_gain[0]:.3f}", flush=True)
    for k, v in net.param_items():
        v[...] = ema[k]

    # Compile all 4 N=16 unitaries into physical shallow-etched MgO:LN/SRN BPM waveguides
    print("\nCompiling 4 x N=16 unitaries into physical MgO:LN/SRN BPM waveguides...", flush=True)
    Us = net.get_unitaries()
    compiled = []
    for l, U_l in enumerate(Us):
        c_res = compile_unitary_to_bpm(U_l, na=0.3, length_um=1250.0, dn_max=5e-3, res_um=1.0, iters=120, seed=l)
        compiled.append(c_res)
        print(f"  Layer {l}: F={c_res['F']:.4f}, F_n={c_res['Fn']:.5f}, eta_actual={c_res['eta_actual']:.4f}", flush=True)

    T_bpm = [c["T_eff"] for c in compiled]
    eta_bpm_last = compiled[-1]["eta_actual"]
    net.cfg.eta_chip = eta_bpm_last

    # Direct field-by-field BPM linearity verification on Layer 0
    import drjit as dr
    from drjit.cuda import Float as FloatNP
    from drjit.cuda.ad import Float as FloatAD
    from pwg.mmwg import MMWGConfig, MultimodeWaveguide
    from pwg.bpm import propagate

    wg0 = MultimodeWaveguide(MMWGConfig(**compiled[0]["cfg"]))
    dn0 = wg0.index_map(FloatAD(compiled[0]["theta"].astype(np.float32)))
    rng_v = np.random.default_rng(77)
    a_test = (rng_v.normal(size=(8, 16)) + 1j * rng_v.normal(size=(8, 16))).astype(np.complex64)
    e_in = a_test @ wg0.modes.astype(np.complex64)
    re_o, im_o = propagate(wg0.prop, dn0, FloatNP(e_in.real.ravel()), FloatNP(e_in.imag.ravel()), 8)
    out_f = (re_o.numpy() + 1j * im_o.numpy()).reshape(8, -1)
    a_out_direct = (out_f @ wg0.modes.conj().T) * wg0.grid.dx
    a_out_direct_eff = (a_out_direct * compiled[0]["D_phase"].conj()[None, :]) * compiled[0]["global_phase"]
    a_out_mat = a_test @ T_bpm[0].T
    bpm_equiv_rel_err = float(np.linalg.norm(a_out_direct_eff - a_out_mat) / np.linalg.norm(a_out_mat))
    print(f"  Direct field-by-field BPM vs T_eff relative error: {bpm_equiv_rel_err:.2e}", flush=True)

    # Sample from trained model: digital, ideal unitary homodyne, physical BPM homodyne, rough BPM homodyne
    det_bpm = HomodyneDetector(eta_qe=0.8, eta_chip=eta_bpm_last, lo_photons_per_mode=1e5, read_noise_e=30.0)

    def make_fn(override_T=None):
        def fn(x, t_vec):
            a_u, ob = net.forward_optical_field(x, t_vec, override_T=override_T)
            return a_u, ob, float(net.out_gain[0])
        return fn

    samp_dig, _ = sample_photonic_diffusion(
        make_fn(None), sched, det_bpm, shape=(2000, 2), n_steps=160,
        rng=np.random.default_rng(801), mode="digital",
    )
    samp_bpm, st_bpm = sample_photonic_diffusion(
        make_fn(T_bpm), sched, det_bpm, shape=(2000, 2), n_steps=160,
        rng=np.random.default_rng(802), mode="homodyne_residual",
        record_steps=[0, 40, 80, 120, 160],
    )

    # Uncalibrated sidewall roughness sigma = 50 nm on all 4 compiled chips
    T_rough50 = [
        evaluate_compiled_bpm(c["theta"], c["cfg"], c["D_phase"], c["global_phase"],
                              rough_sigma_nm=50.0, rough_lc_um=0.2, rough_seed=10 + l)[0]
        for l, c in enumerate(compiled)
    ]
    samp_r50, st_r50 = sample_photonic_diffusion(
        make_fn(T_rough50), sched, det_bpm, shape=(2000, 2), n_steps=160,
        rng=np.random.default_rng(803), mode="homodyne_residual",
    )

    m_net_dig = ring_metrics(samp_dig, ref_samples, gmm)
    m_net_bpm = ring_metrics(samp_bpm, ref_samples, gmm)
    m_net_r50 = ring_metrics(samp_r50, ref_samples, gmm)
    print(f"  N=16 Unitary Digital:      MMD2={m_net_dig['mmd2']:.5f}, r={m_net_dig['r_mean']:.3f}+/-{m_net_dig['r_std']:.3f}")
    print(f"  N=16 Physical BPM Poisson: MMD2={m_net_bpm['mmd2']:.5f}, r={m_net_bpm['r_mean']:.3f}+/-{m_net_bpm['r_std']:.3f}, "
          f"useful={st_bpm['total_useful_photons']:.2f} ph, launched={st_bpm['total_launched_photons']:.2f} ph (dil={st_bpm['dilation_overhead']:.2f}x)")
    print(f"  N=16 BPM + 50nm roughness: MMD2={m_net_r50['mmd2']:.5f}, r={m_net_r50['r_mean']:.3f}+/-{m_net_r50['r_std']:.3f}", flush=True)

    # Save results and sample arrays for plotting
    summary = dict(
        kl_eps_nats=kl_eps,
        kl_1_nats=kl_1,
        delta_kl_nats=delta_kl,
        n_kl_theory=n_kl_theory,
        eta_qe=det_ideal_lo.eta_qe,
        eta_chip=det_ideal_lo.eta_chip,
        sweep_T=sweep_T,
        lo_sweep=lo_sweep,
        bpm_compile=[dict(layer=l, F=c["F"], Fn=c["Fn"], err=c["err"], eta_actual=c["eta_actual"],
                          L_um=c["L_um"], W_um=c["W_um"]) for l, c in enumerate(compiled)],
        bpm_equiv_rel_err=bpm_equiv_rel_err,
        net_digital=m_net_dig,
        net_bpm_poisson=dict(
            metrics=m_net_bpm,
            total_useful_photons=st_bpm["total_useful_photons"],
            total_launched_photons=st_bpm["total_launched_photons"],
            dilation_overhead=st_bpm["dilation_overhead"],
            signal_energy_J=st_bpm["signal_energy_J"],
            lo_energy_J=st_bpm["lo_energy_J"],
        ),
        net_bpm_rough50=dict(metrics=m_net_r50),
        elapsed_s=round(time.time() - t0, 1),
    )
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(
        os.path.join(OUT_DIR, "samples.npz"),
        ref=ref_samples,
        samp_dig=samp_dig,
        samp_bpm=samp_bpm,
        samp_r50=samp_r50,
        traj_0=st_bpm["traj"][0],
        traj_40=st_bpm["traj"][40],
        traj_80=st_bpm["traj"][80],
        traj_120=st_bpm["traj"][120],
        traj_160=st_bpm["traj"][160],
    )
    print(f"Saved Experiment 1 to {OUT_DIR} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
