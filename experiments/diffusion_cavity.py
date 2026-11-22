"""Experiment 4: All-Optical Nonlinear Cavity & Folded Single-Process Photonic Diffusion.

Tests TWO all-optical architectures on 8x8 MNIST (d = 64, N = 64 guided modes)
and compiles them directly onto physical shallow-etched MgO:LN/SRN programmable
multimode waveguides (NA = 0.45, L = 2.5 mm, 0.30 dB/cm vertical SRN leakage,
dn_max = 5e-3, 1 um resolution) via split-step BPM adjoints on CUDA:

  Part A — All-Optical Hidden-Nonlinearity Cavity (1 balanced-homodyne readout/step;
           ZERO intermediate O-E-O conversions):
    - A1: 1 shared physical PMW in Kerr + SA ring cavity (6 round trips) + 1 output PMW
    - A2: 1 shared physical PMW in pure lossless chi^(3) Kerr ring cavity (6 round trips)
    - A3: 1 shared physical PMW in pure saturable-absorber ring cavity (6 round trips)
    - A4: 6 cascaded all-optical Kerr + SA PMWs (zero intermediate O-E-O)

  Part B — Folded Single-Process All-Optical Cavity (ZERO intermediate O-E-O across
           ALL T reverse steps; ONLY ONE final 8-port readout at t = 0!):
    - B1: Complex-mode (d/2 = 32 complex pixel modes + 32 conditioning modes) shared
          1-PMW Kerr + SA score branch inside a 100% passive static storage ring
          (uniform-s schedule: constant passive loss beta_step = 1 - exp(-Delta_s))
    - B2: Cascaded 4-stage complex-mode Kerr + SA score branch (no inner loop delay
          constraint; 1 transit through side-branch per storage-ring round trip)

Hardware stress tests on compiled physical MgO:LN/SRN BPM waveguides:
  - Direct field-by-field BPM propagation verification
  - Exact Poisson balanced-homodyne detection + intracavity loss/SA vacuum noise
  - Operating nonlinear photon number sweep n_op in {10^2, 10^3, 10^4, 10^5} ph/mode
    (with exact branch optical amplifier ASE noise)
  - Cavity round-trip phase-locking jitter sweep sigma_phi in {0, 0.02, 0.05, 0.10, 0.20} rad
  - Waveguide sidewall roughness (sigma = 50 nm)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pwg.cavity_diffusion import (
    FoldedCavityHardware,
    OpticalCavityConfig,
    OpticalCavityScoreNet,
    sample_folded_optical_diffusion,
)
from pwg.diffusion import (
    ComplexAdam,
    HomodyneDetector,
    VPSchedule,
    compile_unitary_to_bpm,
    evaluate_compiled_bpm,
    sample_photonic_diffusion,
)
from experiments.diffusion_mnist import eval_generated, load_mnist_8x8

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "results", "diffusion_cavity")
os.makedirs(OUT_DIR, exist_ok=True)


def train_cavity_net(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    sched: VPSchedule,
    cfg: OpticalCavityConfig,
    n_iters: int = 16000,
    batch_size: int = 256,
    lr_max: float = 4e-3,
    label: str = "cavity",
) -> OpticalCavityScoreNet:
    t0 = time.time()
    net = OpticalCavityScoreNet(cfg)
    opt = ComplexAdam(net, lr=lr_max)
    ema = {k: v.copy() for k, v in net.param_items()}
    ema_decay = 0.998
    rng = np.random.default_rng(cfg.seed + 100)
    n_tr = len(x_tr)

    for it in range(1, n_iters + 1):
        idx = rng.integers(0, n_tr, size=batch_size)
        x0 = x_tr[idx]
        yb = y_tr[idx] if cfg.n_classes > 0 else None
        u = rng.uniform(0.0, 1.0, size=(batch_size,))
        tb = sched.eps + (1.0 - sched.eps) * (u ** 1.35)
        ab, sb = sched.alpha_sigma(tb)
        eb = rng.normal(size=x0.shape)
        xt = ab[:, None] * x0 + sb[:, None] * eb
        tgt = sb[:, None] * x0 - ab[:, None] * eb

        if it <= 300:
            lr = lr_max * (it / 300.0)
        else:
            frac = (it - 300) / max(n_iters - 300, 1)
            lr = 2e-4 + 0.5 * (lr_max - 2e-4) * (1.0 + math.cos(math.pi * frac))

        loss, grads = net.loss_and_grads(xt, tb, tgt, y=yb)
        opt.step(grads, lr=lr)

        if it >= 500:
            for k, v in net.param_items():
                ema[k] = ema_decay * ema[k] + (1.0 - ema_decay) * v
        else:
            for k, v in net.param_items():
                ema[k][...] = v

        if it % 2000 == 0 or it == n_iters:
            print(f"  [{label}] it {it:5d}/{n_iters}: loss={loss:.4f}, lr={lr:.4g} ({time.time()-t0:.1f}s)", flush=True)

    for k, v in net.param_items():
        v[...] = ema[k]
    return net


def compile_cavity_bpm(net: OpticalCavityScoreNet, compile_iters: int = 140, label: str = "cavity") -> list[dict]:
    """Compile all distinct physical PMWs (U_wg_0..n_wg-1 and U_out) onto MgO:LN/SRN BPM."""
    Us, U_out = net.get_unitaries()
    all_Us = list(Us) + [U_out]
    compiled = []
    for idx, U in enumerate(all_Us):
        name = f"{label}_wg{idx}" if idx < len(Us) else f"{label}_out"
        t0 = time.time()
        c = compile_unitary_to_bpm(
            U, length_um=2500.0, dn_max=5e-3, na=0.45,
            res_um=1.0, iters=compile_iters, seed=200 + idx,
        )
        print(
            f"  [{name}] F={c['F']:.5f}, F_n={c['Fn']:.5f}, err={c['err']:.4f}, "
            f"eta={c['eta_actual']:.4f} ({time.time()-t0:.1f}s)",
            flush=True,
        )
        compiled.append(c)
    return compiled


def save_compiled_list(path: str, compiled: list[dict]):
    payload = dict(n_chips=np.array(len(compiled)))
    for i, c in enumerate(compiled):
        payload[f"theta_{i}"] = c["theta"]
        payload[f"T_eff_{i}"] = c["T_eff"]
        payload[f"D_phase_{i}"] = c["D_phase"]
        payload[f"global_phase_{i}"] = np.array(c["global_phase"])
        payload[f"meta_{i}"] = np.array(json.dumps(dict(
            F=c["F"], Fn=c["Fn"], err=c["err"], eta_actual=c["eta_actual"],
            L_um=c["L_um"], W_um=c["W_um"], cfg=c["cfg"],
        )))
    np.savez(path, **payload)


def load_compiled_list(path: str) -> list[dict]:
    z = np.load(path, allow_pickle=False)
    n = int(z["n_chips"])
    out = []
    for i in range(n):
        meta = json.loads(str(z[f"meta_{i}"]))
        out.append(dict(
            theta=z[f"theta_{i}"],
            T_eff=z[f"T_eff_{i}"],
            D_phase=z[f"D_phase_{i}"],
            global_phase=complex(z[f"global_phase_{i}"]),
            **meta,
        ))
    return out


def verify_bpm_field_equivalence(compiled_entry: dict, U_target: np.ndarray, seed: int = 0) -> float:
    """Verify that direct split-step BPM wave propagation matches compiled T_eff to machine precision."""
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("cuda_ad_mono")
    from drjit.cuda import Float as FloatNP
    from drjit.cuda.ad import Float as FloatAD
    from pwg.bpm import propagate
    from pwg.mmwg import MMWGConfig, MultimodeWaveguide

    c = compiled_entry
    wg = MultimodeWaveguide(MMWGConfig(**c["cfg"]))
    dn = wg.index_map(FloatAD(c["theta"].astype(np.float32)))
    T_eff = c["T_eff"]

    rng = np.random.default_rng(seed)
    N = U_target.shape[0]
    a_demo = (rng.normal(size=(4, N)) + 1j * rng.normal(size=(4, N))).astype(np.complex64)
    e_in = a_demo @ wg.modes.astype(np.complex64)
    re_o, im_o = propagate(wg.prop, dn, FloatNP(e_in.real.ravel()), FloatNP(e_in.imag.ravel()), 4)
    out_f = (re_o.numpy() + 1j * im_o.numpy()).reshape(4, -1)
    a_out_direct = (out_f @ wg.modes.conj().T) * wg.grid.dx
    a_out_direct_eff = (a_out_direct * c["D_phase"].conj()[None, :]) * c["global_phase"]
    a_out_mat = a_demo @ T_eff.T
    return float(np.linalg.norm(a_out_direct_eff - a_out_mat) / max(np.linalg.norm(a_out_mat), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_compile", action="store_true")
    ap.add_argument("--n_iters", type=int, default=16000)
    ap.add_argument("--n_iters_ablation", type=int, default=10000)
    ap.add_argument("--compile_iters", type=int, default=140)
    args = ap.parse_args()

    t_start = time.time()
    x_tr, y_tr, x_te, y_te, mu, sd = load_mnist_8x8()
    sched = VPSchedule(beta_min=0.1, beta_max=20.0, eps=2e-3)
    print(f"Loaded 8x8 MNIST: train={x_tr.shape}, test={x_te.shape}", flush=True)

    # Independent evaluation classifier (identical seed & hyperparams to diffusion_mnist.py)
    tr_imgs = np.clip(x_tr * sd + mu, 0.0, 1.0)
    te_imgs = np.clip(x_te * sd + mu, 0.0, 1.0)
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=60, random_state=0)
    clf.fit(tr_imgs, y_tr)
    real_te_acc = float(clf.score(te_imgs, y_te))
    print(f"Real 8x8 MNIST classifier test acc = {real_te_acc*100:.2f}%", flush=True)

    # ------------------------------------------------------------------
    # PART A: All-Optical Hidden-Nonlinearity Cavities (1 readout/step)
    # ------------------------------------------------------------------
    specs_A = [
        ("A1_shared_kerr_sa", OpticalCavityConfig(
            data_dim=64, n_modes=64, n_roundtrips=6, time_dim=32, n_classes=10,
            nonlinearity="kerr_sa", shared_waveguide=True, complex_state=False,
            cavity_feedback_r=0.82, gamma_kerr=0.65, alpha0_sa=1.6, eta_chip=0.966,
            ancilla_reg=0.002, seed=101,
        ), args.n_iters),
        ("A2_shared_kerr", OpticalCavityConfig(
            data_dim=64, n_modes=64, n_roundtrips=6, time_dim=32, n_classes=10,
            nonlinearity="kerr", shared_waveguide=True, complex_state=False,
            cavity_feedback_r=0.82, gamma_kerr=0.75, alpha0_sa=0.0, eta_chip=0.966,
            ancilla_reg=0.002, seed=102,
        ), args.n_iters_ablation),
        ("A3_shared_sa", OpticalCavityConfig(
            data_dim=64, n_modes=64, n_roundtrips=6, time_dim=32, n_classes=10,
            nonlinearity="sa", shared_waveguide=True, complex_state=False,
            cavity_feedback_r=0.82, gamma_kerr=0.0, alpha0_sa=1.8, eta_chip=0.966,
            ancilla_reg=0.002, seed=103,
        ), args.n_iters_ablation),
        ("A4_cascaded_kerr_sa", OpticalCavityConfig(
            data_dim=64, n_modes=64, n_roundtrips=6, time_dim=32, n_classes=10,
            nonlinearity="kerr_sa", shared_waveguide=False, complex_state=False,
            cavity_feedback_r=0.82, gamma_kerr=0.65, alpha0_sa=1.6, eta_chip=0.966,
            ancilla_reg=0.002, seed=104,
        ), args.n_iters_ablation),
    ]

    # ------------------------------------------------------------------
    # PART B: Folded Single-Process All-Optical Cavities (1 readout TOTAL!)
    # ------------------------------------------------------------------
    specs_B = [
        ("B1_folded_shared_kerr_sa", OpticalCavityConfig(
            data_dim=64, n_modes=64, n_roundtrips=6, time_dim=32, n_classes=10,
            nonlinearity="kerr_sa", shared_waveguide=True, complex_state=True,
            cavity_feedback_r=0.82, gamma_kerr=0.65, alpha0_sa=1.6, eta_chip=0.966,
            ancilla_reg=0.002, seed=201,
        ), args.n_iters),
        ("B2_folded_cascaded4_kerr_sa", OpticalCavityConfig(
            data_dim=64, n_modes=64, n_roundtrips=4, time_dim=32, n_classes=10,
            nonlinearity="kerr_sa", shared_waveguide=False, complex_state=True,
            cavity_feedback_r=0.82, gamma_kerr=0.65, alpha0_sa=1.6, eta_chip=0.966,
            ancilla_reg=0.002, seed=202,
        ), args.n_iters),
    ]

    nets = {}
    for name, cfg, nit in specs_A + specs_B:
        p = os.path.join(OUT_DIR, f"{name}.npz")
        if args.skip_train and os.path.exists(p):
            print(f"Loading {name} from {p}", flush=True)
            nets[name] = OpticalCavityScoreNet.load(p)
        else:
            print(f"\nTraining {name} (n_wg={1 if cfg.shared_waveguide else cfg.n_roundtrips}, "
                  f"M_rt={cfg.n_roundtrips}, nl={cfg.nonlinearity}, complex={cfg.complex_state})...", flush=True)
            nets[name] = train_cavity_net(x_tr, y_tr, sched, cfg, n_iters=nit, batch_size=256, lr_max=4e-3, label=name)
            nets[name].save(p)

    # ------------------------------------------------------------------
    # Evaluate Part A (ideal unitaries first)
    # ------------------------------------------------------------------
    results = {"real_test_acc": real_te_acc, "part_A": {}, "part_B": {}, "bpm_compile": {}}
    y_eval = np.repeat(np.arange(10), 100)  # 1,000 samples (100 per digit)
    det_ideal = HomodyneDetector(eta_qe=0.80, eta_chip=0.966, lo_photons_per_mode=1e5, read_noise_e=30.0)

    print("\n=== Evaluating Part A: All-Optical Hidden Cavities (1 readout/step) ===", flush=True)
    for name, _, _ in specs_A:
        net = nets[name]
        rng_s = np.random.default_rng(700)
        # Exact Poisson homodyne + intracavity vacuum noise at 1e4 photons/mode
        samps, sinfo = sample_photonic_diffusion(
            lambda x, tv, n=net, r=rng_s: n.forward_optical_field(x, tv, y=y_eval, cav_photons_per_mode=1e4, rng=r),
            sched, det_ideal, shape=(1000, 64), n_steps=60, rng=rng_s, mode="homodyne_residual",
        )
        m_hom = eval_generated(samps, y_eval, x_te, x_tr[:10000], clf, mu, sd)
        # Digital reference
        rng_d = np.random.default_rng(701)
        samps_d, _ = sample_photonic_diffusion(
            lambda x, tv, n=net: n.forward(x, tv, y=y_eval),
            sched, det_ideal, shape=(1000, 64), n_steps=60, rng=rng_d, mode="digital",
        )
        m_dig = eval_generated(samps_d, y_eval, x_te, x_tr[:10000], clf, mu, sd)
        results["part_A"][name] = dict(
            homodyne_T60=m_hom,
            digital_T60=m_dig,
            useful_photons=sinfo["total_useful_photons"],
            launched_photons=sinfo["total_launched_photons"],
            n_distinct_pmws=net.n_wg + 1,
        )
        print(
            f"  [{name}] Homodyne T=60: acc={m_hom['cond_accuracy']*100:.1f}%, FD={m_hom['fd_pixel']:.3f}, "
            f"useful_ph={sinfo['total_useful_photons']:.1f}, launched_ph={sinfo['total_launched_photons']:.1f} | "
            f"Digital: acc={m_dig['cond_accuracy']*100:.1f}%, FD={m_dig['fd_pixel']:.3f}",
            flush=True,
        )

    # Save 1 real test digit per class (0..9) + standardization stats for plotting
    x_real_10 = np.stack([x_te[np.where(y_te == c)[0][0]] for c in range(10)], axis=0)
    np.savez(os.path.join(OUT_DIR, "mnist_ref.npz"), x_real=x_real_10, mu=np.array(mu), sd=np.array(sd))

    # ------------------------------------------------------------------
    # Compile physical MgO:LN/SRN BPM waveguides for A1 (2), B1 (2), and B2 (5)
    # ------------------------------------------------------------------
    print("\n=== Compiling MgO:LN/SRN BPM waveguides (NA=0.45, L=2.5mm, 0.30 dB/cm) ===", flush=True)
    for key in ["A1_shared_kerr_sa", "B1_folded_shared_kerr_sa", "B2_folded_cascaded4_kerr_sa"]:
        cpath = os.path.join(OUT_DIR, f"compiled_{key}.npz")
        net = nets[key]
        if args.skip_compile and os.path.exists(cpath):
            print(f"Loading compiled BPM for {key} from {cpath}", flush=True)
            comp = load_compiled_list(cpath)
        else:
            comp = compile_cavity_bpm(net, compile_iters=args.compile_iters, label=key)
            save_compiled_list(cpath, comp)

        # Verify direct BPM field equivalence on chip 0
        Us, U_out = net.get_unitaries()
        field_err = verify_bpm_field_equivalence(comp[0], Us[0], seed=42)
        T_wgs = [c["T_eff"] for c in comp[:-1]]
        T_out = comp[-1]["T_eff"]
        net.compiled_T_wg = T_wgs
        net.compiled_T_out = T_out
        eta_actual_mean = float(np.mean([c["eta_actual"] for c in comp]))

        # Also evaluate 50 nm sidewall roughness on all compiled chips!
        from pwg.mmwg import fidelity
        T_rough50_wgs = []
        rough_fn0 = None
        for idx, c in enumerate(comp[:-1]):
            Te_r, _ = evaluate_compiled_bpm(
                c["theta"], c["cfg"], c["D_phase"], c["global_phase"],
                rough_sigma_nm=50.0, rough_seed=900 + idx,
            )
            T_rough50_wgs.append(Te_r)
            if idx == 0:
                _, rough_fn0, _ = fidelity(Te_r, Us[0])
        Te_out_r, _ = evaluate_compiled_bpm(
            comp[-1]["theta"], comp[-1]["cfg"], comp[-1]["D_phase"], comp[-1]["global_phase"],
            rough_sigma_nm=50.0, rough_seed=950,
        )
        T_rough50 = (T_rough50_wgs, Te_out_r)

        results["bpm_compile"][key] = dict(
            chips=[dict(F=c["F"], Fn=c["Fn"], err=c["err"], eta_actual=c["eta_actual"]) for c in comp],
            field_equiv_rel_err=field_err,
            eta_actual_mean=eta_actual_mean,
            rough50_Fn_wg0=float(rough_fn0),
        )
        print(f"  [{key}] Direct BPM field equiv rel_err = {field_err:.2e}, mean eta = {eta_actual_mean:.4f}", flush=True)

        # Save BPM rough50 matrices in memory for stress tests
        net._T_rough50 = T_rough50

    # ------------------------------------------------------------------
    # Full BPM hardware evaluation for A1 (1 readout/step, 2 compiled chips)
    # ------------------------------------------------------------------
    print("\n=== BPM Hardware Evaluation: A1 All-Optical Hidden Cavity ===", flush=True)
    net_A1 = nets["A1_shared_kerr_sa"]
    eta_A1 = results["bpm_compile"]["A1_shared_kerr_sa"]["eta_actual_mean"]
    det_A1 = HomodyneDetector(eta_qe=0.80, eta_chip=eta_A1, lo_photons_per_mode=1e5, read_noise_e=30.0)
    A1_hw = {}

    for T_steps in [60, 120]:
        rng_a = np.random.default_rng(800 + T_steps)
        samps, sinfo = sample_photonic_diffusion(
            lambda x, tv: net_A1.forward_optical_field(x, tv, y=y_eval, cav_photons_per_mode=1e4, rng=rng_a),
            sched, det_A1, shape=(1000, 64), n_steps=T_steps, rng=rng_a, mode="homodyne_residual",
        )
        m = eval_generated(samps, y_eval, x_te, x_tr[:10000], clf, mu, sd)
        A1_hw[f"bpm_homodyne_T{T_steps}"] = dict(
            **m,
            useful_photons=sinfo["total_useful_photons"],
            launched_photons=sinfo["total_launched_photons"],
        )
        if T_steps == 60:
            per_dig = np.stack([samps[c * 100] for c in range(10)], axis=0)
            np.save(os.path.join(OUT_DIR, "samples_A1_per_digit.npy"), per_dig)
            np.save(os.path.join(OUT_DIR, "samples_A1_bpm_T60.npy"), samps[::10])
        print(f"  [A1 BPM T={T_steps}] acc={m['cond_accuracy']*100:.1f}%, FD={m['fd_pixel']:.3f}, "
              f"useful_ph={sinfo['total_useful_photons']:.1f}, launched_ph={sinfo['total_launched_photons']:.1f}", flush=True)

    # A1 stress test: 50 nm sidewall roughness + intracavity photon sweep
    rng_r = np.random.default_rng(850)
    samps_r, _ = sample_photonic_diffusion(
        lambda x, tv: net_A1.forward_optical_field(
            x, tv, y=y_eval, override_T=net_A1._T_rough50, cav_photons_per_mode=1e4, rng=rng_r,
        ),
        sched, det_A1, shape=(1000, 64), n_steps=60, rng=rng_r, mode="homodyne_residual",
    )
    A1_hw["bpm_rough50_T60"] = eval_generated(samps_r, y_eval, x_te, x_tr[:10000], clf, mu, sd)
    print(f"  [A1 BPM 50nm rough] acc={A1_hw['bpm_rough50_T60']['cond_accuracy']*100:.1f}%, "
          f"FD={A1_hw['bpm_rough50_T60']['fd_pixel']:.3f}", flush=True)

    for ncav in [1e2, 1e3, 1e4, 1e5]:
        rng_c = np.random.default_rng(860 + int(math.log10(ncav)))
        sc, _ = sample_photonic_diffusion(
            lambda x, tv: net_A1.forward_optical_field(x, tv, y=y_eval, cav_photons_per_mode=ncav, rng=rng_c),
            sched, det_A1, shape=(1000, 64), n_steps=60, rng=rng_c, mode="homodyne_residual",
        )
        mc = eval_generated(sc, y_eval, x_te, x_tr[:10000], clf, mu, sd)
        A1_hw[f"bpm_ncav_{int(ncav)}"] = mc
        print(f"  [A1 BPM n_cav={ncav:.0e}] acc={mc['cond_accuracy']*100:.1f}%, FD={mc['fd_pixel']:.3f}", flush=True)

    results["part_A"]["A1_bpm_hardware"] = A1_hw

    # ------------------------------------------------------------------
    # PART B: Folded Single-Process All-Optical Cavity (1 readout TOTAL!)
    # ------------------------------------------------------------------
    print("\n=== Evaluating Part B: Folded Single-Process All-Optical Cavity (1 readout TOTAL) ===", flush=True)
    for name in ["B1_folded_shared_kerr_sa", "B2_folded_cascaded4_kerr_sa"]:
        net_B = nets[name]
        eta_B = results["bpm_compile"][name]["eta_actual_mean"] if name in results["bpm_compile"] else 0.966
        det_B = HomodyneDetector(eta_qe=0.80, eta_chip=eta_B, lo_photons_per_mode=1e5, read_noise_e=30.0)
        b_res = {}

        # Nominal strictly-passive uniform-s storage ring (T=60, s_max=8.0 -> beta_step=12.48% > 11.1% passive loss!)
        hw_nom = FoldedCavityHardware(
            n_op_photons_per_mode=1e4, ring_chip_trans=eta_B,
            rho_tap=0.04, rho_rec=0.04, amp_nsp=1.5, phase_jitter_rad=0.0,
        )
        for grid_mode, n_steps in [("uniform_s", 60), ("uniform_s", 80), ("uniform_t", 60)]:
            rng_b = np.random.default_rng(900 + n_steps)
            samps_b, binfo = sample_folded_optical_diffusion(
                net_B, sched, hw_nom, det_B, n_samples=1000, n_steps=n_steps,
                y=y_eval, rng=rng_b, grid_mode=grid_mode, s_max=8.0, s_min=0.002,
                record_steps=[0, 15, 30, 45, 55, 60] if (name == "B1_folded_shared_kerr_sa" and grid_mode == "uniform_s" and n_steps == 60) else None,
            )
            mb = eval_generated(samps_b, y_eval, x_te, x_tr[:10000], clf, mu, sd)
            traj_saved = binfo.pop("trajectory")
            b_res[f"{grid_mode}_T{n_steps}"] = dict(**mb, **binfo)
            print(
                f"  [{name} {grid_mode} T={n_steps}] acc={mb['cond_accuracy']*100:.1f}%, FD={mb['fd_pixel']:.3f}, "
                f"passive_ring={binfo['ring_strictly_passive']}, beta_step={binfo['beta_step_min']*100:.1f}%, "
                f"G_br={binfo['branch_amp_gain_mean_dB']:.1f}dB, wallplug={binfo['total_wallplug_nJ']:.2f}nJ, "
                f"lat={binfo['total_latency_ns']:.1f}ns",
                flush=True,
            )
            if grid_mode == "uniform_s" and n_steps == 60:
                tag = "B1" if name == "B1_folded_shared_kerr_sa" else "B2"
                per_dig_b = np.stack([samps_b[c * 100] for c in range(10)], axis=0)
                np.save(os.path.join(OUT_DIR, f"samples_{tag}_per_digit.npy"), per_dig_b)
                np.save(os.path.join(OUT_DIR, f"samples_{tag}_folded_bpm_T60.npy"), samps_b[::10])
                if name == "B1_folded_shared_kerr_sa":
                    # Save trajectories for 1 sample of each class (0..9)
                    np.savez(
                        os.path.join(OUT_DIR, "traj_B1_folded_bpm_T60.npz"),
                        **{f"step_{k}": np.stack([v[c * 100] for c in range(10)], axis=0) for k, v in traj_saved.items()},
                    )

        # Hardware stress sweeps for BPM-compiled B1
        if name == "B1_folded_shared_kerr_sa":
            # 1. Operating photon number n_op sweep (including branch OPA ASE noise!)
            nop_sweep = {}
            for n_op in [1e2, 3e2, 1e3, 1e4, 1e5]:
                hw_n = FoldedCavityHardware(
                    n_op_photons_per_mode=n_op, ring_chip_trans=eta_B,
                    rho_tap=0.04, rho_rec=0.04, amp_nsp=1.5, phase_jitter_rad=0.0,
                )
                rng_n = np.random.default_rng(1000 + int(n_op))
                sn, bn = sample_folded_optical_diffusion(
                    net_B, sched, hw_n, det_B, n_samples=1000, n_steps=60,
                    y=y_eval, rng=rng_n, grid_mode="uniform_s", s_max=8.0, s_min=0.002,
                )
                bn.pop("trajectory")
                mn = eval_generated(sn, y_eval, x_te, x_tr[:10000], clf, mu, sd)
                nop_sweep[f"nop_{int(n_op)}"] = dict(**mn, **bn)
                print(
                    f"  [B1 BPM n_op={n_op:.0e}] acc={mn['cond_accuracy']*100:.1f}%, FD={mn['fd_pixel']:.3f}, "
                    f"amp_noise_frac={bn['amp_noise_fraction']*100:.2f}%, wallplug={bn['total_wallplug_nJ']:.3f}nJ",
                    flush=True,
                )
            b_res["nop_sweep"] = nop_sweep

            # 2. Cavity round-trip phase-locking jitter sweep
            phi_sweep = {}
            for sphi in [0.0, 0.02, 0.05, 0.10, 0.20]:
                hw_p = FoldedCavityHardware(
                    n_op_photons_per_mode=1e4, ring_chip_trans=eta_B,
                    rho_tap=0.04, rho_rec=0.04, amp_nsp=1.5, phase_jitter_rad=sphi,
                )
                rng_p = np.random.default_rng(1100 + int(sphi * 1000))
                sp, bp = sample_folded_optical_diffusion(
                    net_B, sched, hw_p, det_B, n_samples=1000, n_steps=60,
                    y=y_eval, rng=rng_p, grid_mode="uniform_s", s_max=8.0, s_min=0.002,
                )
                bp.pop("trajectory")
                mp = eval_generated(sp, y_eval, x_te, x_tr[:10000], clf, mu, sd)
                phi_sweep[f"phi_{sphi:.2f}"] = dict(**mp, **bp)
                print(
                    f"  [B1 BPM phase_jitter={sphi:.2f} rad] acc={mp['cond_accuracy']*100:.1f}%, FD={mp['fd_pixel']:.3f}",
                    flush=True,
                )
            b_res["phase_jitter_sweep"] = phi_sweep

            # 3. 50 nm BPM sidewall roughness on folded cavity
            rng_rb = np.random.default_rng(1200)
            srb, brb = sample_folded_optical_diffusion(
                net_B, sched, hw_nom, det_B, n_samples=1000, n_steps=60,
                y=y_eval, rng=rng_rb, grid_mode="uniform_s", s_max=8.0, s_min=0.002,
                override_T=net_B._T_rough50,
            )
            brb.pop("trajectory")
            mrb = eval_generated(srb, y_eval, x_te, x_tr[:10000], clf, mu, sd)
            b_res["bpm_rough50_T60"] = dict(**mrb, **brb)
            print(
                f"  [B1 BPM 50nm rough] acc={mrb['cond_accuracy']*100:.1f}%, FD={mrb['fd_pixel']:.3f}",
                flush=True,
            )

        results["part_B"][name] = b_res

    out_json = os.path.join(OUT_DIR, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {out_json} in {time.time() - t_start:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
