"""Experiment 2 & 3: End-to-end physical N=64 programmable multimode waveguide
denoising diffusion model on 8x8 MNIST digits, driven purely by exact Poisson
balanced-homodyne quantum shot noise.

Pipeline:
  1. Load MNIST (60k train / 10k test), crop 2-px blank margin (24x24) -> exact
     3x3 block average -> sharp 8x8 digits (d = 64 pixels), globally standardized.
  2. Train an independent MLP evaluation classifier on real 8x8 MNIST (for objective
     generated-digit accuracy, class balance, and feature-space Frechet distance).
  3. Train (or load via --skip_train) N=64 complex UnitaryScoreNet models:
     (a) Primary: 6-layer class-conditional, split-quadrature SiLU, ancilla_reg=0.002
     (b) Pure-optical nonlinearity: 4-layer class-conditional, photodiode |z|^2
     (c) Unconditional: 4-layer SiLU, no class labels
  4. Compile ALL unitary layers U_1..U_L in U(64) of the primary model into real
     shallow-etched MgO:LN/SRN MultimodeWaveguide chips (NA=0.45, L=2.5mm, 0.30 dB/cm
     vertical SRN leakage, dn_max=5e-3, 1 um resolution) via split-step BPM adjoints
     on CUDA, save compiled_bpm.npz, verify direct field-by-field BPM propagation,
     and record a 2D |E(x,z)|^2 intensity map.
  5. Sample 1,000 digits (100 per class 0..9) through the compiled physical BPM
     waveguides using ONLY exact Poisson balanced-homodyne detection (zero external
     Gaussian PRNG), and run hardware stress tests:
       - Sidewall roughness (sigma = 10, 20, 50 nm)
       - SRN cavity thickness perturbation (4.2 um -> 0.87 dB/cm)
       - Finite programming resolution (r = 2.0, 3.0 um)
       - Finite hidden-layer shot noise
       - Slow correlated temporal drift across reverse steps (vs. open-loop drift)
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
from pwg.diffusion import (
    ComplexAdam,
    HomodyneDetector,
    PhotonicNetConfig,
    UnitaryScoreNet,
    VPSchedule,
    compile_unitary_to_bpm,
    evaluate_compiled_bpm,
    sample_photonic_diffusion,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "results", "diffusion_mnist")
os.makedirs(OUT_DIR, exist_ok=True)


def load_mnist_8x8():
    d = np.load(os.path.join(ROOT, "data", "mnist.npz"))
    def proc(imgs):
        c = imgs[:, 2:26, 2:26].astype(np.float64) / 255.0
        return c.reshape(len(c), 8, 3, 8, 3).mean(axis=(2, 4)).reshape(len(c), 64)
    x_tr = proc(d["x_train"])
    y_tr = d["y_train"].astype(int)
    x_te = proc(d["x_test"])
    y_te = d["y_test"].astype(int)
    mu = float(x_tr.mean())
    sd = float(x_tr.std())
    return (x_tr - mu) / sd, y_tr, (x_te - mu) / sd, y_te, mu, sd


def frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    m1, m2 = x.mean(axis=0), y.mean(axis=0)
    c1 = np.cov(x, rowvar=False) + 1e-6 * np.eye(x.shape[1])
    c2 = np.cov(y, rowvar=False) + 1e-6 * np.eye(y.shape[1])
    diff = m1 - m2
    w1, v1 = np.linalg.eigh(c1)
    c1_sqrt = (v1 * np.sqrt(np.maximum(w1, 0.0))) @ v1.T
    mid = c1_sqrt @ c2 @ c1_sqrt
    mid = 0.5 * (mid + mid.T)
    wm = np.linalg.eigvalsh(mid)
    tr_sqrt = float(np.sum(np.sqrt(np.maximum(wm, 0.0))))
    return float(np.dot(diff, diff) + np.trace(c1) + np.trace(c2) - 2.0 * tr_sqrt)


def eval_generated(samples: np.ndarray, y_cond: np.ndarray | None, x_te: np.ndarray,
                   x_tr_sub: np.ndarray, clf: MLPClassifier, mu: float, sd: float) -> dict:
    imgs = np.clip(samples * sd + mu, 0.0, 1.0)
    te_imgs = np.clip(x_te * sd + mu, 0.0, 1.0)
    tr_imgs = np.clip(x_tr_sub * sd + mu, 0.0, 1.0)

    probs = clf.predict_proba(imgs)
    preds = np.argmax(probs, axis=1)
    conf = float(np.mean(np.max(probs, axis=1)))
    counts = np.bincount(preds, minlength=10).astype(float)
    p_cls = counts / counts.sum()
    cls_ent = float(-np.sum(p_cls * np.log(np.maximum(p_cls, 1e-12))))
    fd_pix = frechet_distance(imgs, te_imgs[:2000])

    d2 = np.sum((imgs[:200, None, :] - tr_imgs[None, :, :]) ** 2, axis=-1)
    nn_dist = float(np.mean(np.sqrt(np.min(d2, axis=1))))

    out = dict(
        fd_pixel=fd_pix,
        mean_clf_confidence=conf,
        class_entropy_nats=cls_ent,
        max_entropy_nats=math.log(10.0),
        nn_l2_dist=nn_dist,
    )
    if y_cond is not None:
        out["cond_accuracy"] = float(np.mean(preds == y_cond))
    return out


def train_score_net(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    sched: VPSchedule,
    cfg: PhotonicNetConfig,
    n_iters: int = 20000,
    batch_size: int = 256,
    lr_max: float = 4e-3,
    label: str = "model",
) -> UnitaryScoreNet:
    t0 = time.time()
    net = UnitaryScoreNet(cfg)
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


def save_compiled(path: str, compiled: list[dict]):
    payload = dict(n_layers=np.array(len(compiled)))
    for l, c in enumerate(compiled):
        payload[f"theta_{l}"] = c["theta"]
        payload[f"T_eff_{l}"] = c["T_eff"]
        payload[f"D_phase_{l}"] = c["D_phase"]
        payload[f"global_phase_{l}"] = np.array(c["global_phase"])
        payload[f"meta_{l}"] = np.array(json.dumps(dict(
            F=c["F"], Fn=c["Fn"], err=c["err"], eta_actual=c["eta_actual"],
            L_um=c["L_um"], W_um=c["W_um"], cfg=c["cfg"],
        )))
    np.savez(path, **payload)


def load_compiled(path: str) -> list[dict]:
    z = np.load(path, allow_pickle=False)
    L = int(z["n_layers"])
    out = []
    for l in range(L):
        meta = json.loads(str(z[f"meta_{l}"]))
        out.append(dict(
            theta=z[f"theta_{l}"],
            T_eff=z[f"T_eff_{l}"],
            D_phase=z[f"D_phase_{l}"],
            global_phase=complex(z[f"global_phase_{l}"]),
            **meta,
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_compile", action="store_true")
    ap.add_argument("--main_model", default="net_main_6L.npz")
    ap.add_argument("--compile_iters", type=int, default=140)
    args = ap.parse_args()

    t_start = time.time()
    x_tr, y_tr, x_te, y_te, mu, sd = load_mnist_8x8()
    sched = VPSchedule(beta_min=0.1, beta_max=20.0, eps=2e-3)
    print(f"Loaded 8x8 MNIST: train={x_tr.shape}, test={x_te.shape}, mean={mu:.4f}, std={sd:.4f}", flush=True)

    # 1. Train independent evaluation classifier on real 8x8 MNIST
    print("Training evaluation classifier on real 8x8 MNIST...", flush=True)
    tr_imgs = np.clip(x_tr * sd + mu, 0.0, 1.0)
    te_imgs = np.clip(x_te * sd + mu, 0.0, 1.0)
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=60, random_state=0)
    clf.fit(tr_imgs, y_tr)
    real_te_acc = float(clf.score(te_imgs, y_te))
    real_ref_metrics = eval_generated(x_te[:1000], y_te[:1000], x_te[1000:], x_tr[:10000], clf, mu, sd)
    print(f"  Real 8x8 MNIST classifier test acc = {real_te_acc*100:.2f}%, "
          f"real-vs-real FD={real_ref_metrics['fd_pixel']:.4f}, nn_dist={real_ref_metrics['nn_l2_dist']:.3f}", flush=True)

    # 2. Load or train models
    main_path = os.path.join(OUT_DIR, args.main_model)
    sq_path = os.path.join(OUT_DIR, "net_square.npz")
    unc_path = os.path.join(OUT_DIR, "net_uncond.npz")

    if args.skip_train and os.path.exists(main_path):
        print(f"Loading pre-trained primary model from {main_path}...", flush=True)
        net_main = UnitaryScoreNet.load(main_path)
    else:
        print("\nTraining primary 6-layer N=64 UnitaryScoreNet (SiLU, class-conditional)...", flush=True)
        cfg_main = PhotonicNetConfig(
            data_dim=64, n_modes=64, n_layers=6, time_dim=32, n_classes=10,
            nonlinearity="silu", eta_chip=0.97, ancilla_reg=0.002, seed=42,
        )
        net_main = train_score_net(x_tr, y_tr, sched, cfg_main, n_iters=20000, batch_size=256, lr_max=4e-3, label="6L-silu")
        net_main.save(main_path)

    if args.skip_train and os.path.exists(sq_path):
        net_sq = UnitaryScoreNet.load(sq_path)
    else:
        cfg_sq = PhotonicNetConfig(
            data_dim=64, n_modes=64, n_layers=4, time_dim=32, n_classes=10,
            nonlinearity="square", eta_chip=0.97, ancilla_reg=0.002, seed=43,
        )
        net_sq = train_score_net(x_tr, y_tr, sched, cfg_sq, n_iters=12000, batch_size=256, lr_max=3.5e-3, label="optical-sq")
        net_sq.save(sq_path)

    if args.skip_train and os.path.exists(unc_path):
        net_unc = UnitaryScoreNet.load(unc_path)
    else:
        cfg_unc = PhotonicNetConfig(
            data_dim=64, n_modes=64, n_layers=4, time_dim=32, n_classes=0,
            nonlinearity="silu", eta_chip=0.97, ancilla_reg=0.002, seed=44,
        )
        net_unc = train_score_net(x_tr, y_tr, sched, cfg_unc, n_iters=14000, batch_size=256, lr_max=4e-3, label="uncond-silu")
        net_unc.save(unc_path)

    # 3. Compile ALL unitary layers of net_main into real shallow-etched MgO:LN/SRN BPM waveguides
    comp_path = os.path.join(OUT_DIR, "compiled_bpm.npz")
    if args.skip_compile and os.path.exists(comp_path):
        print(f"Loading compiled BPM waveguides from {comp_path}...", flush=True)
        compiled = load_compiled(comp_path)
    else:
        Us = net_main.get_unitaries()
        print(f"\nCompiling {len(Us)} x N=64 unitaries into physical MgO:LN/SRN BPM waveguides (NA=0.45, L=2.5mm)...", flush=True)
        compiled = []
        for l, U_l in enumerate(Us):
            c_res = compile_unitary_to_bpm(
                U_l, na=0.45, length_um=2500.0, dn_max=5e-3, res_um=1.0,
                iters=args.compile_iters, lr=0.15, seed=100 + l,
            )
            compiled.append(c_res)
            print(f"  Compiled Layer {l}: F={c_res['F']:.4f}, F_n={c_res['Fn']:.5f}, "
                  f"err={c_res['err']:.4f}, eta_actual={c_res['eta_actual']:.4f}", flush=True)
        save_compiled(comp_path, compiled)

    T_bpm = [c["T_eff"] for c in compiled]
    eta_bpm_last = float(compiled[-1]["eta_actual"])
    net_main.compiled_T = T_bpm
    net_main.cfg.eta_chip = eta_bpm_last
    net_main.save(os.path.join(OUT_DIR, "net_main_compiled.npz"))

    # Direct field-by-field BPM propagation check + 2D |E(x,z)|^2 intensity snapshot on Layer 0
    import drjit as dr
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("cuda_ad_mono")
    from drjit.cuda import Float as FloatNP
    from drjit.cuda.ad import Float as FloatAD
    from pwg.mmwg import MMWGConfig, MultimodeWaveguide
    from pwg.bpm import propagate

    wg0 = MultimodeWaveguide(MMWGConfig(**compiled[0]["cfg"]))
    dn0 = wg0.index_map(FloatAD(compiled[0]["theta"].astype(np.float32)))
    x_demo = x_te[:4]
    y_demo = y_te[:4]
    cond_demo = net_main._cond_vec(np.full(4, 0.3), y_demo)
    a_demo = (net_main.in_scale[0] * x_demo + 1j * (cond_demo @ net_main.W_in_im + net_main.b_in_im[None, :])).astype(np.complex64)
    e_in = a_demo @ wg0.modes.astype(np.complex64)
    re_o, im_o = propagate(wg0.prop, dn0, FloatNP(e_in.real.ravel()), FloatNP(e_in.imag.ravel()), 4)
    out_f = (re_o.numpy() + 1j * im_o.numpy()).reshape(4, -1)
    a_out_direct = (out_f @ wg0.modes.conj().T) * wg0.grid.dx
    a_out_direct_eff = (a_out_direct * compiled[0]["D_phase"].conj()[None, :]) * compiled[0]["global_phase"]
    a_out_mat = a_demo @ T_bpm[0].T
    bpm_equiv_rel_err = float(np.linalg.norm(a_out_direct_eff - a_out_mat) / np.linalg.norm(a_out_mat))
    print(f"  Direct field-by-field BPM vs T_eff relative error (N=64): {bpm_equiv_rel_err:.2e}", flush=True)

    z_snaps = list(range(0, wg0.grid.nz, 10))
    rec = {k: None for k in z_snaps}
    e_single = e_in[0:1]
    dn0_np = dn0.numpy()
    wg0.prop.forward(FloatNP(dn0_np), FloatNP(e_single.real.ravel()), FloatNP(e_single.imag.ravel()), record=rec)
    field_map_2d = np.stack([rec[k] for k in z_snaps], axis=0).astype(np.float32)
    dn_layer0_ds = dn0_np.reshape(wg0.grid.nz, wg0.grid.nx)[::10, :].astype(np.float32)

    # 4. Evaluate all sampling conditions (1,000 samples: 100 per class 0..9)
    y_eval = np.repeat(np.arange(10), 100)
    det_bpm = HomodyneDetector(eta_qe=0.8, eta_chip=eta_bpm_last, lo_photons_per_mode=1e5, read_noise_e=30.0)
    det_ideal = HomodyneDetector(eta_qe=0.8, eta_chip=0.97, lo_photons_per_mode=1e5, read_noise_e=30.0)

    def make_fn(net_obj, y_arr, override_T=None, hidden_noise_std=0.0, rng_h=None):
        def fn(x, t_vec):
            a_u, ob = net_obj.forward_optical_field(
                x, t_vec, y=y_arr, override_T=override_T,
                hidden_noise_std=hidden_noise_std, rng=rng_h,
            )
            return a_u, ob, float(net_obj.out_gain[0])
        return fn

    results = dict(
        n_layers=net_main.cfg.n_layers,
        real_test_clf_acc=real_te_acc,
        real_reference=real_ref_metrics,
        bpm_compile=[dict(layer=l, F=c["F"], Fn=c["Fn"], err=c["err"], eta_actual=c["eta_actual"],
                          L_um=c["L_um"], W_um=c["W_um"]) for l, c in enumerate(compiled)],
        bpm_equiv_rel_err=bpm_equiv_rel_err,
        conditions={},
        stress_tests={},
    )

    print("\nSampling 1,000 8x8 MNIST digits across sampling modes...", flush=True)
    # (a) Digital PRNG reference (T=120)
    s_dig, _ = sample_photonic_diffusion(
        make_fn(net_main, y_eval, None), sched, det_bpm, shape=(1000, 64), n_steps=120,
        rng=np.random.default_rng(1001), mode="digital",
    )
    results["conditions"]["digital_prng_T120"] = eval_generated(s_dig, y_eval, x_te, x_tr[:10000], clf, mu, sd)

    # (b) Ideal unitary + exact Poisson balanced homodyne (T=120)
    s_uni_hom, st_uni_hom = sample_photonic_diffusion(
        make_fn(net_main, y_eval, None), sched, det_bpm, shape=(1000, 64), n_steps=120,
        rng=np.random.default_rng(1002), mode="homodyne_residual",
    )
    results["conditions"]["unitary_homodyne_T120"] = dict(
        eval_generated(s_uni_hom, y_eval, x_te, x_tr[:10000], clf, mu, sd),
        total_useful_photons=st_uni_hom["total_useful_photons"],
        total_launched_photons=st_uni_hom["total_launched_photons"],
        dilation_overhead=st_uni_hom["dilation_overhead"],
        signal_energy_fJ=st_uni_hom["signal_energy_J"] * 1e15,
        lo_energy_fJ=st_uni_hom["lo_energy_J"] * 1e15,
    )

    # (c) FULL PHYSICAL BPM + exact Poisson balanced homodyne (T=60 & T=120)
    s_bpm60, st_bpm60 = sample_photonic_diffusion(
        make_fn(net_main, y_eval, T_bpm), sched, det_bpm, shape=(1000, 64), n_steps=60,
        rng=np.random.default_rng(1003), mode="homodyne_residual",
    )
    results["conditions"]["bpm_poisson_T60"] = dict(
        eval_generated(s_bpm60, y_eval, x_te, x_tr[:10000], clf, mu, sd),
        total_useful_photons=st_bpm60["total_useful_photons"],
        total_launched_photons=st_bpm60["total_launched_photons"],
        dilation_overhead=st_bpm60["dilation_overhead"],
        signal_energy_fJ=st_bpm60["signal_energy_J"] * 1e15,
        useful_energy_fJ=st_bpm60["useful_energy_J"] * 1e15,
        lo_energy_fJ=st_bpm60["lo_energy_J"] * 1e15,
    )

    s_bpm120, st_bpm120 = sample_photonic_diffusion(
        make_fn(net_main, y_eval, T_bpm), sched, det_bpm, shape=(1000, 64), n_steps=120,
        rng=np.random.default_rng(1004), mode="homodyne_residual",
        record_steps=[0, 30, 60, 90, 108, 120],
    )
    results["conditions"]["bpm_poisson_T120"] = dict(
        eval_generated(s_bpm120, y_eval, x_te, x_tr[:10000], clf, mu, sd),
        total_useful_photons=st_bpm120["total_useful_photons"],
        total_launched_photons=st_bpm120["total_launched_photons"],
        dilation_overhead=st_bpm120["dilation_overhead"],
        signal_energy_fJ=st_bpm120["signal_energy_J"] * 1e15,
        useful_energy_fJ=st_bpm120["useful_energy_J"] * 1e15,
        lo_energy_fJ=st_bpm120["lo_energy_J"] * 1e15,
        max_excess_lo_var_frac=st_bpm120["max_excess_lo_var_frac"],
        mean_excess_read_var_frac=st_bpm120["mean_excess_read_var_frac"],
    )

    # (d) Architecture A (full-state) photon comparison at T=60 & T=120 on MNIST (fixed y_eval[:200]!)
    y_200 = np.repeat(np.arange(10), 20)
    _, st_full60 = sample_photonic_diffusion(
        make_fn(net_main, y_200, T_bpm), sched, det_bpm, shape=(200, 64), n_steps=60,
        rng=np.random.default_rng(1005), mode="homodyne_fullstate",
    )
    _, st_full120 = sample_photonic_diffusion(
        make_fn(net_main, y_200, T_bpm), sched, det_bpm, shape=(200, 64), n_steps=120,
        rng=np.random.default_rng(1006), mode="homodyne_fullstate",
    )
    results["fullstate_mnist_comparison"] = dict(
        T60_photons=st_full60["total_launched_photons"],
        T120_photons=st_full120["total_launched_photons"],
    )

    # (e) Pure-optical square-law photodiode nonlinearity + Poisson homodyne
    s_sq, st_sq = sample_photonic_diffusion(
        make_fn(net_sq, y_eval, None), sched, det_ideal, shape=(1000, 64), n_steps=120,
        rng=np.random.default_rng(1007), mode="homodyne_residual",
    )
    results["conditions"]["optical_square_homodyne"] = dict(
        eval_generated(s_sq, y_eval, x_te, x_tr[:10000], clf, mu, sd),
        total_useful_photons=st_sq["total_useful_photons"],
        total_launched_photons=st_sq["total_launched_photons"],
        dilation_overhead=st_sq["dilation_overhead"],
    )

    # (f) Unconditional SiLU + Poisson homodyne
    s_unc, st_unc = sample_photonic_diffusion(
        make_fn(net_unc, None, None), sched, det_ideal, shape=(1000, 64), n_steps=120,
        rng=np.random.default_rng(1008), mode="homodyne_residual",
    )
    results["conditions"]["uncond_homodyne"] = dict(
        eval_generated(s_unc, None, x_te, x_tr[:10000], clf, mu, sd),
        total_useful_photons=st_unc["total_useful_photons"],
        total_launched_photons=st_unc["total_launched_photons"],
        dilation_overhead=st_unc["dilation_overhead"],
    )

    for k, v in results["conditions"].items():
        acc_str = f"acc={v['cond_accuracy']*100:.1f}%, " if "cond_accuracy" in v else ""
        ph_str = f", useful={v['total_useful_photons']:.1f} ph, launch={v['total_launched_photons']:.1f} ph" if "total_useful_photons" in v else ""
        print(f"  {k:26s}: {acc_str}FD={v['fd_pixel']:.4f}, conf={v['mean_clf_confidence']:.3f}, ent={v['class_entropy_nats']:.3f}{ph_str}", flush=True)
    print(f"  Full-state MNIST photons: T=60 -> {st_full60['total_launched_photons']:.0f} ph, "
          f"T=120 -> {st_full120['total_launched_photons']:.0f} ph", flush=True)

    # 5. Physical Hardware Disorder, Hidden-Layer Noise, and Temporal Drift Stress Tests
    print("\nRunning physical hardware stress tests on compiled N=64 BPM chips...", flush=True)

    # 5a. Sidewall roughness (uncalibrated sigma = 10, 20, 50 nm)
    rough_res = []
    for sig in (10.0, 20.0, 50.0):
        T_r = [
            evaluate_compiled_bpm(c["theta"], c["cfg"], c["D_phase"], c["global_phase"],
                                  rough_sigma_nm=sig, rough_lc_um=0.2, rough_seed=50 + l)[0]
            for l, c in enumerate(compiled)
        ]
        sr, _ = sample_photonic_diffusion(
            make_fn(net_main, y_eval, T_r), sched, det_bpm, shape=(1000, 64), n_steps=120,
            rng=np.random.default_rng(2000 + int(sig)), mode="homodyne_residual",
        )
        m = eval_generated(sr, y_eval, x_te, x_tr[:10000], clf, mu, sd)
        rough_res.append(dict(sigma_nm=sig, **m))
        print(f"  Roughness sigma={sig:4.0f} nm: acc={m['cond_accuracy']*100:.1f}%, FD={m['fd_pixel']:.4f}", flush=True)
    results["stress_tests"]["roughness"] = rough_res

    # 5b. SRN thickness perturbation (4.2 um -> 0.87 dB/cm, +0.57 dB/cm excess loss, UNCALIBRATED)
    T_srn = [
        evaluate_compiled_bpm(c["theta"], c["cfg"], c["D_phase"], c["global_phase"],
                              extra_loss_db_per_cm=0.57)[0]
        for c in compiled
    ]
    s_srn, _ = sample_photonic_diffusion(
        make_fn(net_main, y_eval, T_srn), sched, det_bpm, shape=(1000, 64), n_steps=120,
        rng=np.random.default_rng(2100), mode="homodyne_residual",
    )
    m_srn = eval_generated(s_srn, y_eval, x_te, x_tr[:10000], clf, mu, sd)
    results["stress_tests"]["srn_4p2um_uncalibrated"] = m_srn
    print(f"  SRN 4.2um (0.87 dB/cm):    acc={m_srn['cond_accuracy']*100:.1f}%, FD={m_srn['fd_pixel']:.4f}", flush=True)

    # 5c. Transverse programming PSF blur (uncalibrated res_um = 1.5, 2.0, 3.0 um)
    res_blur = []
    for r_um in (1.5, 2.0, 3.0):
        T_b = [
            evaluate_compiled_bpm(c["theta"], c["cfg"], c["D_phase"], c["global_phase"],
                                  res_um_override=r_um)[0]
            for c in compiled
        ]
        sb, _ = sample_photonic_diffusion(
            make_fn(net_main, y_eval, T_b), sched, det_bpm, shape=(1000, 64), n_steps=120,
            rng=np.random.default_rng(2200 + int(r_um * 10)), mode="homodyne_residual",
        )
        m = eval_generated(sb, y_eval, x_te, x_tr[:10000], clf, mu, sd)
        res_blur.append(dict(res_um=r_um, **m))
        print(f"  Uncalibrated PSF r={r_um:.1f}um: acc={m['cond_accuracy']*100:.1f}%, FD={m['fd_pixel']:.4f}", flush=True)
    results["stress_tests"]["programming_blur"] = res_blur

    # 5d. Finite hidden-layer shot noise (hidden_noise_std = 1 / (2 sqrt(eta_tot * n_hid)))
    hid_res = []
    for n_hid in (10, 50, 200, 1000):
        h_std = 1.0 / (2.0 * math.sqrt(det_bpm.eta_tot * n_hid))
        rng_h = np.random.default_rng(2300 + n_hid)
        sh, _ = sample_photonic_diffusion(
            make_fn(net_main, y_eval, T_bpm, hidden_noise_std=h_std, rng_h=rng_h),
            sched, det_bpm, shape=(1000, 64), n_steps=120,
            rng=np.random.default_rng(2400 + n_hid), mode="homodyne_residual",
        )
        m = eval_generated(sh, y_eval, x_te, x_tr[:10000], clf, mu, sd)
        hid_res.append(dict(n_hid_photons_per_mode=n_hid, hidden_noise_std=h_std, **m))
        print(f"  Hidden shot noise n={n_hid:4d}: acc={m['cond_accuracy']*100:.1f}%, FD={m['fd_pixel']:.4f}", flush=True)
    results["stress_tests"]["hidden_shot_noise"] = hid_res

    # 5e. Slow correlated temporal drift across the 120 reverse steps (AR(1), rho=0.95)
    drift_res = []
    for d_std in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50):
        sd_samp, _ = sample_photonic_diffusion(
            make_fn(net_main, y_eval, T_bpm), sched, det_bpm, shape=(1000, 64), n_steps=120,
            rng=np.random.default_rng(2500 + int(d_std * 100)), mode="homodyne_residual",
            drift_std_per_step=d_std, drift_corr=0.95,
        )
        m = eval_generated(sd_samp, y_eval, x_te, x_tr[:10000], clf, mu, sd)
        drift_res.append(dict(drift_std=d_std, **m))
        print(f"  Temporal drift std={d_std:.2f}:   acc={m['cond_accuracy']*100:.1f}%, FD={m['fd_pixel']:.4f}", flush=True)
    results["stress_tests"]["temporal_drift"] = drift_res

    results["elapsed_s"] = round(time.time() - t_start, 1)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    idx50 = np.concatenate([np.arange(c * 100, c * 100 + 5) for c in range(10)])
    np.savez(
        os.path.join(OUT_DIR, "visuals.npz"),
        mu=mu, sd=sd,
        real_50=np.stack([x_te[y_te == c][:5] for c in range(10)], axis=0).reshape(50, 64),
        dig_50=s_dig[idx50],
        bpm_50=s_bpm120[idx50],
        sq_50=s_sq[idx50],
        unc_50=s_unc[:50],
        traj_steps=np.array([0, 30, 60, 90, 108, 120]),
        traj_bpm=np.stack([st_bpm120["traj"][k][[0, 100, 200, 300, 400, 500, 600, 700, 800, 900]]
                           for k in (0, 30, 60, 90, 108, 120)], axis=0),
        per_step_useful=np.array(st_bpm120["per_step_useful"]),
        per_step_launched=np.array(st_bpm120["per_step_photons"]),
        field_map_2d=field_map_2d,
        dn_layer0_ds=dn_layer0_ds,
        wg0_x=wg0.grid.x().astype(np.float32),
        wg0_z=wg0.grid.z()[::10].astype(np.float32),
        wg0_W=float(wg0.W),
    )
    print(f"\nSaved Experiment 2 & 3 to {OUT_DIR} in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
