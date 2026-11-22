"""Mitsuba 3 path-traced 3D beauty render of the Folded All-Optical Photonic
Diffusion Chip (passive uniform-log-SNR storage ring + open-port ASE coupler +
4% tap/recombiner + 14.2 dB OPA + cutaway MgO:LN/SRN programmable multimode
waveguide textured with REAL compiled B1 |E(x,z)|^2 and Delta n(x,z) +
Kerr/SA nonlinear slab + final 8-port balanced-homodyne receiver).

Runs on CUDA GPU via Dr.Jit / Mitsuba 3 and saves high-res sRGB PNG + raw float32.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES_DIR = os.path.join(ROOT, "results", "diffusion_cavity")


def extract_b1_bpm_maps():
    """Extract REAL compiled Delta n(x,z) and 1550 nm |E(x,z)|^2 from B1 chip 0."""
    import mitsuba as mi
    mi.set_variant("cuda_ad_mono")
    from drjit.cuda import Float as FloatNP
    from drjit.cuda.ad import Float as FloatAD
    from pwg.mmwg import MMWGConfig, MultimodeWaveguide

    z = np.load(os.path.join(RES_DIR, "compiled_B1_folded_shared_kerr_sa.npz"), allow_pickle=False)
    meta0 = json.loads(str(z["meta_0"]))
    theta0 = z["theta_0"]
    wg = MultimodeWaveguide(MMWGConfig(**meta0["cfg"]))
    dn_ad = wg.index_map(FloatAD(theta0.astype(np.float32)))
    dn_np = dn_ad.numpy().reshape(wg.grid.nz, wg.grid.nx)

    # Propagate a realistic 64-mode complex superposition (32 state + 32 cond modes)
    rng = np.random.default_rng(7)
    a_in = (rng.normal(size=(1, 64)) + 1j * rng.normal(size=(1, 64))).astype(np.complex64)
    a_in /= np.linalg.norm(a_in)
    e_in = a_in @ wg.modes.astype(np.complex64)

    z_snaps = list(range(0, wg.grid.nz, 4))
    rec = {k: None for k in z_snaps}
    wg.prop.forward(FloatNP(dn_np.ravel()), FloatNP(e_in.real.ravel()), FloatNP(e_in.imag.ravel()), record=rec)
    field_2d = np.stack([rec[k] for k in z_snaps], axis=0)  # [nz_sub, nx]

    # Crop transversely to waveguide core |x| <= W/2
    x = wg.grid.x()
    xs = np.abs(x) <= (wg.W / 2.0 + 4.0)
    I_core = field_2d[:, xs].T          # [nx_core, nz_sub]
    dn_core = dn_np[::4, :][:, xs].T    # [nx_core, nz_sub]

    # Normalize to rich physical color maps matching fig_device_render
    I_n = I_core / max(np.percentile(I_core, 99.3), 1e-12)
    a_ = np.clip(np.sqrt(np.clip(I_n, 0.0, None)), 0.0, 1.0)[..., None]
    light_rgb = (
        a_ * np.array([1.0, 0.20, 0.04])[None, None, :]
        + (a_ ** 3) * np.array([0.0, 0.58, 0.36])[None, None, :]
    )
    # 525 nm DMD green intensity proportional to dn_core / dn_max
    dn_n = np.clip(dn_core / max(float(meta0["cfg"]["dn_max"]), 1e-6), 0.0, 1.0)
    dmd_rgb = dn_n[..., None] * np.array([0.05, 0.62, 0.08])[None, None, :]
    return light_rgb[::-1].astype(np.float32), dmd_rgb[::-1].astype(np.float32)


def build_folded_chip_scene(light_rgb: np.ndarray, dmd_rgb: np.ndarray, res=(2400, 1080), spp=256):
    """Mitsuba 3 scene of the integrated Folded All-Optical Photonic Diffusion Chip.

    World coordinates (mm, schematic layout scaled cleanly for visual clarity):
      X in [-5.8, 6.2]  : propagation axis (left -> right)
      Z in [-3.4, 3.4]  : transverse / loop axis (back storage arm -> front score branch)
      Y >= 0            : vertical thin-film stack (Si, BOX, Z-cut LN, TOX, SRN, Au)
    """
    import mitsuba as mi
    T = mi.ScalarTransform4f

    # Clean vertical stack heights (visualized with thin-film exaggeration)
    t_si, t_box, t_ln, t_wg = 0.42, 0.14, 0.10, 0.08
    y_si0, y_si1 = 0.0, t_si
    y_box0, y_box1 = y_si1, y_si1 + t_box
    y_ln0, y_ln1 = y_box1, y_box1 + t_ln
    y_top = y_ln1

    def plastic(rgb, rough=0.22):
        return {
            "type": "roughplastic", "distribution": "ggx", "alpha": rough,
            "diffuse_reflectance": {"type": "rgb", "value": rgb}, "int_ior": 1.5,
        }

    def emissive_box(x0, x1, y0, y1, z0, z1, base_rgb, emit_rgb=None, rough=0.20):
        d = {
            "type": "cube",
            "bsdf": plastic(base_rgb, rough),
            "to_world": T().translate([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
            @ T().scale([(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2]),
        }
        if emit_rgb is not None:
            d["emitter"] = {"type": "area", "radiance": {"type": "rgb", "value": emit_rgb}}
        return d

    def metal_box(x0, x1, y0, y1, z0, z1, material="Au", alpha=0.24):
        return {
            "type": "cube",
            "bsdf": {"type": "roughconductor", "material": material, "alpha": alpha},
            "to_world": T().translate([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
            @ T().scale([(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2]),
        }

    def tex_rect(x0, x1, z0, z1, yy, tex, scale, bsdf):
        return {
            "type": "rectangle",
            "to_world": T().translate([(x0 + x1) / 2, yy, (z0 + z1) / 2])
            @ T().rotate([1, 0, 0], -90)
            @ T().scale([(x1 - x0) / 2, (z1 - z0) / 2, 1.0]),
            "bsdf": bsdf,
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "bitmap", "raw": True, "filter_type": "bilinear",
                    "data": mi.TensorXf(np.ascontiguousarray(tex * scale, dtype=np.float32)),
                },
            },
        }

    def add_arc_segments(sc, prefix, cx, cz, R, th0, th1, n_seg, half_w, y0, y1, base_rgb, emit_rgb):
        """Approximate a smooth glowing curved waveguide bend with overlapping oriented cuboids."""
        ths = np.linspace(th0, th1, n_seg + 1)
        for i in range(n_seg):
            tm = 0.5 * (ths[i] + ths[i + 1])
            dth = abs(ths[i + 1] - ths[i])
            seg_len = R * dth * 1.15
            px = cx + R * math.cos(tm)
            pz = cz + R * math.sin(tm)
            # Tangent angle in (X, Z) plane: (-sin(tm), cos(tm))
            ang_deg = math.degrees(math.atan2(math.cos(tm), -math.sin(tm)))
            sc[f"{prefix}_{i}"] = {
                "type": "cube",
                "bsdf": plastic(base_rgb, 0.18),
                "emitter": {"type": "area", "radiance": {"type": "rgb", "value": emit_rgb}},
                "to_world": T().translate([px, 0.5 * (y0 + y1), pz])
                @ T().rotate([0, 1, 0], -ang_deg)
                @ T().scale([0.5 * seg_len, 0.5 * (y1 - y0), half_w]),
            }

    sc = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 8},
        "sensor": {
            "type": "perspective", "fov": 36, "fov_axis": "x",
            "to_world": T().look_at(origin=[0.15, 8.8, 16.5], target=[0.15, 0.30, -0.10], up=[0, 1, 0]),
            "film": {"type": "hdrfilm", "width": res[0], "height": res[1], "rfilter": {"type": "gaussian"}},
            "sampler": {"type": "independent", "sample_count": spp},
        },
        "env": {"type": "constant", "radiance": {"type": "rgb", "value": [0.58, 0.60, 0.63]}},
        "key": {
            "type": "rectangle",
            "to_world": T().look_at(origin=[-4.0, 12.0, 8.0], target=[0.0, 0.5, 0.0], up=[0, 0, 1]) @ T().scale([5, 5, 1]),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [2.3, 2.25, 2.15]}},
        },
        "fill": {
            "type": "rectangle",
            "to_world": T().look_at(origin=[11.0, 6.0, -5.0], target=[0.0, 0.5, 0.0], up=[0, 1, 0]) @ T().scale([4, 4, 1]),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [0.85, 0.90, 1.0]}},
        },
        "floor": {
            "type": "rectangle",
            "to_world": T().translate([0, -0.002, 0]) @ T().rotate([1, 0, 0], -90) @ T().scale([50, 50, 1]),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.86, 0.86, 0.85]}},
        },
    }

    # --- 1. Main Photonic Die Substrate Stack (p++Si / 2 um SiO2 / 0.7 um Z-cut MgO:LN) ---
    cx0, cx1 = -5.6, 5.9
    cz0, cz1 = -3.15, 3.15
    sc["sub_si"] = emissive_box(cx0, cx1, y_si0, y_si1, cz0, cz1, [0.16, 0.17, 0.20], rough=0.18)
    sc["sub_box"] = emissive_box(cx0, cx1, y_box0, y_box1, cz0, cz1, [0.79, 0.85, 0.91], rough=0.28)
    sc["sub_ln"] = emissive_box(cx0, cx1, y_ln0, y_ln1, cz0, cz1, [0.44, 0.25, 0.19], rough=0.20)

    # --- 2. Passive Static Uniform-log-SNR Storage Ring (Racetrack Waveguide Bus, glowing warm amber-orange) ---
    # Back arm: Z = -1.75; Front return arm: Z = -0.35; bends at X = -3.6 and X = +3.3 (R = 0.70)
    z_ring_back = -1.75
    z_ring_front = -0.35
    R_ring = 0.5 * (z_ring_front - z_ring_back)  # 0.70
    z_ring_mid = 0.5 * (z_ring_front + z_ring_back)  # -1.05
    x_rl, x_rr = -3.45, 3.15
    wg_hw = 0.13
    ring_col = [0.85, 0.32, 0.08]
    ring_emit = [1.45, 0.36, 0.06]

    sc["ring_back"] = emissive_box(x_rl, x_rr, y_top, y_top + t_wg, z_ring_back - wg_hw, z_ring_back + wg_hw, ring_col, ring_emit)
    sc["ring_front"] = emissive_box(x_rl, x_rr, y_top, y_top + t_wg, z_ring_front - wg_hw, z_ring_front + wg_hw, ring_col, ring_emit)
    add_arc_segments(sc, "ring_arc_l", x_rl, z_ring_mid, R_ring, math.pi / 2, 3 * math.pi / 2, 22, wg_hw, y_top, y_top + t_wg, ring_col, ring_emit)
    add_arc_segments(sc, "ring_arc_r", x_rr, z_ring_mid, R_ring, -math.pi / 2, math.pi / 2, 22, wg_hw, y_top, y_top + t_wg, ring_col, ring_emit)

    # --- 3. Open-Port Optical ASE Brownian Noise Coupler (Back Edge -> Storage Ring) ---
    ase_col = [0.16, 0.62, 0.85]
    ase_emit = [0.22, 1.15, 1.85]
    sc["ase_in_bus"] = emissive_box(-2.0, -0.3, y_top, y_top + t_wg, -2.14, -1.94, ase_col, ase_emit)
    add_arc_segments(sc, "ase_in_bend", -2.0, -2.54, 0.50, math.pi / 2, math.pi, 12, 0.10, y_top, y_top + t_wg, ase_col, ase_emit)
    sc["ase_in_lead"] = emissive_box(-2.60, -2.40, y_top, y_top + t_wg, -3.15, -2.54, ase_col, ase_emit)
    # Sleek metallic-blue ASE source package with top window
    sc["ase_src_block"] = emissive_box(-3.15, -1.85, y_top, y_top + 0.30, -3.12, -2.62, [0.16, 0.30, 0.44], None, rough=0.18)
    sc["ase_src_win"] = emissive_box(-2.95, -2.05, y_top + 0.30, y_top + 0.33, -3.00, -2.74, ase_col, [0.20, 0.95, 1.55], rough=0.12)
    # Dump/loss port exiting right of ASE coupler
    add_arc_segments(sc, "ase_out_bend", -0.3, -2.54, 0.50, 0.0, math.pi / 2, 12, 0.09, y_top, y_top + t_wg, [0.55, 0.28, 0.15], [0.45, 0.12, 0.03])
    sc["ase_dump_lead"] = emissive_box(0.11, 0.29, y_top, y_top + t_wg, -3.12, -2.54, [0.55, 0.28, 0.15], [0.35, 0.10, 0.02])

    # --- 4. Passive 4% Tap & Recombiner Couplers into Front Nonlinear Score Side-Branch ---
    z_br = 1.55
    br_col = [0.88, 0.36, 0.10]
    br_emit = [1.55, 0.42, 0.08]
    # Left 4% tap directional coupler & smooth bend from storage ring front (Z=-0.35) to branch (Z=+1.55)
    sc["tap_coupler"] = emissive_box(-3.00, -2.25, y_top, y_top + t_wg, -0.06, 0.12, br_col, [1.05, 0.28, 0.05])
    add_arc_segments(sc, "tap_bend_top", -3.00, 0.38, 0.35, math.pi, 1.5 * math.pi, 10, 0.09, y_top, y_top + t_wg, br_col, [1.05, 0.28, 0.05])
    sc["tap_drop"] = emissive_box(-3.44, -3.26, y_top, y_top + t_wg, 0.38, 1.20, br_col, [1.05, 0.28, 0.05])
    add_arc_segments(sc, "tap_bend_bot", -3.00, 1.20, 0.35, 0.5 * math.pi, math.pi, 10, 0.09, y_top, y_top + t_wg, br_col, [1.05, 0.28, 0.05])
    sc["tap_into_opa"] = emissive_box(-3.00, -2.82, y_top, y_top + t_wg, z_br - 0.10, z_br + 0.10, br_col, [1.05, 0.28, 0.05])

    # Right 4% recombining coupler & smooth bend from branch (Z=+1.55) back to storage ring front (Z=-0.35)
    sc["rec_coupler"] = emissive_box(2.05, 2.75, y_top, y_top + t_wg, -0.06, 0.12, br_col, [1.45, 0.40, 0.08])
    add_arc_segments(sc, "rec_bend_top", 2.75, 0.38, 0.35, 1.5 * math.pi, 2.0 * math.pi, 10, 0.09, y_top, y_top + t_wg, br_col, [1.45, 0.40, 0.08])
    sc["rec_rise"] = emissive_box(3.01, 3.19, y_top, y_top + t_wg, 0.38, 1.20, br_col, [1.45, 0.40, 0.08])
    add_arc_segments(sc, "rec_bend_bot", 2.75, 1.20, 0.35, 0.0, 0.5 * math.pi, 10, 0.09, y_top, y_top + t_wg, br_col, [1.45, 0.40, 0.08])

    # --- 5. Side-Branch Optical Elements along Z = +1.55 ---
    # (a) Optical conditioning port a_cond(s_m, y) entering from front-left edge with smooth bend
    cond_col = [0.16, 0.68, 0.42]
    cond_emit = [0.15, 1.25, 0.55]
    sc["cond_lead"] = emissive_box(-4.22, -4.04, y_top, y_top + t_wg, 1.90, 3.15, cond_col, cond_emit)
    add_arc_segments(sc, "cond_bend", -3.78, 1.90, 0.35, math.pi, 1.5 * math.pi, 10, 0.09, y_top, y_top + t_wg, cond_col, cond_emit)
    sc["cond_join"] = emissive_box(-3.78, -2.80, y_top, y_top + t_wg, z_br - 0.10, z_br + 0.10, cond_col, cond_emit)

    # (b) Pre-OPA / SOA chiplet (14.2 dB gain) — sleek dark-slate optical amp package with gold contact & glowing seam
    sc["opa_body"] = emissive_box(-2.85, -2.02, y_top, y_top + 0.32, z_br - 0.40, z_br + 0.40, [0.18, 0.26, 0.38], None, rough=0.20)
    sc["opa_glow"] = emissive_box(-2.86, -2.01, y_top + 0.02, y_top + 0.09, z_br - 0.14, z_br + 0.14, [0.20, 0.65, 0.95], [0.25, 1.1, 1.9], rough=0.15)
    sc["opa_pad"] = metal_box(-2.73, -2.14, y_top + 0.32, y_top + 0.37, z_br - 0.28, z_br + 0.28, "Au", 0.22)
    sc["opa_to_pmw"] = emissive_box(-2.02, -1.65, y_top, y_top + t_wg, z_br - 0.15, z_br + 0.15, br_col, br_emit)

    # (c) Cutaway MgO:LN/SRN Programmable Multimode Waveguide (PMW, X in [-1.65, +1.05])
    px0, px_cut, px1 = -1.65, -0.30, 1.05
    pw = 0.56
    y_tox0, y_tox1 = y_top, y_top + 0.08
    y_srn0, y_srn1 = y_tox1, y_tox1 + 0.24
    y_au0, y_au1 = y_srn1, y_srn1 + 0.04
    sc["pmw_tox"] = emissive_box(px0, px_cut, y_tox0, y_tox1, z_br - pw - 0.16, z_br + pw + 0.16, [0.80, 0.86, 0.92], rough=0.28)
    sc["pmw_srn"] = emissive_box(px0, px_cut, y_srn0, y_srn1, z_br - pw - 0.16, z_br + pw + 0.16, [0.46, 0.55, 0.41], rough=0.28)
    sc["pmw_au"] = metal_box(px0, px_cut, y_au0, y_au1, z_br - pw - 0.16, z_br + pw + 0.16, "Au", 0.24)
    eps = 2e-3
    nd = dmd_rgb.shape[1]
    nl = light_rgb.shape[1]
    sc["pmw_dmd_tex"] = tex_rect(
        px0 + 0.04, px_cut - 0.02, z_br - pw, z_br + pw, y_au1 + eps,
        dmd_rgb[:, :nd // 2], 1.75,
        {"type": "roughconductor", "material": "Au", "alpha": 0.24},
    )
    sc["pmw_light_tex"] = tex_rect(
        px_cut, px1, z_br - pw, z_br + pw, y_top + eps,
        light_rgb[:, nl // 2:], 8.5,
        plastic([0.44, 0.25, 0.19], 0.20),
    )

    # (d) Passive All-Optical Kerr / Saturable-Absorber Nonlinear Slab — translucent plum-violet dielectric crystal + glowing core
    sc["pmw_to_nl"] = emissive_box(1.05, 1.22, y_top, y_top + t_wg, z_br - 0.20, z_br + 0.20, br_col, br_emit)
    sc["kerr_sa_slab"] = emissive_box(1.22, 2.02, y_top, y_top + 0.26, z_br - 0.46, z_br + 0.46, [0.36, 0.14, 0.42], [0.18, 0.05, 0.24], rough=0.10)
    sc["kerr_sa_core"] = emissive_box(1.20, 2.04, y_top + 0.02, y_top + 0.10, z_br - 0.16, z_br + 0.16, [0.85, 0.28, 0.72], [1.4, 0.30, 0.95], rough=0.10)
    sc["nl_to_rec"] = emissive_box(2.02, 2.75, y_top, y_top + t_wg, z_br - 0.10, z_br + 0.10, br_col, br_emit)
    sc["voa_heater"] = metal_box(2.16, 2.62, y_top + t_wg, y_top + t_wg + 0.07, z_br - 0.16, z_br + 0.16, "Au", 0.22)

    # --- 6. t = 0 Output Dump Switch & Single 8-Port Dual-Quadrature Balanced-Homodyne Receiver ---
    sc["out_switch_bus"] = emissive_box(3.85, 4.65, y_top, y_top + t_wg, z_ring_mid - 0.11, z_ring_mid + 0.11, ring_col, [1.45, 0.36, 0.06])
    sc["out_switch_pad"] = metal_box(3.98, 4.42, y_top + t_wg, y_top + t_wg + 0.08, z_ring_mid - 0.20, z_ring_mid + 0.20, "Au", 0.22)
    lo_col = [0.88, 0.72, 0.18]
    lo_emit = [1.55, 1.15, 0.20]
    sc["lo_lead"] = emissive_box(4.85, 5.05, y_top, y_top + t_wg, -3.15, z_ring_mid - 0.35, lo_col, lo_emit)
    sc["homodyne_hybrid"] = emissive_box(4.65, 5.55, y_top, y_top + 0.34, z_ring_mid - 0.72, z_ring_mid + 0.72, [0.22, 0.24, 0.28], None, rough=0.22)
    for k, zp in enumerate([-0.48, -0.16, 0.16, 0.48]):
        sc[f"pd_pad_{k}"] = metal_box(4.80, 5.42, y_top + 0.34, y_top + 0.39, z_ring_mid + zp - 0.10, z_ring_mid + zp + 0.10, "Au", 0.20)

    return sc


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Extracting REAL compiled B1 BPM |E(x,z)|^2 and Delta n(x,z)...", flush=True)
    light_rgb, dmd_rgb = extract_b1_bpm_maps()

    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb")
    print("Rendering 3D Folded Photonic Diffusion Chip with Mitsuba 3 (2400x1080, 256 spp)...", flush=True)
    sc = build_folded_chip_scene(light_rgb, dmd_rgb, res=(2400, 1080), spp=256)
    scene = mi.load_dict(sc)
    img = mi.render(scene).numpy()

    # Filmic tone map + sRGB conversion (exact match to fig_device_render)
    v = img / (1.0 + img * 0.16)
    v = np.clip(v, 0.0, None)
    srgb = np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)
    srgb = np.clip(srgb, 0.0, 1.0)

    out_png = os.path.join(RES_DIR, "cavity_chip_render_raw.png")
    plt.imsave(out_png, srgb)
    print(f"Saved {out_png} (shape={srgb.shape})", flush=True)


if __name__ == "__main__":
    main()
