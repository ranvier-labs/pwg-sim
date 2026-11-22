"""High-craft Mitsuba 3 path-traced 3D beauty render of the Integrated Folded
All-Optical Photonic Diffusion Chip on the shallow-etched MgO:LN/SRN platform.

100% physically rendered in Mitsuba 3 (cuda_ad_rgb, 512 spp) using:
  - Smooth extruded beveled triangular meshes for all curved waveguides, S-bend
    directional couplers, and the passive uniform-log-SNR storage ring
  - Real dielectric glass BSDFs (int_ior = 2.14 LN core, roughdielectric) wrapping
    internal guided-mode emission cores, brushed Au conductors, and layered
    p++Si / SiO2 / Z-cut MgO:LN wafer cross-sections
  - REAL compiled B1 MgO:LN/SRN BPM 1550 nm guided intensity |E(x,z)|^2 and
    525 nm DMD Delta n(x,z) refractive-index maps extracted directly from
    results/diffusion_cavity/compiled_B1_folded_shared_kerr_sa.npz
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
MESH_DIR = os.path.join(RES_DIR, "_meshes")
os.makedirs(MESH_DIR, exist_ok=True)


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

    rng = np.random.default_rng(7)
    a_in = (rng.normal(size=(1, 64)) + 1j * rng.normal(size=(1, 64))).astype(np.complex64)
    a_in /= np.linalg.norm(a_in)
    e_in = a_in @ wg.modes.astype(np.complex64)

    z_snaps = list(range(0, wg.grid.nz, 4))
    rec = {k: None for k in z_snaps}
    wg.prop.forward(FloatNP(dn_np.ravel()), FloatNP(e_in.real.ravel()), FloatNP(e_in.imag.ravel()), record=rec)
    field_2d = np.stack([rec[k] for k in z_snaps], axis=0)

    x = wg.grid.x()
    xs = np.abs(x) <= (wg.W / 2.0 + 3.0)
    I_core = field_2d[:, xs].T
    dn_core = dn_np[::4, :][:, xs].T

    # Rich glowing 1550 nm intensity palette (warm terracotta -> luminous gold-white filaments)
    I_n = I_core / max(np.percentile(I_core, 99.4), 1e-12)
    a_ = np.clip(np.sqrt(np.clip(I_n, 0.0, None)), 0.0, 1.0)[..., None]
    light_rgb = (
        a_ * np.array([1.0, 0.22, 0.03])[None, None, :]
        + (a_ ** 2.6) * np.array([0.0, 0.62, 0.34])[None, None, :]
    )
    dn_n = np.clip(dn_core / max(float(meta0["cfg"]["dn_max"]), 1e-6), 0.0, 1.0)
    dmd_rgb = dn_n[..., None] * np.array([0.06, 0.68, 0.14])[None, None, :]
    return light_rgb[::-1].astype(np.float32), dmd_rgb[::-1].astype(np.float32)


# ---------------------------------------------------------------------------
# Procedural Smooth Waveguide Mesh Builder (beveled trapezoidal/rounded rib)
# ---------------------------------------------------------------------------

def write_obj(path: str, verts: np.ndarray, faces: np.ndarray):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def _dedup_path(pts: np.ndarray, closed: bool = False, tol: float = 1e-4) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    keep = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - keep[-1]) > tol:
            keep.append(p)
    if closed and len(keep) > 2 and np.linalg.norm(keep[-1] - keep[0]) <= tol:
        keep.pop()
    return np.array(keep, dtype=np.float64)


def extrude_waveguide_obj(
    path: str,
    pts_xz: np.ndarray,
    y0: float,
    height: float,
    half_w: float,
    bevel: float = 0.22,
    closed: bool = False,
):
    """Extrude a smooth, beveled 6-vertex waveguide cross-section along an (X, Z) path."""
    pts = _dedup_path(pts_xz, closed=closed)
    M = len(pts)
    profile = np.array([
        [-half_w, 0.0],
        [-half_w, (1.0 - bevel) * height],
        [-half_w * (1.0 - bevel), height],
        [+half_w * (1.0 - bevel), height],
        [+half_w, (1.0 - bevel) * height],
        [+half_w, 0.0],
    ], dtype=np.float64)
    K = len(profile)

    # Smooth unit tangents & miter-clamped normals in (X, Z)
    tang = np.zeros_like(pts)
    if closed:
        for i in range(M):
            d1 = pts[i] - pts[(i - 1) % M]
            d2 = pts[(i + 1) % M] - pts[i]
            t1 = d1 / max(np.linalg.norm(d1), 1e-12)
            t2 = d2 / max(np.linalg.norm(d2), 1e-12)
            tm = t1 + t2
            tang[i] = tm / max(np.linalg.norm(tm), 1e-12)
    else:
        for i in range(M):
            if i == 0:
                d = pts[1] - pts[0]
                tang[i] = d / max(np.linalg.norm(d), 1e-12)
            elif i == M - 1:
                d = pts[-1] - pts[-2]
                tang[i] = d / max(np.linalg.norm(d), 1e-12)
            else:
                d1 = pts[i] - pts[i - 1]
                d2 = pts[i + 1] - pts[i]
                t1 = d1 / max(np.linalg.norm(d1), 1e-12)
                t2 = d2 / max(np.linalg.norm(d2), 1e-12)
                tm = t1 + t2
                tang[i] = tm / max(np.linalg.norm(tm), 1e-12)
    norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)

    verts = []
    for i in range(M):
        px, pz = pts[i]
        nx, nz = norm[i]
        for k in range(K):
            off, dy = profile[k]
            verts.append([px + off * nx, y0 + dy, pz + off * nz])
    verts = np.array(verts, dtype=np.float64)

    # Outward-facing triangle winding (CCW when viewed from outside)
    faces = []
    n_seg = M if closed else (M - 1)
    for i in range(n_seg):
        i_next = (i + 1) % M
        b0 = i * K
        b1 = i_next * K
        for k in range(K):
            kn = (k + 1) % K
            faces.append([b0 + k, b0 + kn, b1 + kn])
            faces.append([b0 + k, b1 + kn, b1 + k])

    if not closed:
        for k in range(1, K - 1):
            faces.append([0, k, k + 1])
            b = (M - 1) * K
            faces.append([b, b + k + 1, b + k])

    write_obj(path, verts, np.array(faces, dtype=np.int32))
    return path


def cubic_bezier(p0, p1, p2, p3, n=40):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


def s_bend_pts(x0, z0, x1, z1, n=36):
    xm = 0.5 * (x0 + x1)
    return cubic_bezier(
        np.array([x0, z0]), np.array([xm, z0]), np.array([xm, z1]), np.array([x1, z1]), n=n
    )


def racetrack_pts(x_l, x_r, z_b, z_f, n_arc=48):
    R = 0.5 * (z_f - z_b)
    zm = 0.5 * (z_f + z_b)
    # Right arc: -pi/2 -> +pi/2
    th_r = np.linspace(-math.pi / 2, math.pi / 2, n_arc, endpoint=False)
    arc_r = np.stack([x_r + R * np.cos(th_r), zm + R * np.sin(th_r)], axis=1)
    # Front straight: x_r -> x_l
    front = np.stack([np.linspace(x_r, x_l, 24, endpoint=False), np.full(24, z_f)], axis=1)
    # Left arc: +pi/2 -> 3*pi/2
    th_l = np.linspace(math.pi / 2, 1.5 * math.pi, n_arc, endpoint=False)
    arc_l = np.stack([x_l + R * np.cos(th_l), zm + R * np.sin(th_l)], axis=1)
    # Back straight: x_l -> x_r
    back = np.stack([np.linspace(x_l, x_r, 24, endpoint=False), np.full(24, z_b)], axis=1)
    return np.concatenate([arc_r, front, arc_l, back], axis=0)


def build_folded_chip_scene(light_rgb: np.ndarray, dmd_rgb: np.ndarray, res=(2400, 1080), spp=512):
    """Studio-lit Mitsuba 3 scene with smooth dielectric waveguides and real compiled BPM maps."""
    import mitsuba as mi
    T = mi.ScalarTransform4f

    # Clean thin-film stack heights
    t_si, t_box, t_ln = 0.44, 0.14, 0.10
    y_si0, y_si1 = 0.0, t_si
    y_box0, y_box1 = y_si1, y_si1 + t_box
    y_ln0, y_ln1 = y_box1, y_box1 + t_ln
    y_top = y_ln1

    def plastic(rgb, rough=0.18, ior=1.5):
        return {
            "type": "roughplastic", "distribution": "ggx", "alpha": rough,
            "diffuse_reflectance": {"type": "rgb", "value": rgb}, "int_ior": ior,
        }

    def glass(alpha=0.05, int_ior=2.0, tint=None):
        d = {"type": "roughdielectric", "distribution": "ggx", "alpha": alpha, "int_ior": int_ior, "ext_ior": 1.0}
        if tint is not None:
            d["specular_transmittance"] = {"type": "rgb", "value": tint}
        return d

    def metal(material="Au", alpha=0.18):
        return {"type": "roughconductor", "material": material, "alpha": alpha}

    def box(x0, x1, y0, y1, z0, z1, bsdf, emit_rgb=None):
        d = {
            "type": "cube", "bsdf": bsdf,
            "to_world": T().translate([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
            @ T().scale([(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2]),
        }
        if emit_rgb is not None:
            d["emitter"] = {"type": "area", "radiance": {"type": "rgb", "value": emit_rgb}}
        return d

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

    def add_glowing_waveguide(sc, name, pts_xz, half_w, height, emit_rgb, glass_tint=None, closed=False):
        """Real glass dielectric ridge waveguide wrapping an internal guided-mode emission core!"""
        p_core = os.path.join(MESH_DIR, f"{name}_core.obj")
        p_clad = os.path.join(MESH_DIR, f"{name}_clad.obj")
        extrude_waveguide_obj(p_core, pts_xz, y_top + 0.012, height * 0.62, half_w * 0.68, bevel=0.30, closed=closed)
        extrude_waveguide_obj(p_clad, pts_xz, y_top + 0.001, height, half_w, bevel=0.28, closed=closed)
        sc[f"{name}_core"] = {
            "type": "obj", "filename": p_core,
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.95, 0.90, 0.85]}},
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": emit_rgb}},
        }
        sc[f"{name}_clad"] = {
            "type": "obj", "filename": p_clad,
            "bsdf": glass(alpha=0.05, int_ior=1.52, tint=glass_tint),
        }

    # --- Studio Scene & Full-Die 3/4 Isometric Camera (all 4 corners cleanly framed!) ---
    sc = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 12},
        "sensor": {
            "type": "perspective", "fov": 31, "fov_axis": "x",
            "to_world": T().look_at(origin=[-6.6, 13.8, 21.8], target=[0.20, 0.25, 0.0], up=[0, 1, 0]),
            "film": {"type": "hdrfilm", "width": res[0], "height": res[1], "rfilter": {"type": "gaussian"}},
            "sampler": {"type": "independent", "sample_count": spp},
        },
        # Bright clean studio environment + warm key + cool rim fill
        "env": {"type": "constant", "radiance": {"type": "rgb", "value": [0.68, 0.70, 0.73]}},
        "key": {
            "type": "rectangle",
            "to_world": T().look_at(origin=[-8.0, 14.0, 9.0], target=[0.0, 0.4, 0.0], up=[0, 0, 1]) @ T().scale([6.5, 6.5, 1]),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [2.6, 2.55, 2.48]}},
        },
        "fill": {
            "type": "rectangle",
            "to_world": T().look_at(origin=[13.0, 8.0, -7.0], target=[0.0, 0.4, 0.0], up=[0, 1, 0]) @ T().scale([5.0, 5.0, 1]),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [1.0, 1.08, 1.22]}},
        },
        "rim": {
            "type": "rectangle",
            "to_world": T().look_at(origin=[0.0, 9.0, -14.0], target=[0.0, 0.4, 0.0], up=[0, 1, 0]) @ T().scale([7.0, 3.0, 1]),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [0.95, 0.98, 1.05]}},
        },
        "floor": {
            "type": "rectangle",
            "to_world": T().translate([0, -0.002, 0]) @ T().rotate([1, 0, 0], -90) @ T().scale([80, 80, 1]),
            "bsdf": plastic([0.92, 0.93, 0.94], rough=0.45, ior=1.35),
        },
    }

    # --- 1. Layered Semiconductor Die (p++Si / 2 um SiO2 / 0.7 um Z-cut MgO:LN with crisp optical cladding top) ---
    cx0, cx1 = -5.4, 5.8
    cz0, cz1 = -3.0, 3.0
    sc["sub_si"] = box(cx0, cx1, y_si0, y_si1, cz0, cz1, plastic([0.14, 0.15, 0.18], rough=0.14, ior=3.4))
    sc["sub_box"] = box(cx0, cx1, y_box0, y_box1, cz0, cz1, plastic([0.74, 0.84, 0.92], rough=0.14, ior=1.45))
    # Warm copper-amber LN film layer visible along beveled wafer edges
    sc["sub_ln"] = box(cx0, cx1, y_ln0, y_ln1 - 0.015, cz0, cz1, plastic([0.62, 0.34, 0.21], rough=0.16, ior=2.1))
    # Ultra-clean satin pale silica/LN top optical surface (giving high contrast for waveguides & reflections!)
    sc["sub_top_clad"] = box(cx0 + 0.02, cx1 - 0.02, y_ln1 - 0.015, y_ln1, cz0 + 0.02, cz1 - 0.02,
                             plastic([0.86, 0.87, 0.89], rough=0.15, ior=1.46))

    # --- 2. Strictly Passive Static Uniform-log-SNR Storage Ring (Racetrack Waveguide) ---
    z_rb, z_rf = -1.65, -0.25
    x_rl, x_rr = -3.20, 3.15
    z_rm = 0.5 * (z_rb + z_rf)  # -0.95
    R_ring = 0.5 * (z_rf - z_rb)  # 0.70

    ring_pts = racetrack_pts(x_rl, x_rr, z_rb, z_rf, n_arc=64)
    add_glowing_waveguide(
        sc, "storage_ring", ring_pts, half_w=0.125, height=0.115,
        emit_rgb=[5.4, 1.65, 0.22], glass_tint=[1.0, 0.88, 0.72], closed=True,
    )

    # --- 3. Open-Port Optical ASE Directional Coupler (Back Edge -> Storage Ring) ---
    p_ase_in = np.concatenate([
        np.array([[-5.4, -2.52], [-2.55, -2.52]]),
        s_bend_pts(-2.55, -2.52, -1.65, -1.94, n=32),
        np.array([[-0.55, -1.94]]),
    ], axis=0)
    add_glowing_waveguide(
        sc, "ase_in", p_ase_in, half_w=0.095, height=0.098,
        emit_rgb=[0.55, 2.95, 4.80], glass_tint=[0.76, 0.95, 1.0],
    )
    sc["ase_collar_base"] = box(-4.55, -3.70, y_top, y_top + 0.16, -2.82, -2.22, plastic([0.18, 0.22, 0.28], 0.18))
    sc["ase_collar_gold"] = box(-4.52, -3.73, y_top + 0.16, y_top + 0.22, -2.78, -2.26, metal("Au", 0.16))
    p_ase_dump = np.concatenate([
        np.array([[-0.55, -1.94]]),
        s_bend_pts(-0.55, -1.94, 0.45, -2.58, n=32),
        np.array([[0.95, -2.58]]),
    ], axis=0)
    add_glowing_waveguide(
        sc, "ase_dump", p_ase_dump, half_w=0.08, height=0.088,
        emit_rgb=[1.65, 0.55, 0.15], glass_tint=[0.95, 0.85, 0.78],
    )

    # --- 4. Co-Propagating 4% Passive Tap & Recombiner into Front Nonlinear Score Branch ---
    z_br = 1.48
    p_tap = np.concatenate([
        s_bend_pts(-3.55, 0.28, -3.20, 0.04, n=24),
        np.array([[-2.75, 0.04]]),
        cubic_bezier(
            np.array([-2.75, 0.04]),
            np.array([-2.40, 0.04]),
            np.array([-2.82, z_br]),
            np.array([-2.46, z_br]),
            n=40,
        ),
    ], axis=0)
    add_glowing_waveguide(
        sc, "tap_wg", p_tap, half_w=0.088, height=0.098,
        emit_rgb=[4.2, 1.30, 0.18], glass_tint=[1.0, 0.88, 0.72],
    )

    p_cond = np.concatenate([
        np.array([[-5.4, 2.28], [-2.10, 2.28]]),
        s_bend_pts(-2.10, 2.28, -1.22, z_br + 0.26, n=36),
    ], axis=0)
    add_glowing_waveguide(
        sc, "cond_wg", p_cond, half_w=0.088, height=0.098,
        emit_rgb=[0.28, 2.85, 1.55], glass_tint=[0.78, 1.0, 0.88],
    )

    # (a) Pre-OPA / SOA chiplet (14.2 dB) at X in [-2.48, -1.68]
    sc["opa_sub"] = box(-2.48, -1.68, y_top, y_top + 0.16, z_br - 0.36, z_br + 0.36, plastic([0.16, 0.20, 0.28], 0.16))
    sc["opa_gold"] = box(-2.43, -1.73, y_top + 0.16, y_top + 0.24, z_br - 0.31, z_br + 0.31, metal("Au", 0.15))
    add_glowing_waveguide(
        sc, "opa_to_pmw", np.array([[-1.68, z_br], [-1.20, z_br]]), half_w=0.11, height=0.10,
        emit_rgb=[5.0, 1.55, 0.22], glass_tint=[1.0, 0.88, 0.72],
    )

    # (b) Cutaway 2.5 mm MgO:LN/SRN Programmable Multimode Waveguide (PMW, X in [-1.22, +1.58], W_vis = 1.06)
    px0, px_cut, px1 = -1.22, 0.18, 1.58
    pw = 0.52
    # Lower LN slab base under cutaway
    sc["pmw_ln_slab"] = box(px0, px1, y_top, y_top + 0.11, z_br - pw - 0.08, z_br + pw + 0.08,
                            plastic([0.52, 0.28, 0.18], rough=0.14, ior=2.1))
    # Left half: full MgO:LN/SRN/Ti-Au upper stack
    y_tox0, y_tox1 = y_top + 0.11, y_top + 0.17
    y_srn0, y_srn1 = y_tox1, y_tox1 + 0.22
    y_au0, y_au1 = y_srn1, y_srn1 + 0.045
    sc["pmw_tox"] = box(px0, px_cut, y_tox0, y_tox1, z_br - pw - 0.10, z_br + pw + 0.10, plastic([0.78, 0.86, 0.93], 0.16))
    sc["pmw_srn"] = box(px0, px_cut, y_srn0, y_srn1, z_br - pw - 0.10, z_br + pw + 0.10, plastic([0.46, 0.54, 0.42], 0.22))
    sc["pmw_au"] = box(px0, px_cut, y_au0, y_au1, z_br - pw - 0.10, z_br + pw + 0.10, metal("Au", 0.18))

    # REAL compiled B1 525 nm DMD Delta n(x,z) map on top of left electrode
    eps = 2e-3
    nd = dmd_rgb.shape[1]
    nl = light_rgb.shape[1]
    sc["pmw_dmd_tex"] = tex_rect(
        px0 + 0.04, px_cut - 0.02, z_br - pw + 0.02, z_br + pw - 0.02, y_au1 + eps,
        dmd_rgb[:, :nd // 2], 1.65, metal("Au", 0.20),
    )
    # REAL compiled B1 1550 nm BPM |E(x,z)|^2 intensity map on right cutaway LN core + clear glass cover
    sc["pmw_light_tex"] = tex_rect(
        px_cut, px1 - 0.02, z_br - pw + 0.02, z_br + pw - 0.02, y_top + 0.11 + eps,
        light_rgb[:, nl // 2:], 9.0, plastic([0.48, 0.24, 0.16], 0.15),
    )
    # Translucent refractive glass cap over the cutaway LN core for genuine optical depth!
    sc["pmw_glass_cap"] = box(px_cut, px1, y_top + 0.12, y_top + 0.20, z_br - pw - 0.06, z_br + pw + 0.06,
                              glass(alpha=0.04, int_ior=1.65, tint=[1.0, 0.95, 0.90]))

    # (c) Passive All-Optical Kerr + SA Nonlinear Crystal (real translucent violet dielectric glass + internal core!)
    add_glowing_waveguide(
        sc, "pmw_to_kerr", np.array([[px1, z_br], [2.62, z_br]]), half_w=0.105, height=0.095,
        emit_rgb=[3.1, 0.92, 0.22], glass_tint=[1.0, 0.90, 0.80],
    )
    sc["kerr_crystal"] = box(1.80, 2.52, y_top + 0.002, y_top + 0.28, z_br - 0.36, z_br + 0.36,
                             glass(alpha=0.045, int_ior=1.82, tint=[0.72, 0.56, 0.98]))
    # Subtle internal violet nonlinear glow core inside the crystal
    sc["kerr_core_glow"] = box(1.84, 2.48, y_top + 0.02, y_top + 0.10, z_br - 0.14, z_br + 0.14,
                               plastic([0.65, 0.35, 0.95], 0.12), emit_rgb=[1.2, 0.35, 1.85])

    # (d) Scalar VOA gold micro-heater + smooth sweeping Co-propagating 4% Recombiner back into Storage Ring
    p_rec = np.concatenate([
        np.array([[2.52, z_br], [2.95, z_br]]),
        cubic_bezier(
            np.array([2.95, z_br]),
            np.array([3.42, z_br]),
            np.array([2.65, 0.04]),
            np.array([3.05, 0.04]),
            n=44,
        ),
        np.array([[3.48, 0.04]]),
        s_bend_pts(3.48, 0.04, 3.78, 0.30, n=22),
    ], axis=0)
    add_glowing_waveguide(
        sc, "rec_wg", p_rec, half_w=0.088, height=0.098,
        emit_rgb=[4.8, 1.45, 0.20], glass_tint=[1.0, 0.88, 0.72],
    )
    sc["voa_heater"] = box(2.58, 2.92, y_top + 0.098, y_top + 0.16, z_br - 0.15, z_br + 0.15, metal("Au", 0.15))

    # --- 5. t = 0 Output Dump Switch & Single 8-Port Balanced-Homodyne Receiver (Right Side) ---
    p_out = np.array([[x_rr + R_ring, z_rm], [4.58, z_rm]])
    add_glowing_waveguide(
        sc, "out_bus", p_out, half_w=0.095, height=0.098,
        emit_rgb=[4.8, 1.42, 0.20], glass_tint=[1.0, 0.88, 0.72],
    )
    sc["dump_switch"] = box(4.02, 4.38, y_top + 0.098, y_top + 0.16, z_rm - 0.16, z_rm + 0.16, metal("Au", 0.15))

    p_lo = np.array([[4.95, -3.00], [4.95, z_rm - 0.58]])
    add_glowing_waveguide(
        sc, "lo_wg", p_lo, half_w=0.095, height=0.098,
        emit_rgb=[4.4, 3.25, 0.52], glass_tint=[1.0, 0.94, 0.74],
    )

    # Dark-slate integrated 8-port 90-degree hybrid + balanced PD receiver with 4 brushed-gold pads
    sc["rx_body"] = box(4.56, 5.42, y_top, y_top + 0.28, z_rm - 0.66, z_rm + 0.66, plastic([0.16, 0.19, 0.24], 0.18))
    for k, zp in enumerate([-0.44, -0.15, 0.15, 0.44]):
        sc[f"rx_pad_{k}"] = box(4.70, 5.30, y_top + 0.28, y_top + 0.33, z_rm + zp - 0.085, z_rm + zp + 0.085, metal("Au", 0.14))

    return sc


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Extracting REAL compiled B1 BPM |E(x,z)|^2 and Delta n(x,z)...", flush=True)
    light_rgb, dmd_rgb = extract_b1_bpm_maps()

    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb")
    print("Rendering studio-grade Mitsuba 3 Folded Photonic Chip (2400x1080, 512 spp)...", flush=True)
    sc = build_folded_chip_scene(light_rgb, dmd_rgb, res=(2400, 1080), spp=512)
    scene = mi.load_dict(sc)
    img = mi.render(scene).numpy()

    # Clean filmic tone map + sRGB
    v = img / (1.0 + img * 0.18)
    v = np.clip(v, 0.0, None)
    srgb = np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)
    srgb = np.clip(srgb, 0.0, 1.0)

    out_png = os.path.join(RES_DIR, "cavity_chip_render_studio.png")
    plt.imsave(out_png, srgb)
    print(f"Saved {out_png} (shape={srgb.shape})", flush=True)


if __name__ == "__main__":
    main()
