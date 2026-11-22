"""DMD projection optics rendered with Mitsuba 3.

The index pattern is written by imaging a DMD (1024x768, 13.7 um pitch) onto
the photoconductor with 1.5x demagnification (9 um pixels on chip).  We model
the projection lens as an ideal thin lens with finite aperture and render, with
Mitsuba's ``thinlens`` sensor, the DMD plane (an area emitter whose radiance is
the DMD pattern texture) as seen from the chip plane.  By reciprocity the
rendered radiance, averaged over the lens aperture, is proportional to the
irradiance the projector deposits on the chip, including

* finite-aperture geometric blur from defocus (chip flatness / tilt / focus
  error), with the correct longitudinal magnification M^2,
* the DMD fill factor (micromirror gaps), and
* pixel-shape / sampling effects.

Wave-optical diffraction is not part of a ray tracer; it is added analytically
as an Airy-equivalent Gaussian (FWHM 0.51 lambda/NA) in :mod:`pwg.device`.

Rendering is differentiable (Mitsuba ``prb`` integrator), so gradients flow
from the photonic loss all the way back to DMD mirror duty cycles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import drjit as dr
import mitsuba as mi
import numpy as np

_VARIANT = "cuda_ad_mono"


def _ensure_variant():
    if mi.variant() != _VARIANT:
        mi.set_variant(_VARIANT)


@dataclass
class ProjectorOptics:
    n_rows: int = 111             # DMD pixels across the waveguide (x)
    n_cols: int = 1012            # DMD pixels along propagation (z)
    pitch_um: float = 13.7        # DMD mirror pitch
    fill_factor: float = 0.92     # mirror area fraction (gaps between micromirrors)
    demag: float = 1.5            # DMD -> chip demagnification
    na_chip: float = 0.10         # image-side numerical aperture of projection lens
    defocus_um: float = 0.0       # chip-side focus error
    object_distance_mm: float = 150.0
    supersample: int = 2          # rendered chip pixels per DMD pixel
    texture_subpixels: int = 1    # texels per mirror (>=26 resolves mirror gaps)
    wavelength_um: float = 0.525  # DMD illumination

    @property
    def chip_pixel_um(self) -> float:
        return self.pitch_um / self.demag

    @property
    def render_pixel_um(self) -> float:
        return self.chip_pixel_um / self.supersample

    def diffraction_sigma_um(self) -> float:
        return 0.51 * self.wavelength_um / self.na_chip / 2.3548

    def build(self, pattern: np.ndarray | None = None):
        """Create the Mitsuba scene.  ``pattern`` is [n_rows, n_cols] in [0, 1]."""
        _ensure_variant()
        if pattern is None:
            pattern = np.zeros((self.n_rows, self.n_cols), np.float32)
        # Texture sub-pixels per mirror.  The inter-mirror gaps of a modern DMD
        # (~0.3 um on chip) are far below the optical blur, so by default (ss=1)
        # the fill factor only enters as an efficiency; ss>=26 resolves the gaps.
        ss = self.texture_subpixels
        self._ss = ss
        g = int(round(ss * (1 - math.sqrt(self.fill_factor)) / 2))
        self._gap = g
        self._ff_eff = ((ss - 2 * g) / ss) ** 2
        W = self.n_cols * self.pitch_um * 1e-3   # mm, along z
        H = self.n_rows * self.pitch_um * 1e-3   # mm, along x
        d_o = self.object_distance_mm
        m = 1.0 / self.demag
        na_obj = self.na_chip * m
        aperture = d_o * math.tan(math.asin(na_obj))
        # chip-side defocus maps to object-side defocus via longitudinal magnification
        focus = d_o + self.defocus_um * 1e-3 / m**2
        fov = 2 * math.degrees(math.atan(W / 2 / d_o))
        self.scene = mi.load_dict({
            "type": "scene",
            "integrator": {"type": "prb", "max_depth": 2},
            "sensor": {
                "type": "thinlens",
                "fov": fov, "fov_axis": "x",
                "aperture_radius": aperture,
                "focus_distance": focus,
                "near_clip": 1.0, "far_clip": 10 * d_o,
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=[0, 0, d_o], target=[0, 0, 0], up=[0, 1, 0]),
                "film": {"type": "hdrfilm",
                         "width": self.n_cols * self.supersample,
                         "height": self.n_rows * self.supersample,
                         "rfilter": {"type": "box"},
                         "pixel_format": "luminance"},
                "sampler": {"type": "independent"},
            },
            "dmd": {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f().scale([W / 2, H / 2, 1]),
                "emitter": {"type": "area", "radiance": {
                    "type": "bitmap", "filter_type": "nearest", "raw": True,
                    "data": mi.TensorXf(np.ascontiguousarray(self._texture(pattern)[..., None])),
                }},
            },
        })
        self.params = mi.traverse(self.scene)
        self.key = "dmd.emitter.radiance.data"
        return self

    def _texture(self, pattern):
        """Upsample mirror duty cycles to sub-pixels and black out the gaps."""
        ss = self._ss
        tex = np.repeat(np.repeat(np.asarray(pattern, np.float32), ss, 0), ss, 1)
        g = self._gap
        if g > 0:
            mask = np.ones((ss, ss), np.float32)
            mask[:g], mask[-g:], mask[:, :g], mask[:, -g:] = 0, 0, 0, 0
            tex *= np.tile(mask, (pattern.shape[0], pattern.shape[1]))
        return tex[::-1]  # bitmap row 0 is the top (+y)

    def texture_dr(self, pattern_dr):
        """Differentiable version of :meth:`_texture` for a [rows*cols] AD Float."""
        ss = self._ss
        R, C = self.n_rows, self.n_cols
        Ht, Wt = R * ss, C * ss
        idx = dr.arange(mi.UInt32, Ht * Wt)
        ty, tx = idx // Wt, idx % Wt
        r = (R - 1) - ty // ss
        c = tx // ss
        val = dr.gather(mi.Float, pattern_dr, r * C + c)
        g = self._gap
        sy, sx = ty % ss, tx % ss
        inside = (sy >= g) & (sy < ss - g) & (sx >= g) & (sx < ss - g)
        val = dr.select(inside, val, 0.0)
        return mi.TensorXf(val, shape=(Ht, Wt, 1))

    def render(self, pattern_dr, spp=64, seed=0):
        """Render chip irradiance (relative, 1 = fully-on mirror) for a DMD pattern.

        ``pattern_dr``: flat [rows*cols] Dr.Jit array (may carry gradients).
        Returns a TensorXf of shape [rows*ss, cols*ss] (x, z).
        """
        self.params[self.key] = self.texture_dr(pattern_dr)
        self.params.update()
        img = mi.render(self.scene, self.params, spp=spp, seed=seed)
        img = img[..., 0] if img.ndim == 3 else img
        return img * (1.0 / self._ff_eff)
