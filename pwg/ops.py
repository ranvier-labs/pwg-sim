"""Small differentiable array helpers on flat row-major Dr.Jit arrays."""
from __future__ import annotations

import math

import drjit as dr
import numpy as np
from drjit.cuda.ad import Float, UInt32, Int32


def gaussian_taps(sigma_px: float, truncate: float = 4.0) -> np.ndarray:
    if sigma_px <= 1e-3:
        return np.array([1.0])
    r = max(1, int(math.ceil(truncate * sigma_px)))
    t = np.arange(-r, r + 1)
    w = np.exp(-0.5 * (t / sigma_px) ** 2)
    return w / w.sum()


def conv_rows(a: Float, rows: int, cols: int, taps: np.ndarray, axis: int) -> Float:
    """Convolve a [rows, cols] array along ``axis`` (1 = within rows) with clamped edges."""
    if len(taps) == 1:
        return a * float(taps[0])
    r = len(taps) // 2
    idx = dr.arange(Int32, rows * cols)
    i, j = idx // cols, idx % cols
    out = dr.zeros(Float, rows * cols)
    for t, w in zip(range(-r, r + 1), taps):
        if axis == 1:
            jj = dr.clip(j + t, 0, cols - 1)
            src = i * cols + jj
        else:
            ii = dr.clip(i + t, 0, rows - 1)
            src = ii * cols + j
        out += float(w) * dr.gather(Float, a, UInt32(src))
    return out


def resample_rows(a: Float, rows: int, src_x: np.ndarray, dst_x: np.ndarray, fill=0.0) -> Float:
    """Linear interpolation along the column axis from grid ``src_x`` to ``dst_x``.

    ``a`` is [rows, len(src_x)]; returns [rows, len(dst_x)]; outside -> ``fill``.
    """
    ns, nd = len(src_x), len(dst_x)
    d0 = src_x[1] - src_x[0]
    u = (dst_x - src_x[0]) / d0
    i0 = np.clip(np.floor(u).astype(np.int64), 0, ns - 1)
    i1 = np.clip(i0 + 1, 0, ns - 1)
    f = (u - np.floor(u)).astype(np.float32)
    inside = ((u >= -0.5) & (u <= ns - 0.5)).astype(np.float32)
    idx = dr.arange(UInt32, rows * nd)
    r, c = idx // nd, idx % nd
    I0 = dr.gather(UInt32, UInt32(i0.astype(np.uint32)), c)
    I1 = dr.gather(UInt32, UInt32(i1.astype(np.uint32)), c)
    F = dr.gather(Float, Float(f), c)
    M = dr.gather(Float, Float(inside), c)
    v = dr.gather(Float, a, r * ns + I0) * (1 - F) + dr.gather(Float, a, r * ns + I1) * F
    return v * M + (1 - M) * fill


def pool_rows(a: Float, rows: int, cols: int, k: int) -> Float:
    """Average groups of ``k`` consecutive rows: [rows, cols] -> [rows//k, cols]."""
    if k == 1:
        return a
    ro = rows // k
    idx = dr.arange(UInt32, ro * cols)
    i, j = idx // cols, idx % cols
    out = dr.zeros(Float, ro * cols)
    for t in range(k):
        out += dr.gather(Float, a, (i * k + t) * cols + j)
    return out * (1.0 / k)


def transpose(a: Float, rows: int, cols: int) -> Float:
    idx = dr.arange(UInt32, rows * cols)
    i, j = idx // rows, idx % rows   # output is [cols, rows]
    return dr.gather(Float, a, j * cols + i)
