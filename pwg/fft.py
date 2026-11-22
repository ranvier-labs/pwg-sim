"""Batched complex FFT written directly in Dr.Jit.

Dr.Jit has no FFT primitive, so we implement a radix-2 Stockham autosort FFT
where every stage is a single gather-only kernel over the whole batch.  The
data layout is row-major ``[batch, n]`` flattened, with real and imaginary
parts stored in separate ``Float`` arrays.  ``n`` must be a power of two.

The functions operate on non-differentiable arrays (they are used inside the
hand-written adjoint of :mod:`pwg.bpm`), but they would also work on AD types.
"""
from __future__ import annotations

import math

import drjit as dr
from drjit.cuda import Float, UInt32


def _check_pow2(n: int) -> int:
    m = int(round(math.log2(n)))
    if 1 << m != n:
        raise ValueError(f"FFT size must be a power of two, got {n}")
    return m


def fft(re: Float, im: Float, n: int, inverse: bool = False, normalize: bool = True):
    """Complex FFT along the last axis of a ``[batch, n]`` array.

    Forward transform uses ``exp(-2 pi i k x / n)``; the inverse uses the
    conjugate kernel and (if ``normalize``) divides by ``n``.
    """
    m = _check_pow2(n)
    total = dr.width(re)
    g = dr.arange(UInt32, total)
    base = g & ~UInt32(n - 1)          # start of this row
    o = g & UInt32(n - 1)              # output index within row
    half = n // 2
    sign = 1.0 if inverse else -1.0

    l = half  # noqa: E741
    s = 0      # log2(m_stride)
    while l >= 1:
        mstride = 1 << s
        k = o & UInt32(mstride - 1)
        jj = o >> (s + 1)
        r = (o >> s) & UInt32(1)
        t = base + k + jj * mstride
        a_re = dr.gather(Float, re, t)
        a_im = dr.gather(Float, im, t)
        b_re = dr.gather(Float, re, t + half)
        b_im = dr.gather(Float, im, t + half)
        # twiddle w = exp(sign * 2 pi i jj / (2 l))
        ang = Float(jj) * (sign * math.pi / l)
        s_, c_ = dr.sincos(ang)
        d_re = a_re - b_re
        d_im = a_im - b_im
        sel = dr.eq(r, 0) if hasattr(dr, "eq") else (r == 0)
        re_new = dr.select(sel, a_re + b_re, d_re * c_ - d_im * s_)
        im_new = dr.select(sel, a_im + b_im, d_re * s_ + d_im * c_)
        re, im = re_new, im_new
        dr.eval(re, im)
        l //= 2  # noqa: E741
        s += 1
    if inverse and normalize:
        re = re * (1.0 / n)
        im = im * (1.0 / n)
    return re, im


def ifft(re: Float, im: Float, n: int):
    return fft(re, im, n, inverse=True)
