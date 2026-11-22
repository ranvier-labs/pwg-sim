import math
import numpy as np
import drjit as dr
from drjit.cuda import Float
from drjit.cuda.ad import Float as FloatAD
from pwg.bpm import Grid, Propagator, propagate


def gaussian(grid, x0, w0, batch_x0=None):
    x = grid.x()
    return np.exp(-((x - x0) / w0) ** 2)


def test_gaussian_diffraction_matches_analytic():
    g = Grid(nx=1024, dx=0.5, nz=200, dz=5.0, absorber_width=0.0)
    p = Propagator(g)
    w0 = 4.0
    e = gaussian(g, 0, w0).astype(np.complex64)
    re, im, _ = p.forward(Float(np.zeros(g.nz * g.nx, np.float32)), Float(e.real), Float(e.imag))
    I = re.numpy() ** 2 + im.numpy() ** 2
    zR = math.pi * w0**2 * g.n0 / g.wavelength
    wz = w0 * math.sqrt(1 + (g.length / zR) ** 2)
    x = g.x()
    w_meas = 2 * math.sqrt(np.sum(I * x**2) / np.sum(I))
    assert abs(w_meas - wz) / wz < 0.02


def test_index_gradient_deflects_beam():
    g = Grid(nx=2048, dx=0.5, nz=400, dz=5.0)
    p = Propagator(g)
    grad = 2e-6  # dn/dx [1/um]
    dn = np.tile((grad * g.x()).astype(np.float32), g.nz)
    e = gaussian(g, 0, 20.0).astype(np.complex64)
    re, im, _ = p.forward(Float(dn), Float(e.real), Float(e.imag))
    I = re.numpy() ** 2 + im.numpy() ** 2
    xc = np.sum(I * g.x()) / np.sum(I)
    expected = 0.5 * grad / g.n0 * g.length**2  # ray in linear index gradient
    assert abs(xc - expected) / expected < 0.03


def test_adjoint_matches_finite_differences():
    g = Grid(nx=128, dx=1.0, nz=37, dz=10.0, loss_db_per_cm=3.0)
    p = Propagator(g, checkpoint_every=5)
    rng = np.random.default_rng(1)
    B = 3
    e = np.stack([gaussian(g, x0, 5.0) for x0 in (-20, 0, 15)]).astype(np.complex64)
    target = rng.normal(size=(B, g.nx)) + 1j * rng.normal(size=(B, g.nx))
    dn0 = (1e-3 * rng.random(g.nz * g.nx)).astype(np.float64)

    def loss_np(dn):
        dnv = FloatAD(dn.astype(np.float32))
        re, im = propagate(p, dnv, Float(e.real.ravel()), Float(e.imag.ravel()), B)
        out = re.numpy() + 1j * im.numpy()
        return float(np.sum(np.real(np.conj(target.ravel()) * out)) + np.sum(np.abs(out) ** 4))

    dnv = FloatAD(dn0.astype(np.float32))
    dr.enable_grad(dnv)
    re, im = propagate(p, dnv, Float(e.real.ravel()), Float(e.imag.ravel()), B)
    tr, ti = FloatAD(target.real.ravel()), FloatAD(target.imag.ravel())
    I = re * re + im * im
    L = dr.sum(tr * re + ti * im) + dr.sum(I * I)
    dr.backward(L)
    gad = dr.grad(dnv).numpy()
    for idx in [5, g.nx * 10 + 64, g.nx * 36 + 70, g.nx * 20 + 50]:
        h = 2e-4
        dp = dn0.copy(); dp[idx] += h
        dm = dn0.copy(); dm[idx] -= h
        fd = (loss_np(dp) - loss_np(dm)) / (2 * h)
        assert abs(fd - gad[idx]) <= 2e-2 * max(abs(fd), 1e-1), (idx, fd, gad[idx])
