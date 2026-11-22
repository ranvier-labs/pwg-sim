import numpy as np
from drjit.cuda import Float
from pwg.fft import fft, ifft


def test_fft_matches_numpy():
    rng = np.random.default_rng(0)
    for n in [2, 8, 64, 1024]:
        B = 3
        x = rng.normal(size=(B, n)) + 1j * rng.normal(size=(B, n))
        re, im = fft(Float(x.real.ravel()), Float(x.imag.ravel()), n)
        y = (re.numpy() + 1j * im.numpy()).reshape(B, n)
        np.testing.assert_allclose(y, np.fft.fft(x, axis=1), rtol=1e-4, atol=1e-3 * np.sqrt(n))
        re2, im2 = ifft(re, im, n)
        z = (re2.numpy() + 1j * im2.numpy()).reshape(B, n)
        np.testing.assert_allclose(z, x, atol=1e-4)
