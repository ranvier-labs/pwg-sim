"""Detection noise.

Photonic matrix-vector products are read out by photodetectors; at low optical
energy the dominant and *exactly modelled* noise is photon shot noise (Poisson
statistics of photoelectrons), plus detector read noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H = 6.62607015e-34
C = 299792458.0


@dataclass
class Detector:
    energy_per_inference_J: float = 1e-13   # optical energy launched per input vector
    wavelength_m: float = 1.55e-6
    quantum_efficiency: float = 0.8
    read_noise_e: float = 30.0              # rms electrons per output channel
    rin_db_hz: float | None = None          # laser RIN (optional)
    bandwidth_hz: float = 1e9

    def photons(self, P):
        return P * self.energy_per_inference_J / (H * C / self.wavelength_m)

    def measure(self, P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """``P``: power fraction per output (relative to launched power).

        Returns detected photo-electrons (shot + read noise, optional RIN)."""
        mean = self.quantum_efficiency * self.photons(np.clip(P, 0, None))
        if self.rin_db_hz is not None:
            rin = 10 ** (self.rin_db_hz / 10) * self.bandwidth_hz
            mean = mean * (1 + rng.normal(0, np.sqrt(rin), size=mean.shape[:-1] + (1,)))
        e = rng.poisson(np.clip(mean, 0, None)).astype(float)
        return e + rng.normal(0, self.read_noise_e, size=e.shape)
