"""Checks of the device physics against the paper's numbers and numerical convergence."""
import math
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from pwg.stack import Stack, core_mode, eo_overlap
from pwg.electro import LocalResponse, lateral_lsf


def test_tm0_effective_index_and_loss():
    m = core_mode(Stack.wright2024())
    assert abs(m.n_eff.real - 1.94) < 0.02          # paper: n_eff = 1.94
    assert 0.05 < m.loss_db_per_cm < 1.0            # paper: COMSOL 0.23 dB/cm, measured < 1


def test_photoconductor_calibration_and_resolution():
    r = LocalResponse(Stack.wright2024()).calibrate()
    assert abs(r.dn_ln(0.0) - 4.0e-3) < 1e-5
    assert abs(r.dn_ln(50.0) - 5.2e-3) < 1e-5
    prog = r.dn_eff(57.0) - r.dn_eff(0.0)
    assert 0.7e-3 < prog < 1.5e-3                   # paper: ~1e-3 programmable
    _, _, fwhm = lateral_lsf(r)
    assert 1.5 < fwhm < 5.0                         # paper: ~3.5 um electrostatic limit


def test_dz_convergence_of_programmed_unitary():
    """A design found at the default dz must give the same matrix when re-simulated at dz/2."""
    import drjit as dr
    from drjit.cuda.ad import Float
    from unitary_scaling import design
    from pwg.mmwg import MMWGConfig, MultimodeWaveguide, fidelity

    N, L = 8, 700.0
    res, wg, theta = design(N, L, 120, 0.1, verbose=False)
    th = theta.numpy().reshape(wg.grid.nz, wg.n_px)
    fine = MultimodeWaveguide(MMWGConfig(n_modes=N, length_um=L, dz=wg.grid.dz / 2))
    th2 = np.repeat(th, 2, axis=0)
    with dr.suspend_grad():
        a = wg.transfer(wg.index_map(Float(th.ravel())))
        b = fine.transfer(fine.index_map(Float(th2.ravel())))
    Ta = wg.matrix(a[0].numpy(), a[1].numpy())
    Tb = fine.matrix(b[0].numpy(), b[1].numpy())
    # same physical device => same matrix up to the (dz-dependent) mode phases
    Ta = Ta / np.exp(1j * wg.beta * wg.grid.length)[:, None]
    Tb = Tb / np.exp(1j * fine.beta * fine.grid.length)[:, None]
    F, Fn, err = fidelity(Tb, Ta / np.linalg.norm(Ta) * math.sqrt(N))
    assert Fn > 0.99, (F, Fn, err)


def test_effective_index_reduction_matches_2d_cross_section():
    from pwg.lateral import LocalIndexTable, EtchedGuide, check_against_2d
    tab = LocalIndexTable()
    g = EtchedGuide(width=10.0, etch_depth=tab.etch_for_na(0.3), sidewall_deg=70, table=tab)
    b2, b1 = check_against_2d(g)
    rel = np.abs(b1[1:] - b2[1:]) / b2[1:]
    assert rel.max() < 0.01


def test_static_modes_converged_with_physical_edge():
    """No split-step leakage: unprogrammed mode loss equals the physical local loss."""
    from drjit.cuda.ad import Float
    from pwg.mmwg import MMWGConfig, MultimodeWaveguide
    wg = MultimodeWaveguide(MMWGConfig(n_modes=16, length_um=2000))
    alpha = -np.log(wg.amp_x) / wg.grid.dz
    expect = np.exp(-2 * ((np.abs(wg.modes) ** 2 * alpha).sum(1) * wg.grid.dx) * 2000)
    re, im = wg.transfer(wg.index_map(Float(np.full(wg.n_par, -20, np.float32))))
    P = (np.abs(wg.matrix(re.numpy(), im.numpy())) ** 2).sum(0)
    excess_db_cm = -10 * np.log10(P / expect) / 0.2
    assert excess_db_cm.max() < 0.02
