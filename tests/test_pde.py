"""Tests for the IMEX PDE solver."""

from __future__ import annotations

import numpy as np
import pytest

from hft_jd import ModelParams, ProblemParams, solve_acquisition, solve_liquidation


def _acq_problem(**kwargs) -> ProblemParams:
    base = dict(
        kind="acquisition",
        T=1.0,
        N_units=390,
        kappa=1e-4,
        alpha=0.01,
        phi=1e-5,
        S0=30.97,
        S_min=29.0,
        S_max=31.1,
    )
    base.update(kwargs)
    return ProblemParams(**base)


def _liq_problem(**kwargs) -> ProblemParams:
    base = dict(
        kind="liquidation",
        T=1.0,
        N_units=390,
        kappa=1e-4,
        alpha=0.01,
        phi=1e-5,
        S0=30.97,
        S_min=30.8,
        S_max=33.0,
    )
    base.update(kwargs)
    return ProblemParams(**base)


# ---------- Boundary / terminal conditions ----------


def test_acquisition_terminal_condition():
    sol = solve_acquisition(_acq_problem(), ModelParams(sigma=0.1041), N=100, M=200)
    assert np.allclose(sol.h[-1, :], 0.01)


def test_acquisition_dirichlet_boundary_at_S_max():
    sol = solve_acquisition(_acq_problem(), ModelParams(sigma=0.1041), N=100, M=200)
    assert np.allclose(sol.h[:, -1], 0.01)


def test_liquidation_terminal_condition():
    sol = solve_liquidation(_liq_problem(), ModelParams(sigma=0.1041), N=100, M=200)
    assert np.allclose(sol.h[-1, :], 0.01)


def test_liquidation_dirichlet_boundary_at_S_min():
    sol = solve_liquidation(_liq_problem(), ModelParams(sigma=0.1041), N=100, M=200)
    assert np.allclose(sol.h[:, 0], 0.01)


# ---------- Solver invariants ----------


def test_h_is_finite_and_nonnegative():
    sol = solve_acquisition(_acq_problem(), ModelParams(sigma=0.1041), N=200, M=200)
    assert np.all(np.isfinite(sol.h))
    # h is positive in the regimes we care about for this problem (Cartea et al. 2015).
    assert np.all(sol.h >= -1e-10)


def test_acquisition_h_increases_toward_terminal():
    """For the acquisition problem with no inventory penalty far from S_max,
    h grows as t → T to match the terminal penalty α."""
    sol = solve_acquisition(_acq_problem(phi=0.0), ModelParams(sigma=0.1041), N=200, M=200)
    # Pick an interior price at S0 and watch h(t, S0) along time.
    i_S0 = np.argmin(np.abs(sol.S_grid - sol.problem.S0))
    h_along_t = sol.h[:, i_S0]
    # Should be monotone non-decreasing in t (i.e., increasing toward terminal α).
    assert np.all(np.diff(h_along_t) >= -1e-8)


def test_pure_diffusion_baseline_recovered_when_jump_terms_zero():
    """Setting σ̄ = ς = 0 should give a Cartea-et-al baseline. Compare a couple
    of grid points against running with tiny σ̄ — should agree closely."""
    sol_pure = solve_acquisition(
        _acq_problem(), ModelParams(sigma=0.1041, sigma_bar=0.0, varsigma=0.0), N=200, M=200
    )
    sol_eps = solve_acquisition(
        _acq_problem(), ModelParams(sigma=0.1041, sigma_bar=1e-8, varsigma=1e-8), N=200, M=200
    )
    diff = np.max(np.abs(sol_pure.h - sol_eps.h))
    assert diff < 1e-6


def test_higher_total_variance_increases_optimal_speed_at_S0():
    """Adding ς > 0 should raise the speed-of-trading scale h(0, S0) when the
    boundary cap at S_max is non-trivial. (Section 4 in the paper.)"""
    base = solve_acquisition(_acq_problem(), ModelParams(sigma=0.1041), N=200, M=200)
    boost = solve_acquisition(
        _acq_problem(), ModelParams(sigma=0.1041, sigma_bar=0.01598, varsigma=0.2), N=200, M=200
    )
    i_S0 = np.argmin(np.abs(base.S_grid - base.problem.S0))
    # at t=0 (start), the boosted version should not be smaller than the baseline.
    assert boost.h[0, i_S0] >= base.h[0, i_S0] - 1e-9


def test_solve_kind_mismatch_raises():
    with pytest.raises(ValueError):
        solve_acquisition(_liq_problem(), ModelParams(sigma=0.1))
    with pytest.raises(ValueError):
        solve_liquidation(_acq_problem(), ModelParams(sigma=0.1))


def test_value_interpolation_in_range():
    sol = solve_acquisition(_acq_problem(), ModelParams(sigma=0.1041), N=100, M=100)
    v = sol.value(0.5, 31.0)
    assert np.isfinite(v)
    assert v >= 0.0
