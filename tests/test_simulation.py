"""Tests for the strategy simulation engine."""

from __future__ import annotations

import numpy as np

from hft_jd import (
    ModelParams,
    ProblemParams,
    simulate,
    solve_acquisition,
    solve_liquidation,
)


def _acq_setup():
    problem = ProblemParams(
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
    model = ModelParams(sigma=0.1041, sigma_bar=0.01598, varsigma=0.1323)
    return problem, model


def _liq_setup():
    problem = ProblemParams(
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
    model = ModelParams(sigma=0.1041, sigma_bar=0.01598, varsigma=0.1323)
    return problem, model


def test_simulation_shapes():
    problem, model = _acq_setup()
    sol = solve_acquisition(problem, model, N=100, M=200)
    res = simulate(sol, n_paths=50, n_steps=200, seed=0)
    assert res.S.shape == (50, 201)
    assert res.Q.shape == (50, 201)
    assert res.cash.shape == (50, 201)
    assert res.speed.shape == (50, 201)
    assert res.avg_traded_price.shape == (50,)


def test_acquisition_inventory_reaches_target():
    problem, model = _acq_setup()
    sol = solve_acquisition(problem, model, N=100, M=200)
    res = simulate(sol, n_paths=200, n_steps=200, seed=42)
    final_inventory = res.Q[:, -1]
    # After applying terminal penalty the agent must have the full target.
    assert np.allclose(final_inventory, problem.N_units, atol=1e-6)


def test_liquidation_inventory_reaches_zero():
    problem, model = _liq_setup()
    sol = solve_liquidation(problem, model, N=100, M=200)
    res = simulate(sol, n_paths=200, n_steps=200, seed=42)
    final_inventory = res.Q[:, -1]
    assert np.allclose(final_inventory, 0.0, atol=1e-6)


def test_acquisition_speeds_nonnegative():
    problem, model = _acq_setup()
    sol = solve_acquisition(problem, model, N=100, M=200)
    res = simulate(sol, n_paths=100, n_steps=200, seed=1)
    assert np.all(res.speed >= -1e-10)


def test_avg_price_finite_and_bounded():
    problem, model = _acq_setup()
    sol = solve_acquisition(problem, model, N=100, M=200)
    res = simulate(sol, n_paths=200, n_steps=200, seed=7)
    assert np.all(np.isfinite(res.avg_traded_price))
    # Average price should be in a sensible range around S0.
    assert np.all(res.avg_traded_price > problem.S_min)
    assert np.all(res.avg_traded_price < problem.S_max + problem.alpha * problem.N_units + 1e-3)


def test_higher_jump_variance_speeds_up_trading():
    """Per Fig. 5 of the paper: with larger σ̄, ς the agent trades faster,
    so the average traded price ends closer to S0 and the histogram tightens.
    We check this by terminating-step distribution (earlier ⇒ faster) and
    the mean-speed at t=0."""
    problem, _ = _acq_setup()
    low = ModelParams(sigma=0.1041)
    high = ModelParams(sigma=0.1041, sigma_bar=0.01598, varsigma=0.2)
    sol_low = solve_acquisition(problem, low, N=100, M=200)
    sol_high = solve_acquisition(problem, high, N=100, M=200)
    res_low = simulate(sol_low, n_paths=300, n_steps=200, seed=11)
    res_high = simulate(sol_high, n_paths=300, n_steps=200, seed=11)
    assert np.mean(res_high.speed[:, 0]) > np.mean(res_low.speed[:, 0])
    assert np.median(res_high.terminated_at) <= np.median(res_low.terminated_at)
