"""Tests for the Semi-Markov and Hawkes coefficient formulas (Eq. 4–6, 10–12)."""

from __future__ import annotations

import math

import pytest

from hft_jd.coefficients import (
    HawkesParams,
    SemiMarkovParams,
    hawkes_coefficients,
    semi_markov_coefficients,
)


# ---------- Semi-Markov ----------


def test_semi_markov_pi_star_balanced():
    """Symmetric chain (p = p') has π* = 1/2."""
    params = SemiMarkovParams(
        delta=0.01,
        p_cont=0.4,
        p_cont_prime=0.4,
        m_tau=1.0,
        sigma_diffusion_sq=0.01,
    )
    coefs = semi_markov_coefficients(params)
    assert math.isclose(coefs.pi_star, 0.5, abs_tol=1e-12)


def test_semi_markov_eta_zero_for_balanced_chain():
    """In the balanced market (π* = 1/2) the drift η = δ(2π*−1)/m_tau = 0."""
    params = SemiMarkovParams(
        delta=0.01,
        p_cont=0.45,
        p_cont_prime=0.45,
        m_tau=2.0,
        sigma_diffusion_sq=0.01,
    )
    coefs = semi_markov_coefficients(params)
    assert math.isclose(coefs.eta, 0.0, abs_tol=1e-12)


def test_semi_markov_sigma_bar_increases_with_jump_variance():
    base = SemiMarkovParams(
        delta=0.01,
        p_cont=0.5,
        p_cont_prime=0.4,
        m_tau=1.0,
        sigma_diffusion_sq=0.01,
    )
    larger_delta = SemiMarkovParams(
        delta=0.02,
        p_cont=0.5,
        p_cont_prime=0.4,
        m_tau=1.0,
        sigma_diffusion_sq=0.01,
    )
    a = semi_markov_coefficients(base)
    b = semi_markov_coefficients(larger_delta)
    assert b.sigma_bar > a.sigma_bar
    assert b.varsigma > a.varsigma


def test_semi_markov_rejects_invalid_probs():
    with pytest.raises(ValueError):
        SemiMarkovParams(
            delta=0.01, p_cont=0.0, p_cont_prime=0.4, m_tau=1.0, sigma_diffusion_sq=0.01
        )
    with pytest.raises(ValueError):
        SemiMarkovParams(
            delta=0.01, p_cont=0.4, p_cont_prime=1.0, m_tau=1.0, sigma_diffusion_sq=0.01
        )


# ---------- Hawkes ----------


def test_hawkes_eta_scales_with_lambda():
    """η_HP = a* · λ / (1 − μ̂) should be linear in λ."""
    base = HawkesParams(
        delta=0.01,
        p=0.5,
        p_prime=0.5,
        a_star=0.005,
        lam=1.0,
        mu_hat=0.3,
        sigma_hat_sq=0.001,
    )
    doubled = HawkesParams(**{**base.__dict__, "lam": 2.0})
    a = hawkes_coefficients(base)
    b = hawkes_coefficients(doubled)
    assert math.isclose(b.eta / a.eta, 2.0, rel_tol=1e-12)


def test_hawkes_blowup_as_mu_to_one():
    """η_HP, ς_HP → ∞ as μ̂ → 1 (criticality)."""
    p1 = HawkesParams(
        delta=0.01, p=0.5, p_prime=0.5, a_star=0.005, lam=1.0, mu_hat=0.5, sigma_hat_sq=0.001
    )
    p2 = HawkesParams(
        delta=0.01, p=0.5, p_prime=0.5, a_star=0.005, lam=1.0, mu_hat=0.9, sigma_hat_sq=0.001
    )
    a = hawkes_coefficients(p1)
    b = hawkes_coefficients(p2)
    assert b.eta > a.eta
    assert b.varsigma > a.varsigma


def test_hawkes_rejects_supercritical():
    with pytest.raises(ValueError):
        HawkesParams(
            delta=0.01, p=0.5, p_prime=0.5, a_star=0.005, lam=1.0, mu_hat=1.0, sigma_hat_sq=0.001
        )
