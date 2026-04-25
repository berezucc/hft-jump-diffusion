"""Closed-form coefficients (η, σ̄, ς) for the Semi-Markov and Hawkes cases.

References (Lalor & Swishchuk 2025):
    Eq. 4: η_SM = s* / (m·τ),   s* = δ(2π* − 1).
    Eq. 5: (σ*_SM)² = 4δ² · (1 − p'_cont + π*(p'_cont − p_cont)) / (p_cont + p'_cont − 2)².
    Eq. 6: σ̄_SM = sqrt[(σ*)² / (m·τ) + Π·σ² / (m·τ)].
    Eq. 10: η_HP = a* · λ / (1 − μ̂).
    Eq. 11: σ*_HP = sqrt[4δ² · ((1 − p' + π*(p' − p))/(p+p'−2)² − π*(1−π*))].
    Eq. 12: σ̄_HP = sqrt[(σ*)² + a* · sqrt(λ / (1 − μ̂))].

The defining quantity for the reduced PDE is the *aggregated* diffusion
σ_total² = σ² + σ̄² + ς². See `hft_jd.params.ModelParams.total_variance`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Semi-Markov
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemiMarkovParams:
    """Calibrated inputs for the Semi-Markov diffusion approximation.

    Attributes:
        delta: Tick size δ (price increments take values ±δ).
        p_cont: P[X_{k+1}=δ | X_k=δ]  — probability of an up-tick following an up-tick.
        p_cont_prime: P[X_{k+1}=−δ | X_k=−δ] — probability of a down-tick following a
            down-tick.
        m_tau: m·τ in Eq. 6 — mean inter-arrival × scaling. Treated here as a single
            calibrated quantity (the paper computes it from order-book interarrival
            statistics; see Swishchuk & Vadori 2017, Sec. 4.2).
        sigma_star_sq: (σ*)² from Eq. 5. If left as None, computed from p_cont,
            p_cont_prime and delta.
        sigma_diffusion_sq: σ² of the Brownian component (used in Eq. 6 σ̄ formula).
        Pi: scaling Π in Eq. 6 (typically the long-run probability matrix factor).
            Defaults to 1 — many calibrations fold this into m_tau.
    """

    delta: float
    p_cont: float
    p_cont_prime: float
    m_tau: float
    sigma_diffusion_sq: float
    sigma_star_sq: float | None = None
    Pi: float = 1.0

    def __post_init__(self) -> None:
        if self.delta <= 0:
            raise ValueError("delta must be > 0")
        for p in (self.p_cont, self.p_cont_prime):
            if not 0.0 < p < 1.0:
                raise ValueError("p_cont and p_cont_prime must be in (0, 1)")
        if self.m_tau <= 0:
            raise ValueError("m_tau must be > 0")


def _pi_star_two_state(p: float, p_prime: float) -> float:
    """Stationary up-tick probability for the two-state chain.

    π* = (p' − 1) / (p + p' − 2).  See Eq. 5 footnote in the paper.
    """
    denom = p + p_prime - 2.0
    if math.isclose(denom, 0.0):
        raise ValueError("Degenerate Markov chain: p + p' = 2 (absorbing states).")
    return (p_prime - 1.0) / denom


def _semi_markov_sigma_star_sq(delta: float, p: float, p_prime: float) -> float:
    """Eq. 5: (σ*_SM)² = 4δ² · (1 − p' + π*(p' − p)) / (p + p' − 2)²."""
    pi_star = _pi_star_two_state(p, p_prime)
    numer = 1.0 - p_prime + pi_star * (p_prime - p)
    denom = (p + p_prime - 2.0) ** 2
    return 4.0 * delta**2 * numer / denom


@dataclass(frozen=True)
class SemiMarkovCoefficients:
    eta: float
    sigma_bar: float
    varsigma: float
    pi_star: float
    sigma_star_sq: float


def semi_markov_coefficients(p: SemiMarkovParams) -> SemiMarkovCoefficients:
    """Compute (η_SM, σ̄_SM, ς_SM) from calibrated Semi-Markov inputs.

    Returns the trio used by the reduced PDE plus the intermediate π* and (σ*)²
    so callers can introspect the calibration.
    """
    pi_star = _pi_star_two_state(p.p_cont, p.p_cont_prime)
    sigma_star_sq = (
        p.sigma_star_sq
        if p.sigma_star_sq is not None
        else _semi_markov_sigma_star_sq(p.delta, p.p_cont, p.p_cont_prime)
    )

    s_star = p.delta * (2.0 * pi_star - 1.0)
    eta = s_star / p.m_tau
    sigma_bar = math.sqrt(sigma_star_sq / p.m_tau + p.Pi * p.sigma_diffusion_sq / p.m_tau)
    # ς_SM = σ* / sqrt(τ).  We don't separate τ from m·τ in the calibration object,
    # so we expose ς via its squared contribution: σ*² / τ ≈ σ*² · m / m_tau.
    # When the user supplies sigma_star_sq directly, they can override varsigma in
    # ModelParams. Conservatively, we report ς = sqrt(σ*² / m_tau), matching the
    # canonical Eq. 6 simplification when m=1.
    varsigma = math.sqrt(sigma_star_sq / p.m_tau)
    return SemiMarkovCoefficients(
        eta=eta,
        sigma_bar=sigma_bar,
        varsigma=varsigma,
        pi_star=pi_star,
        sigma_star_sq=sigma_star_sq,
    )


# ---------------------------------------------------------------------------
# Hawkes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HawkesParams:
    """Calibrated inputs for the Hawkes diffusion approximation.

    Attributes:
        delta: Tick size δ.
        p: P[X_{k+1}=δ | X_k=δ].
        p_prime: P[X_{k+1}=−δ | X_k=−δ].
        a_star: a* := Σ π_i a(i) — the ergodic mean of the price-movement map.
        lam: Background intensity λ of the Hawkes process.
        mu_hat: ∫ μ(s) ds, the integrated excitation function (must be in [0, 1)).
        sigma_hat_sq: σ̂² := Σ π_i v(i), variance of state transitions (Eq. after 12).
    """

    delta: float
    p: float
    p_prime: float
    a_star: float
    lam: float
    mu_hat: float
    sigma_hat_sq: float

    def __post_init__(self) -> None:
        if self.delta <= 0:
            raise ValueError("delta must be > 0")
        for prob in (self.p, self.p_prime):
            if not 0.0 < prob < 1.0:
                raise ValueError("p and p_prime must be in (0, 1)")
        if not 0.0 <= self.mu_hat < 1.0:
            raise ValueError("mu_hat must satisfy 0 ≤ μ̂ < 1")
        if self.lam <= 0:
            raise ValueError("lam must be > 0")
        if self.sigma_hat_sq < 0:
            raise ValueError("sigma_hat_sq must be ≥ 0")


@dataclass(frozen=True)
class HawkesCoefficients:
    eta: float
    sigma_bar: float
    varsigma: float
    pi_star: float
    sigma_star_sq: float


def _hawkes_sigma_star_sq(delta: float, p: float, p_prime: float) -> float:
    """Eq. 11: (σ*_HP)² = 4δ²·[(1 − p' + π*(p' − p))/(p + p' − 2)² − π*(1 − π*)]."""
    pi_star = _pi_star_two_state(p, p_prime)
    a = (1.0 - p_prime + pi_star * (p_prime - p)) / ((p + p_prime - 2.0) ** 2)
    b = pi_star * (1.0 - pi_star)
    return 4.0 * delta**2 * (a - b)


def hawkes_coefficients(h: HawkesParams) -> HawkesCoefficients:
    """Compute (η_HP, σ̄_HP, ς_HP) from calibrated Hawkes inputs."""
    pi_star = _pi_star_two_state(h.p, h.p_prime)
    sigma_star_sq = _hawkes_sigma_star_sq(h.delta, h.p, h.p_prime)
    one_minus_mu = 1.0 - h.mu_hat

    eta = h.a_star * h.lam / one_minus_mu
    # ς_HP = σ̂ · sqrt(λ / (1 − μ̂))  — Eq. 12 footnote.
    varsigma = math.sqrt(h.sigma_hat_sq * h.lam / one_minus_mu)
    # σ̄_HP = sqrt[(σ*)² + a* · sqrt(λ / (1 − μ̂))]  — Eq. 12.
    sigma_bar = math.sqrt(sigma_star_sq + h.a_star * math.sqrt(h.lam / one_minus_mu))
    return HawkesCoefficients(
        eta=eta,
        sigma_bar=sigma_bar,
        varsigma=varsigma,
        pi_star=pi_star,
        sigma_star_sq=sigma_star_sq,
    )
