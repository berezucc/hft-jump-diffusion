"""Problem and model parameter containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProblemKind = Literal["acquisition", "liquidation"]


@dataclass(frozen=True)
class ModelParams:
    """Reduced (post-coefficient-aggregation) parameters for the h(t, S) PDE.

    Attributes:
        sigma: Brownian diffusion coefficient σ of the midprice.
        sigma_bar: Diffusion-approximation coefficient σ̄ from the jump intensity
            (Eq. 6 / Eq. 12). Set to 0 to recover the pure-diffusion baseline.
        varsigma: Diffusion-approximation coefficient ς from the jump tick size
            (Eq. 5 → ςSM, Eq. 11 → ςHP). Set to 0 to recover the pure-diffusion
            baseline.
        eta: Drift coefficient η carried by the permanent-impact term.
            Defaults to 0 because the paper sets the permanent impact b=0
            for the numerical scheme (so η drops out of the reduced PDE).
    """

    sigma: float
    sigma_bar: float = 0.0
    varsigma: float = 0.0
    eta: float = 0.0

    @property
    def total_variance(self) -> float:
        """σ² + σ̄² + ς² — the diffusion coefficient of the PDE."""
        return self.sigma**2 + self.sigma_bar**2 + self.varsigma**2

    def __post_init__(self) -> None:
        for name in ("sigma", "sigma_bar", "varsigma"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class ProblemParams:
    """Trading problem parameters.

    Attributes:
        kind: 'acquisition' or 'liquidation'.
        T: Trading horizon.
        N_units: Number of units to acquire / liquidate.
        kappa: Temporary impact coefficient κ (>0). Optimal max trading speed
            is α/κ.
        alpha: Terminal/boundary penalty α (≥0).
        phi: Running inventory penalty φ (≥0). The paper uses 1e-5.
        S0: Initial midprice.
        S_min: Lower price boundary of the grid (also the price floor in
            liquidation problems).
        S_max: Upper price boundary of the grid (also the price cap in
            acquisition problems).
    """

    kind: ProblemKind
    T: float
    N_units: int
    kappa: float
    alpha: float
    phi: float
    S0: float
    S_min: float
    S_max: float

    def __post_init__(self) -> None:
        if self.kind not in ("acquisition", "liquidation"):
            raise ValueError(f"kind must be 'acquisition' or 'liquidation', got {self.kind!r}")
        if self.kappa <= 0:
            raise ValueError("kappa must be > 0")
        if not (self.S_min < self.S0 < self.S_max):
            raise ValueError("Require S_min < S0 < S_max")
        if self.T <= 0 or self.N_units <= 0 or self.alpha < 0 or self.phi < 0:
            raise ValueError("T, N_units must be > 0; alpha, phi must be ≥ 0")
