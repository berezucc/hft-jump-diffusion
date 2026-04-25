"""IMEX finite-difference solver for the reduced HJB PDE in h(t, S).

After the ansatz H(t, S, q) = qS + q²·h(t, S) and setting permanent impact b=0,
both acquisition and liquidation problems reduce to (Eq. 29 / Eq. 43):

    ∂_t h + (1/2)·σ_total² · ∂_SS h − (1/κ)·h² ± φ = 0

with σ_total² := σ² + σ̄² + ς², '+φ' for acquisition, '−φ' for liquidation.

Terminal condition:                   h(T, S) = α        (both problems)
Dirichlet boundary on the active cap: h(t, S_max) = α    (acquisition)
                                      h(t, S_min) = α    (liquidation)
Neumann ∂_SS h = 0 on the inactive boundary               (Cartea et al. 2015 Sec. 6).

We march **backward in real time** from n = N (terminal) to n = 0.
Diffusion is treated implicitly (tridiagonal solve via Thomas algorithm),
the quadratic h² reaction is treated explicitly using h^{n+1} — this is the
IMEX split used by the paper (Eq. 49–54).

Note on the paper's notation: Eq. 50 prints `(1−β)` on the diagonal, but
deriving the implicit step from Eq. 49 yields `(1+β)`. We follow the
mathematically consistent form `(1+β)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hft_jd.params import ModelParams, ProblemParams


@dataclass
class HSolution:
    """Solution grid for h(t, S).

    Attributes:
        t_grid: Time grid, shape (N+1,), values 0, Δt, ..., T.
        S_grid: Price grid, shape (M+1,), values S_min, S_min+ΔS, ..., S_max.
        h: Solution array, shape (N+1, M+1). h[n, i] ≈ h(t_grid[n], S_grid[i]).
        problem: The ProblemParams used.
        model: The ModelParams used.
    """

    t_grid: np.ndarray
    S_grid: np.ndarray
    h: np.ndarray
    problem: ProblemParams
    model: ModelParams

    @property
    def dt(self) -> float:
        return float(self.t_grid[1] - self.t_grid[0])

    @property
    def dS(self) -> float:
        return float(self.S_grid[1] - self.S_grid[0])

    def value(self, t: float, S: float) -> float:
        """Bilinear interpolation of h at an arbitrary (t, S) inside the grid."""
        n_idx, n_w = _interp_index(t, self.t_grid)
        i_idx, i_w = _interp_index(S, self.S_grid)
        h = self.h
        v00 = h[n_idx, i_idx]
        v01 = h[n_idx, i_idx + 1]
        v10 = h[n_idx + 1, i_idx]
        v11 = h[n_idx + 1, i_idx + 1]
        return float(
            (1 - n_w) * ((1 - i_w) * v00 + i_w * v01)
            + n_w * ((1 - i_w) * v10 + i_w * v11)
        )


def _interp_index(x: float, grid: np.ndarray) -> tuple[int, float]:
    """Return (left_index, weight_in_[0,1]) for linear interp on a uniform grid."""
    x = float(np.clip(x, grid[0], grid[-1]))
    n = len(grid) - 1
    pos = (x - grid[0]) / (grid[-1] - grid[0]) * n
    idx = int(np.clip(np.floor(pos), 0, n - 1))
    w = pos - idx
    return idx, w


def _thomas(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Tridiagonal solve via the Thomas algorithm.

    sub, diag, sup are the sub-, main, and super-diagonals of size m. sub[0] and
    sup[-1] are unused (no element off the matrix). rhs has size m.
    """
    m = diag.size
    cp = np.empty(m)
    dp = np.empty(m)
    cp[0] = sup[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for k in range(1, m):
        denom = diag[k] - sub[k] * cp[k - 1]
        cp[k] = sup[k] / denom if k < m - 1 else 0.0
        dp[k] = (rhs[k] - sub[k] * dp[k - 1]) / denom
    x = np.empty(m)
    x[-1] = dp[-1]
    for k in range(m - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return x


def _build_grid(problem: ProblemParams, N: int, M: int) -> tuple[np.ndarray, np.ndarray]:
    t_grid = np.linspace(0.0, problem.T, N + 1)
    S_grid = np.linspace(problem.S_min, problem.S_max, M + 1)
    return t_grid, S_grid


def _solve(
    problem: ProblemParams,
    model: ModelParams,
    N: int,
    M: int,
    *,
    sign_phi: int,
    dirichlet_at: str,
) -> HSolution:
    """Generic IMEX backward march. sign_phi=+1 for acquisition, −1 for liquidation.

    dirichlet_at = 'high' (i=M) or 'low' (i=0) — the side that has h=α boundary.
    """
    t_grid, S_grid = _build_grid(problem, N, M)
    dt = t_grid[1] - t_grid[0]
    dS = S_grid[1] - S_grid[0]

    var = model.total_variance
    alpha_coef = var * dt / (2.0 * dS * dS)
    beta_coef = 2.0 * alpha_coef  # = var * dt / dS²

    h = np.empty((N + 1, M + 1))
    h[N, :] = problem.alpha  # terminal condition

    if dirichlet_at == "high":
        # acquisition: h(t, S_max) = α → fix the rightmost column.
        h[:, M] = problem.alpha
        interior_lo, interior_hi = 1, M  # interior indices [1, M-1]
        neumann_idx = 0
    elif dirichlet_at == "low":
        # liquidation: h(t, S_min) = α → fix the leftmost column.
        h[:, 0] = problem.alpha
        interior_lo, interior_hi = 1, M
        neumann_idx = M
    else:
        raise ValueError(dirichlet_at)

    # Implicit tridiagonal matrix on interior nodes (size M-1 when both ends fixed,
    # but only one end is Dirichlet — the other is handled via the boundary ODE).
    # We treat the Neumann-side index as a regular unknown updated by an explicit
    # ODE step (no diffusion contribution there, since ∂_SS h = 0).
    inner_size = M - 1
    sub = np.full(inner_size, -alpha_coef)
    diag = np.full(inner_size, 1.0 + beta_coef)
    sup = np.full(inner_size, -alpha_coef)
    sub[0] = 0.0  # unused
    sup[-1] = 0.0  # unused

    for n in range(N - 1, -1, -1):
        h_next = h[n + 1, :]

        # Boundary update via the boundary ODE (∂_SS h = 0, drop diffusion).
        # ∂_t h − h²/κ + sign_phi·φ = 0  →  h^n = h^{n+1} − dt/κ·(h^{n+1})² + sign_phi·dt·φ
        h_neumann = (
            h_next[neumann_idx]
            - dt / problem.kappa * h_next[neumann_idx] ** 2
            + sign_phi * dt * problem.phi
        )

        # Build the RHS for interior nodes using IMEX: explicit reaction at h^{n+1}.
        rhs = (
            h_next[interior_lo:interior_hi]
            - dt / problem.kappa * h_next[interior_lo:interior_hi] ** 2
            + sign_phi * dt * problem.phi
        ).astype(float, copy=True)

        # Account for the known Dirichlet column on whichever side it sits.
        if dirichlet_at == "high":
            rhs[-1] += alpha_coef * h[n, M]  # h_M is fixed at α
            # Left edge i=1 connects to neumann node h_0 (which we will fill below
            # from the boundary ODE).
            rhs[0] += alpha_coef * h_neumann
        else:  # dirichlet_at == "low"
            rhs[0] += alpha_coef * h[n, 0]  # h_0 is fixed at α
            rhs[-1] += alpha_coef * h_neumann

        sol = _thomas(sub, diag, sup, rhs)
        h[n, interior_lo:interior_hi] = sol
        h[n, neumann_idx] = h_neumann

    return HSolution(t_grid=t_grid, S_grid=S_grid, h=h, problem=problem, model=model)


def solve_acquisition(
    problem: ProblemParams, model: ModelParams, *, N: int = 500, M: int = 1000
) -> HSolution:
    """Solve the acquisition problem PDE (Eq. 29) on a uniform grid.

    Args:
        problem: Trading problem parameters; expects kind='acquisition'.
        model: Reduced model parameters (σ, σ̄, ς).
        N: Number of time steps (grid is N+1 points from 0 to T).
        M: Number of space steps (grid is M+1 points from S_min to S_max).
    """
    if problem.kind != "acquisition":
        raise ValueError("solve_acquisition expects ProblemParams.kind == 'acquisition'")
    return _solve(problem, model, N, M, sign_phi=+1, dirichlet_at="high")


def solve_liquidation(
    problem: ProblemParams, model: ModelParams, *, N: int = 500, M: int = 1000
) -> HSolution:
    """Solve the liquidation problem PDE (Eq. 43) on a uniform grid."""
    if problem.kind != "liquidation":
        raise ValueError("solve_liquidation expects ProblemParams.kind == 'liquidation'")
    return _solve(problem, model, N, M, sign_phi=-1, dirichlet_at="low")
