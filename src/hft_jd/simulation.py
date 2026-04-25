"""Strategy simulation engine for acquisition and liquidation.

Discretises the controlled processes from Sec. 3 of Lalor & Swishchuk (2025):

    Inventory:        dQ_t = ±ν_t dt
    Midprice:         dS_t = ±g(ν_t)·η dt + sqrt(σ² + σ̄² + ς²) dW_t
    Execution price:  Ŝ_t = S_t + (Δ ± f(ν_t)) / 2
    Cash:             dC_t = Ŝ_t · ν_t dt

with the linear impact model g(ν) = bν, f(ν) = κν. We simulate many paths
in parallel using the precomputed feedback policy ν* = q·h(t,S)/κ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hft_jd.pde import HSolution
from hft_jd.policy import FeedbackPolicy


@dataclass
class SimulationResult:
    """Outputs from a Monte-Carlo strategy simulation.

    Each array's first axis is path, second is time. A path that hits a
    stopping condition before T is held constant past that point and the
    `terminated_at` array records the step at which it stopped.

    Attributes:
        t_grid: shape (N+1,)
        S: shape (n_paths, N+1) — midprice paths.
        Q: shape (n_paths, N+1) — inventory.
        S_exec: shape (n_paths, N+1) — execution prices Ŝ.
        cash: shape (n_paths, N+1) — running cash C.
        speed: shape (n_paths, N+1) — trading speed ν.
        terminated_at: shape (n_paths,) — step index of termination
            (= N+1 if the path ran all the way to T without stopping).
        avg_traded_price: shape (n_paths,) — total cash divided by total
            units traded (= N_units for acquisition, target liq for
            liquidation), accounting for the terminal penalty.
    """

    t_grid: np.ndarray
    S: np.ndarray
    Q: np.ndarray
    S_exec: np.ndarray
    cash: np.ndarray
    speed: np.ndarray
    terminated_at: np.ndarray
    avg_traded_price: np.ndarray


def simulate(
    solution: HSolution,
    *,
    n_paths: int = 1_000,
    n_steps: int | None = None,
    spread: float = 0.0,
    b_permanent: float = 0.0,
    seed: int | None = None,
) -> SimulationResult:
    """Run n_paths Monte-Carlo simulations of the optimal strategy.

    Args:
        solution: Output of `solve_acquisition` or `solve_liquidation`.
        n_paths: Number of independent simulated paths.
        n_steps: Time-discretisation for the simulation. Defaults to the
            same N as the solution grid.
        spread: Constant bid-ask spread Δ in Eq. 18.
        b_permanent: Permanent impact coefficient b. The PDE solution sets
            b=0; passing b>0 here re-introduces a small drift in the
            simulated price (it does *not* invalidate the policy because
            the paper argues b is small relative to κ).
        seed: PRNG seed.
    """
    problem = solution.problem
    model = solution.model
    is_acquisition = problem.kind == "acquisition"
    sign = +1.0 if is_acquisition else -1.0

    N = n_steps if n_steps is not None else len(solution.t_grid) - 1
    dt = problem.T / N
    t_grid = np.linspace(0.0, problem.T, N + 1)
    sigma_total = float(np.sqrt(model.total_variance))

    policy = FeedbackPolicy(solution)

    rng = np.random.default_rng(seed)

    S = np.full((n_paths, N + 1), problem.S0, dtype=float)
    if is_acquisition:
        Q = np.zeros((n_paths, N + 1), dtype=float)
        Y = np.full((n_paths, N + 1), problem.N_units, dtype=float)  # remaining to acquire
    else:
        Q = np.full((n_paths, N + 1), problem.N_units, dtype=float)  # remaining to liquidate
        Y = Q  # liquidation uses Q directly
    S_exec = np.full((n_paths, N + 1), problem.S0, dtype=float)
    cash = np.zeros((n_paths, N + 1), dtype=float)
    speed = np.zeros((n_paths, N + 1), dtype=float)

    active = np.ones(n_paths, dtype=bool)
    terminated_at = np.full(n_paths, N + 1, dtype=int)

    sqrt_dt = np.sqrt(dt)

    for n in range(N):
        t_n = t_grid[n]

        # remaining units → drives feedback control
        remaining = Y[:, n] if is_acquisition else Q[:, n]
        nu = policy.trading_speed(t_n, S[:, n], remaining)
        nu = np.where(active, nu, 0.0)
        speed[:, n] = nu

        # Execution price: Ŝ = S + (Δ ± κν) / 2
        S_exec[:, n] = S[:, n] + (spread + sign * problem.kappa * nu) / 2.0

        # Cash update: dC = Ŝ · ν dt for acquisition (cash decreases with sign? — the
        # paper integrates +Ŝν dt for acquisition cost, +Ŝν dt for liquidation
        # revenue. We keep both as positive quantities and interpret as 'cost paid'
        # for acquisition and 'revenue received' for liquidation.
        cash[:, n + 1] = cash[:, n] + S_exec[:, n] * nu * dt

        # Inventory update
        if is_acquisition:
            Q[:, n + 1] = Q[:, n] + nu * dt
            Y[:, n + 1] = np.maximum(Y[:, n] - nu * dt, 0.0)
        else:
            Q[:, n + 1] = np.maximum(Q[:, n] - nu * dt, 0.0)
            Y[:, n + 1] = Q[:, n + 1]

        # Midprice update with diffusion + permanent impact drift
        dW = rng.standard_normal(n_paths) * sqrt_dt
        drift = sign * b_permanent * nu * dt
        S[:, n + 1] = S[:, n] + drift + sigma_total * dW
        # When inactive, freeze the state.
        S[~active, n + 1] = S[~active, n]
        Q[~active, n + 1] = Q[~active, n]
        Y[~active, n + 1] = Y[~active, n]
        cash[~active, n + 1] = cash[~active, n]

        # Check stopping conditions:
        if is_acquisition:
            hit_cap = S[:, n + 1] >= problem.S_max
            done = Y[:, n + 1] <= 1e-12
        else:
            hit_cap = S[:, n + 1] <= problem.S_min
            done = Q[:, n + 1] <= 1e-12

        newly_stopped = active & (hit_cap | done)
        if newly_stopped.any():
            terminated_at[newly_stopped] = n + 1
            active &= ~newly_stopped

        if not active.any():
            # Carry forward last values to the end so all arrays are well-defined.
            for k in range(n + 2, N + 1):
                S[:, k] = S[:, n + 1]
                Q[:, k] = Q[:, n + 1]
                Y[:, k] = Y[:, n + 1]
                cash[:, k] = cash[:, n + 1]
            break

    # Apply terminal penalty for any path that hasn't completed: the agent
    # acquires/liquidates the remainder at S_τ ± α·remaining (Eq. 22, 36).
    if is_acquisition:
        remaining_final = np.maximum(problem.N_units - Q[:, -1], 0.0)
        s_at_stop = _gather_at_stop(S, terminated_at, N)
        cash[:, -1] += remaining_final * (s_at_stop + problem.alpha * remaining_final)
        Q[:, -1] += remaining_final
        total_units = problem.N_units
    else:
        remaining_final = np.maximum(Q[:, -1], 0.0)
        s_at_stop = _gather_at_stop(S, terminated_at, N)
        cash[:, -1] += remaining_final * (s_at_stop - problem.alpha * remaining_final)
        Q[:, -1] -= remaining_final
        total_units = problem.N_units

    avg_traded_price = cash[:, -1] / total_units

    return SimulationResult(
        t_grid=t_grid,
        S=S,
        Q=Q,
        S_exec=S_exec,
        cash=cash,
        speed=speed,
        terminated_at=terminated_at,
        avg_traded_price=avg_traded_price,
    )


def _gather_at_stop(S: np.ndarray, terminated_at: np.ndarray, N: int) -> np.ndarray:
    """Return S at the step the path stopped (or final step if it never did)."""
    idx = np.clip(terminated_at, 0, N)
    return S[np.arange(S.shape[0]), idx]
