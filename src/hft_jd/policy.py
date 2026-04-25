"""Optimal feedback control built from a precomputed h(t, S) grid.

For both problems the optimal trading speed reduces to (Eq. 32, Eq. 46):

    ν*(t, S, q) = q · h(t, S) / κ

with q = remaining units to acquire (acquisition) or remaining units to
liquidate (liquidation). The PDE solver returns h on a (t, S) grid; this
module wraps it with bilinear interpolation so the simulator can evaluate
ν* at any (t, S).
"""

from __future__ import annotations

import numpy as np

from hft_jd.pde import HSolution


class FeedbackPolicy:
    """Vectorised feedback control ν* from an HSolution."""

    def __init__(self, solution: HSolution) -> None:
        self.solution = solution
        self.kappa = solution.problem.kappa
        self._t_grid = solution.t_grid
        self._S_grid = solution.S_grid
        self._h = solution.h
        self._N = len(self._t_grid) - 1
        self._M = len(self._S_grid) - 1
        self._t0 = self._t_grid[0]
        self._tN = self._t_grid[-1]
        self._S0 = self._S_grid[0]
        self._SM = self._S_grid[-1]

    def h_value(self, t: float | np.ndarray, S: float | np.ndarray) -> np.ndarray:
        """Bilinear interp of h on the (t, S) grid; broadcasts over arrays."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        S_arr = np.atleast_1d(np.asarray(S, dtype=float))
        t_clip = np.clip(t_arr, self._t0, self._tN)
        S_clip = np.clip(S_arr, self._S0, self._SM)

        t_pos = (t_clip - self._t0) / (self._tN - self._t0) * self._N
        S_pos = (S_clip - self._S0) / (self._SM - self._S0) * self._M

        n_idx = np.clip(np.floor(t_pos).astype(int), 0, self._N - 1)
        i_idx = np.clip(np.floor(S_pos).astype(int), 0, self._M - 1)
        n_w = t_pos - n_idx
        i_w = S_pos - i_idx

        h = self._h
        v00 = h[n_idx, i_idx]
        v01 = h[n_idx, i_idx + 1]
        v10 = h[n_idx + 1, i_idx]
        v11 = h[n_idx + 1, i_idx + 1]
        result = (1 - n_w) * ((1 - i_w) * v00 + i_w * v01) + n_w * (
            (1 - i_w) * v10 + i_w * v11
        )
        return result

    def trading_speed(
        self, t: float | np.ndarray, S: float | np.ndarray, q: float | np.ndarray
    ) -> np.ndarray:
        """ν*(t, S, q) = q · h(t, S) / κ, clipped at 0 (no negative speeds)."""
        h_val = self.h_value(t, S)
        speed = np.maximum(q * h_val / self.kappa, 0.0)
        return speed
