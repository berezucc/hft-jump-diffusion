"""Streamlit dashboard for exploring Semi-Markov & Hawkes jump-diffusion solutions.

Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hft_jd import ModelParams, ProblemParams, simulate, solve_acquisition, solve_liquidation
from hft_jd.coefficients import (
    HawkesParams,
    SemiMarkovParams,
    hawkes_coefficients,
    semi_markov_coefficients,
)


st.set_page_config(
    page_title="HFT Jump-Diffusion",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Semi-Markov & Hawkes Jump-Diffusion — HFT Acquisition / Liquidation")
st.caption(
    "Interactive companion to Lalor & Swishchuk (2025), arXiv:2409.12776. "
    "Solves the reduced HJB PDE in h(t, S) and simulates the optimal trading "
    "strategy under non-Markovian jump dynamics."
)


# ---------------------------------------------------------------------------
# Sidebar: problem & model parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Problem")
    kind = st.radio("Type", ["acquisition", "liquidation"], horizontal=True)

    T = st.number_input("Horizon T", 0.1, 10.0, 1.0, 0.1)
    N_units = st.number_input("Units to trade N", 1, 10_000, 390, 1)
    kappa = st.number_input("Temporary impact κ", 1e-6, 1e-1, 1e-4, format="%.6f")
    alpha = st.number_input("Terminal penalty α", 0.0, 1.0, 0.01, 0.001, format="%.4f")
    phi = st.number_input("Running penalty φ", 0.0, 1.0, 1e-5, 1e-5, format="%.6f")

    S0 = st.number_input("Initial midprice S₀", 0.0, 1e6, 30.97, 0.01)
    if kind == "acquisition":
        S_min = st.number_input("S_min (lower grid)", 0.0, 1e6, 29.0, 0.01)
        S_max = st.number_input("S_max (price cap)", 0.0, 1e6, 31.1, 0.01)
    else:
        S_min = st.number_input("S_min (price floor)", 0.0, 1e6, 30.8, 0.01)
        S_max = st.number_input("S_max (upper grid)", 0.0, 1e6, 33.0, 0.01)

    st.header("Model")
    process = st.selectbox(
        "Jump process",
        ["Manual (σ̄, ς)", "Semi-Markov calibration", "Hawkes calibration"],
    )
    sigma = st.number_input("Brownian σ", 0.0, 10.0, 0.1041, 0.001, format="%.4f")

    if process == "Manual (σ̄, ς)":
        sigma_bar = st.number_input("σ̄", 0.0, 5.0, 0.01598, 0.001, format="%.5f")
        varsigma = st.number_input("ς", 0.0, 5.0, 0.1323, 0.001, format="%.5f")
        eta_view = 0.0
        cal_info = None
    elif process == "Semi-Markov calibration":
        delta = st.number_input("Tick size δ", 1e-4, 1.0, 0.01, format="%.4f")
        p = st.slider("p_cont = P[Xₖ₊₁=δ | Xₖ=δ]", 0.01, 0.99, 0.5, 0.01)
        p_prime = st.slider(
            "p'_cont = P[Xₖ₊₁=−δ | Xₖ=−δ]", 0.01, 0.99, 0.4, 0.01
        )
        m_tau = st.number_input("m·τ", 1e-3, 1e6, 100.0, format="%.3f")
        sm = SemiMarkovParams(
            delta=delta,
            p_cont=p,
            p_cont_prime=p_prime,
            m_tau=m_tau,
            sigma_diffusion_sq=sigma**2,
        )
        coefs = semi_markov_coefficients(sm)
        sigma_bar = coefs.sigma_bar
        varsigma = coefs.varsigma
        eta_view = coefs.eta
        cal_info = {
            "η_SM": coefs.eta,
            "σ̄_SM": coefs.sigma_bar,
            "ς_SM": coefs.varsigma,
            "π*": coefs.pi_star,
            "(σ*)²": coefs.sigma_star_sq,
        }
    else:  # Hawkes
        delta = st.number_input("Tick size δ", 1e-4, 1.0, 0.01, format="%.4f")
        p = st.slider("p", 0.01, 0.99, 0.5, 0.01)
        p_prime = st.slider("p'", 0.01, 0.99, 0.5, 0.01)
        a_star = st.number_input("a*", -1.0, 1.0, 0.005, 0.0001, format="%.5f")
        lam = st.number_input("Background intensity λ", 1e-4, 1e4, 1.0, format="%.4f")
        mu_hat = st.slider("μ̂ (excitation)", 0.0, 0.99, 0.3, 0.01)
        sigma_hat_sq = st.number_input("σ̂²", 0.0, 1.0, 1e-3, format="%.5f")
        hp = HawkesParams(
            delta=delta,
            p=p,
            p_prime=p_prime,
            a_star=a_star,
            lam=lam,
            mu_hat=mu_hat,
            sigma_hat_sq=sigma_hat_sq,
        )
        coefs = hawkes_coefficients(hp)
        sigma_bar = coefs.sigma_bar
        varsigma = coefs.varsigma
        eta_view = coefs.eta
        cal_info = {
            "η_HP": coefs.eta,
            "σ̄_HP": coefs.sigma_bar,
            "ς_HP": coefs.varsigma,
            "π*": coefs.pi_star,
            "(σ*)²": coefs.sigma_star_sq,
        }

    st.header("Numerics")
    N_t = st.slider("Time grid steps N", 50, 1000, 200, 50)
    N_S = st.slider("Space grid steps M", 50, 1000, 300, 50)
    n_paths = st.slider("Monte-Carlo paths", 100, 10_000, 1_000, 100)
    seed = st.number_input("Seed", 0, 10**6, 0)


# Display calibration outputs when applicable
if cal_info is not None:
    cols = st.columns(len(cal_info))
    for col, (k, v) in zip(cols, cal_info.items()):
        col.metric(k, f"{v:.6g}")


# ---------------------------------------------------------------------------
# Solve & simulate (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=True)
def _solve(
    kind: str,
    T: float,
    N_units: int,
    kappa: float,
    alpha: float,
    phi: float,
    S0: float,
    S_min: float,
    S_max: float,
    sigma: float,
    sigma_bar: float,
    varsigma: float,
    N_t: int,
    N_S: int,
):
    problem = ProblemParams(
        kind=kind,
        T=T,
        N_units=int(N_units),
        kappa=kappa,
        alpha=alpha,
        phi=phi,
        S0=S0,
        S_min=S_min,
        S_max=S_max,
    )
    model = ModelParams(sigma=sigma, sigma_bar=sigma_bar, varsigma=varsigma)
    solver = solve_acquisition if kind == "acquisition" else solve_liquidation
    sol = solver(problem, model, N=N_t, M=N_S)
    return sol


@st.cache_data(show_spinner=True)
def _sim(_sol, n_paths: int, seed: int, sig_key: tuple):
    return simulate(_sol, n_paths=n_paths, seed=int(seed))


try:
    sol = _solve(
        kind, T, int(N_units), kappa, alpha, phi, S0, S_min, S_max,
        sigma, sigma_bar, varsigma, N_t, N_S,
    )
except ValueError as e:
    st.error(f"Invalid parameters: {e}")
    st.stop()


# Cache key needs to react to model/problem changes
sig_key = (
    kind, T, int(N_units), kappa, alpha, phi, S0, S_min, S_max,
    sigma, sigma_bar, varsigma, N_t, N_S, n_paths, seed,
)
sim_result = _sim(sol, n_paths, seed, sig_key)


# ---------------------------------------------------------------------------
# Top: h(t, S) heatmap
# ---------------------------------------------------------------------------

st.subheader("Optimal trading speed scaling, h(t, S)")
st.caption(
    "Reduced HJB solution. ν*(t,S,q) = q·h(t,S)/κ. Hotter colours ⇒ faster trading."
)

heatmap = go.Figure(
    data=go.Heatmap(
        z=sol.h,
        x=sol.S_grid,
        y=sol.t_grid,
        colorscale="Viridis",
        colorbar=dict(title="h"),
    )
)
heatmap.update_layout(
    height=420,
    margin=dict(l=40, r=20, t=10, b=40),
    xaxis_title="Asset price S",
    yaxis_title="Time t",
)
st.plotly_chart(heatmap, use_container_width=True)

# ---------------------------------------------------------------------------
# Middle row: sample paths + speed
# ---------------------------------------------------------------------------

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Sample midprice paths")
    n_show = min(8, n_paths)
    paths = go.Figure()
    for i in range(n_show):
        paths.add_trace(
            go.Scatter(
                x=sim_result.t_grid,
                y=sim_result.S[i],
                mode="lines",
                name=f"Path {i+1}",
                opacity=0.85,
            )
        )
    boundary = sol.problem.S_max if kind == "acquisition" else sol.problem.S_min
    label = "S_max" if kind == "acquisition" else "S_min"
    paths.add_hline(
        y=boundary,
        line_dash="dash",
        line_color="black",
        annotation_text=label,
        annotation_position="top right",
    )
    paths.update_layout(
        height=380,
        margin=dict(l=40, r=20, t=10, b=40),
        xaxis_title="t",
        yaxis_title="S",
    )
    st.plotly_chart(paths, use_container_width=True)

with col_b:
    st.subheader("Inventory evolution (median ± IQR)")
    qs = np.percentile(sim_result.Q, [25, 50, 75], axis=0)
    inv = go.Figure()
    inv.add_trace(
        go.Scatter(
            x=sim_result.t_grid,
            y=qs[1],
            mode="lines",
            name="Median",
            line=dict(color="#1f77b4"),
        )
    )
    inv.add_trace(
        go.Scatter(
            x=np.r_[sim_result.t_grid, sim_result.t_grid[::-1]],
            y=np.r_[qs[2], qs[0][::-1]],
            fill="toself",
            fillcolor="rgba(31,119,180,0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="IQR",
        )
    )
    inv.update_layout(
        height=380,
        margin=dict(l=40, r=20, t=10, b=40),
        xaxis_title="t",
        yaxis_title="Inventory Q",
    )
    st.plotly_chart(inv, use_container_width=True)


# ---------------------------------------------------------------------------
# Bottom row: avg traded price histogram + summary stats
# ---------------------------------------------------------------------------

col_c, col_d = st.columns([2, 1])

with col_c:
    st.subheader("Distribution of average traded price")
    hist = go.Figure(
        data=go.Histogram(
            x=sim_result.avg_traded_price,
            nbinsx=60,
            marker_color="#2ca02c",
        )
    )
    hist.add_vline(
        x=float(np.mean(sim_result.avg_traded_price)),
        line_dash="dash",
        line_color="black",
        annotation_text="mean",
    )
    hist.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=10, b=40),
        xaxis_title="Average price",
        yaxis_title="Frequency",
    )
    st.plotly_chart(hist, use_container_width=True)

with col_d:
    st.subheader("Summary")
    df = pd.DataFrame(
        {
            "Metric": [
                "Mean avg price",
                "Std avg price",
                "Median avg price",
                "% paths hit boundary",
                "% paths reached terminal time",
                "Mean terminal cash",
                "Total variance σ²+σ̄²+ς²",
            ],
            "Value": [
                f"{np.mean(sim_result.avg_traded_price):.4f}",
                f"{np.std(sim_result.avg_traded_price):.4f}",
                f"{np.median(sim_result.avg_traded_price):.4f}",
                f"{100 * np.mean(sim_result.terminated_at < len(sim_result.t_grid)):.1f}%",
                f"{100 * np.mean(sim_result.terminated_at >= len(sim_result.t_grid)):.1f}%",
                f"{np.mean(sim_result.cash[:, -1]):.2f}",
                f"{sol.model.total_variance:.4g}",
            ],
        }
    )
    st.table(df)


with st.expander("Math reference"):
    st.markdown(
        r"""
**Reduced PDE** (after the ansatz $H(t,S,q) = qS + q^2 h(t,S)$, $b=0$):

$$
\partial_t h + \tfrac{1}{2}(\sigma^2 + \bar\sigma^2 + \varsigma^2)\,\partial_{SS}h
- \tfrac{1}{\kappa}h^2 \pm \phi = 0,
$$

with $+\phi$ for acquisition, $-\phi$ for liquidation. Terminal condition $h(T,S)=\alpha$.

**Optimal control** (Eq. 32 / Eq. 46):
$\nu^\*(t,S,q) = q \cdot h(t,S) / \kappa.$

**Semi-Markov coefficients** (Eq. 4–6):
$\eta_{SM} = \frac{\delta(2\pi^\*-1)}{m\tau},\quad
 \bar\sigma_{SM}^2 = \tfrac{(\sigma^\*)^2}{m\tau} + \tfrac{\Pi\sigma^2}{m\tau},\quad
 \varsigma_{SM}^2 = \tfrac{(\sigma^\*)^2}{m\tau}$.

**Hawkes coefficients** (Eq. 10–12):
$\eta_{HP} = \tfrac{a^\*\lambda}{1-\hat\mu},\quad
 \bar\sigma_{HP}^2 = (\sigma^\*)^2 + a^\*\sqrt{\tfrac{\lambda}{1-\hat\mu}},\quad
 \varsigma_{HP}^2 = \hat\sigma^2 \tfrac{\lambda}{1-\hat\mu}$.
        """
    )
