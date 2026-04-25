# HFT Jump-Diffusion

Reference Python implementation of:

> **Lalor & Swishchuk (2025).** *Algorithmic and High-Frequency Trading Problems for Semi-Markov and Hawkes Jump-Diffusion Models.* arXiv:2409.12776v2.

The paper extends the optimal acquisition/liquidation problems of Cartea, Jaimungal & Penalva (2015) with non-Markovian jump dynamics - modelled via Semi-Markov or Hawkes processes - and solved through a diffusion approximation. This package gives you the math in clean, modular, tested code, plus a Streamlit dashboard for exploring the solutions.

## What's in the box

- `hft_jd.coefficients` - closed-form coefficients (η, σ̄, ς) for both Semi-Markov and Hawkes cases.
- `hft_jd.pde` - IMEX finite-difference solver for the reduced PDE in `h(t, S)` for both acquisition (price cap) and liquidation (price floor) problems.
- `hft_jd.simulation` - Monte-Carlo strategy simulation: price paths, inventory, cash, execution prices.
- `hft_jd.policy` - Optimal feedback control `ν*(t, S, q) = q · h(t, S) / κ`.
- `dashboard/app.py` - Streamlit UI.
- `tests/` - pytest suite covering coefficients, PDE invariants, and simulation properties.

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Run

```bash
pytest                                # tests
streamlit run dashboard/app.py        # interactive UI
```

## Math, briefly

The midprice dynamics under both jump regimes reduce to

$$ dS_t = \pm g(\nu_t)\,\eta\,dt + \sqrt{\sigma^2 + \bar\sigma^2 + \varsigma^2}\,dW_t $$

so the Hamilton–Jacobi–Bellman PDE collapses (via the ansatz `H(t,S,q) = qS + q²h(t,S)`) to

$$ \partial_t h + \tfrac{1}{2}(\sigma^2+\bar\sigma^2+\varsigma^2)\partial_{SS} h - \tfrac{1}{\kappa}h^2 \pm \phi = 0 $$

with `+φ` for acquisition and `−φ` for liquidation, and terminal/boundary value `α`. The optimal trading speed in feedback form is `ν* = q · h(t,S) / κ`.

See the docstrings in `hft_jd/pde.py` for the IMEX scheme.
