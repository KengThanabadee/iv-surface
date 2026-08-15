# implied-vol-surface

`implied-vol-surface` is being developed into a Deribit-backed crypto-options
analytics workspace. The current `main` branch contains the reusable,
Streamlit-independent visualization layer while the new market-data path is
built in reviewable components.

The earlier Bybit price-to-IV learning implementation—including its
Black-Scholes solver, collector, tests, manual check script, and original
documentation—is preserved in the Git tag `bybit-solved-iv-study`.

## Current functionality

`iv_surface.visualization` currently provides:

- conversion of an expiry-by-strike IV grid into chart-ready long-form data;
- optional aligned UTC expiry display metadata;
- a bounded and inspectable ATM proxy;
- Plotly IV heatmap, smile, and ATM term-structure figures;
- explicit `NaN` gaps without implicit interpolation.

The Deribit adapter, normalized option-chain snapshot, Call/Put composite IV,
and Streamlit workspace are not implemented yet.

## Installation

Create or activate the project environment, then install the package with the
dashboard and development dependencies:

```bash
.venv/bin/python -m pip install -e '.[dashboard,dev]'
```

Plotly remains optional for consumers that do not import the visualization
module.

## Tests

```bash
.venv/bin/python -m pytest
```

## Archived Bybit study

To inspect the complete earlier implementation without changing the current
branch:

```bash
git show bybit-solved-iv-study:README.md
git show bybit-solved-iv-study:iv_surface/solver.py
```

If that implementation is needed again, create a separate branch from the tag
instead of mixing its market-data assumptions into the Deribit path:

```bash
git switch -c revive-bybit-study bybit-solved-iv-study
```
