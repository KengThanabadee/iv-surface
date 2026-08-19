# implied-vol-surface

A small personal quant library for learning and building implied-volatility tooling from first principles.

`implied-vol-surface` provides Black-Scholes pricing helpers, implied-volatility root finding, simple surface construction, bilinear interpolation, and basic Bybit option-chain fetching.

The implied-volatility root-finding logic is implemented without `scipy.optimize`.

## Status

This is an early local package in a personal quant-library ecosystem. It is useful for experiments and for feeding downstream projects, but it is not production trading infrastructure and is not financial advice.

## What It Does

- Prices European calls and puts with Black-Scholes.
- Solves implied volatility with Newton-Raphson and bisection fallback.
- Builds a 2D IV surface from a grid of option prices.
- Interpolates IV inside an expiry/strike grid with bilinear interpolation.
- Fetches and parses Bybit option-chain ticker data into a DataFrame, keeping market quotes separate from exchange mark values.
- Converts usable option-chain mid prices into the expiry/strike grid used by the surface builder.

## Core Data Shape

IV surface arrays use expiry rows and strike columns:

```text
iv_surface.shape == (len(expiries), len(strikes))
iv_surface[i, j] == IV at expiries[i], strikes[j]
```

Example layout:

```text
              K=90   K=100   K=110
T=0.10        0.18    0.20    0.22
T=0.25        0.19    0.21    0.23
T=0.50        0.20    0.22    0.24
```

Accordingly, the surface APIs take `expiries` before `strikes`, and interpolation uses `target_T` and `target_K`.

## Installation

Install locally in editable mode:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Example

```python
import numpy as np

from iv_surface.solver import bs_price, build_surface, interpolate_iv

spot_price = 100
r = 0.03
sigma = 0.20

expiries = [0.10, 0.25, 0.50]
strikes = [90, 100, 110]

option_price_grid = np.array([
    [bs_price(spot_price, K, T, r, sigma, "call") for K in strikes]
    for T in expiries
])

iv_surface = build_surface(
    option_price_grid, spot_price, expiries, strikes, r, flag="call"
)
iv = interpolate_iv(iv_surface, expiries, strikes, target_T=0.18, target_K=97)

print(iv_surface)
print(iv)
```

## API

### `bs_price(spot_price, K, T, r, sigma, flag, t=0)`

Returns the Black-Scholes price for a European call or put.

### `solve_iv(option_price, spot_price, K, T, r, flag, sigma_low=1e-6, sigma_high=10.0)`

Returns implied volatility for a single option price. Raises `ValueError` for invalid inputs, expired options, or prices outside the bracket implied by the sigma bounds.

### `build_surface(option_price_grid, spot_price, expiries, strikes, r=0, flag="call")`

Returns a 2D NumPy array of implied volatilities with shape `(n_expiries, n_strikes)`. Cells that cannot be solved are filled with `NaN`.

### `interpolate_iv(iv_surface, expiries, strikes, target_T, target_K)`

Returns bilinearly interpolated IV at the target `(T, K)`. Raises `ValueError` if the target point is outside the grid or if the grid is not valid.

### `fetch_chain(underlying="BTC")`

Fetches Bybit option tickers for an underlying and returns a tidy pandas DataFrame with parsed symbol fields, bid/ask prices, bid/ask IVs, mid price, mark price, mark IV, index price, and time to expiry.

`mid_price` is computed only from valid positive bid/ask quotes. If the bid/ask quote is not usable, `mid_price` is `NaN` and `quote_source` is `"none"`; the fetcher does not silently fall back to `mark_price`.

### `prepare_surface_inputs(chain, flag="call")`

Filters a fetched option-chain DataFrame to usable bid/ask mid prices and returns `option_price_grid`, `spot_price`, `expiries`, and `strikes` for `build_surface`.

`spot_price` is the median `index_price` across usable rows. If usable rows have different `index_price` values, a warning is emitted.

### `build_surface_from_chain(chain, flag="call", r=0)`

Convenience wrapper for:

```text
fetch_chain() -> prepare_surface_inputs() -> build_surface()
```

Returns the IV surface together with the option price grid, spot price, expiries, and strikes. Missing usable quotes remain `NaN` in the price grid and IV surface.

## Assumptions And Limitations

- Uses Black-Scholes European option assumptions.
- Time to expiry is measured in years.
- No dividend yield parameter yet.
- IV solving requires `T > 0`; implied volatility is undefined at expiry.
- `build_surface` fills failed IV solves with `NaN`.
- Market IV surfaces should be built from usable bid/ask mid prices; `mark_price` and `mark_iv` are kept for comparison.
- Interpolation requires at least two strictly increasing expiries and strikes.
- Interpolation is in-grid only; no extrapolation.
- The surface is a raw interpolated grid, not an arbitrage-free fitted surface.

## Tests

```bash
pytest
```

## Manual Live Check

Unit tests use mocked data and do not call Bybit. To sanity-check the live option-chain pipeline manually:

```bash
python scripts/check_bybit_chain.py --underlying BTC
```

This fetches the live chain from Bybit's `api.bytick.com` endpoint by default, builds call and put surfaces from usable bid/ask mid prices, and prints quote counts, grid shape, and `NaN` ratios.

The live check uses `--r 0.03` by default as a diagnostic risk-free-rate assumption. Override it explicitly when comparing model assumptions:

```bash
python scripts/check_bybit_chain.py --underlying BTC --r 0
python scripts/check_bybit_chain.py --underlying BTC --r 0.05
```

To choose an explicit Bybit REST host or timeout:

```bash
python scripts/check_bybit_chain.py --underlying BTC --base-url https://api.bybit.com
python scripts/check_bybit_chain.py --underlying BTC --base-url https://api.bytick.com --timeout 20
```

`api.bytick.com` is the default because it works in the current development environment. The script does not silently fall back between hosts; use `--base-url` to make the endpoint explicit.
