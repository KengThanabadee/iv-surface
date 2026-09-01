from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from iv_surface.solver import build_surface


_REQUIRED_COLUMNS = {
    "flag",
    "tau",
    "strike",
    "mid_price",
    "quote_source",
    "index_price",
}
_USABLE_CHAIN_NUMERIC_COLUMNS = ["tau", "strike", "mid_price", "index_price"]
_VALID_FLAGS = {"call", "put"}


@dataclass(frozen=True)
class SurfaceInputs:
    option_price_grid: np.ndarray
    spot_price: float
    expiries: list[float]
    strikes: list[float]


@dataclass(frozen=True)
class SurfaceResult:
    iv_surface: np.ndarray
    option_price_grid: np.ndarray
    spot_price: float
    expiries: list[float]
    strikes: list[float]


@dataclass(frozen=True)
class CombinedSurfaceResult:
    iv_surface: np.ndarray
    source_grid: np.ndarray
    call_iv_surface: np.ndarray
    put_iv_surface: np.ndarray
    call_option_price_grid: np.ndarray
    put_option_price_grid: np.ndarray
    spot_price: float
    expiries: list[float]
    strikes: list[float]


def _validate_chain(chain: pd.DataFrame, flag: str) -> None:
    if not isinstance(chain, pd.DataFrame):
        raise TypeError("chain must be a pandas DataFrame")
    if flag not in _VALID_FLAGS:
        raise ValueError("flag must be 'call' or 'put'")

    missing = sorted(_REQUIRED_COLUMNS - set(chain.columns))
    if missing:
        raise ValueError(f"chain is missing required columns: {missing}")


def _filter_usable_chain_rows(chain: pd.DataFrame, flag: str) -> pd.DataFrame:
    data = chain.copy()
    for column in _USABLE_CHAIN_NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data[
        (data["flag"] == flag)
        & (data["quote_source"] == "mid")
        & np.isfinite(data["mid_price"])
        & np.isfinite(data["tau"])
        & np.isfinite(data["strike"])
        & np.isfinite(data["index_price"])
        & (data["mid_price"] > 0)
        & (data["tau"] > 0)
        & (data["strike"] > 0)
        & (data["index_price"] > 0)
    ].copy()


def prepare_surface_inputs(chain: pd.DataFrame, flag: str = "call") -> SurfaceInputs:
    _validate_chain(chain, flag)

    usable = _filter_usable_chain_rows(chain, flag)

    if usable.empty:
        raise ValueError(f"chain has no usable {flag} rows after surface-input filters")

    duplicates = usable.duplicated(subset=["tau", "strike"], keep=False)
    if duplicates.any():
        raise ValueError("chain contains duplicate rows for the same tau and strike")

    expiries = sorted(usable["tau"].unique().tolist())
    strikes = sorted(usable["strike"].unique().tolist())

    price_grid = usable.pivot(index="tau", columns="strike", values="mid_price")
    option_price_grid = price_grid.reindex(index=expiries, columns=strikes).to_numpy(
        dtype=float
    )
    spot_price = float(usable["index_price"].median())
    if usable["index_price"].nunique() > 1:
        warnings.warn(
            "usable rows have different index_price values; using median "
            f"spot_price={spot_price}",
            UserWarning,
            stacklevel=2,
        )

    return SurfaceInputs(
        option_price_grid=option_price_grid,
        spot_price=spot_price,
        expiries=expiries,
        strikes=strikes,
    )


def build_surface_from_chain(
    chain: pd.DataFrame, flag: str = "call", r: float = 0
) -> SurfaceResult:
    inputs = prepare_surface_inputs(chain, flag=flag)
    iv_surface = build_surface(
        inputs.option_price_grid,
        inputs.spot_price,
        inputs.expiries,
        inputs.strikes,
        r,
        flag,
    )

    return SurfaceResult(
        iv_surface=iv_surface,
        option_price_grid=inputs.option_price_grid,
        spot_price=inputs.spot_price,
        expiries=inputs.expiries,
        strikes=inputs.strikes,
    )


def _combined_price_grid(
    usable: pd.DataFrame, expiries: list[float], strikes: list[float]
) -> np.ndarray:
    grid = np.full((len(expiries), len(strikes)), np.nan)
    expiry_positions = {tau: i for i, tau in enumerate(expiries)}
    strike_positions = {strike: j for j, strike in enumerate(strikes)}
    for row in usable.itertuples(index=False):
        grid[expiry_positions[row.tau], strike_positions[row.strike]] = row.mid_price
    return grid


def build_combined_surface_from_chain(
    chain: pd.DataFrame, r: float = 0
) -> CombinedSurfaceResult:
    """Build a spot-based Call/Put IV surface with auditable cell provenance."""
    _validate_chain(chain, "call")

    data = chain.copy()
    for column in _USABLE_CHAIN_NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    identity_rows = data[
        data["flag"].isin(_VALID_FLAGS)
        & np.isfinite(data["tau"])
        & np.isfinite(data["strike"])
        & (data["tau"] > 0)
        & (data["strike"] > 0)
    ]
    if identity_rows.empty:
        raise ValueError("chain has no valid Call or Put expiry/strike rows")

    expiries = sorted(identity_rows["tau"].unique().tolist())
    strikes = sorted(identity_rows["strike"].unique().tolist())
    call_rows = _filter_usable_chain_rows(data, "call")
    put_rows = _filter_usable_chain_rows(data, "put")
    usable = pd.concat([call_rows, put_rows], ignore_index=True)
    if usable.empty:
        raise ValueError("chain has no usable Call or Put rows after surface-input filters")

    duplicates = usable.duplicated(subset=["flag", "tau", "strike"], keep=False)
    if duplicates.any():
        raise ValueError("chain contains duplicate rows for the same flag, tau, and strike")

    spot_price = float(usable["index_price"].median())
    if usable["index_price"].nunique() > 1:
        warnings.warn(
            "usable rows have different index_price values; using shared median "
            f"spot_price={spot_price}",
            UserWarning,
            stacklevel=2,
        )

    call_prices = _combined_price_grid(call_rows, expiries, strikes)
    put_prices = _combined_price_grid(put_rows, expiries, strikes)
    call_iv = build_surface(call_prices, spot_price, expiries, strikes, r, "call")
    put_iv = build_surface(put_prices, spot_price, expiries, strikes, r, "put")

    combined_iv = np.full(call_iv.shape, np.nan)
    source_grid = np.full(call_iv.shape, "missing", dtype=object)
    for i in range(len(expiries)):
        for j, strike in enumerate(strikes):
            call_value = call_iv[i, j]
            put_value = put_iv[i, j]
            call_finite = np.isfinite(call_value)
            put_finite = np.isfinite(put_value)
            moneyness = strike / spot_price

            if 0.98 <= moneyness <= 1.02:
                if call_finite and put_finite:
                    combined_iv[i, j] = (call_value + put_value) / 2
                    source_grid[i, j] = "near_atm_average"
                elif call_finite:
                    combined_iv[i, j] = call_value
                    source_grid[i, j] = "near_atm_call_only"
                elif put_finite:
                    combined_iv[i, j] = put_value
                    source_grid[i, j] = "near_atm_put_only"
            elif moneyness < 0.98:
                if put_finite:
                    combined_iv[i, j] = put_value
                    source_grid[i, j] = "put"
                elif call_finite:
                    combined_iv[i, j] = call_value
                    source_grid[i, j] = "call_fallback"
            else:
                if call_finite:
                    combined_iv[i, j] = call_value
                    source_grid[i, j] = "call"
                elif put_finite:
                    combined_iv[i, j] = put_value
                    source_grid[i, j] = "put_fallback"

    return CombinedSurfaceResult(
        iv_surface=combined_iv,
        source_grid=source_grid,
        call_iv_surface=call_iv,
        put_iv_surface=put_iv,
        call_option_price_grid=call_prices,
        put_option_price_grid=put_prices,
        spot_price=spot_price,
        expiries=expiries,
        strikes=strikes,
    )
