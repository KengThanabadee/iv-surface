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
