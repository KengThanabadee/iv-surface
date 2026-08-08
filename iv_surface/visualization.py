import numpy as np
import pandas as pd
import plotly.graph_objects as go


_LONG_FRAME_COLUMNS = {"tau", "strike", "iv", "moneyness"}
_ATM_FRAME_COLUMNS = {
    "tau",
    "strike",
    "iv",
    "moneyness",
    "atm_distance",
    "selection_status",
}
_ATM_TOLERANCE = 1e-12


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")

    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def surface_to_long_frame(
    iv_surface, expiries, strikes, spot_price
) -> pd.DataFrame:
    """Convert an expiry-by-strike IV grid to chart-ready long-form data."""
    try:
        surface = np.asarray(iv_surface, dtype=float)
        expiry_values = np.asarray(list(expiries), dtype=float)
        strike_values = np.asarray(list(strikes), dtype=float)
        spot_value = float(spot_price)
    except (TypeError, ValueError) as exc:
        raise ValueError("surface inputs must be numeric") from exc

    expected_shape = (len(expiry_values), len(strike_values))
    if surface.ndim != 2 or surface.shape != expected_shape:
        raise ValueError(
            "iv_surface shape must equal (len(expiries), len(strikes))"
        )
    if not np.isfinite(spot_value) or spot_value <= 0:
        raise ValueError("spot_price must be finite and positive")

    tau = np.repeat(expiry_values, len(strike_values))
    strike = np.tile(strike_values, len(expiry_values))

    return pd.DataFrame(
        {
            "tau": tau,
            "strike": strike,
            "iv": surface.reshape(-1),
            "spot_price": spot_value,
            "moneyness": strike / spot_value,
        }
    )


def atm_term_structure_frame(
    long_frame: pd.DataFrame, max_moneyness_distance: float = 0.05
) -> pd.DataFrame:
    """Select a bounded, transparent ATM proxy independently for each expiry."""
    _require_columns(long_frame, _LONG_FRAME_COLUMNS, "long_frame")

    try:
        distance_bound = float(max_moneyness_distance)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "max_moneyness_distance must be finite and non-negative"
        ) from exc
    if not np.isfinite(distance_bound) or distance_bound < 0:
        raise ValueError("max_moneyness_distance must be finite and non-negative")

    rows = []
    for tau, expiry_frame in long_frame.groupby("tau", sort=True, dropna=False):
        iv = expiry_frame["iv"].to_numpy(dtype=float)
        moneyness = expiry_frame["moneyness"].to_numpy(dtype=float)
        finite = np.isfinite(iv) & np.isfinite(moneyness)

        if not finite.any():
            rows.append(
                {
                    "tau": tau,
                    "strike": np.nan,
                    "iv": np.nan,
                    "moneyness": np.nan,
                    "atm_distance": np.nan,
                    "selection_status": "no_finite_iv",
                }
            )
            continue

        candidates = expiry_frame.loc[finite].copy()
        candidates["atm_distance"] = np.abs(candidates["moneyness"] - 1.0)
        minimum_distance = float(candidates["atm_distance"].min())
        within_bound = minimum_distance < distance_bound or np.isclose(
            minimum_distance,
            distance_bound,
            atol=_ATM_TOLERANCE,
            rtol=0,
        )

        if not within_bound:
            rows.append(
                {
                    "tau": tau,
                    "strike": np.nan,
                    "iv": np.nan,
                    "moneyness": np.nan,
                    "atm_distance": minimum_distance,
                    "selection_status": "outside_bound",
                }
            )
            continue

        tied = candidates[
            np.isclose(
                candidates["atm_distance"],
                minimum_distance,
                atol=_ATM_TOLERANCE,
                rtol=0,
            )
        ]
        rows.append(
            {
                "tau": tau,
                "strike": float(tied["strike"].mean()),
                "iv": float(tied["iv"].mean()),
                "moneyness": float(tied["moneyness"].mean()),
                "atm_distance": minimum_distance,
                "selection_status": (
                    "symmetric_average" if len(tied) > 1 else "nearest"
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "tau",
            "strike",
            "iv",
            "moneyness",
            "atm_distance",
            "selection_status",
        ],
    )


def make_iv_heatmap(long_frame: pd.DataFrame) -> go.Figure:
    """Create a moneyness-by-expiry heatmap without filling missing IV cells."""
    _require_columns(long_frame, _LONG_FRAME_COLUMNS, "long_frame")

    iv_grid = (
        long_frame.pivot(index="tau", columns="moneyness", values="iv")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    strike_grid = (
        long_frame.pivot(index="tau", columns="moneyness", values="strike")
        .reindex(index=iv_grid.index, columns=iv_grid.columns)
    )

    trace = go.Heatmap(
        x=iv_grid.columns.to_numpy(dtype=float),
        y=iv_grid.index.to_numpy(dtype=float),
        z=iv_grid.to_numpy(dtype=float),
        customdata=strike_grid.to_numpy(dtype=float),
        connectgaps=False,
        colorbar={"title": "implied volatility", "tickformat": ".0%"},
        hovertemplate=(
            "K / S: %{x:.4f}<br>"
            "tau: %{y:.4f} years<br>"
            "strike: %{customdata:g}<br>"
            "IV: %{z:.2%}<extra></extra>"
        ),
    )
    figure = go.Figure(data=[trace])
    figure.update_layout(xaxis_title="K / S", yaxis_title="tau (years)")
    return figure


def make_smile_figure(
    long_frame: pd.DataFrame, selected_expiries=None
) -> go.Figure:
    """Create one IV smile trace per selected expiry."""
    _require_columns(long_frame, _LONG_FRAME_COLUMNS, "long_frame")

    selected = None if selected_expiries is None else set(selected_expiries)
    figure = go.Figure()
    for tau in sorted(long_frame["tau"].dropna().unique()):
        if selected is not None and tau not in selected:
            continue
        smile = long_frame[long_frame["tau"] == tau].sort_values("moneyness")
        figure.add_trace(
            go.Scatter(
                x=smile["moneyness"].to_numpy(dtype=float),
                y=smile["iv"].to_numpy(dtype=float),
                customdata=smile["strike"].to_numpy(dtype=float),
                mode="lines+markers",
                name=f"tau={tau:g}",
                connectgaps=False,
                hovertemplate=(
                    "K / S: %{x:.4f}<br>"
                    "strike: %{customdata:g}<br>"
                    "IV: %{y:.2%}<extra>%{fullData.name}</extra>"
                ),
            )
        )

    figure.add_vline(x=1.0, line_dash="dash", line_color="gray")
    figure.update_layout(xaxis_title="K / S", yaxis_title="implied volatility")
    figure.update_yaxes(tickformat=".0%")
    return figure


def make_atm_term_structure_figure(atm_frame: pd.DataFrame) -> go.Figure:
    """Create an ATM-proxy term structure while preserving rejected expiries."""
    _require_columns(atm_frame, _ATM_FRAME_COLUMNS, "atm_frame")

    data = atm_frame.sort_values("tau")
    customdata = data[
        ["strike", "moneyness", "atm_distance", "selection_status"]
    ].to_numpy()
    trace = go.Scatter(
        x=data["tau"].to_numpy(dtype=float),
        y=data["iv"].to_numpy(dtype=float),
        customdata=customdata,
        mode="lines+markers",
        connectgaps=False,
        hovertemplate=(
            "tau: %{x:.4f} years<br>"
            "IV: %{y:.2%}<br>"
            "strike: %{customdata[0]}<br>"
            "K / S: %{customdata[1]}<br>"
            "ATM distance: %{customdata[2]}<br>"
            "selection: %{customdata[3]}<extra></extra>"
        ),
    )
    figure = go.Figure(data=[trace])
    figure.update_layout(
        xaxis_title="tau (years)", yaxis_title="implied volatility"
    )
    figure.update_yaxes(tickformat=".0%")
    return figure
