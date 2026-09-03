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


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")

    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _format_expiry_label(tau, expiry_datetime=pd.NaT) -> str:
    if pd.isna(expiry_datetime):
        return f"tau={tau:g}"

    timestamp = pd.Timestamp(expiry_datetime)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%d")


def _expiry_labels(frame: pd.DataFrame, taus) -> list[str]:
    labels = []
    has_metadata = "expiry_datetime" in frame.columns
    for tau in taus:
        expiry_datetime = pd.NaT
        if has_metadata:
            values = frame.loc[
                frame["tau"] == tau, "expiry_datetime"
            ].dropna()
            if not values.empty:
                expiry_datetime = values.iloc[0]
        labels.append(_format_expiry_label(tau, expiry_datetime))
    return labels


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
    figure.update_yaxes(
        tickmode="array",
        tickvals=iv_grid.index.to_numpy(dtype=float),
        ticktext=_expiry_labels(long_frame, iv_grid.index),
    )
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
        expiry_label = _expiry_labels(long_frame, [tau])[0]
        figure.add_trace(
            go.Scatter(
                x=smile["moneyness"].to_numpy(dtype=float),
                y=smile["iv"].to_numpy(dtype=float),
                customdata=smile["strike"].to_numpy(dtype=float),
                mode="lines+markers",
                name=expiry_label,
                meta=tau,
                connectgaps=False,
                hovertemplate=(
                    "tau: %{meta:.4f} years<br>"
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
    figure.update_xaxes(
        tickmode="array",
        tickvals=data["tau"].to_numpy(dtype=float),
        ticktext=_expiry_labels(data, data["tau"]),
    )
    figure.update_yaxes(tickformat=".0%")
    return figure
