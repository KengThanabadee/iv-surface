from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from iv_surface.visualization import (
    surface_to_long_frame,
    atm_term_structure_frame,
    make_iv_heatmap,
    make_smile_figure,
    make_atm_term_structure_figure,
)


def _long_frame():
    return surface_to_long_frame(
        np.array([[0.20, np.nan], [0.25, 0.30]]),
        expiries=[0.5, 0.25],
        strikes=[110, 90],
        spot_price=100,
    )


def test_surface_to_long_frame_preserves_shape_order_and_nan():
    frame = _long_frame()

    assert len(frame) == 4
    assert {"tau", "strike", "iv", "spot_price", "moneyness"} <= set(
        frame.columns
    )
    assert frame[["tau", "strike"]].values.tolist() == [
        [0.5, 110.0],
        [0.5, 90.0],
        [0.25, 110.0],
        [0.25, 90.0],
    ]
    assert frame["moneyness"].tolist() == [1.1, 0.9, 1.1, 0.9]
    assert np.isnan(frame.loc[1, "iv"])


def test_surface_to_long_frame_rejects_bad_shape_and_spot_price():
    with pytest.raises(ValueError, match="shape"):
        surface_to_long_frame(np.ones((1, 2)), [0.25, 0.5], [90, 100], 100)

    for spot_price in [0, -1, np.nan, np.inf]:
        with pytest.raises(ValueError, match="spot_price"):
            surface_to_long_frame(np.ones((1, 1)), [0.25], [100], spot_price)


def test_surface_to_long_frame_aligns_and_normalizes_expiry_datetimes():
    surface = np.array([[0.20], [0.25], [np.nan]])
    expiries = [0.5, 0.25, 0.125]
    without_metadata = surface_to_long_frame(surface, expiries, [100], 100)

    with_metadata = surface_to_long_frame(
        surface,
        expiries,
        [100],
        100,
        expiry_datetimes=[
            "2026-06-30T20:00:00-04:00",
            datetime(2026, 7, 2, 8),
            None,
        ],
    )

    assert str(with_metadata["expiry_datetime"].dtype) == "datetime64[ns, UTC]"
    assert with_metadata["expiry_datetime"].tolist() == [
        pd.Timestamp("2026-07-01T00:00:00Z"),
        pd.Timestamp("2026-07-02T08:00:00Z"),
        pd.NaT,
    ]
    assert "expiry_datetime" not in without_metadata.columns
    pd.testing.assert_frame_equal(
        with_metadata.drop(columns="expiry_datetime"),
        without_metadata,
    )


def test_surface_to_long_frame_rejects_invalid_expiry_datetimes():
    with pytest.raises(ValueError, match=r"expiry_datetimes.*len\(expiries\)"):
        surface_to_long_frame(
            np.ones((2, 1)),
            [0.25, 0.5],
            [100],
            100,
            expiry_datetimes=["2026-07-01"],
        )

    with pytest.raises(ValueError, match="expiry_datetimes.*valid"):
        surface_to_long_frame(
            np.ones((1, 1)),
            [0.25],
            [100],
            100,
            expiry_datetimes=["not-a-date"],
        )


def test_surface_to_long_frame_accepts_mixed_valid_datetime_formats():
    frame = surface_to_long_frame(
        np.ones((2, 1)),
        [0.25, 0.5],
        [100],
        100,
        expiry_datetimes=["2026-07-01", "2026-07-02T08:00:00Z"],
    )

    assert frame["expiry_datetime"].tolist() == [
        pd.Timestamp("2026-07-01T00:00:00Z"),
        pd.Timestamp("2026-07-02T08:00:00Z"),
    ]


def test_atm_term_structure_selects_nearest_and_accepts_bound():
    frame = pd.DataFrame(
        {
            "tau": [0.5, 0.25, 0.25],
            "strike": [105, 94, 102],
            "iv": [0.30, 0.24, 0.20],
            "moneyness": [1.05, 0.94, 1.02],
        }
    )

    result = atm_term_structure_frame(frame)

    assert result["tau"].tolist() == [0.25, 0.5]
    assert result["strike"].tolist() == [102.0, 105.0]
    assert result["selection_status"].tolist() == ["nearest", "nearest"]
    assert np.allclose(result["atm_distance"], [0.02, 0.05])


@pytest.mark.parametrize("right_moneyness", [1.05, 1.0500000000005])
def test_atm_term_structure_averages_exact_and_near_ties(right_moneyness):
    frame = pd.DataFrame(
        {
            "tau": [0.25, 0.25],
            "strike": [95, 105],
            "iv": [0.20, 0.24],
            "moneyness": [0.95, right_moneyness],
        }
    )

    result = atm_term_structure_frame(frame).iloc[0]

    assert result["selection_status"] == "symmetric_average"
    assert result["strike"] == 100
    assert result["iv"] == pytest.approx(0.22)
    assert result["moneyness"] == pytest.approx(
        (0.95 + right_moneyness) / 2
    )


def test_atm_term_structure_does_not_average_distinct_distances():
    frame = pd.DataFrame(
        {
            "tau": [0.25, 0.25],
            "strike": [95, 105],
            "iv": [0.20, 0.24],
            "moneyness": [0.95, 1.050000000002],
        }
    )

    result = atm_term_structure_frame(frame).iloc[0]

    assert result["selection_status"] == "nearest"
    assert result["strike"] == 95
    assert result["iv"] == 0.20


def test_atm_term_structure_preserves_outside_bound_and_no_finite_iv():
    frame = pd.DataFrame(
        {
            "tau": [0.25, 0.5],
            "strike": [120, 100],
            "iv": [0.30, np.nan],
            "moneyness": [1.20, 1.0],
        }
    )

    result = atm_term_structure_frame(frame)
    outside = result.iloc[0]
    missing = result.iloc[1]

    assert outside["selection_status"] == "outside_bound"
    assert np.isnan(outside[["strike", "iv", "moneyness"]].astype(float)).all()
    assert outside["atm_distance"] == pytest.approx(0.20)
    assert missing["selection_status"] == "no_finite_iv"
    assert np.isnan(
        missing[["strike", "iv", "moneyness", "atm_distance"]].astype(float)
    ).all()


def test_atm_term_structure_preserves_expiry_metadata_for_rejected_rows():
    long_frame = pd.DataFrame(
        {
            "tau": [0.5, 0.25, 0.125],
            "strike": [120, 100, 100],
            "iv": [0.30, np.nan, 0.20],
            "moneyness": [1.20, 1.0, 1.0],
        }
    )
    with_metadata = long_frame.assign(
        expiry_datetime=pd.to_datetime(
            [
                "2026-07-01T08:00:00Z",
                "2026-06-01T08:00:00Z",
                "2026-05-01T08:00:00Z",
            ],
            utc=True,
        )
    )

    baseline = atm_term_structure_frame(long_frame)
    result = atm_term_structure_frame(with_metadata)

    pd.testing.assert_frame_equal(
        result.drop(columns="expiry_datetime"), baseline
    )
    assert result["expiry_datetime"].tolist() == [
        pd.Timestamp("2026-05-01T08:00:00Z"),
        pd.Timestamp("2026-06-01T08:00:00Z"),
        pd.Timestamp("2026-07-01T08:00:00Z"),
    ]
    assert result["selection_status"].tolist() == [
        "nearest",
        "no_finite_iv",
        "outside_bound",
    ]


@pytest.mark.parametrize("distance", [-0.01, np.nan, np.inf, -np.inf])
def test_atm_term_structure_rejects_invalid_distance_bound(distance):
    with pytest.raises(ValueError, match="finite and non-negative"):
        atm_term_structure_frame(_long_frame(), distance)


def test_make_iv_heatmap_sorts_grid_keeps_nan_and_strike_hover():
    figure = make_iv_heatmap(_long_frame())

    assert isinstance(figure, go.Figure)
    trace = figure.data[0]
    assert isinstance(trace, go.Heatmap)
    assert trace.x.tolist() == [0.9, 1.1]
    assert trace.y.tolist() == [0.25, 0.5]
    assert np.asarray(trace.z).shape == (2, 2)
    assert np.asarray(trace.z)[0].tolist() == [0.30, 0.25]
    assert np.isnan(np.asarray(trace.z)[1, 0])
    assert np.asarray(trace.z)[1, 1] == 0.20
    assert np.asarray(trace.customdata).tolist() == [[90.0, 110.0], [90.0, 110.0]]
    assert trace.connectgaps is False
    assert "strike" in trace.hovertemplate
    assert figure.layout.xaxis.title.text == "K / S"
    assert figure.layout.yaxis.title.text == "tau (years)"


def test_make_smile_figure_filters_and_sorts_without_changing_iv_values():
    figure = make_smile_figure(_long_frame(), selected_expiries=[0.5])

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1
    trace = figure.data[0]
    assert trace.name == "tau=0.5"
    assert trace.x.tolist() == [0.9, 1.1]
    assert np.isnan(trace.y[0])
    assert trace.y[1] == 0.20
    assert trace.customdata.tolist() == [90.0, 110.0]
    assert trace.connectgaps is False
    assert figure.layout.yaxis.tickformat == ".0%"
    assert len(figure.layout.shapes) == 1
    assert figure.layout.shapes[0].x0 == 1.0
    assert figure.layout.shapes[0].x1 == 1.0


def test_make_smile_figure_orders_all_expiries_ascending():
    figure = make_smile_figure(_long_frame())

    assert [trace.name for trace in figure.data] == ["tau=0.25", "tau=0.5"]
    assert all(trace.connectgaps is False for trace in figure.data)


def test_make_atm_term_structure_figure_keeps_gap_and_hover_fields():
    atm_frame = atm_term_structure_frame(
        pd.DataFrame(
            {
                "tau": [0.5, 0.25],
                "strike": [120, 100],
                "iv": [0.30, np.nan],
                "moneyness": [1.20, 1.0],
            }
        )
    )

    figure = make_atm_term_structure_figure(atm_frame)

    assert isinstance(figure, go.Figure)
    trace = figure.data[0]
    assert trace.x.tolist() == [0.25, 0.5]
    assert np.isnan(trace.y).all()
    assert trace.connectgaps is False
    assert np.asarray(trace.customdata).shape == (2, 4)
    for label in ["tau", "IV", "strike", "K / S", "ATM distance", "selection"]:
        assert label in trace.hovertemplate
    assert figure.layout.yaxis.tickformat == ".0%"


def test_figures_use_date_labels_and_tau_fallback_without_moving_coordinates():
    long_frame = surface_to_long_frame(
        np.array([[0.20, np.nan], [0.25, 0.30]]),
        expiries=[0.5, 0.25],
        strikes=[110, 90],
        spot_price=100,
        expiry_datetimes=["2026-07-01T08:00:00Z", None],
    )

    heatmap = make_iv_heatmap(long_frame)
    heatmap_trace = heatmap.data[0]
    assert heatmap_trace.y.tolist() == [0.25, 0.5]
    assert list(heatmap.layout.yaxis.tickvals) == [0.25, 0.5]
    assert list(heatmap.layout.yaxis.ticktext) == ["tau=0.25", "2026-07-01"]
    assert np.isnan(np.asarray(heatmap_trace.z)[1, 0])
    assert heatmap_trace.connectgaps is False

    smiles = make_smile_figure(long_frame)
    assert [trace.name for trace in smiles.data] == [
        "tau=0.25",
        "2026-07-01",
    ]
    assert [trace.meta for trace in smiles.data] == [0.25, 0.5]
    assert all("tau:" in trace.hovertemplate for trace in smiles.data)
    assert all(trace.connectgaps is False for trace in smiles.data)

    selected = make_smile_figure(long_frame, selected_expiries=[0.5])
    assert len(selected.data) == 1
    assert selected.data[0].name == "2026-07-01"
    assert selected.data[0].x.tolist() == [0.9, 1.1]
    assert np.isnan(selected.data[0].y[0])
    assert selected.data[0].y[1] == 0.20

    atm = make_atm_term_structure_figure(atm_term_structure_frame(long_frame))
    atm_trace = atm.data[0]
    assert atm_trace.x.tolist() == [0.25, 0.5]
    assert list(atm.layout.xaxis.tickvals) == [0.25, 0.5]
    assert list(atm.layout.xaxis.ticktext) == ["tau=0.25", "2026-07-01"]
    assert np.isnan(atm_trace.y).all()
    assert atm_trace.connectgaps is False
