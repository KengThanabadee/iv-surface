import numpy as np
import pandas as pd
import pytest

from iv_surface.surface_builder import (
    CombinedSurfaceResult,
    SurfaceInputs,
    SurfaceResult,
    build_combined_surface_from_chain,
    build_surface_from_chain,
    prepare_surface_inputs,
)
from iv_surface.solver import bs_price


def _row(
    flag="call",
    tau=0.25,
    strike=100,
    mid_price=10,
    quote_source="mid",
    index_price=100,
):
    return {
        "flag": flag,
        "tau": tau,
        "strike": strike,
        "mid_price": mid_price,
        "quote_source": quote_source,
        "index_price": index_price,
    }


def test_prepare_surface_inputs_filters_sorts_and_keeps_missing_cells():
    chain = pd.DataFrame(
        [
            _row(tau=0.5, strike=100, mid_price=11),
            _row(tau=0.25, strike=90, mid_price=10),
            _row(tau=0.25, strike=100, mid_price=12, index_price=100),
            _row(flag="put", tau=0.25, strike=90, mid_price=99),
            _row(tau=0.5, strike=110, mid_price=np.nan),
            _row(tau=0.5, strike=120, mid_price=20, quote_source="none"),
        ]
    )

    result = prepare_surface_inputs(chain, flag="call")

    assert isinstance(result, SurfaceInputs)
    assert result.spot_price == 100
    assert result.expiries == [0.25, 0.5]
    assert result.strikes == [90, 100]
    assert result.option_price_grid.shape == (2, 2)
    assert result.option_price_grid[0, 0] == 10
    assert result.option_price_grid[0, 1] == 12
    assert np.isnan(result.option_price_grid[1, 0])
    assert result.option_price_grid[1, 1] == 11


def test_prepare_surface_inputs_warns_when_index_prices_differ():
    chain = pd.DataFrame(
        [
            _row(tau=0.25, strike=90, mid_price=10, index_price=99),
            _row(tau=0.25, strike=100, mid_price=12, index_price=100),
            _row(tau=0.5, strike=100, mid_price=11, index_price=101),
        ]
    )

    with pytest.warns(UserWarning, match="different index_price"):
        result = prepare_surface_inputs(chain, flag="call")

    assert result.spot_price == 100


def test_prepare_surface_inputs_uses_selected_put_flag():
    chain = pd.DataFrame(
        [
            _row(flag="call", tau=0.25, strike=100, mid_price=10),
            _row(flag="put", tau=0.25, strike=100, mid_price=8),
        ]
    )

    result = prepare_surface_inputs(chain, flag="put")

    assert result.option_price_grid.tolist() == [[8.0]]


def test_prepare_surface_inputs_rejects_invalid_flag():
    chain = pd.DataFrame([_row()])

    with pytest.raises(ValueError):
        prepare_surface_inputs(chain, flag="straddle")


def test_prepare_surface_inputs_rejects_missing_columns():
    chain = pd.DataFrame([_row()]).drop(columns=["quote_source"])

    with pytest.raises(ValueError):
        prepare_surface_inputs(chain)


def test_prepare_surface_inputs_rejects_missing_index_price():
    chain = pd.DataFrame([_row()]).drop(columns=["index_price"])

    with pytest.raises(ValueError):
        prepare_surface_inputs(chain)


def test_prepare_surface_inputs_rejects_no_usable_rows():
    chain = pd.DataFrame([_row(mid_price=np.nan), _row(quote_source="none")])

    with pytest.raises(ValueError):
        prepare_surface_inputs(chain)


def test_prepare_surface_inputs_filters_invalid_index_prices():
    chain = pd.DataFrame(
        [
            _row(tau=0.25, strike=90, mid_price=10, index_price=np.nan),
            _row(tau=0.25, strike=100, mid_price=12, index_price=0),
            _row(tau=0.5, strike=100, mid_price=11, index_price=100),
        ]
    )

    result = prepare_surface_inputs(chain)

    assert result.spot_price == 100
    assert result.expiries == [0.5]
    assert result.strikes == [100]
    assert result.option_price_grid.tolist() == [[11.0]]


def test_prepare_surface_inputs_rejects_duplicate_tau_strike():
    chain = pd.DataFrame(
        [
            _row(tau=0.25, strike=100, mid_price=10),
            _row(tau=0.25, strike=100, mid_price=11),
        ]
    )

    with pytest.raises(ValueError):
        prepare_surface_inputs(chain)


def test_build_surface_from_chain_solves_iv_surface_from_mid_prices():
    spot_price = 100
    r = 0.01
    sigma = 0.2
    expiries = [0.25, 0.5]
    strikes = [90, 100]
    rows = []

    for T in expiries:
        for K in strikes:
            rows.append(
                _row(
                    tau=T,
                    strike=K,
                    mid_price=bs_price(spot_price, K, T, r, sigma, "call"),
                    index_price=spot_price,
                )
            )

    result = build_surface_from_chain(pd.DataFrame(rows), flag="call", r=r)

    assert isinstance(result, SurfaceResult)
    assert result.spot_price == spot_price
    assert result.expiries == expiries
    assert result.strikes == strikes
    assert np.allclose(result.iv_surface, sigma, rtol=0.0, atol=1e-4)


def _priced_row(flag, strike, sigma, *, tau=0.25, spot_price=100, r=0):
    return _row(
        flag=flag,
        tau=tau,
        strike=strike,
        mid_price=bs_price(spot_price, strike, tau, r, sigma, flag),
        index_price=spot_price,
    )


def test_combined_surface_blends_near_atm_and_prefers_otm_wings():
    rows = []
    call_sigmas = {95: 0.21, 98: 0.22, 100: 0.24, 102: 0.26, 105: 0.28}
    put_sigmas = {95: 0.31, 98: 0.32, 100: 0.34, 102: 0.36, 105: 0.38}
    for strike in call_sigmas:
        rows.extend(
            [
                _priced_row("call", strike, call_sigmas[strike]),
                _priced_row("put", strike, put_sigmas[strike]),
            ]
        )

    result = build_combined_surface_from_chain(pd.DataFrame(rows))

    assert isinstance(result, CombinedSurfaceResult)
    assert result.strikes == [95, 98, 100, 102, 105]
    assert result.source_grid.tolist() == [[
        "put",
        "near_atm_average",
        "near_atm_average",
        "near_atm_average",
        "call",
    ]]
    assert np.allclose(
        result.iv_surface[0],
        [0.31, 0.27, 0.29, 0.31, 0.28],
        rtol=0,
        atol=1e-4,
    )


def test_combined_surface_uses_labeled_opposite_side_fallbacks_and_gaps():
    rows = [
        _priced_row("call", 95, 0.20),
        _row(flag="put", strike=95, mid_price=np.nan, quote_source="none"),
        _priced_row("put", 100, 0.30),
        _row(flag="call", strike=100, mid_price=np.nan, quote_source="none"),
        _priced_row("put", 105, 0.40),
        _row(flag="call", strike=105, mid_price=np.nan, quote_source="none"),
        _row(flag="call", strike=110, mid_price=np.nan, quote_source="none"),
        _row(flag="put", strike=110, mid_price=np.nan, quote_source="none"),
    ]

    result = build_combined_surface_from_chain(pd.DataFrame(rows))

    assert result.source_grid.tolist() == [[
        "call_fallback",
        "near_atm_put_only",
        "put_fallback",
        "missing",
    ]]
    assert np.allclose(result.iv_surface[0, :3], [0.20, 0.30, 0.40], atol=1e-4)
    assert np.isnan(result.iv_surface[0, 3])


def test_combined_surface_wing_selection_stays_spot_based_when_r_is_nonzero():
    rows = [
        _priced_row("call", 103, 0.20, tau=1.0, r=0.10),
        _priced_row("put", 103, 0.40, tau=1.0, r=0.10),
    ]

    result = build_combined_surface_from_chain(pd.DataFrame(rows), r=0.10)

    assert result.source_grid.tolist() == [["call"]]
    assert result.iv_surface[0, 0] == pytest.approx(0.20, abs=1e-4)


def test_combined_surface_aligns_union_grid_and_uses_one_shared_spot():
    rows = [
        _priced_row("put", 90, 0.25, tau=0.25),
        _priced_row("call", 110, 0.30, tau=0.5),
    ]

    result = build_combined_surface_from_chain(pd.DataFrame(rows))

    assert result.spot_price == 100
    assert result.expiries == [0.25, 0.5]
    assert result.strikes == [90, 110]
    assert result.iv_surface.shape == (2, 2)
    assert result.source_grid.tolist() == [
        ["put", "missing"],
        ["missing", "call"],
    ]


def test_combined_surface_rejects_duplicate_side_contracts_and_no_usable_quotes():
    duplicate = pd.DataFrame(
        [_priced_row("call", 100, 0.20), _priced_row("call", 100, 0.21)]
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_combined_surface_from_chain(duplicate)

    no_quotes = pd.DataFrame(
        [
            _row(flag="call", mid_price=np.nan, quote_source="none"),
            _row(flag="put", mid_price=np.nan, quote_source="none"),
        ]
    )
    with pytest.raises(ValueError, match="no usable Call or Put"):
        build_combined_surface_from_chain(no_quotes)
