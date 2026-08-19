from datetime import datetime, timezone

import numpy as np
import pytest

from iv_surface import fetcher
from iv_surface.fetcher import (
    _bybit_tickers_url,
    _compute_mid_price,
    _to_float,
    compute_tau,
    fetch_chain,
    parse_symbol,
)


def test_to_float_returns_nan_for_bad_values():
    assert _to_float("123.45") == 123.45
    assert np.isnan(_to_float(None))
    assert np.isnan(_to_float("not-a-number"))
    assert np.isnan(_to_float(np.inf))


def test_compute_mid_price_uses_valid_bid_ask_only():
    assert _compute_mid_price(100, 110) == 105
    assert np.isnan(_compute_mid_price(0, 110))
    assert np.isnan(_compute_mid_price(100, 0))
    assert np.isnan(_compute_mid_price(110, 100))
    assert np.isnan(_compute_mid_price(np.nan, 110))


def test_parse_symbol_call_and_put():
    call = parse_symbol("BTC-27JUN25-100000-C")
    put = parse_symbol("BTC-27JUN25-100000-P")
    usdt_call = parse_symbol("BTC-27JUN25-100000-C-USDT")

    assert call["underlying"] == "BTC"
    assert call["expiry_dt"] == datetime(2025, 6, 27, 8, tzinfo=timezone.utc)
    assert call["strike"] == 100000.0
    assert call["flag"] == "call"
    assert put["flag"] == "put"
    assert usdt_call["flag"] == "call"
    assert usdt_call["strike"] == 100000.0


def test_parse_symbol_rejects_invalid_flag():
    with pytest.raises(ValueError):
        parse_symbol("BTC-27JUN25-100000-X")


def test_compute_tau_clamps_expired_time_to_zero():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert compute_tau(now, now) == 0.0
    assert compute_tau(datetime(2025, 12, 31, tzinfo=timezone.utc), now) == 0.0


def test_compute_tau_one_year():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    expiry = datetime(2027, 1, 1, 6, tzinfo=timezone.utc)

    assert abs(compute_tau(expiry, now) - 1.0) < 1e-12


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_chain_uses_mid_price_as_quote_source(monkeypatch):
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "BTC-27JUN30-100000-C-USDT",
                    "bid1Price": "100",
                    "ask1Price": "120",
                    "bid1Iv": "0.55",
                    "ask1Iv": "0.65",
                    "markPrice": "108",
                    "markIv": "0.60",
                    "indexPrice": "99900",
                    "underlyingPrice": "100000",
                }
            ]
        }
    }

    def fake_get(url, params, timeout):
        assert url == f"{fetcher.DEFAULT_BYBIT_BASE_URL}/v5/market/tickers"
        assert params == {"category": "option", "baseCoin": "BTC"}
        assert timeout == fetcher.DEFAULT_BYBIT_TIMEOUT
        return _FakeResponse(payload)

    monkeypatch.setattr(fetcher.requests, "get", fake_get)

    df = fetch_chain("BTC")
    row = df.iloc[0]

    assert row["bid_price"] == 100
    assert row["ask_price"] == 120
    assert row["mid_price"] == 110
    assert row["quote_source"] == "mid"
    assert row["bid_iv"] == 0.55
    assert row["ask_iv"] == 0.65
    assert row["mark_price"] == 108
    assert row["mark_iv"] == 0.60
    assert row["index_price"] == 99900


def test_fetch_chain_accepts_custom_base_url_and_timeout(monkeypatch):
    payload = {"result": {"list": []}}

    def fake_get(url, params, timeout):
        assert url == "https://api.bybit.com/v5/market/tickers"
        assert params == {"category": "option", "baseCoin": "ETH"}
        assert timeout == 3
        return _FakeResponse(payload)

    monkeypatch.setattr(fetcher.requests, "get", fake_get)

    df = fetch_chain("ETH", base_url="https://api.bybit.com", timeout=3)

    assert df.empty


def test_bybit_tickers_url_normalizes_trailing_slash():
    assert (
        _bybit_tickers_url("https://api.bytick.com/")
        == "https://api.bytick.com/v5/market/tickers"
    )


def test_fetch_chain_does_not_fallback_to_mark_price(monkeypatch):
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "BTC-27JUN30-100000-C-USDT",
                    "bid1Price": "0",
                    "ask1Price": "120",
                    "markPrice": "108",
                    "markIv": "0.60",
                    "indexPrice": "99900",
                    "underlyingPrice": "100000",
                }
            ]
        }
    }

    def fake_get(url, params, timeout):
        return _FakeResponse(payload)

    monkeypatch.setattr(fetcher.requests, "get", fake_get)

    df = fetch_chain("BTC")
    row = df.iloc[0]

    assert np.isnan(row["mid_price"])
    assert row["quote_source"] == "none"
    assert row["mark_price"] == 108
