from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from iv_surface import fetcher
from iv_surface.fetcher import (
    compute_tau,
    load_chain_snapshot,
    parse_chain_snapshot,
)


SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "bybit_btc_options_snapshot.json"
)
EXPECTED_TICKER_COUNT = 660
EXPECTED_MISSING_MID_COUNT = 30
REQUIRED_CHAIN_COLUMNS = {
    "symbol",
    "underlying",
    "expiry_dt",
    "strike",
    "flag",
    "bid_price",
    "ask_price",
    "mid_price",
    "quote_source",
    "bid_iv",
    "ask_iv",
    "mark_price",
    "mark_iv",
    "index_price",
    "underlying_price",
    "tau",
}


def _load_fixture():
    snapshot = load_chain_snapshot(SNAPSHOT_PATH)
    return snapshot, parse_chain_snapshot(snapshot)


def test_raw_snapshot_and_parsed_chain_reconcile():
    snapshot, chain = _load_fixture()
    raw_tickers = snapshot["response"]["result"]["list"]

    assert snapshot["response"]["retCode"] == 0
    assert snapshot["request"]["base_url"] == "https://api.bytick.com"
    assert snapshot["request"]["params"] == {
        "category": "option",
        "baseCoin": "BTC",
    }
    assert len(raw_tickers) == EXPECTED_TICKER_COUNT
    assert len(chain) == EXPECTED_TICKER_COUNT
    assert chain["symbol"].nunique() == EXPECTED_TICKER_COUNT
    assert REQUIRED_CHAIN_COLUMNS <= set(chain.columns)
    assert set(chain["flag"]) == {"call", "put"}
    assert set(chain["underlying"]) == {"BTC"}


def test_snapshot_has_no_duplicate_surface_identities():
    _, chain = _load_fixture()

    duplicates = chain.duplicated(
        subset=["flag", "tau", "strike"],
        keep=False,
    )

    assert not duplicates.any()


def test_snapshot_tau_is_deterministic_and_uses_utc_metadata():
    snapshot, chain = _load_fixture()
    captured_at = datetime.fromisoformat(
        snapshot["captured_at_utc"].replace("Z", "+00:00")
    )
    expected_tau = np.array(
        [compute_tau(expiry_dt, captured_at) for expiry_dt in chain["expiry_dt"]]
    )

    assert captured_at.tzinfo == timezone.utc
    assert all(expiry_dt.tzinfo == timezone.utc for expiry_dt in chain["expiry_dt"])
    assert np.array_equal(chain["tau"].to_numpy(), expected_tau)


def test_snapshot_retains_missing_quotes_instead_of_filtering_them():
    _, chain = _load_fixture()
    missing_mid = chain["mid_price"].isna()

    assert int(missing_mid.sum()) == EXPECTED_MISSING_MID_COUNT
    assert (chain.loc[missing_mid, "quote_source"] == "none").all()


def test_snapshot_load_and_parse_do_not_use_the_network(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("offline fixture loading must not call Bybit")

    monkeypatch.setattr(fetcher.requests, "get", fail_if_called)

    snapshot = load_chain_snapshot(SNAPSHOT_PATH)
    chain = parse_chain_snapshot(snapshot)

    assert len(chain) == EXPECTED_TICKER_COUNT
