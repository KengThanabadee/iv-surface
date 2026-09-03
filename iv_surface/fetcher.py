from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DEFAULT_BYBIT_BASE_URL = "https://api.bytick.com"
DEFAULT_BYBIT_TIMEOUT = 10
_EXPIRY_FMT = "%d%b%y"
SNAPSHOT_SCHEMA_VERSION = 1

_CHAIN_COLUMNS = [
    "symbol",
    "underlying",
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
    "expiry_dt",
]


def _bybit_tickers_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/v5/market/tickers"


def _to_float(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _compute_mid_price(bid_price, ask_price):
    if not np.isfinite(bid_price) or not np.isfinite(ask_price):
        return np.nan
    if bid_price <= 0 or ask_price <= 0:
        return np.nan
    if ask_price < bid_price:
        return np.nan
    return (bid_price + ask_price) / 2


def parse_symbol(symbol: str) -> dict:
    """Parse Bybit option symbols with optional settle coin suffix."""
    parts = symbol.split("-")
    if len(parts) == 4:
        underlying, expiry_str, strike_str, flag_char = parts
    elif len(parts) == 5:
        underlying, expiry_str, strike_str, flag_char, _settle_coin = parts
    else:
        raise ValueError(f"Unexpected symbol format: {symbol!r}")
    expiry_dt = datetime.strptime(expiry_str, _EXPIRY_FMT).replace(
        hour=8, tzinfo=timezone.utc
    )
    strike = float(strike_str)
    if flag_char.upper() == "C":
        flag = "call"
    elif flag_char.upper() == "P":
        flag = "put"
    else:
        raise ValueError(f"Unexpected option flag: {flag_char!r}")
    return {"underlying": underlying, "expiry_dt": expiry_dt, "strike": strike, "flag": flag}


def compute_tau(expiry_dt: datetime, now: datetime) -> float:
    """Time to expiry in years."""
    diff = expiry_dt - now
    seconds = diff.total_seconds()
    return max(seconds / (365.25 * 24 * 3600), 0.0)


def _parse_utc_timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("captured_at_utc must be an ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("captured_at_utc must be a valid ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("captured_at_utc must use UTC")
    return parsed.astimezone(timezone.utc)


def _parse_bybit_tickers_response(
    data: dict, captured_at_utc: datetime
) -> pd.DataFrame:
    """Parse a Bybit option-ticker response at one fixed observation time."""
    if captured_at_utc.tzinfo is None:
        raise ValueError("captured_at_utc must be timezone-aware")
    captured_at_utc = captured_at_utc.astimezone(timezone.utc)

    result = data.get("result", {}) if isinstance(data, dict) else {}
    tickers = result.get("list") if isinstance(result, dict) else None
    if not isinstance(tickers, list):
        raise RuntimeError(f"Unexpected response structure: {data}")

    rows = []
    unsupported_symbols = []
    for ticker in tickers:
        if not isinstance(ticker, dict):
            raise RuntimeError("Unexpected ticker structure: ticker must be an object")
        symbol = ticker.get("symbol", "")
        try:
            parsed = parse_symbol(symbol)
        except ValueError:
            unsupported_symbols.append(symbol)
            continue

        bid_price = _to_float(ticker.get("bid1Price"))
        ask_price = _to_float(ticker.get("ask1Price"))
        bid_iv = _to_float(ticker.get("bid1Iv"))
        ask_iv = _to_float(ticker.get("ask1Iv"))
        mark_price = _to_float(ticker.get("markPrice"))
        mark_iv = _to_float(ticker.get("markIv"))
        index_price = _to_float(ticker.get("indexPrice"))
        underlying_price = _to_float(ticker.get("underlyingPrice"))
        mid_price = _compute_mid_price(bid_price, ask_price)
        quote_source = "mid" if np.isfinite(mid_price) else "none"
        tau = compute_tau(parsed["expiry_dt"], captured_at_utc)

        rows.append(
            {
                "symbol": symbol,
                "underlying": parsed["underlying"],
                "expiry_dt": parsed["expiry_dt"],
                "strike": parsed["strike"],
                "flag": parsed["flag"],
                "bid_price": bid_price,
                "ask_price": ask_price,
                "mid_price": mid_price,
                "quote_source": quote_source,
                "bid_iv": bid_iv,
                "ask_iv": ask_iv,
                "mark_price": mark_price,
                "mark_iv": mark_iv,
                "index_price": index_price,
                "underlying_price": underlying_price,
                "tau": tau,
            }
        )

    if unsupported_symbols:
        displayed_symbols = ", ".join(repr(symbol) for symbol in unsupported_symbols)
        raise ValueError(
            f"Bybit ticker response contains unsupported symbols: {displayed_symbols}"
        )

    if not rows:
        return pd.DataFrame(columns=_CHAIN_COLUMNS)

    expiry_dts = [row.pop("expiry_dt") for row in rows]
    df = pd.DataFrame(rows)
    df["expiry_dt"] = pd.Series(expiry_dts, dtype=object)

    return df.sort_values(["tau", "strike"]).reset_index(drop=True)


def fetch_chain_snapshot(
    underlying: str = "BTC",
    base_url: str = DEFAULT_BYBIT_BASE_URL,
    timeout: float = DEFAULT_BYBIT_TIMEOUT,
) -> dict:
    """Fetch one raw Bybit response and wrap it with reproducibility metadata."""
    params = {"category": "option", "baseCoin": underlying}
    resp = requests.get(_bybit_tickers_url(base_url), params=params, timeout=timeout)
    resp.raise_for_status()

    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"JSON decode failed: {exc}") from exc

    captured_at_utc = datetime.now(timezone.utc)
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "captured_at_utc": captured_at_utc.isoformat().replace("+00:00", "Z"),
        "request": {
            "base_url": base_url.rstrip("/"),
            "endpoint": "/v5/market/tickers",
            "params": params,
        },
        "response": data,
    }


def _validate_snapshot_envelope(snapshot: dict) -> datetime:
    """Validate snapshot metadata and return its frozen UTC capture time."""
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be a JSON object")
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"snapshot schema_version must be {SNAPSHOT_SCHEMA_VERSION}"
        )
    request = snapshot.get("request")
    if not isinstance(request, dict):
        raise ValueError("snapshot is missing request metadata")
    if not isinstance(request.get("base_url"), str) or not request["base_url"]:
        raise ValueError("snapshot request base_url must be a non-empty string")
    if request.get("endpoint") != "/v5/market/tickers":
        raise ValueError("snapshot request endpoint must be /v5/market/tickers")
    params = request.get("params")
    if not isinstance(params, dict):
        raise ValueError("snapshot request params must be a JSON object")
    if params.get("category") != "option":
        raise ValueError("snapshot request category must be option")
    if not isinstance(params.get("baseCoin"), str) or not params["baseCoin"]:
        raise ValueError("snapshot request baseCoin must be a non-empty string")
    if not isinstance(snapshot.get("response"), dict):
        raise ValueError("snapshot response must be a JSON object")
    return _parse_utc_timestamp(snapshot.get("captured_at_utc"))


def parse_chain_snapshot(snapshot: dict) -> pd.DataFrame:
    """Parse a snapshot envelope using its frozen capture time."""
    captured_at_utc = _validate_snapshot_envelope(snapshot)
    return _parse_bybit_tickers_response(snapshot["response"], captured_at_utc)


def load_chain_snapshot(path) -> dict:
    """Load and validate a snapshot envelope without network access."""
    with Path(path).open(encoding="utf-8") as snapshot_file:
        snapshot = json.load(snapshot_file)
    _validate_snapshot_envelope(snapshot)
    return snapshot


def fetch_chain(
    underlying: str = "BTC",
    base_url: str = DEFAULT_BYBIT_BASE_URL,
    timeout: float = DEFAULT_BYBIT_TIMEOUT,
) -> pd.DataFrame:
    """Fetch live option chain from Bybit and return a tidy DataFrame."""
    snapshot = fetch_chain_snapshot(
        underlying=underlying,
        base_url=base_url,
        timeout=timeout,
    )
    return parse_chain_snapshot(snapshot)
