from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

DEFAULT_BYBIT_BASE_URL = "https://api.bytick.com"
DEFAULT_BYBIT_TIMEOUT = 10
_EXPIRY_FMT = "%d%b%y"


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


def fetch_chain(
    underlying: str = "BTC",
    base_url: str = DEFAULT_BYBIT_BASE_URL,
    timeout: float = DEFAULT_BYBIT_TIMEOUT,
) -> pd.DataFrame:
    """Fetch live option chain from Bybit and return a tidy DataFrame."""
    params = {"category": "option", "baseCoin": underlying}
    resp = requests.get(_bybit_tickers_url(base_url), params=params, timeout=timeout)
    resp.raise_for_status()

    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"JSON decode failed: {exc}") from exc

    result = data.get("result", {})
    tickers = result.get("list") if isinstance(result, dict) else None
    if not isinstance(tickers, list):
        raise RuntimeError(f"Unexpected response structure: {data}")

    now = datetime.now(timezone.utc)
    rows = []
    for ticker in tickers:
        symbol = ticker.get("symbol", "")
        try:
            parsed = parse_symbol(symbol)
        except ValueError:
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
        tau = compute_tau(parsed["expiry_dt"], now)

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

    if not rows:
        return pd.DataFrame(rows)

    expiry_dts = [row.pop("expiry_dt") for row in rows]
    df = pd.DataFrame(rows)
    df["expiry_dt"] = pd.Series(expiry_dts, dtype=object)

    df = df.sort_values(["tau", "strike"]).reset_index(drop=True)
    return df
