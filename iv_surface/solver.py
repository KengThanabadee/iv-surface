import bisect
import numpy as np
from scipy.stats import norm


def _is_strictly_increasing(values):
    return all(values[i] < values[i + 1] for i in range(len(values) - 1))


def _compute_d1_d2(spot_price, K, T, t, r, sigma):
        for name, value in {
            "spot_price": spot_price,
            "K": K,
            "T": T,
            "t": t,
            "r": r,
            "sigma": sigma,
        }.items():
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite, got {value}")
        if np.any(spot_price <= 0):
            raise ValueError(f"spot_price must be > 0, got {spot_price}")
        if np.any(K <= 0):
            raise ValueError(f"K must be > 0, got {K}")
        if np.any(sigma <= 0):
            raise ValueError(f"sigma must be > 0, got {sigma}")
        tau = np.maximum(T - t, 1e-12)
        d1 = (np.log(spot_price / K) + (r + 0.5 * sigma ** 2) * (tau)) / (sigma * np.sqrt(tau))
        d2 = d1 - (sigma * np.sqrt(tau))
        return d1, d2

def bs_price(spot_price, K, T, r, sigma, flag, t=0):
    if flag not in ("call", "put"):
            raise ValueError(f"flag must be 'call' or 'put', got '{flag}'")

    if t >= T:
        if flag == "call" :
            return np.maximum(spot_price - K, 0)
        if flag == "put":
            return np.maximum(K - spot_price, 0)

    d1, d2 = _compute_d1_d2(spot_price, K, T, t, r, sigma)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    discount = np.exp(-r * (T - t))
    C = spot_price * N_d1 - K * discount * N_d2
    P = C + K * discount - spot_price # put-call parity

    if flag == "call" :
        return C
    if flag == "put":
        return P

def _vega_compute(spot_price, sigma, T, K, r, t=0):
    if t >= T:
        return 0
    d1, _ = _compute_d1_d2(spot_price, K, T, t, r, sigma)
    vega = spot_price * norm.pdf(d1) * np.sqrt(T - t)
    return vega

def _newton_step(sigma, spot_price, K, T, r, option_price, flag):
    f_sigma = bs_price(spot_price, K, T, r, sigma, flag) - option_price
    f_diff_sigma = _vega_compute(spot_price, sigma, T, K, r)
    if f_diff_sigma < 1e-8:
        return -1
    sigma_next = sigma - f_sigma / f_diff_sigma
    return sigma_next

def solve_iv(option_price, spot_price, K, T, r, flag, sigma_low=1e-6, sigma_high=10.0):
    tolerance = 1e-4
    for name, value in {
        "option_price": option_price,
        "spot_price": spot_price,
        "K": K,
        "T": T,
        "r": r,
        "sigma_low": sigma_low,
        "sigma_high": sigma_high,
    }.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite, got {value}")
    if T <= 0:
        raise ValueError(f"T must be > 0 for implied volatility, got {T}")
    if sigma_low <= 0 or sigma_high <= 0:
        raise ValueError("sigma bounds must be > 0")
    if sigma_low >= sigma_high:
        raise ValueError("sigma_low must be less than sigma_high")
    interval = sigma_high - sigma_low
    bs_low = bs_price(spot_price, K, T, r, sigma_low, flag)
    bs_high = bs_price(spot_price, K, T, r, sigma_high, flag)

    if abs(bs_low - option_price) < tolerance:
        return sigma_low
    if abs(bs_high - option_price) < tolerance:
        return sigma_high

    if (bs_low - option_price) * (bs_high - option_price) > 0:
        raise ValueError(f"option_price {option_price} is outside no-arbitrage bounds")

    sigma_mid = (sigma_low + sigma_high) / 2
    while interval >= tolerance:
        sigma_nr = _newton_step(sigma_mid, spot_price, K, T, r, option_price, flag)
        if sigma_nr != -1 and np.isfinite(sigma_nr) and abs(sigma_nr - sigma_mid) < tolerance:
            return sigma_nr

        if np.isfinite(sigma_nr) and (sigma_nr >= sigma_low) and (sigma_nr <= sigma_high):
            sigma_mid = sigma_nr
        else:
            sigma_mid = (sigma_low + sigma_high) / 2

        bs_mid = bs_price(spot_price, K, T, r, sigma_mid, flag)

        if abs(bs_mid - option_price) < tolerance:
            return sigma_mid

        if (bs_low - option_price) * (bs_mid - option_price) < 0:
            # root is in left half
            sigma_high = sigma_mid
            bs_high = bs_mid

        else:
            # root is in right half
            sigma_low = sigma_mid
            bs_low = bs_mid

        interval = sigma_high - sigma_low

    return (sigma_low + sigma_high) / 2

def build_surface(option_price_grid, spot_price, expiries, strikes, r=0, flag="call"):
    # option_price_grid: 2D array shape (n_expiries, n_strikes)
    # returns: 2D array same shape, IV at each (expiry, strike)
    expected_shape = (len(expiries), len(strikes))
    if option_price_grid.shape != expected_shape:
        raise ValueError(
            f"option_price_grid shape must be {expected_shape}, got {option_price_grid.shape}"
        )

    iv_surface = np.full((len(expiries), len(strikes)), np.nan)
    for i, T in enumerate(expiries):
        for j, K in enumerate(strikes):
            try:
                iv_surface[i, j] = solve_iv(
                    option_price_grid[i, j], spot_price, K, T, r, flag
                )
            except ValueError:
                pass
    return iv_surface

def interpolate_iv(iv_surface, expiries, strikes, target_T, target_K):
    expected_shape = (len(expiries), len(strikes))
    if iv_surface.shape != expected_shape:
        raise ValueError(f"iv_surface shape must be {expected_shape}, got {iv_surface.shape}")
    if len(expiries) < 2 or len(strikes) < 2:
        raise ValueError("bilinear interpolation requires at least two expiries and two strikes")
    if not _is_strictly_increasing(expiries):
        raise ValueError("expiries must be strictly increasing")
    if not _is_strictly_increasing(strikes):
        raise ValueError("strikes must be strictly increasing")

    # Step 1: validate target point is inside the grid
    if target_T < expiries[0] or target_T > expiries[-1]:
        raise ValueError(f"target_T is outside the bound")
    if target_K < strikes[0] or target_K > strikes[-1]:
        raise ValueError(f"target_K is outside the bound")

    # Step 2: find bracketing indices for target_T and target_K
    i_high = bisect.bisect_right(expiries, target_T)
    i_high = min(i_high, len(expiries) - 1)
    i_low = i_high - 1

    j_high = bisect.bisect_right(strikes, target_K)
    j_high = min(j_high, len(strikes) - 1)
    j_low = j_high - 1

    # Step 3: compute weights
    w_T = (target_T - expiries[i_low]) / (expiries[i_high] - expiries[i_low])
    w_K = (target_K - strikes[j_low]) / (strikes[j_high] - strikes[j_low])

    # Step 4: interpolate along K at T_low and T_high, then interpolate along T
    iv_at_T_low = (
        iv_surface[i_low, j_low]
        + (iv_surface[i_low, j_high] - iv_surface[i_low, j_low]) * w_K
    )
    iv_at_T_high = (
        iv_surface[i_high, j_low]
        + (iv_surface[i_high, j_high] - iv_surface[i_high, j_low]) * w_K
    )

    return iv_at_T_low + (iv_at_T_high - iv_at_T_low) * w_T
