import numpy as np
import pytest
from iv_surface.solver import bs_price, solve_iv, build_surface, interpolate_iv

def test_solve_iv_round_trip():
    spot_price = 100
    K = 100
    T = 1
    r = 0.03
    sigma = 0.2
    flag = "call"

    option_price = bs_price(spot_price, K, T, r, sigma, flag)
    implied_vol = solve_iv(option_price, spot_price, K, T, r, flag)
    assert abs(implied_vol - sigma) < 1e-4

def test_solve_iv_bad_price():
    with pytest.raises(ValueError):
        solve_iv(101, 100, 100, 1, 0.03, "call")

def test_solve_iv_rejects_non_finite_price():
    with pytest.raises(ValueError):
        solve_iv(np.inf, 100, 100, 1, 0.03, "call")

    with pytest.raises(ValueError):
        solve_iv(-np.inf, 100, 100, 1, 0.03, "call")

    with pytest.raises(ValueError):
        solve_iv(np.nan, 100, 100, 1, 0.03, "call")

def test_solve_iv_returns_lower_bracket_endpoint():
    spot_price = 100
    K = 100
    T = 1
    r = 0.03
    sigma_low = 1e-6
    flag = "call"

    option_price = bs_price(spot_price, K, T, r, sigma_low, flag)
    implied_vol = solve_iv(option_price, spot_price, K, T, r, flag)

    assert abs(implied_vol - sigma_low) < 1e-4

def test_solve_iv_returns_upper_bracket_endpoint():
    spot_price = 100
    K = 100
    T = 1
    r = 0.03
    sigma_high = 10.0
    flag = "call"

    option_price = bs_price(spot_price, K, T, r, sigma_high, flag)
    implied_vol = solve_iv(option_price, spot_price, K, T, r, flag)

    assert abs(implied_vol - sigma_high) < 1e-4

def test_solve_iv_rejects_expired_option():
    with pytest.raises(ValueError):
        solve_iv(0, 100, 100, 0, 0.03, "call")

def test_solve_iv_rejects_non_finite_inputs():
    cases = [
        {"option_price": 10, "spot_price": np.nan, "K": 100, "T": 1, "r": 0.03, "flag": "call"},
        {"option_price": 10, "spot_price": 100, "K": np.nan, "T": 1, "r": 0.03, "flag": "call"},
        {"option_price": 10, "spot_price": 100, "K": 100, "T": np.nan, "r": 0.03, "flag": "call"},
        {"option_price": 10, "spot_price": 100, "K": 100, "T": 1, "r": np.nan, "flag": "call"},
    ]

    for kwargs in cases:
        with pytest.raises(ValueError):
            solve_iv(**kwargs)

def test_build_surface_round_trip():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    sigma = 0.2
    spot_price = 100
    r = 0

    option_price_grid = np.array([
        [bs_price(spot_price, K, T, r, sigma, "call") for K in strikes]
        for T in expiries
    ])

    iv_surface = build_surface(option_price_grid, spot_price, expiries, strikes, r)

    assert np.allclose(sigma, iv_surface, rtol=0.0, atol=1e-4)

def test_build_surface_wrong_price_shape():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    option_price_grid = np.ones((2, 3))

    with pytest.raises(ValueError):
        build_surface(
            option_price_grid, spot_price=100, expiries=expiries, strikes=strikes
        )

def test_interpolate_iv_flat_surface_from_prices():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    sigma = 0.2
    spot_price = 100
    r = 0

    option_price_grid = np.array([
        [bs_price(spot_price, K, T, r, sigma, "call") for K in strikes]
        for T in expiries
    ])
    iv_surface = build_surface(option_price_grid, spot_price, expiries, strikes, r)

    # query a point strictly inside the grid
    iv = interpolate_iv(iv_surface, expiries, strikes, target_T=0.18, target_K=97)
    assert abs(iv - sigma) < 1e-4

def test_interpolate_iv_non_flat_surface():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    iv_surface = np.array([
        [0.18, 0.20, 0.22],
        [0.19, 0.21, 0.23],
        [0.20, 0.22, 0.24],
    ])

    iv = interpolate_iv(iv_surface, expiries, strikes, target_T=0.175, target_K=95)

    assert abs(iv - 0.195) < 1e-4

def test_interpolate_iv_out_of_bounds():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    iv_surface = np.full((3, 3), 0.2)

    with pytest.raises(ValueError):
        interpolate_iv(iv_surface, expiries, strikes, target_T=0.25, target_K=85)

    with pytest.raises(ValueError):
        interpolate_iv(iv_surface, expiries, strikes, target_T=0.05, target_K=100)

def test_interpolate_iv_wrong_surface_shape():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    iv_surface = np.full((2, 3), 0.2)

    with pytest.raises(ValueError):
        interpolate_iv(iv_surface, expiries, strikes, target_T=0.25, target_K=100)

def test_interpolate_iv_rejects_singleton_grid_dimension():
    with pytest.raises(ValueError):
        interpolate_iv(np.array([[0.2, 0.3]]), [1], [100, 110], target_T=1, target_K=105)

    with pytest.raises(ValueError):
        interpolate_iv(np.array([[0.2], [0.3]]), [1, 2], [100], target_T=1.5, target_K=100)

def test_interpolate_iv_rejects_unsorted_or_duplicate_grid():
    iv_surface = np.full((3, 3), 0.2)

    with pytest.raises(ValueError):
        interpolate_iv(iv_surface, [0.1, 0.25, 0.5], [90, 110, 100], target_T=0.25, target_K=100)

    with pytest.raises(ValueError):
        interpolate_iv(iv_surface, [0.1, 0.25, 0.25], [90, 100, 110], target_T=0.25, target_K=100)

def test_interpolate_iv_exact_grid_point():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    iv_surface = np.array([
        [0.18, 0.20, 0.22],
        [0.19, 0.21, 0.23],
        [0.20, 0.22, 0.24],
    ])
    iv = interpolate_iv(iv_surface, expiries, strikes, target_T=0.25, target_K=100)

    assert abs(iv - 0.21) < 1e-4

def test_interpolate_iv_upper_boundary_grid_point():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    iv_surface = np.array([
        [0.18, 0.20, 0.22],
        [0.19, 0.21, 0.23],
        [0.20, 0.22, 0.24],
    ])

    iv = interpolate_iv(iv_surface, expiries, strikes, target_T=0.5, target_K=110)

    assert abs(iv - 0.24) < 1e-4

def test_interpolate_iv_lower_boundary_grid_point():
    strikes = [90, 100, 110]
    expiries = [0.1, 0.25, 0.5]
    iv_surface = np.array([
        [0.18, 0.20, 0.22],
        [0.19, 0.21, 0.23],
        [0.20, 0.22, 0.24],
    ])

    iv = interpolate_iv(iv_surface, expiries, strikes, target_T=0.1, target_K=90)

    assert abs(iv - 0.18) < 1e-4
