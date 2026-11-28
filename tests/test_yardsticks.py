"""Unit tests for the closed-form reference points."""
import numpy as np
import pytest

from agrinegotiate import yardsticks as y


def test_worked_example():
    v, o_f, o_b = 1900.0, 900.0, 200.0
    assert y.gains(v, o_f, o_b) == pytest.approx(800)
    assert y.nash_price(v, o_f, o_b) == pytest.approx(1300)
    assert y.gain_share(1300, v, o_f, o_b) == pytest.approx(0.5)
    assert y.gain_share(950, v, o_f, o_b) == pytest.approx(0.0625)
    assert y.target_price(v, o_f, o_b, 0.4) == pytest.approx(1220)


def test_rubinstein_example_and_delay_table():
    assert round(y.rubinstein_share(0.90, 0.98, farmer_first=True), 3) == 0.169
    assert round(y.rubinstein_share(0.90, 0.98, farmer_first=False), 3) == 0.153
    assert round(y.rubinstein_limit(0.90, 0.98), 3) == 0.161
    table = {1: (0.169, 0.153), 3: (0.187, 0.137), 5: (0.206, 0.122)}
    for k, (ff, bf) in table.items():
        assert round(y.rubinstein_share(0.90, 0.98, True, k), 3) == ff
        assert round(y.rubinstein_share(0.90, 0.98, False, k), 3) == bf


def test_symmetric_rubinstein_is_near_half():
    assert y.rubinstein_limit(0.95, 0.95) == pytest.approx(0.5)


def test_max_gains_matches_bruteforce_hand_case():
    v, o_b, cap = [0.95, 0.90], [0.10, 0.05], [1, 1]
    o_f = [0.45, 0.50, 0.30]
    g, pairs = y.max_gains(v, o_f, o_b, cap)
    assert g == pytest.approx(y.max_gains_bruteforce(v, o_f, o_b, cap))
    assert len(pairs) == 2


def test_max_gains_matches_bruteforce_random():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n, m = rng.integers(1, 5), rng.integers(1, 3)
        v = rng.uniform(0.4, 1.1, m); o_b = rng.uniform(0.0, 0.2, m)
        o_f = rng.uniform(0.2, 0.9, n); cap = rng.integers(1, 3, m)
        g, _ = y.max_gains(v, o_f, o_b, cap)
        assert g == pytest.approx(y.max_gains_bruteforce(v, o_f, o_b, cap))


def test_shares_sum_to_one():
    rng = np.random.default_rng(1)
    for _ in range(100):
        v, o_f, o_b = rng.uniform(0.8, 1.1), rng.uniform(0.2, 0.5), rng.uniform(0, 0.2)
        rho, t = rng.uniform(0.95, 1.0), int(rng.integers(0, 10))
        g = y.gains(v, o_f, o_b, rho, t)
        if g <= 0:
            continue
        p = rng.uniform(o_f * rho ** t, v * rho ** t - o_b)
        phi = y.gain_share(p, v, o_f, o_b, rho, t)
        buyer = (v * rho ** t - p - o_b) / g
        assert phi + buyer == pytest.approx(1.0)


def test_no_trade_possible_is_reported_not_divided():
    assert y.nash_price(0.5, 0.45, 0.10) is None
    assert y.gain_share(0.5, 0.5, 0.45, 0.10) is None
    assert y.target_price(0.5, 0.45, 0.10, 0.4) is None
    assert y.max_gains([0.5], [0.45], [0.10], [1]) == (0.0, [])
