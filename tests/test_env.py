"""Protocol, delay-model and metric-identity tests ."""
import os

import numpy as np
import pytest

from agrinegotiate import metrics, scenarios
from agrinegotiate.agents import scripted as S
from agrinegotiate.env.core import Market, F, B
from agrinegotiate.eval.evaluate import evaluate, run_episode

SYM = scenarios.get("SYM")


def manual_env(**kw):
    """1x1 market whose farmer link is controlled by hand."""
    env = Market(SYM.with_(duty=0.5, **kw))
    env.reset(123)
    env.p_fail, env.p_rec = 0.0, 0.0        # link stays wherever the test puts it
    env.link[:] = True
    return env


def acts(env, f=None, b=None):
    a = np.full((2, 1, 1), env.A_WAIT, dtype=int)
    if f is not None: a[F, 0, 0] = f
    if b is not None: a[B, 0, 0] = b
    return a


def mid_bin(env):
    """A bin both sides may legally offer in the symmetric scenario."""
    _, mask, _ = env.observe()
    both = np.nonzero(mask[F, 0, 0, :env.K] & mask[B, 0, 0, :env.K])[0]
    return int(both[len(both) // 2])


def test_message_takes_one_step_and_deal_closes_on_delivery():
    env = manual_env()
    k = mid_bin(env)
    env.step(acts(env, b=k))                          # t=0: buyer bids
    assert env.seen_id[B, 0, 0] >= 0 and env.t == 1   # delivered at t=1
    _, mask, _ = env.observe()
    assert mask[F, 0, 0, env.A_ACCEPT]
    env.step(acts(env, f=env.A_ACCEPT))               # t=1: farmer accepts
    assert env.done and env.deals[0]["t"] == 2 and env.deals[0]["price"] == pytest.approx(env.prices[k])


def test_late_accept_fails():
    env = manual_env(ttl=3)
    k = mid_bin(env)
    env.step(acts(env, b=k))                          # bid sent at t=0, valid through t=3
    env.step(acts(env, f=env.A_ACCEPT))               # accept sent at t=1 ...
    assert env.done                                   # (sanity: with the link up it closes)
    env = manual_env(ttl=3)
    env.step(acts(env, b=k))
    env.link[:] = False                               # ... but now the link drops first
    env.step(acts(env, f=env.A_ACCEPT))
    for _ in range(4):
        env.step(acts(env))
    env.link[:] = True
    env.step(acts(env))                               # accept finally delivered, too late
    assert not env.deals and env.failed_accepts == 1 and not env.pending_accept.any()


def test_cancelled_offer_cannot_be_accepted():
    env = manual_env()
    k = mid_bin(env)
    env.step(acts(env, b=k))
    env.step(acts(env, f=env.A_ACCEPT, b=k - 1))      # buyer replaces its bid as the farmer accepts
    assert not env.deals and env.failed_accepts == 1
    assert env.seen_price[B, 0, 0] == pytest.approx(env.prices[k - 1])


def test_store_and_forward_keeps_order_and_latest_offer_wins():
    env = manual_env()
    _, mask, _ = env.observe()
    valid = np.nonzero(mask[F, 0, 0, :env.K])[0]
    env.link[:] = False
    env.step(acts(env, f=int(valid[-1])))             # ask sent while the link is down
    env.step(acts(env, f=int(valid[-2])))             # replaced one step later
    assert env.seen_id[F, 0, 0] < 0 and len(env.queue[0]) == 2
    env.link[:] = True
    env.step(acts(env))
    assert env.seen_price[F, 0, 0] == pytest.approx(env.prices[valid[-2]]) and not env.queue[0]


def test_drop_mode_loses_messages_sent_while_off():
    env = manual_env(drop_mode=True)
    _, mask, _ = env.observe()
    k = int(np.nonzero(mask[F, 0, 0, :env.K])[0][-1])
    env.link[:] = False
    env.step(acts(env, f=k))
    env.link[:] = True
    env.step(acts(env)); env.step(acts(env))
    assert env.seen_id[F, 0, 0] < 0


def test_walk_at_step_zero_gives_exactly_the_fallback():
    r = evaluate(SYM, S.make("TakeFallback", F, SYM), S.make("TakeFallback", B, SYM), 50)
    assert r["E"] == pytest.approx(0.0) and r["agreement_rate"] == 0.0


def test_oracle_fair_split_is_the_ceiling():
    r = evaluate(SYM, S.make("OracleFairSplit", F, SYM), S.make("OracleFairSplit", B, SYM), 200)
    assert r["E"] > 0.95 and r["agreement_rate"] == 1.0
    assert abs(r["S_F"] - 0.5) < 0.03          # limited by the 0.05 price grid


def test_same_seed_same_episode():
    sc = scenarios.get("RURAL-DTN")
    a = run_episode(sc, S.make("Linear", F, sc), S.make("TitForTat", B, sc), 7)
    b = run_episode(sc, S.make("Linear", F, sc), S.make("TitForTat", B, sc), 7)
    assert np.allclose(a.farmer_gain, b.farmer_gain) and a.deals == b.deals


@pytest.mark.parametrize("name", list(scenarios.SCENARIOS))
def test_identity_and_rationality_under_random_play(name):
    """E = FGC + BGC, and no deal ever closes with negative gains, under random valid actions."""
    sc = scenarios.get(name)
    n = int(os.environ.get("AGRI_RANDOM_EPISODES", 150))
    res = [run_episode(sc, S.make("RandomValid", F, sc), S.make("RandomValid", B, sc), s) for s in range(n)]
    m = metrics.aggregate(res)
    assert m["E"] == pytest.approx(m["FGC"] + m["BGC"])
    for r in res:
        for d in r.deals:
            assert d["g_t"] >= -1e-9 and (d["phi"] is None or -1e-6 <= d["phi"] <= 1 + 1e-6)
        total = r.farmer_gain.sum() + r.buyer_gain.sum()
        assert total <= r.g_star + 1e-9          # nobody can beat the efficiency ceiling


def test_inequality_indices_reject_negative_input():
    with pytest.raises(AssertionError):
        metrics.gini([-1.0, 2.0])
    assert metrics.jain([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert metrics.gini([1.0, 1.0]) == pytest.approx(0.0)
