import pytest

from agrinegotiate import scenarios
from agrinegotiate.agents import scripted as S
from agrinegotiate.env.core import Market, F, B
from agrinegotiate.eval.evaluate import evaluate
from agrinegotiate.mechanisms.fee import FeeMechanism, FloorMechanism

SYM = scenarios.get("SYM")


def test_oracle_shortfall_matches_worked_example():
    mech = FeeMechanism(tau=0.4, lam=1.0)
    env = Market(SYM, mechanism=mech)
    env.reset(0)
    env.v[:], env.o_f[:], env.o_b[:] = 1.9, 0.9, 0.2        # the worked example, scaled by 1/1000
    assert mech.target(env, 0, 0, 0) == pytest.approx(1.22)
    assert mech.shortfall(env, 0, 0, 0.95, 0) == pytest.approx(0.27)
    assert mech.shortfall(env, 0, 0, 1.30, 0) == 0.0


def test_dual_update_moves_towards_constraint_and_is_clipped():
    mech = FeeMechanism(tau=0.4, lam=0.0)
    assert mech.dual_update(0.30) == pytest.approx(0.05 * (0.30 - 0.02))
    assert FeeMechanism(lam=0.0).dual_update(0.0) == 0.0             # cannot go negative
    assert FeeMechanism(lam=4.99).dual_update(10.0) == 5.0           # capped


def test_fee_is_charged_to_hardball_buyer_but_not_counted_as_lost_welfare():
    plain = evaluate(SYM, S.make("Boulware", F, SYM), S.make("HardballBuyer", B, SYM), 100)
    fee = evaluate(SYM, S.make("Boulware", F, SYM), S.make("HardballBuyer", B, SYM), 100,
                   mechanism=FeeMechanism(tau=0.4, lam=1.0))
    assert fee["E"] == pytest.approx(plain["E"])                     # scripted behaviour is unchanged
    assert fee["fees_over_gstar"] > 0 and fee["BGC_net_of_fees"] < fee["BGC"]
    assert fee["j_c"] > 0.02


def test_price_floor_removes_low_bids():
    mech = FloorMechanism(kappa=0.6)
    env = Market(SYM, mechanism=mech)
    env.reset(1)
    _, mask, _ = env.observe()
    allowed = env.prices[mask[B, 0, 0, :env.K]]
    assert allowed.min() >= 0.6 * env.level - 1e-9
    r = evaluate(SYM, S.make("Boulware", F, SYM), S.make("HardballBuyer", B, SYM), 100, mechanism=mech)
    assert r["price_share"] >= 0.6 - 1e-9 or r["agreement_rate"] < 1.0
