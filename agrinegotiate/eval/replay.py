"""Record one episode as a JSON-friendly dict: public and private parameters, the reference
price lines, the full event log, and the outcome. Used by the dashboard and its API."""
from __future__ import annotations

import json
import pathlib

import numpy as np

from .. import metrics, scenarios, yardsticks
from ..agents import scripted as S
from ..env.core import Market, F, B
from ..mechanisms.fee import FeeMechanism, FloorMechanism

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTED = ["ZIC", "Boulware", "Linear", "Conceder", "TitForTat", "OracleFairSplit", "TakeFallback"]


def make_mechanism(spec: dict | None):
    """spec: None | {"kind": "fee", "tau": .., "lam": .., ["observable": bool, "kappa": ..]} | {"kind": "floor", "kappa": ..}"""
    if not spec or spec.get("kind") in (None, "none"):
        return None
    if spec["kind"] == "floor":
        return FloorMechanism(float(spec["kappa"]))
    return FeeMechanism(tau=float(spec.get("tau", 0.4)), lam=float(spec.get("lam", 1.0)),
                        observable=bool(spec.get("observable", False)), kappa=spec.get("kappa"))


def make_agents(sc, farmer: str, buyer: str):
    """Agent names: a scripted class name, or 'run:<path under results/>' for a trained checkpoint
    (the same run supplies whichever side asks for it). Returns (farmer, buyer, delay_aware, lam_fair)."""
    out, delay_aware, lam = {}, True, None
    for role, name in ((F, farmer), (B, buyer)):
        if name.startswith("run:"):
            import torch
            from ..agents.ppo import load_agents
            path = ROOT / "results" / name[4:] / "checkpoint.pt"
            ck = torch.load(path, weights_only=True)
            out[role] = load_agents(path, sc)[role]
            delay_aware = bool(ck.get("delay_aware", True))
            if role == B:
                lam = float(ck.get("lam_fair", 0.0))
        else:
            out[role] = S.make(name, role, sc)
    return out[F], out[B], delay_aware, lam


def record_episode(sc, farmer, buyer, seed: int, mechanism=None, delay_aware: bool = True) -> dict:
    env = Market(sc, mechanism=mechanism, delay_aware=delay_aware, log_events=True)
    env.reset(seed)
    farmer.reset(seed)
    buyer.reset(seed)
    while not env.done:
        obs, mask, active = env.observe()
        env.step(np.stack([farmer.act(env, obs[F], mask[F], active[F]),
                           buyer.act(env, obs[B], mask[B], active[B])]), mask)
    res = env.result()
    T, rho = sc.T, sc.rho
    v, o_f, o_b = float(env.v[0]), float(env.o_f[0]), float(env.o_b[0])
    ts = list(range(T + 1))
    lines = {
        "nash": [yardsticks.nash_price(v, o_f, o_b, rho, t) for t in ts],
        "farmer_fallback": [o_f * rho ** t for t in ts],
        "buyer_walkaway": [v * rho ** t - o_b for t in ts],
    }
    if isinstance(mechanism, FeeMechanism):
        lines["target"] = [mechanism.target(env, 0, 0, t) for t in ts]
    if getattr(mechanism, "floor_kappa", None) is not None:
        lines["floor"] = [mechanism.floor_kappa * env.level * rho ** t for t in ts]
    m = metrics.aggregate([res])
    return {
        "scenario": sc.name, "seed": int(seed), "n_farmers": sc.n_farmers, "n_buyers": sc.n_buyers,
        "T": T, "rho": rho, "ttl": sc.ttl, "duty": sc.duty, "level": env.level,
        "v": env.v.tolist(), "o_f": env.o_f.tolist(), "o_b": env.o_b.tolist(),
        "deadline": env.deadline.tolist(), "p_lo": sc.p_lo, "p_hi": sc.p_hi,
        "lines": lines, "events": res.events, "deals": res.deals,
        "farmer_gain": res.farmer_gain.tolist(), "buyer_gain": res.buyer_gain.tolist(),
        "price_received": res.price_received.tolist(), "fees": res.buyer_fees.tolist(),
        "g_star": res.g_star, "E": m.get("E"), "FGC": m.get("FGC"),
        "farmer": getattr(farmer, "name", "?"), "buyer": getattr(buyer, "name", "?"),
    }


def record(scenario: str, farmer: str, buyer: str, seed: int, mechanism: dict | None = None,
           overrides: dict | None = None) -> dict:
    sc = scenarios.get(scenario).with_(**(overrides or {}))
    f, b, delay_aware, lam = make_agents(sc, farmer, buyer)
    if mechanism and mechanism.get("kind") == "fee" and "lam" not in mechanism and lam is not None:
        mechanism = dict(mechanism, lam=lam)                 # use the fee rate the buyer was trained with
    ep = record_episode(sc, f, b, seed, make_mechanism(mechanism), delay_aware)
    ep["farmer"], ep["buyer"], ep["mechanism"] = farmer, buyer, mechanism
    return json.loads(json.dumps(ep, default=float))
