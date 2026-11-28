"""Scripted negotiators.

Every agent except the two reference points sees only its own observation vector
and action mask, exactly like a learned policy. `OracleFairSplit` is privileged by
design: it is a reference point, not a competitor.
"""
from __future__ import annotations

import numpy as np

from .. import yardsticks
from ..env import core as C
from ..env.core import F, B

EPS = 1e-9


class ScriptedAgent:
    """Base class: one decision per open thread, from that thread's observation and mask."""
    name = "scripted"

    def __init__(self, role: int, sc, **kw):
        self.role, self.sc = role, sc
        self.prices = np.linspace(sc.p_lo, sc.p_hi, sc.n_bins)
        self.K = sc.n_bins
        self.A_ACCEPT, self.A_WALK, self.A_WAIT = self.K, self.K + 1, self.K + 2
        self.ttl = sc.ttl if sc.ttl is not None else sc.T
        self.start = sc.p_hi if role == F else sc.p_lo

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng([seed, self.role, 7919])
        self.mem: dict = {}

    def act(self, env, obs, mask, active) -> np.ndarray:
        acts = np.full(active.shape, self.A_WAIT, dtype=int)
        for i, j in zip(*np.nonzero(active)):
            acts[i, j] = self.act_thread((i, j), obs[i, j], mask[i, j])
        return acts

    # helpers
    def _bin_for(self, target: float, m: np.ndarray):
        """Valid bin closest to the target without being worse for me than the target."""
        valid = np.nonzero(m[:self.K])[0]
        if valid.size == 0:
            return None
        if self.role == F:
            ok = valid[self.prices[valid] >= target - EPS]
            return int(ok.min()) if ok.size else int(valid.max())
        ok = valid[self.prices[valid] <= target + EPS]
        return int(ok.max()) if ok.size else int(valid.min())

    def _beats(self, cp_price: float, mine: float) -> bool:
        return cp_price >= mine - EPS if self.role == F else cp_price <= mine + EPS

    def _offer_or_wait(self, b: int, o: np.ndarray) -> int:
        """Send bin b unless the same offer is already live and not about to expire."""
        same = o[C.O_OWN_LIVE] > 0.5 and abs(o[C.O_OWN_PRICE] - self.prices[b]) < 1e-6
        age = round(float(o[C.O_OWN_AGE]) * self.sc.T)
        if same and age < self.ttl - 1:
            return self.A_WAIT
        return b

    def _decide(self, b, o, m) -> int:
        if b is None:
            return self.A_WAIT
        if m[self.A_ACCEPT] and self._beats(float(o[C.O_CP_PRICE]), float(self.prices[b])):
            return self.A_ACCEPT
        return self._offer_or_wait(b, o)

    def act_thread(self, key, o, m) -> int:
        raise NotImplementedError


class TimeDependent(ScriptedAgent):
    """Faratin et al. time-dependent tactic. beta < 1 Boulware, 1 linear, > 1 conceder."""

    def __init__(self, role, sc, beta: float = 1.0, start: float | None = None, name: str | None = None, **kw):
        super().__init__(role, sc)
        self.beta = beta
        if start is not None:
            self.start = start
        self.name = name or f"TD(beta={beta:g})"

    def target(self, o) -> float:
        t = round(float(o[C.O_TIME]) * self.sc.T)
        left = round(float(o[C.O_DEADLINE]) * self.sc.T)
        horizon = max(1, t + left - 2)             # two steps are needed to close after an offer
        alpha = min(1.0, t / horizon) ** (1.0 / self.beta)
        reserve = float(o[C.O_WALKAWAY])
        return self.start + alpha * (reserve - self.start)

    def act_thread(self, key, o, m) -> int:
        return self._decide(self._bin_for(self.target(o), m), o, m)


TACTIC_BETA = {"Boulware": 0.2, "Linear": 1.0, "Conceder": 5.0}     # <1 holds firm, 1 steady, >1 concedes early


class ZIC(ScriptedAgent):
    """Zero-intelligence with constraint: random individually rational offers (Gode and Sunder)."""
    name = "ZI-C"

    def act_thread(self, key, o, m) -> int:
        valid = np.nonzero(m[:self.K])[0]
        if valid.size == 0:
            return self.A_WAIT
        b = int(self.rng.choice(valid))
        if m[self.A_ACCEPT] and self._beats(float(o[C.O_CP_PRICE]), float(self.prices[b])):
            return self.A_ACCEPT
        # hold each random offer for three steps so the counterpart has time to accept it
        if o[C.O_OWN_LIVE] > 0.5 and round(float(o[C.O_OWN_AGE]) * self.sc.T) < 3:
            return self.A_WAIT
        return b


class RandomValid(ScriptedAgent):
    """Uniform over all currently valid actions. Used only for stress tests."""
    name = "RandomValid"

    def act_thread(self, key, o, m) -> int:
        return int(self.rng.choice(np.nonzero(m)[0]))


class TitForTat(ScriptedAgent):
    """Matches the counterpart's last concession; never concedes less than Boulware would."""
    name = "TitForTat"

    def __init__(self, role, sc, **kw):
        super().__init__(role, sc)
        self._floor_tactic = TimeDependent(role, sc, beta=0.2)

    def act_thread(self, key, o, m) -> int:
        sign = -1.0 if self.role == F else 1.0      # direction in which I concede
        boul = self._floor_tactic.target(o)
        has_cp = o[C.O_CP_HAS] > 0.5
        cp = float(o[C.O_CP_PRICE])
        prev_cp, mine = self.mem.get(key, (None, None))
        if mine is None:
            target = boul
        else:
            step = abs(cp - prev_cp) if (has_cp and prev_cp is not None) else 0.0
            target = mine + sign * step
            target = min(target, boul) if self.role == F else max(target, boul)
        reserve = float(o[C.O_WALKAWAY])
        target = max(target, reserve) if self.role == F else min(target, reserve)
        b = self._bin_for(target, m)
        if b is not None:
            self.mem[key] = (cp if has_cp else prev_cp, float(self.prices[b]))
        return self._decide(b, o, m)


class HardballBuyer(ScriptedAgent):
    """Exploitation probe: bids one bin above its estimate of the farmer's fallback and never moves."""
    name = "Hardball"

    def __init__(self, role, sc, **kw):
        assert role == B, "the hardball probe is a buyer"
        super().__init__(role, sc)

    def act_thread(self, key, o, m) -> int:
        est = self.sc.theta * float(o[C.O_SIGNAL]) * float(o[C.O_FRESH])
        idx = min(self.K - 1, int(np.searchsorted(self.prices, est - EPS)) + 1)
        return self._decide(self._bin_for(float(self.prices[idx]), m), o, m)


class TakeFallback(ScriptedAgent):
    """Reference floor: everyone walks at step 0."""
    name = "TakeFallback"

    def act_thread(self, key, o, m) -> int:
        return self.A_WALK


class OracleFairSplit(ScriptedAgent):
    """Reference ceiling (privileged): optimal matching, Nash price, fastest possible close."""
    name = "OracleFair"

    def act(self, env, obs, mask, active) -> np.ndarray:
        acts = np.full(active.shape, self.A_WAIT, dtype=int)
        if "pairs" not in self.mem:
            self.mem["pairs"] = set(yardsticks.max_gains(env.v, env.o_f, env.o_b, env.cap)[1])
        for i, j in zip(*np.nonzero(active)):
            m = mask[i, j]
            if (i, j) not in self.mem["pairs"]:
                acts[i, j] = self.A_WALK
            elif self.role == F:
                p = yardsticks.nash_price(env.v[j], env.o_f[i], env.o_b[j], env.sc.rho, env.t + 2)
                valid = np.nonzero(m[:self.K])[0]
                stale = env.sent_id[F, i, j] < 0 or env.t >= env.sent_t[F, i, j] + self.ttl - 1
                if p is not None and valid.size and stale and not env.pending_accept[B, i, j]:
                    acts[i, j] = int(valid[np.argmin(np.abs(self.prices[valid] - p))])
            elif m[self.A_ACCEPT]:
                acts[i, j] = self.A_ACCEPT
        return acts


REGISTRY = {c.__name__: c for c in
            [ZIC, RandomValid, TitForTat, HardballBuyer, TakeFallback, OracleFairSplit]}


def make(name: str, role: int, sc, **kw) -> ScriptedAgent:
    if name.startswith("TD:"):                     # e.g. "TD:0.5:1.0" -> beta, start
        parts = name.split(":")
        return TimeDependent(role, sc, beta=float(parts[1]),
                             start=float(parts[2]) if len(parts) > 2 else None)
    if name in TACTIC_BETA:
        return TimeDependent(role, sc, beta=TACTIC_BETA[name], name=name)
    return REGISTRY[name](role, sc, **kw)
