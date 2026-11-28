"""Market environment core.

One `Market` simulates one episode: N farmers, M buyers, one negotiation thread
per (farmer, buyer) pair. Farmers reach the platform over an intermittent link.

Step order, as seen from the caller:
    reset()                      -> state at t = 0
    observe()                    -> observations and action masks at t
    step(actions)                -> messages enqueued with send step t, walk-aways settle,
                                    t += 1, links update, deliverable messages are delivered
                                    (deals close), deadlines apply.

Sides are indexed 0 = farmer, 1 = buyer throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import yardsticks
from ..scenarios import Scenario

F, B = 0, 1
OBS_DIM = 22

# observation indices
(O_ROLE, O_TIME, O_DEADLINE, O_WALKAWAY, O_SIGNAL, O_FRESH, O_OWN_PRICE, O_OWN_LIVE, O_OWN_AGE,
 O_OWN_DELIVERED, O_CP_PRICE, O_CP_HAS, O_CP_AGE, O_CP_VALID, O_LINK, O_ROUNDS, O_ALT_PRICE,
 O_ALT_HAS, O_N_OPEN, O_CAPACITY, O_LAMBDA, O_TAU) = range(OBS_DIM)

# features hidden from delay-unaware agents
DELAY_FEATURES = [O_OWN_AGE, O_OWN_DELIVERED, O_CP_AGE, O_CP_VALID, O_LINK]


@dataclass
class EpisodeResult:
    """Everything the metrics module needs about one finished episode."""
    scenario: str
    seed: int
    level: float
    g_star: float
    n_opt_pairs: int
    farmer_gain: np.ndarray        # U_F - o_F per farmer
    buyer_gain: np.ndarray         # sum of unit payoffs - k*o_B per buyer, before fees
    buyer_fees: np.ndarray         # fees charged per buyer
    price_received: np.ndarray     # per farmer: deal price or decayed fallback price
    sold: np.ndarray               # per farmer: True if sold on the platform
    deals: list = field(default_factory=list)   # dicts: i, j, price, t, phi, shortfall
    failed_accepts: int = 0
    decay_loss: float = 0.0
    events: Optional[list] = None


def _best_other(x: np.ndarray, axis: int) -> np.ndarray:
    """For every entry: the largest value among the *other* entries along `axis` (-inf if none)."""
    srt = np.sort(x, axis=axis)
    top = np.take(srt, [-1], axis=axis)
    second = np.take(srt, [-2], axis=axis) if x.shape[axis] > 1 else np.full_like(top, -np.inf)
    return np.where(x == top, second, top)


class Market:
    def __init__(self, sc: Scenario, mechanism=None, delay_aware: bool = True, log_events: bool = False):
        self.sc = sc
        self.mech = mechanism
        self.delay_aware = delay_aware
        self.log_events = log_events
        self.prices = np.linspace(sc.p_lo, sc.p_hi, sc.n_bins)
        self.K = sc.n_bins
        self.A_ACCEPT, self.A_WALK, self.A_WAIT = self.K, self.K + 1, self.K + 2
        self.n_actions = self.K + 3
        self.N, self.M = sc.n_farmers, sc.n_buyers
        if sc.duty >= 1.0:
            self.p_rec, self.p_fail = 1.0, 0.0
        else:
            self.p_rec = 1.0 / sc.outage_len
            self.p_fail = self.p_rec * (1.0 - sc.duty) / sc.duty
            assert 0.0 <= self.p_fail <= 1.0, "duty cycle / outage length combination is infeasible"

    # reset
    def reset(self, seed: int) -> None:
        sc, N, M = self.sc, self.N, self.M
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        rng = self.rng
        self.level = float(rng.choice(sc.level_samples)) if sc.level_samples else float(rng.uniform(sc.level_lo, sc.level_hi))
        self.v = self.level * (1.0 - rng.uniform(sc.margin_lo, sc.margin_hi, M))
        self.o_b = np.full(M, sc.o_b * self.level)
        theta = rng.uniform(sc.theta - sc.theta_spread, sc.theta + sc.theta_spread, N)
        self.o_f = theta * self.level
        if sc.deadline_lo is None:
            self.deadline = np.full(N, sc.T, dtype=int)
        else:
            self.deadline = rng.integers(sc.deadline_lo, sc.deadline_hi + 1, N)
        self.sig_f = self.level * np.exp(rng.normal(0.0, sc.sigma_info, N)) if sc.sigma_info > 0 \
            else np.full(N, self.level)
        self.cost_f = sc.base_cost + sc.h_farmer
        self.cost_b = sc.base_cost
        self.cap = np.full(M, sc.capacity, dtype=int)
        self.link = rng.random(N) < sc.duty if sc.duty < 1.0 else np.ones(N, dtype=bool)
        self.t = 0

        self.open = np.ones((N, M), dtype=bool)
        shape = (2, N, M)
        self.sent_price = np.zeros(shape)
        self.sent_t = np.full(shape, -1, dtype=int)
        self.sent_id = np.full(shape, -1, dtype=int)
        self.sent_delivered = np.zeros(shape, dtype=bool)
        self.seen_price = np.zeros(shape)       # seen[s] = offer from side s as delivered to the other side
        self.seen_t = np.full(shape, -1, dtype=int)
        self.seen_id = np.full(shape, -1, dtype=int)
        self.pending_accept = np.zeros(shape, dtype=bool)
        self.rounds = np.zeros((N, M), dtype=int)
        self._next_id = 0
        self.queue: list[list[dict]] = [[] for _ in range(N)]

        self.f_done = np.zeros(N, dtype=bool)
        self.f_sold = np.zeros(N, dtype=bool)
        self.f_U = np.zeros(N)
        self.f_price = np.zeros(N)
        self.b_done = np.zeros(M, dtype=bool)
        self.b_U = np.zeros(M)
        self.b_fees = np.zeros(M)
        self.b_cost_units = np.zeros(M)          # fee base (shortfall) accumulated per buyer
        self.deals: list[dict] = []
        self.failed_accepts = 0
        self.decay_loss = 0.0
        self.events: list[dict] = []
        self.g_star, pairs = yardsticks.max_gains(self.v, self.o_f, self.o_b, self.cap)
        self.n_opt_pairs = len(pairs)
        self.done = False
        self._log(kind="reset", level=self.level, link=self.link.tolist())

    # helpers
    def _log(self, **ev) -> None:
        if self.log_events:
            ev["t"] = self.t
            self.events.append(ev)

    def _validity_end(self, send_t):
        """Last step at which an offer sent at send_t can still be accepted on delivery."""
        if self.sc.ttl is None:
            return np.full_like(np.asarray(send_t), self.sc.T)
        return np.minimum(np.asarray(send_t) + self.sc.ttl, self.sc.T)

    def farmer_walkaway(self, t=None) -> np.ndarray:
        t = self.t if t is None else t
        return self.o_f * self.sc.rho ** t

    def buyer_walkaway(self, t=None) -> np.ndarray:
        t = self.t if t is None else t
        return self.v * self.sc.rho ** np.asarray(t) - self.o_b

    def _floor(self, t) -> float:
        if self.mech is not None and getattr(self.mech, "floor_kappa", None) is not None:
            return self.mech.floor_kappa * self.level * self.sc.rho ** t
        return -np.inf

    # observe
    def observe(self):
        """Returns obs[2, N, M, OBS_DIM], mask[2, N, M, n_actions], active[2, N, M]."""
        sc, N, M, t, T = self.sc, self.N, self.M, self.t, self.sc.T
        rho_t = sc.rho ** t
        obs = np.zeros((2, N, M, OBS_DIM), dtype=np.float32)
        mask = np.zeros((2, N, M, self.n_actions), dtype=bool)
        active = np.stack([self.open & ~self.f_done[:, None], self.open & ~self.b_done[None, :]])

        walk = np.stack([np.broadcast_to(self.farmer_walkaway()[:, None], (N, M)),
                         np.broadcast_to(self.buyer_walkaway()[None, :], (N, M))])
        sig = np.stack([np.broadcast_to(self.sig_f[:, None], (N, M)), np.full((N, M), self.level)])
        dl = np.stack([np.broadcast_to((self.deadline - t)[:, None], (N, M)), np.full((N, M), T - t)])
        ttl_norm = float(sc.ttl) if sc.ttl is not None else float(T)

        for s in (F, B):
            o = 1 - s
            has_own = self.sent_id[s] >= 0
            has_cp = self.seen_id[o] >= 0
            cp_valid_left = np.where(has_cp, self._validity_end(self.seen_t[o]) - t, 0)
            ob = obs[s]
            ob[..., O_ROLE] = s
            ob[..., O_TIME] = t / T
            ob[..., O_DEADLINE] = np.maximum(dl[s], 0) / T
            ob[..., O_WALKAWAY] = walk[s]
            ob[..., O_SIGNAL] = sig[s]
            ob[..., O_FRESH] = rho_t
            ob[..., O_OWN_PRICE] = np.where(has_own, self.sent_price[s], 0.0)
            ob[..., O_OWN_LIVE] = has_own
            ob[..., O_OWN_AGE] = np.where(has_own, t - self.sent_t[s], 0) / T
            ob[..., O_CP_PRICE] = np.where(has_cp, self.seen_price[o], 0.0)
            ob[..., O_CP_HAS] = has_cp
            ob[..., O_CP_AGE] = np.where(has_cp, t - self.seen_t[o], 0) / T
            ob[..., O_CP_VALID] = np.clip(cp_valid_left, 0, None) / ttl_norm
            ob[..., O_ROUNDS] = self.rounds / T
            # the agent's other threads: best counterpart offer elsewhere and share of threads still open
            offer = np.where(has_cp & self.open & (cp_valid_left > 0), self.seen_price[o], np.nan)
            ax = 1 if s == F else 0                              # axis along which this agent's threads run
            best = _best_other(np.nan_to_num(offer if s == F else -offer, nan=-np.inf), ax)
            ob[..., O_ALT_PRICE] = np.where(np.isfinite(best), best if s == F else -best, 0.0)
            ob[..., O_ALT_HAS] = np.isfinite(best)
            ob[..., O_N_OPEN] = self.open.sum(axis=ax, keepdims=True) / (M if s == F else N)
            if s == F:
                ob[..., O_OWN_DELIVERED] = self.sent_delivered[F]
                ob[..., O_LINK] = self.link[:, None]
                ob[..., O_CAPACITY] = 1.0
            else:
                ob[..., O_CAPACITY] = (self.cap / sc.capacity)[None, :]
                if self.mech is not None:
                    ob[..., O_LAMBDA] = getattr(self.mech, "lam", 0.0) / getattr(self.mech, "lam_max", 1.0)
            if self.mech is not None:
                ob[..., O_TAU] = getattr(self.mech, "tau", 0.0)

        if not self.delay_aware:
            obs[..., DELAY_FEATURES] = 0.0

        # action masks
        P = self.prices
        floor = self._floor(t)
        # farmer: never ask below own current walk-away price
        f_ok = P[None, None, :] >= (self.farmer_walkaway()[:, None, None] - 1e-9)
        # buyer: never bid above own value at the last step the bid could still be accepted,
        # so every deal is individually rational when it closes
        b_cap = self.buyer_walkaway(self._validity_end(t))                  # [M]
        b_ok = (P[None, None, :] <= b_cap[None, :, None] + 1e-9) & (P[None, None, :] >= floor - 1e-9)
        mask[F, :, :, :self.K] = f_ok
        mask[B, :, :, :self.K] = b_ok
        for s in (F, B):
            o = 1 - s
            has_cp = self.seen_id[o] >= 0
            in_time = (t + 1) <= self._validity_end(self.seen_t[o])
            if s == F:
                rational = self.seen_price[B] >= (self.farmer_walkaway(t + 1)[:, None] - 1e-9)
            else:
                worst = self.buyer_walkaway(self._validity_end(self.seen_t[F]))   # [N, M] via broadcast below
                rational = (self.seen_price[F] <= worst + 1e-9) & (self.seen_price[F] >= floor - 1e-9)
            mask[s, :, :, self.A_ACCEPT] = has_cp & in_time & rational
            mask[s, :, :, self.A_WALK] = True
            mask[s, :, :, self.A_WAIT] = True
            waiting = self.pending_accept[s]
            mask[s][waiting] = False
            mask[s][waiting, self.A_WAIT] = True
            mask[s][~active[s]] = False
            mask[s][~active[s], self.A_WAIT] = True
        return obs, mask, active

    # step
    def step(self, actions: np.ndarray, mask: Optional[np.ndarray] = None) -> bool:
        """actions[2, N, M] integer actions per side and thread. Returns done.

        Pass the mask returned by observe() to have every action checked against it.
        """
        assert not self.done
        t = self.t
        for s in (F, B):
            for i in range(self.N):
                for j in range(self.M):
                    if not self.open[i, j] or (self.f_done[i] if s == F else self.b_done[j]):
                        continue
                    a = int(actions[s, i, j])
                    if mask is not None:
                        assert mask[s, i, j, a], f"masked action {a} by side {s} on thread {(i, j)} at t={t}"
                    self._apply(s, i, j, a, t)

        self._settle_idle(t)                      # walk-aways settle at the step they happen

        # advance time, update links, deliver
        self.t = t = t + 1
        if self.p_fail > 0:
            u = self.rng.random(self.N)
            self.link = np.where(self.link, u >= self.p_fail, u < self.p_rec)
        self._log(kind="links", link=self.link.tolist())
        self._deliver(t)
        self._settle_idle(t)
        # deadlines
        for i in range(self.N):
            if not self.f_done[i] and t >= self.deadline[i]:
                self._farmer_fallback(i, t)
        self._settle_idle(t)
        if t >= self.sc.T:
            for i in range(self.N):
                if not self.f_done[i]:
                    self._farmer_fallback(i, t)
            for j in range(self.M):
                if not self.b_done[j]:
                    self._buyer_end(j, t)
        self.done = bool(self.f_done.all() and self.b_done.all())
        return self.done

    # internals
    def _apply(self, s: int, i: int, j: int, a: int, t: int) -> None:
        """One side's action on one thread: WAIT does nothing, WALK closes the thread, OFFER/ACCEPT queue a message."""
        o = 1 - s
        if a == self.A_WAIT:
            return
        if a == self.A_WALK:
            self.open[i, j] = False
            self.pending_accept[:, i, j] = False
            self._log(kind="walk", side=s, i=i, j=j)
            return
        lost = self.sc.drop_mode and not self.link[i]          # drop mode: sent while the link is down = gone
        if a == self.A_ACCEPT:
            self.pending_accept[s, i, j] = True
            msg = dict(kind="accept", side=s, i=i, j=j, ref=int(self.seen_id[o, i, j]),
                       price=float(self.seen_price[o, i, j]), send_t=t)
        else:
            self._next_id += 1
            self.sent_price[s, i, j] = self.prices[a]
            self.sent_t[s, i, j] = t
            self.sent_id[s, i, j] = self._next_id
            self.sent_delivered[s, i, j] = False
            self.rounds[i, j] += 1
            msg = dict(kind="offer", side=s, i=i, j=j, id=self._next_id, price=float(self.prices[a]), send_t=t)
        self._log(kind="send", lost=bool(lost), **{k: v for k, v in msg.items() if k != "kind"}, msg=msg["kind"])
        if lost:
            if msg["kind"] == "accept":
                self.pending_accept[s, i, j] = False
            return
        self.queue[i].append(msg)

    def _deliver(self, t: int) -> None:
        sc = self.sc
        ready = []
        for i in range(self.N):
            if sc.drop_mode or self.link[i]:
                keep = []
                for m in self.queue[i]:
                    (ready if m["send_t"] < t else keep).append(m)
                self.queue[i] = keep
        if not ready:
            return
        tie = self.rng.random(len(ready))
        order = sorted(range(len(ready)), key=lambda k: (ready[k]["send_t"], tie[k]))
        for k in order:
            m = ready[k]
            s, i, j = m["side"], m["i"], m["j"]
            o = 1 - s
            if not self.open[i, j]:
                continue
            if m["kind"] == "offer":
                if m["id"] != self.sent_id[s, i, j]:
                    continue                                  # superseded before it arrived
                self.seen_price[s, i, j] = m["price"]
                self.seen_t[s, i, j] = m["send_t"]
                self.seen_id[s, i, j] = m["id"]
                self.sent_delivered[s, i, j] = True
                self._log(kind="deliver", msg="offer", side=s, i=i, j=j, price=m["price"], send_t=m["send_t"])
            else:
                valid = (m["ref"] == self.sent_id[o, i, j]
                         and t <= int(self._validity_end(self.sent_t[o, i, j]))
                         and not self.f_done[i] and self.cap[j] > 0)
                if valid:
                    self._deal(i, j, m["price"], t)
                else:
                    self.pending_accept[s, i, j] = False
                    self.failed_accepts += 1
                    self._log(kind="accept_failed", side=s, i=i, j=j, price=m["price"], send_t=m["send_t"])

    def _deal(self, i: int, j: int, p: float, t: int) -> None:
        sc = self.sc
        d = sc.rho ** t
        g_t = yardsticks.gains(self.v[j], self.o_f[i], self.o_b[j], sc.rho, t)
        assert g_t > -1e-9, "deal closed with negative gains from trade; action masking is broken"
        phi = yardsticks.gain_share(p, self.v[j], self.o_f[i], self.o_b[j], sc.rho, t) if g_t > 1e-12 else None
        shortfall = self.mech.shortfall(self, i, j, p, t) if self.mech is not None else 0.0
        fee = getattr(self.mech, "lam", 0.0) * shortfall
        self.f_U[i] = p - self.cost_f * t
        self.f_price[i] = p
        self.f_sold[i] = True
        self.f_done[i] = True
        self.b_U[j] += self.v[j] * d - p - self.cost_b * t
        self.b_fees[j] += fee
        self.b_cost_units[j] += shortfall
        self.cap[j] -= 1
        self.decay_loss += (self.v[j] - self.o_f[i]) * (1.0 - d)
        self.deals.append(dict(i=i, j=j, price=float(p), t=t, phi=phi, shortfall=float(shortfall),
                               fee=float(fee), g_t=float(g_t), level=self.level))
        self._log(kind="deal", i=i, j=j, price=float(p), phi=phi, fee=float(fee))
        self._close_farmer(i)
        if self.cap[j] == 0:
            self._close_buyer(j)

    def _farmer_fallback(self, i: int, t: int) -> None:
        price = self.o_f[i] * self.sc.rho ** t
        self.f_U[i] = price - self.cost_f * t
        self.f_price[i] = price
        self.f_done[i] = True
        self._close_farmer(i)
        self._log(kind="fallback", i=i, price=float(price))

    def _buyer_end(self, j: int, t: int) -> None:
        self.b_U[j] += self.cap[j] * (self.o_b[j] - self.cost_b * t)
        self.b_done[j] = True
        self._close_buyer(j)

    def _close_farmer(self, i: int) -> None:
        """Close every thread of farmer i and cancel accepts still in flight on them."""
        self.open[i, :] = False
        self.pending_accept[:, i, :] = False

    def _close_buyer(self, j: int) -> None:
        self.open[:, j] = False
        self.pending_accept[:, :, j] = False

    def _settle_idle(self, t: int) -> None:
        """Agents with no open thread left are finished at step t."""
        changed = True
        while changed:
            changed = False
            for i in range(self.N):
                if not self.f_done[i] and not self.open[i].any():
                    self._farmer_fallback(i, t)
                    changed = True
            for j in range(self.M):
                if not self.b_done[j] and not self.open[:, j].any():
                    self._buyer_end(j, t)
                    changed = True

    # result
    def rewards(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Terminal rewards: farmer, buyer before fee, buyer fee base (shortfall)."""
        r_f = self.f_U - self.o_f
        r_b = self.b_U - self.sc.capacity * self.o_b
        return r_f, r_b, self.b_cost_units.copy()

    def result(self) -> EpisodeResult:
        assert self.done
        r_f, r_b, _ = self.rewards()
        return EpisodeResult(
            scenario=self.sc.name, seed=self.seed, level=self.level, g_star=self.g_star,
            n_opt_pairs=self.n_opt_pairs, farmer_gain=r_f.copy(), buyer_gain=r_b.copy(),
            buyer_fees=self.b_fees.copy(), price_received=self.f_price.copy(), sold=self.f_sold.copy(),
            deals=list(self.deals), failed_accepts=self.failed_accepts, decay_loss=self.decay_loss,
            events=list(self.events) if self.log_events else None)
