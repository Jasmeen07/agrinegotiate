"""Fairness mechanisms.

FeeMechanism: fee = lam * shortfall, shortfall = max(0, target price - deal price).
  oracle target      p_tau = o_F*rho^t + tau*G_t           (uses private values; simulation only)
  observable target  p_tau = kappa * P_ref * rho^t         (public quantities only)
`lam` is fixed here; the trainer adapts it by dual ascent via `dual_update`.

FloorMechanism: the platform rejects any offer below kappa * P_ref * rho^t.
"""
from __future__ import annotations

from .. import yardsticks


class FeeMechanism:
    floor_kappa = None

    def __init__(self, tau: float = 0.4, lam: float = 0.0, observable: bool = False,
                 kappa: float | None = None, eps_c: float = 0.02, eta: float = 0.05, lam_max: float = 5.0):
        if observable and kappa is None:
            raise ValueError("observable mechanism needs kappa (target price as a share of P_ref)")
        self.tau, self.lam, self.observable, self.kappa = tau, lam, observable, kappa
        self.eps_c, self.eta, self.lam_max = eps_c, eta, lam_max

    def target(self, env, i, j, t):
        if self.observable:
            return self.kappa * env.level * env.sc.rho ** t
        return yardsticks.target_price(env.v[j], env.o_f[i], env.o_b[j], self.tau, env.sc.rho, t)

    def shortfall(self, env, i, j, p, t) -> float:
        target = self.target(env, i, j, t)
        if target is None:
            return 0.0
        return max(0.0, target - p)

    def dual_update(self, j_c_hat: float) -> float:
        """lam <- clip(lam + eta * (J_c - eps_c), 0, lam_max). Returns the new value."""
        self.lam = min(self.lam_max, max(0.0, self.lam + self.eta * (j_c_hat - self.eps_c)))
        return self.lam


class FloorMechanism:
    lam = 0.0
    lam_max = 1.0
    tau = 0.0

    def __init__(self, kappa: float):
        self.floor_kappa = kappa

    def shortfall(self, env, i, j, p, t) -> float:
        return 0.0
