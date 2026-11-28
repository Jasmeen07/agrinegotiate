"""Scenario definitions. All prices are in grid units."""
from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class Scenario:
    name: str = "SYM"
    n_farmers: int = 1
    n_buyers: int = 1
    capacity: int = 1              # lots each buyer can take (k_j)
    T: int = 30                    # steps per episode
    # bargaining-power knobs
    rho: float = 1.0               # freshness kept per step
    deadline_lo: Optional[int] = None   # farmer deadline ~ U{lo..hi}; None -> T
    deadline_hi: Optional[int] = None
    h_farmer: float = 0.0          # extra farmer holding cost per step
    base_cost: float = 0.002       # time cost per step paid by both sides
    theta: float = 0.45            # mean farmer fallback as a share of the reference price
    theta_spread: float = 0.08     # farmer-specific spread around theta
    o_b: float = 0.10              # buyer fallback as a share of the reference price
    margin_lo: float = 0.03        # buyer handling margin ~ U(lo, hi)
    margin_hi: float = 0.08
    level_lo: float = 0.70         # reference price level per episode ~ U(lo, hi), grid units
    level_hi: float = 1.10
    level_samples: Optional[tuple] = None   # if set, the episode's level is drawn from these real-data values instead
    sigma_info: float = 0.0        # log-normal noise on the farmer's view of the reference price
    # link model
    duty: float = 1.0              # long-run share of time the farmer link is ON
    outage_len: float = 2.0        # mean outage length in steps
    ttl: Optional[int] = 6         # offer validity in steps; None = unlimited
    drop_mode: bool = False        # True: messages sent while OFF are lost (ablation)
    # action grid
    n_bins: int = 21
    p_lo: float = 0.20
    p_hi: float = 1.20

    def with_(self, **kw) -> "Scenario":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["level_samples"] = list(self.level_samples) if self.level_samples else None
        return d


_SYM = Scenario()
SCENARIOS: dict[str, Scenario] = {
    "SYM": _SYM,
    "PERISH": _SYM.with_(name="PERISH", rho=0.97),
    "DEADLINE": _SYM.with_(name="DEADLINE", deadline_lo=10, deadline_hi=20),
    "INFO": _SYM.with_(name="INFO", sigma_info=0.20),
    "THIN": _SYM.with_(name="THIN", n_farmers=8, n_buyers=2, capacity=2),
    "RURAL": _SYM.with_(name="RURAL", rho=0.97, deadline_lo=12, deadline_hi=20, h_farmer=0.005,
                        sigma_info=0.15, n_farmers=8, n_buyers=2, capacity=2),
    "RURAL-DTN": _SYM.with_(name="RURAL-DTN", rho=0.97, deadline_lo=12, deadline_hi=20, h_farmer=0.005,
                            sigma_info=0.15, n_farmers=8, n_buyers=2, capacity=2, duty=0.6),
    # variants added for the experiments
    # SYM with equal uncertainty on both sides: farmer fallbacks vary as little as buyer values.
    "SYMX": _SYM.with_(name="SYMX", theta_spread=0.025),
    # Small thin market that trains quickly: three farmers compete for one buyer with one slot.
    "THIN3": _SYM.with_(name="THIN3", n_farmers=3, n_buyers=1, capacity=1),
    # Bilateral version of RURAL (all knobs except market structure), and with intermittent links.
    "RURAL1": _SYM.with_(name="RURAL1", rho=0.97, deadline_lo=12, deadline_hi=20, h_farmer=0.005, sigma_info=0.15),
    "RURAL1-DTN": _SYM.with_(name="RURAL1-DTN", rho=0.97, deadline_lo=12, deadline_hi=20, h_farmer=0.005,
                             sigma_info=0.15, duty=0.6),
    # PERISH with widely varying buyer values (handling margin 3% to 40%). Added after the price floor
    # beat the fee in PERISH, where buyer values barely vary: a floor should cost trades here.
    "PERISH-HET": _SYM.with_(name="PERISH-HET", rho=0.97, margin_lo=0.03, margin_hi=0.40),
    # Stage-2 market (3 farmers, 2 buyers), symmetric knobs.
    "MARKET32": _SYM.with_(name="MARKET32", n_farmers=3, n_buyers=2, capacity=1),
}


# Scenarios calibrated on real Agmarknet lemon prices (run experiments/calibrate_real.py to regenerate the json).
_CAL = pathlib.Path(__file__).resolve().parents[1] / "data" / "real" / "calibration.json"
if _CAL.exists():
    _real = json.loads(_CAL.read_text())
    _R = dict(sigma_info=_real["sigma_info_used"], theta_spread=_real["theta_spread"],
              level_samples=tuple(_real["level_samples"]))
    SCENARIOS["PERISH-REAL"] = _SYM.with_(name="PERISH-REAL", rho=0.97, **_R)
    SCENARIOS["RURAL1-REAL"] = _SYM.with_(name="RURAL1-REAL", rho=0.97, deadline_lo=12, deadline_hi=20, h_farmer=0.005, **_R)
    SCENARIOS["RURAL1-REAL-DTN"] = SCENARIOS["RURAL1-REAL"].with_(name="RURAL1-REAL-DTN", duty=0.6)


def get(name: str) -> Scenario:
    return SCENARIOS[name]


if __name__ == "__main__":  # dump YAML copies into configs/scenarios
    import yaml
    out = pathlib.Path(__file__).resolve().parents[1] / "configs" / "scenarios"
    out.mkdir(parents=True, exist_ok=True)
    for k, sc in SCENARIOS.items():
        (out / f"{k.lower().replace('-', '_')}.yaml").write_text(yaml.safe_dump(sc.to_dict(), sort_keys=False))
    print("wrote", len(SCENARIOS), "scenario files to", out)
