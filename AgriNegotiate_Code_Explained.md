# AgriNegotiate — the code, explained line by line

*A project by Harshita and Jasmeen.*
*Fairness-Constrained Multi-Agent Reinforcement Learning for Delay-Tolerant Autonomous Negotiation in Rural Agricultural Markets.*

How to use this file: Part 1 gives the big picture and the answers to the questions most likely to be asked. Part 2 walks through every Python file in the project, in the order data flows through it. Each block shows the real code with its **real line numbers**, followed by an explanation of what those lines do and why. Part 3 covers the non-Python files. Line numbers match the files in the repository (this is the **cleaned-up version** of the code).

**What the cleanup changed (behaviour is identical — checked on 44 fixed-seed runs, scripted and trained, with bit-for-bit equal results, and all 39 tests pass):**
- `observe()` — the slow Python double loop that finds each agent's best offer on its *other* threads is replaced by a small vectorised helper `_best_other`.
- `step()` — split in two: `step` now only loops and checks masks; the per-action logic lives in `_apply`.
- `ppo.py` — one helper `_step` handles acting for both network types (recurrent hidden-state bookkeeping now exists once); advantage computation moved into `_advantages`; unused imports removed.
- `networks.py` — the feed-forward net now has the same `step()` interface as the GRU net.
- `scripted.py` — Boulware/Linear/Conceder are no longer three near-identical classes but one table (`TACTIC_BETA`); an unused hook was deleted.
- Stale references to an internal planning document ("spec 8.6" etc.) were removed from every docstring; unused imports in tests removed.

**Second cleanup, after the 3M rerun (again no change to any result — 113 fixed-seed scripted evaluations, one fully logged episode and 11 short training runs gave bit-for-bit equal metrics, learning curves and network weights, and every hypothesis verdict and number in `results/summary.md` is unchanged):**
- `core.py` — the four copies of "close this farmer's / buyer's threads and cancel their pending accepts" are now two helpers, `_close_farmer` and `_close_buyer`.
- `ppo.py` — terminal rewards are credited to the threads in one pass instead of one pass per episode; the metric list comes from `stats.KEYS` instead of a second copy; `getattr(mech, "lam", 0.0)` already gives 0 when there is no mechanism, so the extra `if mech is not None` was dropped.
- `stats.py` — the bootstrap is written once (`_bootstrap_iqms`) and used by both `iqm_ci` and the new `iqm_diff_ci` (the difference test that used to live in `make_report.py`), drawing the same random numbers in the same order.
- `make_report.py` — each hypothesis test is one function (`check_h1`, `check_h2`, `check_fee`, `check_floors`, `check_h5`) used for both the short runs and the 3M rerun, instead of two hand-written copies. The only visible change: the evidence text of the `_3m` rows now uses the same wording as the short-run rows (for example it says which bar the interval must clear).

---

# Part 1 — Big picture

## 1.1 What the project is, in five sentences
Farmers and buyers negotiate crop prices. Each side has a private value (a farmer's fallback price, a buyer's willingness to pay), and the crop spoils as time passes, so waiting costs both sides. Rural farmers also have unreliable internet, so their messages are delayed or lost. We train reinforcement-learning agents (PPO) to negotiate on both sides, add a **fairness constraint** so farmers are not squeezed, and compare everything against hand-written baseline negotiators and a theoretical best (the benchmark).

## 1.2 Architecture (who calls whom)
```
scenarios.py      -> parameter sets (SYM, PERISH, RURAL-DTN ...)
yardsticks.py     -> theory: gains, Nash price, Rubinstein, Hungarian best-case G*
env/core.py       -> Market: the simulator (reset / observe / step / result)
agents/scripted.py-> hand-written negotiators (benchmarks, opponents, floor & ceiling; Boulware/Linear/Conceder from one table)
agents/networks.py-> neural nets (feed-forward, GRU)
agents/ppo.py     -> IPPO training loop, dual ascent for the fee
mechanisms/fee.py -> fairness fee and price floor
metrics.py        -> efficiency, farmer share, Gini, Jain ...
eval/*            -> fixed-seed evaluation, IQM + bootstrap CIs, episode recording
experiments/*     -> batch runners, report and figure builders
dashboard/*       -> FastAPI backend + exported data for the website
tests/*           -> proof that the simulator and formulas are right
```

## 1.3 What happens in one episode (memorise this)
1. `reset(seed)` draws a market level, buyer values, farmer fallback values, deadlines, and link states; computes G* (best possible total gains) with the Hungarian algorithm.
2. Loop until done: `observe()` gives each agent a 22-number view and a mask of legal actions → each agent picks one of 24 actions on each of its threads (21 price offers, ACCEPT, WALK, WAIT) → `step(actions)`.
3. In `step`: messages are queued; time moves to t+1; links turn on/off (a two-state Markov chain); messages that are old enough and whose link is up are delivered; an ACCEPT that references the current valid offer closes a deal; deadlines force unsold farmers to their fallback.
4. At the end each agent's reward is its gain over its fallback. PPO updates the networks from those rewards.

## 1.4 The formulas
| Quantity | Formula | Meaning |
|---|---|---|
| Gains from trade at time t | G_t = (v − o_f)·ρ^t − o_b | what a deal at time t creates; shrinks as the crop spoils |
| Nash price | p = o_f·ρ^t + G_t/2 | the "equal split" price |
| Farmer share of a deal | φ = (p − o_f·ρ^t)/G_t | 0.5 = equal split |
| Efficiency | E = (FGC + BGC) | realised gains / best possible gains G* |
| Farmer gain capture | FGC = farmer gains / G* | the farmers' part of E |
| Farmer share | S_F = FGC / E | undefined if E < 0.05 or a side is negative |
| Fee | fee = λ·max(0, target − price), target = o_f·ρ^t + τ·G_t | charged to the buyer when the price is below the fair target |
| Dual ascent | λ ← clip(λ + η(Ĵ_c − ε), 0, 5) | raise the fee while the constraint is violated |
| Buyer advantage | A = A_reward − λ·A_cost | how the constraint enters PPO |
| PPO loss | −min(r·A, clip(r, 0.8, 1.2)·A) + 0.5·value − c·entropy | standard clipped objective |
| GAE | δ_k = r_k + V_{k+1} − V_k;  A_k = δ_k + γλ·A_{k+1}, γ = 1, λ = 0.95 | advantage estimate |
| Link chain | p_rec = 1/outage_len;  p_fail = p_rec(1 − duty)/duty | gives long-run uptime = duty |

Key numbers: 21 price bins from 0.20 to 1.20; T = 30 steps; TTL = 6; hidden size 64; lr 3e-4; 4 epochs × 4 minibatches; ε = 0.02; η = 0.05 (0.25 in the "fast" runs); τ = 0.4.

## 1.5 Questions you may be asked, with answers
**Q1. Where is your benchmark?** The benchmark is the full set of hand-written negotiators (ZIC, Boulware, Linear, Conceder, TitForTat, a tuned time-dependent agent, a hardball buyer), plus two reference points: TakeFallback (floor, E = 0) and OracleFairSplit (ceiling, E ≈ 1 with an equal split). All run on the same fixed seeds with the same metrics (`experiments/x1_scripted.py`, `eval/evaluate.py`). The environment is also exposed through the standard PettingZoo API.

**Q2. Why can't you just train and look at the reward?** Because one seed proves nothing in RL. We report the interquartile mean over 10 seeds with a 95% bootstrap interval, evaluate on 300 fixed episodes at each checkpoint, and count failed (collapsed) runs.

**Q3. How do you know the simulator is correct?** `tests/` checks hand-computed examples, a brute-force check of the Hungarian optimum on 200 random markets, determinism by seed, the E = FGC + BGC identity under random play across every scenario, and that no deal ever has negative gains.

**Q4. Why action masking?** So no agent can ever make an irrational offer or accept a deal that loses money; the learning problem is about *strategy*, not about learning not to lose money.

**Q5. Why a Lagrangian / fee and not just a price floor?** A floor blocks trades below a price, so deals that would have been good for both sides disappear when buyers are heterogeneous. A fee leaves the trade possible and charges for unfairness. Our 10-seed PERISH-HET result is honest: floors lose deals as they tighten, the fee keeps trading but 2 of 10 runs collapsed, and the two reach similar efficiency at similar shares. We did **not** show the fee clearly wins there.

**Q6. What is dual ascent?** The constrained problem is: maximise welfare subject to expected shortfall ≤ ε. The Lagrangian adds λ·(constraint cost). λ is updated by gradient *ascent*: if the measured cost is above ε, λ grows; if below, it shrinks (never below 0).

**Q7. Why IPPO and not MAPPO?** Each role has a decentralised actor and its own critic; with independent critics the algorithm is simpler and scales to unequal numbers of agents. A centralised critic (MAPPO) is listed as future work.

**Q8. Why γ = 1 and terminal rewards?** Each episode is short, the payoff is only known when it ends, and discounting is already modelled *inside* the market by ρ^t (crop spoilage).

**Q9. What does the delay-aware agent do differently?** It sees extra features (own-offer age, whether it was delivered, counterparty offer age/validity, link state) and has a GRU memory. The delay-unaware agent has those five features zeroed (`DELAY_FEATURES`).

**Q10. Why the WALK logit of −3?** In early runs 3 of 10 PERISH agents collapsed into always walking, because random early play teaches "leaving is safe". Biasing the initial policy against WALK fixed it (DECISIONS.md #18).

**Q11. How do you avoid train/test leakage?** Disjoint seed ranges: tuning 0–99, training-run seeds 1000–1009, training episodes 10,000,000+, evaluation 50,000–50,999.

**Q12. Why IQM instead of the mean?** One collapsed seed can drag a mean badly; the IQM drops the lowest and highest 25% of runs (Agarwal et al., rliable).

**Q13. What are the honest limits?** Short training budget (hundreds of iterations, not millions of steps); synthetic scenarios (real mandi data code is written but not run — no network access); mostly 1×1 markets for learned agents; no centralised critic; no collusion tests. These are on the roadmap, not hidden.

**Q14. What does the Hungarian algorithm give you?** The maximum total gains G* over all one-to-one (or capacity-limited) matchings; every efficiency number is divided by it. Capacity-c buyers are handled by repeating the buyer's column c times.

**Q15. Why random tie-breaks in `_deliver`?** So that when two messages arrive at the same step, no agent index is systematically favoured.

---

# Part 2 — The code, file by file


## 2.1  `agrinegotiate/scenarios.py`

**Role of this file.** A *scenario* is one complete description of a market: how many farmers and buyers, how perishable the crop is, how reliable the phone link is, and so on. Everything the simulator needs is stored in one immutable object (`Scenario`) so an experiment can be reproduced exactly from its name. All prices in the whole project are in *grid units* (0.20 to 1.20) so numbers are comparable across scenarios.

**Likely questions.** Why a frozen dataclass? (Immutable, hashable, can be copied with changes via `replace`, saved to YAML.) What does each knob do? (Section "bargaining-power knobs" below.) Why so many named scenarios? (Each isolates one source of unfairness so we can test hypotheses H1 and H2 one factor at a time.)

**Lines 1–7**
```python
   1  """Scenario definitions. All prices are in grid units."""
   2  from __future__ import annotations
   3  
   4  import json
   5  import pathlib
   6  from dataclasses import asdict, dataclass, replace
   7  from typing import Optional
```
- Line 1: module docstring. It says these definitions come from sections 5, 6 and 8 of the written specification and that prices are in grid units.
- Line 2: `from __future__ import annotations` lets type hints refer to classes that are defined later (for example the return type `"Scenario"`) without quotes causing errors, and makes hints lazily evaluated.
- Line 4: imports three helpers from `dataclasses`: `dataclass` (decorator that writes `__init__`, `__repr__` and so on), `asdict` (turn the object into a plain dict) and `replace` (make a copy with a few fields changed).
- Line 5: `Optional` marks fields that may be `None`.

**Lines 4–5**
```python
   4  import json
   5  import pathlib
```
- Lines 4–5: `json` and `pathlib` are needed only by the real-data block at the bottom, which reads `data/real/calibration.json`.

**Lines 10–11**
```python
  10  @dataclass(frozen=True)
  11  class Scenario:
```
- Line 8: `@dataclass(frozen=True)` generates the constructor from the annotated fields below and makes instances read-only. Frozen matters because one Scenario object is shared by the environment, the trainer and the evaluator; nobody can accidentally change it mid-experiment.
- Line 9: the class name. A `Scenario` is only data, with no behaviour besides two tiny helper methods at the bottom.

**Lines 12–17**
```python
  12      name: str = "SYM"
  13      n_farmers: int = 1
  14      n_buyers: int = 1
  15      capacity: int = 1              # lots each buyer can take (k_j)
  16      T: int = 30                    # steps per episode
  17      # bargaining-power knobs
```
- Line 10: `name` identifies the scenario in result folders and plots. Default `"SYM"` (symmetric).
- Line 11: `n_farmers`, number of farmers selling. Default 1.
- Line 12: `n_buyers`, number of buyers. Default 1.
- Line 13: `capacity`, how many lots one buyer can take (called k_j in the spec). With 8 farmers and 2 buyers of capacity 2, only 4 farmers can sell, so there is scarcity.
- Line 14: `T`, the episode length in steps (30). After T steps the episode ends and unsold farmers take their fallback.

**Lines 18–22**
```python
  18      rho: float = 1.0               # freshness kept per step
  19      deadline_lo: Optional[int] = None   # farmer deadline ~ U{lo..hi}; None -> T
  20      deadline_hi: Optional[int] = None
  21      h_farmer: float = 0.0          # extra farmer holding cost per step
  22      base_cost: float = 0.002       # time cost per step paid by both sides
```
These are the **bargaining-power knobs**; each one tilts the market toward one side.
- Line 16: `rho` (ρ) is the freshness kept per step. 1.0 means the crop never spoils; 0.97 means each step loses 3% of its value, so waiting hurts the farmer more than the buyer. This is the PERISH effect.
- Lines 17–18: `deadline_lo` and `deadline_hi`. If both are set, each farmer's personal deadline is drawn uniformly from that range (shorter than T), meaning the farmer must sell sooner. `None` means the deadline is simply T.
- Line 19: `h_farmer`, an extra holding cost the farmer pays per step (storage, spoilage risk).
- Line 20: `base_cost`, a small time cost both sides pay each step, so that dragging out talks is never free.

**Lines 23–32**
```python
  23      theta: float = 0.45            # mean farmer fallback as a share of the reference price
  24      theta_spread: float = 0.08     # farmer-specific spread around theta
  25      o_b: float = 0.10              # buyer fallback as a share of the reference price
  26      margin_lo: float = 0.03        # buyer handling margin ~ U(lo, hi)
  27      margin_hi: float = 0.08
  28      level_lo: float = 0.70         # reference price level per episode ~ U(lo, hi), grid units
  29      level_hi: float = 1.10
  30      level_samples: Optional[tuple] = None   # if set, the episode's level is drawn from these real-data values instead
  31      sigma_info: float = 0.0        # log-normal noise on the farmer's view of the reference price
  32      # link model
```
- Line 21: `theta` (θ), the average farmer fallback price as a fraction of the reference (market) price. 0.45 means a farmer could on average get 45% of today's market price somewhere else.
- Line 22: `theta_spread`, how much individual farmers differ around θ. This one was added later (see DECISIONS.md) because making farmers more uncertain than buyers biased the symmetric test; SYMX reduces it.
- Line 23: `o_b`, the buyer's fallback payoff as a share of the reference price.
- Lines 24–25: `margin_lo` and `margin_hi`, the buyer's handling margin is drawn uniformly in this range. A buyer's value is roughly level × (1 + margin). Wider range means buyers differ more (used by PERISH-HET).
- Lines 26–27: `level_lo` and `level_hi`: each episode draws a market price level uniformly from this range, so agents cannot memorise one price.
- Line 28: `sigma_info`, the standard deviation of log-normal noise on the farmer's view of the reference price. 0 means the farmer sees the true price; 0.2 means a noisy view (the INFO scenario, an information gap).

**Lines 30–30**
```python
  30      level_samples: Optional[tuple] = None   # if set, the episode's level is drawn from these real-data values instead
```
- Line 30: `level_samples` (real-data scenarios only). When set, each episode's market price level is drawn from this list of values measured on real Agmarknet prices instead of being uniform in [`level_lo`, `level_hi`]. `None` keeps the synthetic behaviour, so every older scenario is unchanged.

**Lines 33–37**
```python
  33      duty: float = 1.0              # long-run share of time the farmer link is ON
  34      outage_len: float = 2.0        # mean outage length in steps
  35      ttl: Optional[int] = 6         # offer validity in steps; None = unlimited
  36      drop_mode: bool = False        # True: messages sent while OFF are lost (ablation)
  37      # action grid
```
These are the **link model** settings (section 8).
- Line 30: `duty`, the long-run fraction of time the farmer's phone link is ON. 1.0 is always on; 0.6 is on 60% of the time.
- Line 31: `outage_len`, the mean outage length in steps (2). Together with `duty` it fixes the on/off switching probabilities of a two-state Markov chain (a Gilbert–Elliott link).
- Line 32: `ttl`, time to live. An offer is valid for 6 steps; after that it expires and can no longer be accepted. `None` would mean offers never expire.
- Line 33: `drop_mode`. If True, messages sent while the link is OFF are lost; if False (default), they are queued (store-and-forward) and delivered when the link returns.

**Lines 38–40**
```python
  38      n_bins: int = 21
  39      p_lo: float = 0.20
  40      p_hi: float = 1.20
```
The **action grid** (section 5.6): prices are discretised into `n_bins` = 21 values evenly spaced from `p_lo` = 0.20 to `p_hi` = 1.20, i.e. steps of 0.05. A discrete grid lets the policy network output a simple probability vector instead of a continuous number.

**Lines 42–45**
```python
  42      def with_(self, **kw) -> "Scenario":
  43          return replace(self, **kw)
  44  
  45      def to_dict(self) -> dict:
```
- Lines 39–40: `with_` returns a *new* Scenario with some fields changed, using `dataclasses.replace`. Because the class is frozen we never edit in place; we derive variants, e.g. `_SYM.with_(rho=0.97)`.
- Lines 42–43: `to_dict` converts to a plain dictionary, used when saving `config.json` for every run and when dumping YAML files.

**Lines 45–48**
```python
  45      def to_dict(self) -> dict:
  46          d = asdict(self)
  47          d["level_samples"] = list(self.level_samples) if self.level_samples else None
  48          return d
```
- Lines 45–48: `to_dict` turns the scenario into a plain dict for saving as YAML. The tuple of real-data levels is converted to a list so YAML can store it.

**Lines 51–62**
```python
  51  _SYM = Scenario()
  52  SCENARIOS: dict[str, Scenario] = {
  53      "SYM": _SYM,
  54      "PERISH": _SYM.with_(name="PERISH", rho=0.97),
  55      "DEADLINE": _SYM.with_(name="DEADLINE", deadline_lo=10, deadline_hi=20),
  56      "INFO": _SYM.with_(name="INFO", sigma_info=0.20),
  57      "THIN": _SYM.with_(name="THIN", n_farmers=8, n_buyers=2, capacity=2),
  58      "RURAL": _SYM.with_(name="RURAL", rho=0.97, deadline_lo=12, deadline_hi=20, h_farmer=0.005,
  59                          sigma_info=0.15, n_farmers=8, n_buyers=2, capacity=2),
  60      "RURAL-DTN": _SYM.with_(name="RURAL-DTN", rho=0.97, deadline_lo=12, deadline_hi=20, h_farmer=0.005,
  61                              sigma_info=0.15, n_farmers=8, n_buyers=2, capacity=2, duty=0.6),
  62      # variants added for the experiments
```
- Line 46: `_SYM = Scenario()` is the baseline: all defaults, one farmer, one buyer, nothing perishable, perfect link. Every other scenario is defined as "SYM plus changes".
- Line 47: `SCENARIOS` is a registry (dictionary) from name to Scenario.
- Line 48: SYM, the symmetric control used to test H1.
- Line 49: PERISH, `rho=0.97`. Tests whether perishability alone hurts the farmer (H2).
- Line 50: DEADLINE, farmer deadlines drawn from 10 to 20 steps instead of 30.
- Line 51: INFO, noisy farmer price view (σ=0.20).
- Line 52: THIN, a thin market: 8 farmers, 2 buyers with capacity 2 each (so scarcity: 8 sellers for 4 slots).
- Lines 53–54: RURAL combines the four disadvantages (perishable, short deadline, holding cost, noisy information) in the 8-farmer market. It is the realistic "stacked disadvantages" case.
- Lines 55–56: RURAL-DTN is RURAL plus an intermittent link (`duty=0.6`). DTN means delay-tolerant networking.

**Lines 63–76**
```python
  63      # SYM with equal uncertainty on both sides: farmer fallbacks vary as little as buyer values.
  64      "SYMX": _SYM.with_(name="SYMX", theta_spread=0.025),
  65      # Small thin market that trains quickly: three farmers compete for one buyer with one slot.
  66      "THIN3": _SYM.with_(name="THIN3", n_farmers=3, n_buyers=1, capacity=1),
  67      # Bilateral version of RURAL (all knobs except market structure), and with intermittent links.
  68      "RURAL1": _SYM.with_(name="RURAL1", rho=0.97, deadline_lo=12, deadline_hi=20, h_farmer=0.005, sigma_info=0.15),
  69      "RURAL1-DTN": _SYM.with_(name="RURAL1-DTN", rho=0.97, deadline_lo=12, deadline_hi=20, h_farmer=0.005,
  70                               sigma_info=0.15, duty=0.6),
  71      # PERISH with widely varying buyer values (handling margin 3% to 40%). Added after the price floor
  72      # beat the fee in PERISH, where buyer values barely vary: a floor should cost trades here.
  73      "PERISH-HET": _SYM.with_(name="PERISH-HET", rho=0.97, margin_lo=0.03, margin_hi=0.40),
  74      # Stage-2 market (3 farmers, 2 buyers), symmetric knobs.
  75      "MARKET32": _SYM.with_(name="MARKET32", n_farmers=3, n_buyers=2, capacity=1),
  76  }
```
Variants added later for specific experiments (the comments say why).
- Line 59: SYMX, SYM with `theta_spread` cut to 0.025 so that farmer-side and buyer-side uncertainty are equal. Used to diagnose why SYM is lopsided (x5_diagnosis).
- Line 61: THIN3, 3 farmers and 1 buyer with one slot. A small thin market that trains quickly and still shows competition between farmers.
- Line 63: RURAL1, the bilateral (1 farmer, 1 buyer) version of RURAL: same perishability, deadlines, holding cost and noise but no market structure. It is the learning testbed for delay experiments.
- Lines 64–65: RURAL1-DTN, RURAL1 with `duty=0.6`. Used with duty 1.0, 0.6 and 0.4 for hypotheses H4 and H5.
- Line 68: PERISH-HET, PERISH with buyer margins from 3% to 40%, so buyer values differ a lot. Added after the price floor beat the fee in PERISH: when buyers differ widely a floor should start to cost trades.
- Line 70: MARKET32, 3 farmers and 2 buyers with capacity 1, a stage-2 market from the spec.

**Lines 79–87**
```python
  79  # Scenarios calibrated on real Agmarknet lemon prices (run experiments/calibrate_real.py to regenerate the json).
  80  _CAL = pathlib.Path(__file__).resolve().parents[1] / "data" / "real" / "calibration.json"
  81  if _CAL.exists():
  82      _real = json.loads(_CAL.read_text())
  83      _R = dict(sigma_info=_real["sigma_info_used"], theta_spread=_real["theta_spread"],
  84                level_samples=tuple(_real["level_samples"]))
  85      SCENARIOS["PERISH-REAL"] = _SYM.with_(name="PERISH-REAL", rho=0.97, **_R)
  86      SCENARIOS["RURAL1-REAL"] = _SYM.with_(name="RURAL1-REAL", rho=0.97, deadline_lo=12, deadline_hi=20, h_farmer=0.005, **_R)
  87      SCENARIOS["RURAL1-REAL-DTN"] = SCENARIOS["RURAL1-REAL"].with_(name="RURAL1-REAL-DTN", duty=0.6)
```
- Line 79: comment saying these scenarios come from real prices and that `experiments/calibrate_real.py` regenerates the numbers.
- Line 80: path of `data/real/calibration.json`, two folders above this file's package.
- Lines 81–84: only if that file exists (a fresh checkout without it still works), load it and bundle the three calibrated numbers into `_R`: `sigma_info` (how stale a farmer's price signal is), `theta_spread` (how much farmers' fallback prices differ) and `level_samples` (real price-level values).
- Line 85: `PERISH-REAL` = the symmetric market with a perishable crop (ρ = 0.97, 3% loss per step) plus the real numbers.
- Line 86: `RURAL1-REAL` adds rural conditions: deadlines of 12 to 20 steps and a small holding cost.
- Line 87: `RURAL1-REAL-DTN` is the same with a link that is on only 60% of the time (delay-tolerant case).

**Lines 90–91**
```python
  90  def get(name: str) -> Scenario:
  91      return SCENARIOS[name]
```
`get(name)` looks a scenario up by name. A `KeyError` here means a misspelt scenario.

**Lines 94–100**
```python
  94  if __name__ == "__main__":  # dump YAML copies into configs/scenarios
  95      import yaml
  96      out = pathlib.Path(__file__).resolve().parents[1] / "configs" / "scenarios"
  97      out.mkdir(parents=True, exist_ok=True)
  98      for k, sc in SCENARIOS.items():
  99          (out / f"{k.lower().replace('-', '_')}.yaml").write_text(yaml.safe_dump(sc.to_dict(), sort_keys=False))
 100      print("wrote", len(SCENARIOS), "scenario files to", out)
```
The `__main__` block runs only when the file is executed directly (`python agrinegotiate/scenarios.py`), not when imported.
- Lines 79–81: build the output folder `configs/scenarios` relative to this file (`parents[1]` is the project root).
- Line 82: create it if missing.
- Lines 83–84: for every scenario, write a YAML file named after it (lower case, `-` replaced by `_`). `sort_keys=False` keeps field order readable.
- Line 85: print how many files were written. These YAMLs are copies for humans; the code uses the Python objects.


## 2.2  `agrinegotiate/yardsticks.py`

**Role of this file.** The *yardsticks* are closed-form reference points: the best possible outcome and the fair-split outcome that learned agents are compared against. **Nothing here is learned.** This is the answer to the professor's question "better than what?".

Notation: v = buyer's value of a fresh lot, o_f = farmer's fallback price, o_b = buyer's fallback payoff, ρ (rho) = freshness kept per step, t = step at which the deal closes.

**Key formulas.**
- Gains from trade: G_t = (v − o_f)·ρ^t − o_b.
- Nash (even split) price: o_f·ρ^t + G_t/2.
- Farmer share of gains: φ = (p − o_f·ρ^t) / G_t.

**Lines 1–6**
```python
   1  """Closed-form reference points. Nothing here is learned.
   2  
   3  Conventions: prices are in grid units. v = buyer value of a fresh lot,
   4  o_f = farmer fallback price of a fresh lot, o_b = buyer fallback payoff,
   5  rho = freshness kept per step, t = step at which the deal closes.
   6  """
```
Docstring. Says "closed form" (no learning) and defines the symbols used in every function below, so the reader does not have to guess what `v`, `o_f`, `o_b`, `rho`, `t` mean.

**Lines 7–13**
```python
   7  from __future__ import annotations
   8  
   9  import itertools
  10  from typing import Optional, Sequence
  11  
  12  import numpy as np
  13  from scipy.optimize import linear_sum_assignment
```
Imports. `itertools` is used by the brute-force check. `Optional` and `Sequence` are type hints. `numpy` does the matrix work. `linear_sum_assignment` from SciPy solves the *assignment problem* (best one-to-one matching) with the Hungarian algorithm in polynomial time.

**Lines 16–19**
```python
  16  def gains(v: float, o_f: float, o_b: float, rho: float = 1.0, t: int = 0) -> float:
  17      """Gains from trade G_t = v*rho^t - o_f*rho^t - o_b."""
  18      d = rho ** t
  19      return v * d - o_f * d - o_b
```
`gains(v, o_f, o_b, rho, t)` returns the total surplus a deal at step t would create.
- Line 18: `d = rho ** t` is the freshness remaining after t steps. With ρ = 0.97 and t = 10, d ≈ 0.74.
- Line 19: both the buyer's value and the farmer's fallback decay with freshness, so `v*d - o_f*d`; the buyer's fallback `o_b` does not decay. The result is G_t. If G_t ≤ 0 there is no mutually beneficial price.

**Lines 22–27**
```python
  22  def nash_price(v: float, o_f: float, o_b: float, rho: float = 1.0, t: int = 0) -> Optional[float]:
  23      """Symmetric Nash bargaining price. None when no trade is possible."""
  24      g = gains(v, o_f, o_b, rho, t)
  25      if g <= 0:
  26          return None
  27      return o_f * rho ** t + 0.5 * g
```
`nash_price` is the price that splits the gains 50/50 (symmetric Nash bargaining).
- Line 24: compute G_t.
- Lines 25–26: if there is no surplus, return `None` (no trade possible).
- Line 27: the farmer receives their (decayed) fallback plus half the surplus.

**Lines 30–35**
```python
  30  def target_price(v: float, o_f: float, o_b: float, tau: float, rho: float = 1.0, t: int = 0) -> Optional[float]:
  31      """Price at which the farmer's gain share equals tau."""
  32      g = gains(v, o_f, o_b, rho, t)
  33      if g <= 0:
  34          return None
  35      return o_f * rho ** t + tau * g
```
`target_price` generalises Nash to any farmer share τ (tau). Same code but `tau * g` instead of `0.5 * g`. The fairness fee uses τ = 0.4 as the *target*: "the farmer should get at least 40% of the gains".

**Lines 38–43**
```python
  38  def gain_share(p: float, v: float, o_f: float, o_b: float, rho: float = 1.0, t: int = 0) -> Optional[float]:
  39      """Farmer's share phi of the gains for a deal at price p. None if G_t <= 0."""
  40      g = gains(v, o_f, o_b, rho, t)
  41      if g <= 0:
  42          return None
  43      return (p - o_f * rho ** t) / g
```
`gain_share` goes the other way: given the actual deal price p, returns φ, the farmer's share. Line 43: (price − farmer's decayed fallback) ÷ total gains. φ = 0.5 is the fair split, φ = 1 means the farmer took everything, φ = 0 means the farmer got only their fallback.

**Lines 46–56**
```python
  46  def rubinstein_share(delta_f: float, delta_b: float, farmer_first: bool = True, k: int = 1) -> float:
  47      """Farmer's share in Rubinstein alternating offers.
  48  
  49      delta_* are per-step discount factors; each round takes k steps.
  50      First proposer's share = (1 - d_other) / (1 - d_f * d_b).
  51      """
  52      df, db = delta_f ** k, delta_b ** k
  53      denom = 1.0 - df * db
  54      if farmer_first:
  55          return (1.0 - db) / denom
  56      return 1.0 - (1.0 - df) / denom
```
`rubinstein_share` gives the theoretical outcome of Rubinstein's alternating-offers game, where each side is impatient with a per-step discount factor δ (delta).
- Line 52: `df, db` are the discount factors raised to k, since each round of offers lasts k steps.
- Line 53: `denom = 1 − df·db`, the common denominator.
- Lines 54–55: if the farmer proposes first, their share is (1 − db)/denom; the first mover has an advantage, and the more patient the opponent, the lower the share.
- Line 56: otherwise the buyer moves first and the farmer's share is the remainder, 1 − (1 − df)/denom.

**Lines 59–62**
```python
  59  def rubinstein_limit(delta_f: float, delta_b: float) -> float:
  60      """Short-round limit of the farmer's share: r_B / (r_F + r_B), r = -ln(delta)."""
  61      r_f, r_b = -np.log(delta_f), -np.log(delta_b)
  62      return float(r_b / (r_f + r_b))
```
`rubinstein_limit` is the limit when rounds are very short: share = r_B/(r_F + r_B) where r = −ln δ is the continuous impatience rate. The *more impatient* side (larger r) gets the *smaller* share. Used to predict how perishability should shift the split.

**Lines 65–70**
```python
  65  def pair_gain_matrix(v: Sequence[float], o_f: Sequence[float], o_b: Sequence[float]) -> np.ndarray:
  66      """[N, M] matrix of max(0, v_j - o_f_i - o_b_j) at step 0."""
  67      v = np.asarray(v, float)[None, :]
  68      o_b = np.asarray(o_b, float)[None, :]
  69      o_f = np.asarray(o_f, float)[:, None]
  70      return np.maximum(0.0, v - o_f - o_b)
```
`pair_gain_matrix` builds the table of potential gains for every farmer–buyer pair at step 0.
- Lines 67–69: reshape with `[None, :]` / `[:, None]` so NumPy *broadcasting* subtracts every farmer fallback from every buyer value, producing an N×M matrix (rows farmers, columns buyers).
- Line 70: `np.maximum(0.0, ...)` clips negatives: a pair with no surplus contributes zero.

**Lines 73–87**
```python
  73  def max_gains(v: Sequence[float], o_f: Sequence[float], o_b: Sequence[float],
  74                capacity: Sequence[int]) -> tuple[float, list[tuple[int, int]]]:
  75      """Maximum total gains from trade G* and one optimal matching.
  76  
  77      One lot per farmer, at most capacity[j] lots per buyer, all at step 0.
  78      Returns (G*, [(farmer, buyer), ...]) listing only pairs with positive gain.
  79      """
  80      g = pair_gain_matrix(v, o_f, o_b)
  81      cols = [j for j, k in enumerate(capacity) for _ in range(int(k))]
  82      if g.size == 0 or not cols:
  83          return 0.0, []
  84      expanded = g[:, cols]
  85      rows, cidx = linear_sum_assignment(expanded, maximize=True)
  86      pairs = [(int(i), int(cols[c])) for i, c in zip(rows, cidx) if expanded[i, c] > 0]
  87      return float(sum(g[i, j] for i, j in pairs)), pairs
```
`max_gains` computes G\*, the largest total surplus any matching could achieve. This is the denominator of efficiency E.
- Line 80: the gain matrix.
- Line 81: expand each buyer into `capacity[j]` identical columns. A buyer with capacity 2 becomes two columns, turning a many-slots problem into a plain one-to-one assignment.
- Lines 82–83: if there is nothing to match return 0 and an empty list.
- Line 84: select the expanded columns.
- Line 85: `linear_sum_assignment(..., maximize=True)` returns the optimal row and column indices (Hungarian algorithm).
- Line 86: keep only pairs whose gain is positive, and map the expanded column index back to the original buyer index.
- Line 87: return (total gain, list of (farmer, buyer) pairs).

**Lines 90–98**
```python
  90  def max_gains_bruteforce(v, o_f, o_b, capacity) -> float:
  91      """Exhaustive search over all matchings. For tests on tiny cases only."""
  92      g = pair_gain_matrix(v, o_f, o_b)
  93      n = g.shape[0]
  94      slots = [j for j, k in enumerate(capacity) for _ in range(int(k))] + [None] * n
  95      best = 0.0
  96      for perm in set(itertools.permutations(slots, n)):
  97          best = max(best, sum(g[i, j] for i, j in enumerate(perm) if j is not None))
  98      return float(best)
```
`max_gains_bruteforce` is an independent check used only in tests: try every possible matching and keep the best. It is exponential, so only for tiny cases. The test compares it with the Hungarian result to make sure `max_gains` is right.
- Line 92: gain matrix; Line 93: number of farmers.
- Line 94: all slots plus `None` placeholders so a farmer may stay unmatched.
- Line 96: loop over every distinct permutation of slots; Line 97: add up gains of assigned pairs and keep the maximum; Line 98: return it.


## 2.3  `agrinegotiate/metrics.py`

**Role of this file.** Turns the raw results of many evaluation episodes into the headline numbers (efficiency E, farmer share S_F, agreement rate, Nash distance and so on). A key rule, stated in the docstring: **ratios of sums across episodes, never means of ratios.** Averaging per-episode ratios is biased because episodes with tiny gains get the same weight as large ones.

**Definitions.**
- E = (farmer gain + buyer gain) / G\* (efficiency).
- FGC = farmer gain / G\*, BGC = buyer gain / G\* (gain captured), so E = FGC + BGC.
- S_F = FGC / E = the farmer's share of the captured gain. Undefined when almost nothing was traded or either side lost money.

**Lines 1–8**
```python
   1  """Evaluation metrics. Ratios of sums across episodes, never means of ratios."""
   2  from __future__ import annotations
   3  
   4  from typing import Iterable
   5  
   6  import numpy as np
   7  
   8  E_MIN_FOR_SHARE = 0.05
```
- Line 1: docstring with the ratio-of-sums rule. Line 2: lazy annotations. Line 4: `Iterable` type hint. Line 6: NumPy.
- Line 8: `E_MIN_FOR_SHARE = 0.05`. If efficiency is below 5% the farmer share is a ratio of two tiny numbers and is meaningless, so we report `None` instead. This is why some plots show "n/a".

**Lines 11–16**
```python
  11  def gini(x) -> float:
  12      x = np.asarray(x, float)
  13      assert (x >= -1e-12).all(), "Gini needs non-negative inputs (use price received, not gains)"
  14      if x.size == 0 or x.mean() <= 0:
  15          return float("nan")
  16      return float(np.abs(x[:, None] - x[None, :]).sum() / (2 * x.size ** 2 * x.mean()))
```
`gini` computes the Gini inequality index of a vector (0 = perfect equality, near 1 = one person has everything).
- Line 13: it must be fed *non-negative* values (price received, not gains) or the formula is invalid; the `assert` enforces that with a small tolerance.
- Lines 14–15: if the array is empty or its mean is ≤ 0, return NaN.
- Line 16: Σ|xi − xj| over all pairs divided by 2·n²·mean. `x[:, None] − x[None, :]` builds all pairwise differences by broadcasting.

**Lines 19–24**
```python
  19  def jain(x) -> float:
  20      x = np.asarray(x, float)
  21      assert (x >= -1e-12).all(), "Jain's index needs non-negative inputs"
  22      if x.size == 0 or (x ** 2).sum() <= 0:
  23          return float("nan")
  24      return float(x.sum() ** 2 / (x.size * (x ** 2).sum()))
```
`jain` computes Jain's fairness index: (Σx)² / (n·Σx²). It equals 1 when everyone gets the same and falls toward 1/n when one gets everything. Same non-negativity assert and NaN guard as `gini`.

**Lines 27–34**
```python
  27  def aggregate(results: Iterable) -> dict:
  28      """Aggregate a list of EpisodeResult into the reported metrics."""
  29      results = list(results)
  30      n_total = len(results)
  31      used = [r for r in results if r.g_star > 1e-9]
  32      out = {"episodes": n_total, "episodes_no_trade_possible": n_total - len(used)}
  33      if not used:
  34          return out
```
`aggregate(results)` is the main function. It receives a list of `EpisodeResult` objects (defined in `env/core.py`).
- Line 29: convert any iterable to a list so it can be reused.
- Line 30: `n_total`, number of episodes.
- Line 31: `used` keeps only episodes where some gain was possible (`g_star` greater than 1e-9). Episodes where no trade could help anyone cannot say anything about fairness.
- Line 32: start the output dictionary with how many episodes there were and how many had no possible trade.
- Lines 33–34: if nothing is usable, return early.
- Lines 32–34: the result dict starts with the episode counts; if no episode had any possible gain (`g_star` ≈ 0) it returns early because every ratio would divide by zero.

**Lines 35–40**
```python
  35      g_star = sum(r.g_star for r in used)
  36      f_gain = sum(float(r.farmer_gain.sum()) for r in used)
  37      b_gain = sum(float(r.buyer_gain.sum()) for r in used)
  38      fees = sum(float(r.buyer_fees.sum()) for r in used)
  39      E, FGC, BGC = (f_gain + b_gain) / g_star, f_gain / g_star, b_gain / g_star
  40      assert abs(E - (FGC + BGC)) < 1e-9, "identity E = FGC + BGC violated"
```
Sum across *all* used episodes (the ratio-of-sums rule):
- Line 35: `g_star` total best possible gains. Line 36: total farmer gain. Line 37: total buyer gain. Line 38: total fees buyers paid (from the fairness mechanism).
- Line 39: E, FGC and BGC as sums over G\*.
- Line 40: a sanity `assert` that E = FGC + BGC exactly; if the accounting were wrong the program stops.
- Line 40: a built-in **assertion of the identity E = FGC + BGC**, so any bookkeeping bug in the env crashes loudly instead of producing wrong tables.

**Lines 41–43**
```python
  41      deals = [d for r in used for d in r.deals]
  42      n_opt = sum(r.n_opt_pairs for r in used)
  43      phis = np.array([d["phi"] for d in deals if d["phi"] is not None], float)
```
- Line 41: flatten the list of deals from all episodes (each deal is a dict with price, share `phi`, time `t`, shortfall `shortfall` and gain `g_t`).
- Line 42: `n_opt` is the total number of pairs in the optimal matchings, so the agreement rate is deals made divided by deals that were possible.
- Line 43: collect each deal's farmer share φ into a NumPy array, ignoring `None`.

**Lines 44–53**
```python
  44      out.update(
  45          E=E, FGC=FGC, BGC=BGC, BGC_net_of_fees=(b_gain - fees) / g_star,
  46          S_F=(FGC / E) if (E >= E_MIN_FOR_SHARE and FGC >= 0 and BGC >= 0) else None,
  47          agreement_rate=len(deals) / n_opt if n_opt else None,
  48          deals=len(deals), fees_over_gstar=fees / g_star,
  49          failed_accepts_per_episode=sum(r.failed_accepts for r in used) / len(used),
  50          decay_loss=sum(r.decay_loss for r in used) / g_star,
  51          fallback_rate=float(np.mean(np.concatenate([~r.sold for r in used]))),
  52          price_share=float(np.mean(np.concatenate([r.price_received / r.level for r in used]))),
  53      )
```
`out.update(...)` adds the headline metrics:
- `E, FGC, BGC` as above. `BGC_net_of_fees` is the buyer's gain after paying fees.
- `S_F` is FGC/E only when E ≥ 0.05 and both FGC and BGC are non-negative; otherwise `None`.
- `agreement_rate` deals/possible deals (guard for zero division).
- `deals` the count. `fees_over_gstar` fees as a share of G\*.
- `failed_accepts_per_episode` how often an agent tried to accept an offer that was no longer valid (expired or replaced), a sign of delay trouble.
- `decay_loss` value lost to spoilage as a share of G\*.
- `fallback_rate` share of farmers who ended with no sale (`~r.sold` negates the boolean array).
- `price_share` average price received relative to the market level.

**Lines 54–63**
```python
  54      if deals:
  55          out.update(
  56              phi_mean=float(phis.mean()) if phis.size else None,
  57              phi_median=float(np.median(phis)) if phis.size else None,
  58              phi_p10=float(np.percentile(phis, 10)) if phis.size else None,
  59              nash_distance=float(np.abs(phis - 0.5).mean()) if phis.size else None,
  60              low_share_deal_rate=float((phis < 0.1).mean()) if phis.size else None,
  61              time_to_agreement=float(np.mean([d["t"] for d in deals])),
  62              j_c=float(np.mean([d["shortfall"] / d["g_t"] for d in deals if d["g_t"] > 1e-9] or [0.0])),
  63          )
```
If there was at least one deal, add distribution metrics:
- `phi_mean`, `phi_median`, `phi_p10`: mean, median and 10th percentile of the farmer share per deal (the 10th percentile shows how bad the worst deals are).
- `nash_distance`: mean |φ − 0.5|, the average distance from the even split.
- `low_share_deal_rate`: fraction of deals where the farmer got under 10% of the gain.
- `time_to_agreement`: average step of agreement.
- `j_c`: the constraint cost for the fairness mechanism, the average fee shortfall relative to gains; this is what the dual-ascent update tries to push below ε (0.02).
Every one is guarded with `if phis.size` so empty arrays do not crash.

**Lines 64–66**
```python
  64      else:
  65          out.update(phi_mean=None, phi_median=None, phi_p10=None, nash_distance=None,
  66                     low_share_deal_rate=None, time_to_agreement=None, j_c=None)
```
Else branch: if there were no deals at all, set all the distribution metrics to `None` so the output always has the same keys.

**Lines 67–71**
```python
  67      multi = [r for r in used if r.price_received.size > 1]
  68      if multi:
  69          out["gini_price_received"] = float(np.nanmean([gini(r.price_received) for r in multi]))
  70          out["jain_price_received"] = float(np.nanmean([jain(r.price_received) for r in multi]))
  71      return out
```
Inequality across farmers, only for multi-farmer episodes (`price_received.size > 1`): the average Gini and Jain of the prices farmers received, using `nanmean` to ignore NaNs. For one-farmer scenarios these are skipped because inequality needs at least two people.


## 2.4  `agrinegotiate/env/core.py`

The heart of the project: one `Market` object simulates one episode of the negotiation game. Everything else (training, evaluation, dashboard) calls `reset`, `observe`, `step`, `result`. Sides are indexed 0 = farmer (F), 1 = buyer (B). Every (farmer i, buyer j) pair has its own negotiation *thread*, so arrays are shaped `[side, N, M, ...]`.

**Lines 1–14**
```python
   1  """Market environment core.
   2  
   3  One `Market` simulates one episode: N farmers, M buyers, one negotiation thread
   4  per (farmer, buyer) pair. Farmers reach the platform over an intermittent link.
   5  
   6  Step order, as seen from the caller:
   7      reset()                      -> state at t = 0
   8      observe()                    -> observations and action masks at t
   9      step(actions)                -> messages enqueued with send step t, walk-aways settle,
  10                                      t += 1, links update, deliverable messages are delivered
  11                                      (deals close), deadlines apply.
  12  
  13  Sides are indexed 0 = farmer, 1 = buyer throughout.
  14  """
```
- Module docstring: states the step order. `reset()` → `observe()` → `step(actions)`. In `step`, messages are first *enqueued* with send-time t, walk-aways settle, time advances, the link state updates, then deliverable messages are delivered (deals close) and deadlines apply. **Likely question: "in what order do things happen in one step?"** Answer: send → settle walk-aways → t+1 → links update → deliver → deadlines.

**Lines 15–15**
```python
  15  from __future__ import annotations
```
- `from __future__ import annotations` lets type hints refer to classes not yet defined (and keeps `list[dict]` style hints safe).

**Lines 17–23**
```python
  17  from dataclasses import dataclass, field
  18  from typing import Optional
  19  
  20  import numpy as np
  21  
  22  from .. import yardsticks
  23  from ..scenarios import Scenario
```
- Imports: `dataclass/field` for the result record, `Optional` for nullable types, NumPy for all arrays, `yardsticks` (gains, Nash price, max gains) and `Scenario` (the parameter set).

**Lines 25–26**
```python
  25  F, B = 0, 1
  26  OBS_DIM = 22
```
- `F, B = 0, 1` side indices. `OBS_DIM = 22` is the number of observation features each agent sees.

**Lines 28–31**
```python
  28  # observation indices
  29  (O_ROLE, O_TIME, O_DEADLINE, O_WALKAWAY, O_SIGNAL, O_FRESH, O_OWN_PRICE, O_OWN_LIVE, O_OWN_AGE,
  30   O_OWN_DELIVERED, O_CP_PRICE, O_CP_HAS, O_CP_AGE, O_CP_VALID, O_LINK, O_ROUNDS, O_ALT_PRICE,
  31   O_ALT_HAS, O_N_OPEN, O_CAPACITY, O_LAMBDA, O_TAU) = range(OBS_DIM)
```
- Names for each observation index via tuple-unpacking of `range(22)`: `O_ROLE=0 … O_TAU=21`. Using names instead of magic numbers makes the observe code readable. Groups: role/time/deadline/walk-away/signal/freshness (task basics), own offer (price/live/age/delivered), counterparty offer (price/has/age/valid), link state, rounds, best alternative price on other threads, number of open threads, capacity, fee λ and floor τ (mechanism info).

**Lines 33–34**
```python
  33  # features hidden from delay-unaware agents
  34  DELAY_FEATURES = [O_OWN_AGE, O_OWN_DELIVERED, O_CP_AGE, O_CP_VALID, O_LINK]
```
- `DELAY_FEATURES`: the five features that reveal the network state (own offer age, whether it was delivered, counterparty offer age/validity, link up/down). A **delay-unaware** agent has these zeroed in `observe`, so we can test whether delay-awareness helps (hypothesis H5).

**Lines 37–53**
```python
  37  @dataclass
  38  class EpisodeResult:
  39      """Everything the metrics module needs about one finished episode."""
  40      scenario: str
  41      seed: int
  42      level: float
  43      g_star: float
  44      n_opt_pairs: int
  45      farmer_gain: np.ndarray        # U_F - o_F per farmer
  46      buyer_gain: np.ndarray         # sum of unit payoffs - k*o_B per buyer, before fees
  47      buyer_fees: np.ndarray         # fees charged per buyer
  48      price_received: np.ndarray     # per farmer: deal price or decayed fallback price
  49      sold: np.ndarray               # per farmer: True if sold on the platform
  50      deals: list = field(default_factory=list)   # dicts: i, j, price, t, phi, shortfall
  51      failed_accepts: int = 0
  52      decay_loss: float = 0.0
  53      events: Optional[list] = None
```
- `EpisodeResult` dataclass: everything `metrics.aggregate` needs from one finished episode.
- `level` = that episode's market price level; `g_star` = best achievable total gains (Hungarian); `n_opt_pairs` = number of pairs in that optimum.
- `farmer_gain`, `buyer_gain` (before fees), `buyer_fees` are per-agent arrays; `price_received` is the deal price, or the decayed fallback price if the farmer did not sell; `sold` flags platform sales.
- `deals` is a list of dicts (i, j, price, t, phi, shortfall, fee, g_t, level). `failed_accepts` counts accepts that arrived too late/invalid; `decay_loss` is the value lost to spoilage from waiting; `events` is an optional log used by the dashboard replay.
- `field(default_factory=list)` avoids the shared-mutable-default bug.

**Lines 56–61**
```python
  56  def _best_other(x: np.ndarray, axis: int) -> np.ndarray:
  57      """For every entry: the largest value among the *other* entries along `axis` (-inf if none)."""
  58      srt = np.sort(x, axis=axis)
  59      top = np.take(srt, [-1], axis=axis)
  60      second = np.take(srt, [-2], axis=axis) if x.shape[axis] > 1 else np.full_like(top, -np.inf)
  61      return np.where(x == top, second, top)
```
- `_best_other(x, axis)`: for every entry, the largest value among the *other* entries along an axis. Sort once; the maximum is `top`, the runner-up is `second`; an entry that *is* the maximum gets the runner-up, every other entry gets the maximum. Ties work because the runner-up then equals the maximum. With one thread there is nothing else, so −∞ is returned. This replaces a slow Python double loop and gives identical numbers.

**Lines 64–64**
```python
  64  class Market:
```
- `Market`: one episode simulator.

**Lines 65–69**
```python
  65      def __init__(self, sc: Scenario, mechanism=None, delay_aware: bool = True, log_events: bool = False):
  66          self.sc = sc
  67          self.mech = mechanism
  68          self.delay_aware = delay_aware
  69          self.log_events = log_events
```
- Constructor stores the scenario, optional mechanism (fee / floor object), whether the agent is delay-aware, and whether to log events (only the dashboard needs logs; logging off keeps training fast).

**Lines 70–70**
```python
  70          self.prices = np.linspace(sc.p_lo, sc.p_hi, sc.n_bins)
```
- The price grid: 21 evenly spaced prices from 0.20 to 1.20 (step 0.05). Actions index into this array.

**Lines 71–73**
```python
  71          self.K = sc.n_bins
  72          self.A_ACCEPT, self.A_WALK, self.A_WAIT = self.K, self.K + 1, self.K + 2
  73          self.n_actions = self.K + 3
```
- `K` = number of price bins. Action ids: 0..K−1 = OFFER(price k); K = ACCEPT; K+1 = WALK; K+2 = WAIT. So `n_actions = K+3 = 24`.

**Lines 74–74**
```python
  74          self.N, self.M = sc.n_farmers, sc.n_buyers
```
- N farmers, M buyers from the scenario.

**Lines 75–82**
```python
  75          if sc.duty >= 1.0:
  76              self.p_rec, self.p_fail = 1.0, 0.0
  77          else:
  78              self.p_rec = 1.0 / sc.outage_len
  79              self.p_fail = self.p_rec * (1.0 - sc.duty) / sc.duty
  80              assert 0.0 <= self.p_fail <= 1.0, "duty cycle / outage length combination is infeasible"
  81  
  82      # reset
```
- Gilbert–Elliott link parameters. If duty ≥ 1 the link is always up (`p_rec=1`, `p_fail=0`). Otherwise recovery probability `p_rec = 1/outage_len` (mean outage length = outage_len steps), and `p_fail` is chosen so the long-run fraction of time up equals duty: stationary up-prob = p_rec/(p_rec+p_fail) = duty ⇒ p_fail = p_rec·(1−duty)/duty. The assert rejects infeasible combinations (p_fail>1). **Likely question: how is the duty cycle enforced?** Via this algebra on a two-state Markov chain.

**Lines 83–87**
```python
  83      def reset(self, seed: int) -> None:
  84          sc, N, M = self.sc, self.N, self.M
  85          self.seed = int(seed)
  86          self.rng = np.random.default_rng(seed)
  87          rng = self.rng
```
- `reset(seed)`: starts a new episode. Stores the seed and builds a seeded NumPy generator, so every episode is exactly reproducible (needed for fixed evaluation sets and paired comparisons).

**Lines 88–88**
```python
  88          self.level = float(rng.choice(sc.level_samples)) if sc.level_samples else float(rng.uniform(sc.level_lo, sc.level_hi))
```
- `level`: this episode's market price level, uniform in [0.70, 1.10] — a common shock (a good or bad market day) that shifts everyone's values.

**Lines 89–89**
```python
  89          self.v = self.level * (1.0 - rng.uniform(sc.margin_lo, sc.margin_hi, M))
```
- Buyer values `v_j = level·(1 − margin)`, margin ~ U(margin_lo, margin_hi) per buyer. Margin is how far below the market level the buyer values the crop; wide margins (PERISH-HET: 0.03–0.40) create heterogeneous buyers.

**Lines 90–90**
```python
  90          self.o_b = np.full(M, sc.o_b * self.level)
```
- Buyer's outside option `o_b = 0.10·level` for every capacity unit (value of not buying).

**Lines 91–92**
```python
  91          theta = rng.uniform(sc.theta - sc.theta_spread, sc.theta + sc.theta_spread, N)
  92          self.o_f = theta * self.level
```
- Farmer reservation values `o_f = θ·level`, θ ~ U(0.45 ± spread). This is what the farmer gets by selling elsewhere (the fallback market), scaled by the level.

**Lines 93–96**
```python
  93          if sc.deadline_lo is None:
  94              self.deadline = np.full(N, sc.T, dtype=int)
  95          else:
  96              self.deadline = rng.integers(sc.deadline_lo, sc.deadline_hi + 1, N)
```
- Deadlines: if the scenario has none, every farmer's deadline is T (30). Otherwise each farmer gets an integer deadline in [lo, hi] (DEADLINE scenario 10–20): perishable crops must be sold by then.

**Lines 97–98**
```python
  97          self.sig_f = self.level * np.exp(rng.normal(0.0, sc.sigma_info, N)) if sc.sigma_info > 0 \
  98              else np.full(N, self.level)
```
- Farmer's private *signal* of the market level: `level·exp(N(0,σ))` (log-noise) when σ>0 (INFO scenario), else exact. Farmers see this noisy signal; buyers see the true level — information asymmetry.

**Lines 99–100**
```python
  99          self.cost_f = sc.base_cost + sc.h_farmer
 100          self.cost_b = sc.base_cost
```
- Per-step holding costs: farmers pay `base_cost + h_farmer` per step (spoilage/storage), buyers pay `base_cost`.

**Lines 101–101**
```python
 101          self.cap = np.full(M, sc.capacity, dtype=int)
```
- `cap`: remaining purchase capacity of each buyer (units they can still buy).

**Lines 102–102**
```python
 102          self.link = rng.random(N) < sc.duty if sc.duty < 1.0 else np.ones(N, dtype=bool)
```
- Initial link state: each farmer's link is up with probability `duty` (sampled from the stationary distribution) or always up if duty=1.

**Lines 103–103**
```python
 103          self.t = 0
```
- Clock `t = 0`.

**Lines 105–105**
```python
 105          self.open = np.ones((N, M), dtype=bool)
```
- `open[i,j]`: thread (i,j) is still alive. Starts all True.

**Lines 106–110**
```python
 106          shape = (2, N, M)
 107          self.sent_price = np.zeros(shape)
 108          self.sent_t = np.full(shape, -1, dtype=int)
 109          self.sent_id = np.full(shape, -1, dtype=int)
 110          self.sent_delivered = np.zeros(shape, dtype=bool)
```
- Message state arrays of shape `(2, N, M)`: **sent_*** = the latest offer a side has sent on a thread (price, send time, unique id, whether delivered yet). The sender sees its own latest offer immediately; the counterparty only after delivery.

**Lines 111–113**
```python
 111          self.seen_price = np.zeros(shape)       # seen[s] = offer from side s as delivered to the other side
 112          self.seen_t = np.full(shape, -1, dtype=int)
 113          self.seen_id = np.full(shape, -1, dtype=int)
```
- **seen_*** = the latest offer *as delivered to the other side*: `seen_price[s]` is side s's offer as visible to the opposite side. This split between sent and seen is exactly what models network delay.

**Lines 114–114**
```python
 114          self.pending_accept = np.zeros(shape, dtype=bool)
```
- `pending_accept[s,i,j]`: side s has sent an ACCEPT that is still in flight; they must wait (no further actions).

**Lines 115–116**
```python
 115          self.rounds = np.zeros((N, M), dtype=int)
 116          self._next_id = 0
```
- `rounds[i,j]`: number of offers made on the thread; `_next_id` gives each offer a unique id so accepts can reference the exact offer they accept.

**Lines 117–117**
```python
 117          self.queue: list[list[dict]] = [[] for _ in range(N)]
```
- `queue[i]`: per-farmer outbox of in-flight messages. Every message goes through the farmer's link, so one queue per farmer.

**Lines 119–130**
```python
 119          self.f_done = np.zeros(N, dtype=bool)
 120          self.f_sold = np.zeros(N, dtype=bool)
 121          self.f_U = np.zeros(N)
 122          self.f_price = np.zeros(N)
 123          self.b_done = np.zeros(M, dtype=bool)
 124          self.b_U = np.zeros(M)
 125          self.b_fees = np.zeros(M)
 126          self.b_cost_units = np.zeros(M)          # fee base (shortfall) accumulated per buyer
 127          self.deals: list[dict] = []
 128          self.failed_accepts = 0
 129          self.decay_loss = 0.0
 130          self.events: list[dict] = []
```
- Per-agent bookkeeping: farmer done/sold flags, utility `f_U`, final price `f_price`; buyer done flag, utility `b_U`, accumulated fees `b_fees`, and `b_cost_units` (the fee base: total shortfall). Also the deals list, failed-accept counter, spoilage-loss counter and event log.

**Lines 131–132**
```python
 131          self.g_star, pairs = yardsticks.max_gains(self.v, self.o_f, self.o_b, self.cap)
 132          self.n_opt_pairs = len(pairs)
```
- Computes `G*` once per episode with the Hungarian algorithm (best possible matching) and the number of pairs in it. This is the yardstick for efficiency.

**Lines 133–136**
```python
 133          self.done = False
 134          self._log(kind="reset", level=self.level, link=self.link.tolist())
 135  
 136      # helpers
```
- `done=False`; logs a reset event (used by the replay).

**Lines 137–140**
```python
 137      def _log(self, **ev) -> None:
 138          if self.log_events:
 139              ev["t"] = self.t
 140              self.events.append(ev)
```
- `_log`: appends an event dict with the current time, only when logging is enabled.

**Lines 142–146**
```python
 142      def _validity_end(self, send_t):
 143          """Last step at which an offer sent at send_t can still be accepted on delivery."""
 144          if self.sc.ttl is None:
 145              return np.full_like(np.asarray(send_t), self.sc.T)
 146          return np.minimum(np.asarray(send_t) + self.sc.ttl, self.sc.T)
```
- `_validity_end(send_t)`: last step an offer sent at `send_t` can still be accepted: `min(send_t + TTL, T)`. TTL (6) stops stale offers being accepted long after the market moved. With ttl=None offers live until T.

**Lines 148–150**
```python
 148      def farmer_walkaway(self, t=None) -> np.ndarray:
 149          t = self.t if t is None else t
 150          return self.o_f * self.sc.rho ** t
```
- Farmer's walk-away price at time t: `o_f·ρ^t`. Their outside option decays with ρ (the crop spoils), so the minimum acceptable price falls over time.

**Lines 152–154**
```python
 152      def buyer_walkaway(self, t=None) -> np.ndarray:
 153          t = self.t if t is None else t
 154          return self.v * self.sc.rho ** np.asarray(t) - self.o_b
```
- Buyer's walk-away (max price they'd pay): `v·ρ^t − o_b` — discounted value minus the outside option. Accepts above this would give them negative gains.

**Lines 156–161**
```python
 156      def _floor(self, t) -> float:
 157          if self.mech is not None and getattr(self.mech, "floor_kappa", None) is not None:
 158              return self.mech.floor_kappa * self.level * self.sc.rho ** t
 159          return -np.inf
 160  
 161      # observe
```
- Hard price floor (mechanism): `κ·level·ρ^t` when the mechanism has `floor_kappa`, otherwise −∞ (no floor). Buyer offers and accepts below it are masked out.

**Lines 162–165**
```python
 162      def observe(self):
 163          """Returns obs[2, N, M, OBS_DIM], mask[2, N, M, n_actions], active[2, N, M]."""
 164          sc, N, M, t, T = self.sc, self.N, self.M, self.t, self.sc.T
 165          rho_t = sc.rho ** t
```
- `observe()` builds all agents' observations at once. Shapes: `obs[2,N,M,22]`, `mask[2,N,M,24]`, `active[2,N,M]`. `rho_t = ρ^t` is the freshness factor.

**Lines 166–168**
```python
 166          obs = np.zeros((2, N, M, OBS_DIM), dtype=np.float32)
 167          mask = np.zeros((2, N, M, self.n_actions), dtype=bool)
 168          active = np.stack([self.open & ~self.f_done[:, None], self.open & ~self.b_done[None, :]])
```
- Allocates zero arrays. `active[s]`: the thread is open and that side's agent isn't finished (farmer done / buyer done broadcasted along the right axis).

**Lines 170–174**
```python
 170          walk = np.stack([np.broadcast_to(self.farmer_walkaway()[:, None], (N, M)),
 171                           np.broadcast_to(self.buyer_walkaway()[None, :], (N, M))])
 172          sig = np.stack([np.broadcast_to(self.sig_f[:, None], (N, M)), np.full((N, M), self.level)])
 173          dl = np.stack([np.broadcast_to((self.deadline - t)[:, None], (N, M)), np.full((N, M), T - t)])
 174          ttl_norm = float(sc.ttl) if sc.ttl is not None else float(T)
```
- Precomputes per-side arrays broadcast to `[N,M]`: walk-away price, signal (farmer: noisy; buyer: true level), steps left to deadline (farmer's own; buyer's is T−t), and `ttl_norm` to normalise validity.

**Lines 176–180**
```python
 176          for s in (F, B):
 177              o = 1 - s
 178              has_own = self.sent_id[s] >= 0
 179              has_cp = self.seen_id[o] >= 0
 180              cp_valid_left = np.where(has_cp, self._validity_end(self.seen_t[o]) - t, 0)
```
- Loop over both sides. `o = 1−s` is the opposite side. `has_own`/`has_cp` say whether this side has sent / the counterparty's offer has been seen. `cp_valid_left` = steps left before the counterparty's offer expires.

**Lines 181–195**
```python
 181              ob = obs[s]
 182              ob[..., O_ROLE] = s
 183              ob[..., O_TIME] = t / T
 184              ob[..., O_DEADLINE] = np.maximum(dl[s], 0) / T
 185              ob[..., O_WALKAWAY] = walk[s]
 186              ob[..., O_SIGNAL] = sig[s]
 187              ob[..., O_FRESH] = rho_t
 188              ob[..., O_OWN_PRICE] = np.where(has_own, self.sent_price[s], 0.0)
 189              ob[..., O_OWN_LIVE] = has_own
 190              ob[..., O_OWN_AGE] = np.where(has_own, t - self.sent_t[s], 0) / T
 191              ob[..., O_CP_PRICE] = np.where(has_cp, self.seen_price[o], 0.0)
 192              ob[..., O_CP_HAS] = has_cp
 193              ob[..., O_CP_AGE] = np.where(has_cp, t - self.seen_t[o], 0) / T
 194              ob[..., O_CP_VALID] = np.clip(cp_valid_left, 0, None) / ttl_norm
 195              ob[..., O_ROUNDS] = self.rounds / T
```
- Fills the per-thread features (all normalised to roughly [0,1]): role, time/T, remaining deadline/T, own walk-away, signal, freshness, own last price/live/age, counterparty last price/has/age, validity left, rounds/T. For each side, `o = 1−s` is the opposite side; `has_own`/`has_cp` say whether this side has sent / the counterparty's offer has been seen; `cp_valid_left` = steps left before the counterparty's offer expires.

**Lines 196–202**
```python
 196              # the agent's other threads: best counterpart offer elsewhere and share of threads still open
 197              offer = np.where(has_cp & self.open & (cp_valid_left > 0), self.seen_price[o], np.nan)
 198              ax = 1 if s == F else 0                              # axis along which this agent's threads run
 199              best = _best_other(np.nan_to_num(offer if s == F else -offer, nan=-np.inf), ax)
 200              ob[..., O_ALT_PRICE] = np.where(np.isfinite(best), best if s == F else -best, 0.0)
 201              ob[..., O_ALT_HAS] = np.isfinite(best)
 202              ob[..., O_N_OPEN] = self.open.sum(axis=ax, keepdims=True) / (M if s == F else N)
```
- *The agent's other threads* (multi-thread awareness). `offer` = the counterparty's currently valid offers (NaN where there is none). `ax` is the axis along which this agent's threads run (a farmer's threads run across buyers, a buyer's across farmers). A farmer wants the **highest** alternative offer, a buyer the **lowest**, so buyers use the negated prices and the same max routine. `O_ALT_PRICE` gets the best offer elsewhere (0 if none), `O_ALT_HAS` flags whether one exists, `O_N_OPEN` is the fraction of this agent's threads still open. This lets agents compare partners, which matters in thin markets.

**Lines 203–206**
```python
 203              if s == F:
 204                  ob[..., O_OWN_DELIVERED] = self.sent_delivered[F]
 205                  ob[..., O_LINK] = self.link[:, None]
 206                  ob[..., O_CAPACITY] = 1.0
```
- Farmer-only features: whether own offer was delivered, whether the link is up, capacity = 1.

**Lines 207–210**
```python
 207              else:
 208                  ob[..., O_CAPACITY] = (self.cap / sc.capacity)[None, :]
 209                  if self.mech is not None:
 210                      ob[..., O_LAMBDA] = getattr(self.mech, "lam", 0.0) / getattr(self.mech, "lam_max", 1.0)
```
- Buyer features: remaining capacity share; if a mechanism exists, the current λ normalised by its maximum (so the buyer can see the fee price).

**Lines 211–212**
```python
 211              if self.mech is not None:
 212                  ob[..., O_TAU] = getattr(self.mech, "tau", 0.0)
```
- Mechanism's target-share τ is shown to both sides when there is a mechanism.

**Lines 214–215**
```python
 214          if not self.delay_aware:
 215              obs[..., DELAY_FEATURES] = 0.0
```
- Delay-unaware agents get the five delay features zeroed.

**Lines 217–221**
```python
 217          # action masks
 218          P = self.prices
 219          floor = self._floor(t)
 220          # farmer: never ask below own current walk-away price
 221          f_ok = P[None, None, :] >= (self.farmer_walkaway()[:, None, None] - 1e-9)
```
- **Action masks.** Farmer: may only offer prices ≥ own current walk-away (never sell below reservation).

**Lines 222–227**
```python
 222          # buyer: never bid above own value at the last step the bid could still be accepted,
 223          # so every deal is individually rational when it closes
 224          b_cap = self.buyer_walkaway(self._validity_end(t))                  # [M]
 225          b_ok = (P[None, None, :] <= b_cap[None, :, None] + 1e-9) & (P[None, None, :] >= floor - 1e-9)
 226          mask[F, :, :, :self.K] = f_ok
 227          mask[B, :, :, :self.K] = b_ok
```
- Buyer: may only bid up to the walk-away *at the last step the offer could be accepted* (so every deal is individually rational even if delivery is delayed), and not below the price floor. Offers masks are copied to both sides' first K actions.

**Lines 228–231**
```python
 228          for s in (F, B):
 229              o = 1 - s
 230              has_cp = self.seen_id[o] >= 0
 231              in_time = (t + 1) <= self._validity_end(self.seen_t[o])
```
- For each side: does the counterparty have an offer, and is it still valid next step (`t+1 ≤ validity_end`)?

**Lines 232–236**
```python
 232              if s == F:
 233                  rational = self.seen_price[B] >= (self.farmer_walkaway(t + 1)[:, None] - 1e-9)
 234              else:
 235                  worst = self.buyer_walkaway(self._validity_end(self.seen_t[F]))   # [N, M] via broadcast below
 236                  rational = (self.seen_price[F] <= worst + 1e-9) & (self.seen_price[F] >= floor - 1e-9)
```
- ACCEPT rationality: farmer accepts only if the buyer's offer ≥ their walk-away at t+1; buyer accepts only if the farmer's price ≤ worst-case walk-away and ≥ floor. This guarantees no deal with negative gains (asserted in `_deal`).

**Lines 237–239**
```python
 237              mask[s, :, :, self.A_ACCEPT] = has_cp & in_time & rational
 238              mask[s, :, :, self.A_WALK] = True
 239              mask[s, :, :, self.A_WAIT] = True
```
- ACCEPT allowed iff has offer ∧ in time ∧ rational; WALK and WAIT always allowed.

**Lines 240–242**
```python
 240              waiting = self.pending_accept[s]
 241              mask[s][waiting] = False
 242              mask[s][waiting, self.A_WAIT] = True
```
- If this side already has an accept in flight, it can only WAIT.

**Lines 243–244**
```python
 243              mask[s][~active[s]] = False
 244              mask[s][~active[s], self.A_WAIT] = True
```
- Inactive threads (closed or agent finished) can only WAIT (a harmless no-op).

**Lines 245–245**
```python
 245          return obs, mask, active
```
- Returns observations, masks, active flags.

**Lines 247–263**
```python
 247      # step
 248      def step(self, actions: np.ndarray, mask: Optional[np.ndarray] = None) -> bool:
 249          """actions[2, N, M] integer actions per side and thread. Returns done.
 250  
 251          Pass the mask returned by observe() to have every action checked against it.
 252          """
 253          assert not self.done
 254          t = self.t
 255          for s in (F, B):
 256              for i in range(self.N):
 257                  for j in range(self.M):
 258                      if not self.open[i, j] or (self.f_done[i] if s == F else self.b_done[j]):
 259                          continue
 260                      a = int(actions[s, i, j])
 261                      if mask is not None:
 262                          assert mask[s, i, j, a], f"masked action {a} by side {s} on thread {(i, j)} at t={t}"
 263                      self._apply(s, i, j, a, t)
```
- `step(actions, mask)`: applies all agents' simultaneous actions. `actions[2,N,M]`. If a mask is passed, every action is asserted legal (a debug safety net). Loops over side, farmer i, buyer j, skips closed threads or finished agents, checks the action, and hands it to `_apply`.

**Lines 265–265**
```python
 265          self._settle_idle(t)                      # walk-aways settle at the step they happen
```
- Walk-aways settle immediately: any agent with no open thread is finished at this step.

**Lines 267–273**
```python
 267          # advance time, update links, deliver
 268          self.t = t = t + 1
 269          if self.p_fail > 0:
 270              u = self.rng.random(self.N)
 271              self.link = np.where(self.link, u >= self.p_fail, u < self.p_rec)
 272          self._log(kind="links", link=self.link.tolist())
 273          self._deliver(t)
```
- Time advances. If the link can fail, sample a uniform `u` per farmer: an up link stays up if `u ≥ p_fail`; a down link recovers if `u < p_rec` (vectorised Markov-chain update). Then `_deliver(t)` moves ready messages (those sent before t, over links that are up) and may close deals.

**Lines 274–279**
```python
 274          self._settle_idle(t)
 275          # deadlines
 276          for i in range(self.N):
 277              if not self.f_done[i] and t >= self.deadline[i]:
 278                  self._farmer_fallback(i, t)
 279          self._settle_idle(t)
```
- Settle idle agents again; farmers whose deadline has passed fall back to their outside option; settle again.

**Lines 280–286**
```python
 280          if t >= self.sc.T:
 281              for i in range(self.N):
 282                  if not self.f_done[i]:
 283                      self._farmer_fallback(i, t)
 284              for j in range(self.M):
 285                  if not self.b_done[j]:
 286                      self._buyer_end(j, t)
```
- At t ≥ T everything still open is closed: farmers fall back, buyers end (unused capacity gets the outside option).

**Lines 287–288**
```python
 287          self.done = bool(self.f_done.all() and self.b_done.all())
 288          return self.done
```
- The episode is done when all farmers and buyers are done; returns that flag.

**Lines 290–290**
```python
 290      # internals
```
- Helpers that change market state follow: `_apply`, `_deliver`, `_deal`, and the fallback/settle routines.

**Lines 291–319**
```python
 291      def _apply(self, s: int, i: int, j: int, a: int, t: int) -> None:
 292          """One side's action on one thread: WAIT does nothing, WALK closes the thread, OFFER/ACCEPT queue a message."""
 293          o = 1 - s
 294          if a == self.A_WAIT:
 295              return
 296          if a == self.A_WALK:
 297              self.open[i, j] = False
 298              self.pending_accept[:, i, j] = False
 299              self._log(kind="walk", side=s, i=i, j=j)
 300              return
 301          lost = self.sc.drop_mode and not self.link[i]          # drop mode: sent while the link is down = gone
 302          if a == self.A_ACCEPT:
 303              self.pending_accept[s, i, j] = True
 304              msg = dict(kind="accept", side=s, i=i, j=j, ref=int(self.seen_id[o, i, j]),
 305                         price=float(self.seen_price[o, i, j]), send_t=t)
 306          else:
 307              self._next_id += 1
 308              self.sent_price[s, i, j] = self.prices[a]
 309              self.sent_t[s, i, j] = t
 310              self.sent_id[s, i, j] = self._next_id
 311              self.sent_delivered[s, i, j] = False
 312              self.rounds[i, j] += 1
 313              msg = dict(kind="offer", side=s, i=i, j=j, id=self._next_id, price=float(self.prices[a]), send_t=t)
 314          self._log(kind="send", lost=bool(lost), **{k: v for k, v in msg.items() if k != "kind"}, msg=msg["kind"])
 315          if lost:
 316              if msg["kind"] == "accept":
 317                  self.pending_accept[s, i, j] = False
 318              return
 319          self.queue[i].append(msg)
```
- `_apply(s, i, j, a, t)`: one side's action on one thread. **WAIT** does nothing. **WALK** closes the thread and cancels pending accepts on it. For OFFER and ACCEPT: `lost` is true in drop mode when the farmer's link is down (the message vanishes). ACCEPT marks the accept as pending and builds an accept message that references the exact offer id and price being accepted. OFFER increments the global id, records the offer as the side's latest sent one (price from the grid, send time, id, undelivered), bumps the round count and builds the offer message. The send is logged; a lost accept loses its pending flag; any other message is appended to the farmer's queue, to be delivered later by `_deliver`.

**Lines 321–329**
```python
 321      def _deliver(self, t: int) -> None:
 322          sc = self.sc
 323          ready = []
 324          for i in range(self.N):
 325              if sc.drop_mode or self.link[i]:
 326                  keep = []
 327                  for m in self.queue[i]:
 328                      (ready if m["send_t"] < t else keep).append(m)
 329                  self.queue[i] = keep
```
- `_deliver`: for each farmer whose link is up (or in drop mode), messages with `send_t < t` are *ready*; others stay queued. This is store-and-forward: queued messages are released when the link returns.

**Lines 330–333**
```python
 330          if not ready:
 331              return
 332          tie = self.rng.random(len(ready))
 333          order = sorted(range(len(ready)), key=lambda k: (ready[k]["send_t"], tie[k]))
```
- Random tie-break values, then sort by (send time, tie) so earlier messages resolve first and ties are fair/random, not index-biased.

**Lines 334–339**
```python
 334          for k in order:
 335              m = ready[k]
 336              s, i, j = m["side"], m["i"], m["j"]
 337              o = 1 - s
 338              if not self.open[i, j]:
 339                  continue
```
- For each ready message: skip if the thread closed in the meantime.

**Lines 340–347**
```python
 340              if m["kind"] == "offer":
 341                  if m["id"] != self.sent_id[s, i, j]:
 342                      continue                                  # superseded before it arrived
 343                  self.seen_price[s, i, j] = m["price"]
 344                  self.seen_t[s, i, j] = m["send_t"]
 345                  self.seen_id[s, i, j] = m["id"]
 346                  self.sent_delivered[s, i, j] = True
 347                  self._log(kind="deliver", msg="offer", side=s, i=i, j=j, price=m["price"], send_t=m["send_t"])
```
- Offers: ignored if superseded by a newer offer from the same side (id mismatch). Otherwise written into the `seen_*` arrays so the counterparty can now see it; marks own offer as delivered.

**Lines 348–351**
```python
 348              else:
 349                  valid = (m["ref"] == self.sent_id[o, i, j]
 350                           and t <= int(self._validity_end(self.sent_t[o, i, j]))
 351                           and not self.f_done[i] and self.cap[j] > 0)
```
- Accepts are valid only if they reference the counterparty's *current* offer id, arrive before its TTL expires, the farmer is not already done, and the buyer still has capacity.

**Lines 352–357**
```python
 352                  if valid:
 353                      self._deal(i, j, m["price"], t)
 354                  else:
 355                      self.pending_accept[s, i, j] = False
 356                      self.failed_accepts += 1
 357                      self._log(kind="accept_failed", side=s, i=i, j=j, price=m["price"], send_t=m["send_t"])
```
- Valid → `_deal`. Invalid → the accept is cancelled, `failed_accepts` incremented (the cost of delay/staleness), event logged.

**Lines 359–364**
```python
 359      def _deal(self, i: int, j: int, p: float, t: int) -> None:
 360          sc = self.sc
 361          d = sc.rho ** t
 362          g_t = yardsticks.gains(self.v[j], self.o_f[i], self.o_b[j], sc.rho, t)
 363          assert g_t > -1e-9, "deal closed with negative gains from trade; action masking is broken"
 364          phi = yardsticks.gain_share(p, self.v[j], self.o_f[i], self.o_b[j], sc.rho, t) if g_t > 1e-12 else None
```
- `_deal`: discount factor `d = ρ^t`; gains from trade `g_t`; assert ≥ 0 (guards the masking logic). Gain share φ computed if gains are positive.

**Lines 365–366**
```python
 365          shortfall = self.mech.shortfall(self, i, j, p, t) if self.mech is not None else 0.0
 366          fee = getattr(self.mech, "lam", 0.0) * shortfall
```
- Asks the mechanism for the *shortfall* (how far below the target price the deal was) and computes the fee = λ·shortfall. Without a mechanism the shortfall is 0 and `getattr(None, "lam", 0.0)` is 0, so the fee is 0.

**Lines 367–370**
```python
 367          self.f_U[i] = p - self.cost_f * t
 368          self.f_price[i] = p
 369          self.f_sold[i] = True
 370          self.f_done[i] = True
```
- Farmer: utility = price − holding cost·t; recorded as sold and finished.

**Lines 371–374**
```python
 371          self.b_U[j] += self.v[j] * d - p - self.cost_b * t
 372          self.b_fees[j] += fee
 373          self.b_cost_units[j] += shortfall
 374          self.cap[j] -= 1
```
- Buyer: gets `v·d − p − cost·t`, pays the fee, accumulates shortfall units, uses one unit of capacity.

**Lines 375–375**
```python
 375          self.decay_loss += (self.v[j] - self.o_f[i]) * (1.0 - d)
```
- Spoilage loss: `(v − o_f)(1 − d)` — value destroyed by closing at t rather than 0.

**Lines 376–378**
```python
 376          self.deals.append(dict(i=i, j=j, price=float(p), t=t, phi=phi, shortfall=float(shortfall),
 377                                 fee=float(fee), g_t=float(g_t), level=self.level))
 378          self._log(kind="deal", i=i, j=j, price=float(p), phi=phi, fee=float(fee))
```
- Records the deal dict and an event.

**Lines 379–381**
```python
 379          self._close_farmer(i)
 380          if self.cap[j] == 0:
 381              self._close_buyer(j)
```
- Closes all of the farmer's threads; if the buyer's capacity is now used up, closes all of that buyer's threads too. Pending accepts on those threads are cancelled by the two helpers below.

**Lines 383–389**
```python
 383      def _farmer_fallback(self, i: int, t: int) -> None:
 384          price = self.o_f[i] * self.sc.rho ** t
 385          self.f_U[i] = price - self.cost_f * t
 386          self.f_price[i] = price
 387          self.f_done[i] = True
 388          self._close_farmer(i)
 389          self._log(kind="fallback", i=i, price=float(price))
```
- `_farmer_fallback`: farmer sells at the decayed outside price `o_f·ρ^t` (utility minus holding cost); closes their threads.

**Lines 391–394**
```python
 391      def _buyer_end(self, j: int, t: int) -> None:
 392          self.b_U[j] += self.cap[j] * (self.o_b[j] - self.cost_b * t)
 393          self.b_done[j] = True
 394          self._close_buyer(j)
```
- `_buyer_end`: unused capacity earns the outside option `o_b − cost·t` each; closes the buyer's threads.

**Lines 396–403**
```python
 396      def _close_farmer(self, i: int) -> None:
 397          """Close every thread of farmer i and cancel accepts still in flight on them."""
 398          self.open[i, :] = False
 399          self.pending_accept[:, i, :] = False
 400  
 401      def _close_buyer(self, j: int) -> None:
 402          self.open[:, j] = False
 403          self.pending_accept[:, :, j] = False
```
- `_close_farmer` / `_close_buyer`: one place for "this agent is finished": mark every thread of farmer *i* (row *i*) or buyer *j* (column *j*) closed, and clear any accept still in flight on those threads, for both sides. Used by `_deal`, `_farmer_fallback` and `_buyer_end`.

**Lines 405–419**
```python
 405      def _settle_idle(self, t: int) -> None:
 406          """Agents with no open thread left are finished at step t."""
 407          changed = True
 408          while changed:
 409              changed = False
 410              for i in range(self.N):
 411                  if not self.f_done[i] and not self.open[i].any():
 412                      self._farmer_fallback(i, t)
 413                      changed = True
 414              for j in range(self.M):
 415                  if not self.b_done[j] and not self.open[:, j].any():
 416                      self._buyer_end(j, t)
 417                      changed = True
 418  
 419      # result
```
- `_settle_idle`: loops until no change: any farmer/buyer with no open thread is ended now. Looping handles cascades (a buyer ending can leave a farmer with no threads).

**Lines 420–424**
```python
 420      def rewards(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
 421          """Terminal rewards: farmer, buyer before fee, buyer fee base (shortfall)."""
 422          r_f = self.f_U - self.o_f
 423          r_b = self.b_U - self.sc.capacity * self.o_b
 424          return r_f, r_b, self.b_cost_units.copy()
```
- `rewards`: terminal rewards. Farmer: `U − o_f` (gain over the fallback). Buyer: `U − capacity·o_b`. The third array is the buyer's fee base (shortfall), used by the Lagrangian constraint.

**Lines 426–434**
```python
 426      def result(self) -> EpisodeResult:
 427          assert self.done
 428          r_f, r_b, _ = self.rewards()
 429          return EpisodeResult(
 430              scenario=self.sc.name, seed=self.seed, level=self.level, g_star=self.g_star,
 431              n_opt_pairs=self.n_opt_pairs, farmer_gain=r_f.copy(), buyer_gain=r_b.copy(),
 432              buyer_fees=self.b_fees.copy(), price_received=self.f_price.copy(), sold=self.f_sold.copy(),
 433              deals=list(self.deals), failed_accepts=self.failed_accepts, decay_loss=self.decay_loss,
 434              events=list(self.events) if self.log_events else None)
```
- `result()`: packs everything into an `EpisodeResult` (copies arrays so later resets don't change them). Only valid when `done`.


## 2.5  `agrinegotiate/env/pettingzoo_env.py`

Wraps `Market` in the standard PettingZoo `ParallelEnv` API so the benchmark works with off-the-shelf multi-agent tooling. One agent per farmer and per buyer; an agent's action is a vector with one thread-level action per counterpart.

**Lines 1–3**
```python
   1  """PettingZoo ParallelEnv wrapper around the market core, so the benchmark can be used
   2  with standard multi-agent tooling. One agent per farmer and per buyer; an agent's action
   3  is one thread-level action for each counterpart."""
```
- Docstring: standard multi-agent wrapper; one agent per farmer/buyer; one action per counterpart thread.

**Lines 4–12**
```python
   4  from __future__ import annotations
   5  
   6  import functools
   7  
   8  import numpy as np
   9  from gymnasium import spaces
  10  from pettingzoo import ParallelEnv
  11  
  12  from .core import Market, OBS_DIM, F, B
```
- Imports: `functools` (caching), NumPy, gymnasium `spaces`, PettingZoo `ParallelEnv`, and the core env.

**Lines 15–16**
```python
  15  class AgriNegotiateEnv(ParallelEnv):
  16      metadata = {"name": "agrinegotiate_v0"}
```
- `AgriNegotiateEnv(ParallelEnv)` with metadata name `agrinegotiate_v0`.

**Lines 18–24**
```python
  18      def __init__(self, scenario, mechanism=None, delay_aware: bool = True, reward_scale: float = 1.0):
  19          self.sc = scenario
  20          self.core = Market(scenario, mechanism=mechanism, delay_aware=delay_aware)
  21          self.reward_scale = reward_scale
  22          self.possible_agents = [f"farmer_{i}" for i in range(scenario.n_farmers)] + \
  23                                 [f"buyer_{j}" for j in range(scenario.n_buyers)]
  24          self.agents = []
```
- Constructor builds a `Market`, stores `reward_scale`, and lists agent names `farmer_0..`, `buyer_0..`. `agents` (currently alive) starts empty until `reset`.

**Lines 26–27**
```python
  26      def _n_threads(self, agent: str) -> int:
  27          return self.sc.n_buyers if agent.startswith("farmer") else self.sc.n_farmers
```
- `_n_threads`: a farmer has M threads (one per buyer); a buyer has N threads.

**Lines 29–35**
```python
  29      @functools.lru_cache(maxsize=None)
  30      def observation_space(self, agent):
  31          n = self._n_threads(agent)
  32          return spaces.Dict({
  33              "observation": spaces.Box(-np.inf, np.inf, (n, OBS_DIM), np.float32),
  34              "action_mask": spaces.MultiBinary([n, self.core.n_actions]),
  35          })
```
- `observation_space`: a Dict of the observation Box `[n_threads, 22]` and a MultiBinary action mask `[n_threads, n_actions]`. `lru_cache` is required by PettingZoo so spaces are stable objects.

**Lines 37–39**
```python
  37      @functools.lru_cache(maxsize=None)
  38      def action_space(self, agent):
  39          return spaces.MultiDiscrete([self.core.n_actions] * self._n_threads(agent))
```
- `action_space`: MultiDiscrete — one 24-way choice per thread.

**Lines 41–51**
```python
  41      def _views(self):
  42          obs, mask, _ = self.core.observe()
  43          self._mask = mask
  44          out = {}
  45          for a in self.agents:
  46              k = int(a.split("_")[1])
  47              if a.startswith("farmer"):
  48                  out[a] = {"observation": obs[F, k], "action_mask": mask[F, k].astype(np.int8)}
  49              else:
  50                  out[a] = {"observation": obs[B, :, k], "action_mask": mask[B, :, k].astype(np.int8)}
  51          return out
```
- `_views`: calls `observe()`, stores the mask for validation, and slices per-agent views: farmer k gets `obs[F, k]` (row k), buyer k gets `obs[B, :, k]` (column k).

**Lines 53–56**
```python
  53      def reset(self, seed=None, options=None):
  54          self.core.reset(0 if seed is None else int(seed))
  55          self.agents = list(self.possible_agents)
  56          return self._views(), {a: {} for a in self.agents}
```
- `reset`: resets the core (seed default 0), makes all agents alive, returns observations and empty infos.

**Lines 58–66**
```python
  58      def step(self, actions):
  59          N, M = self.sc.n_farmers, self.sc.n_buyers
  60          acts = np.full((2, N, M), self.core.A_WAIT, dtype=int)
  61          for a, act in actions.items():
  62              k = int(a.split("_")[1])
  63              if a.startswith("farmer"):
  64                  acts[F, k, :] = act
  65              else:
  66                  acts[B, :, k] = act
```
- `step`: starts from all-WAIT and writes each agent's action vector into its row (farmer) or column (buyer).

**Lines 67–67**
```python
  67          done = self.core.step(acts, self._mask)
```
- Steps the core, passing the stored mask so illegal actions are caught.

**Lines 68–74**
```python
  68          rewards = {a: 0.0 for a in self.agents}
  69          if done:
  70              r_f, r_b, c_b = self.core.rewards()
  71              lam = getattr(self.core.mech, "lam", 0.0) if self.core.mech is not None else 0.0
  72              for a in self.agents:
  73                  k = int(a.split("_")[1])
  74                  rewards[a] = self.reward_scale * float(r_f[k] if a.startswith("farmer") else r_b[k] - lam * c_b[k])
```
- Rewards are zero until the episode ends; then farmer reward = gain over fallback, buyer reward = gain − λ·fee-base (the Lagrangian reward), scaled by `reward_scale`.

**Lines 75–81**
```python
  75          obs = self._views()
  76          terms = {a: done for a in self.agents}
  77          truncs = {a: False for a in self.agents}
  78          infos = {a: {} for a in self.agents}
  79          if done:
  80              self.agents = []
  81          return obs, rewards, terms, truncs, infos
```
- Returns new observations, rewards, termination flags (all agents terminate together), no truncation, infos; clears `agents` at the end as PettingZoo requires.


## 2.6  `agrinegotiate/agents/scripted.py`

Hand-written (non-learning) negotiators. They serve three roles: training opponents, benchmarks (the professor's 'no benchmark' objection), and reference points (floor and ceiling). All except `OracleFairSplit` see only their own observation and mask — exactly like a learned policy.

**Lines 1–6**
```python
   1  """Scripted negotiators.
   2  
   3  Every agent except the two reference points sees only its own observation vector
   4  and action mask, exactly like a learned policy. `OracleFairSplit` is privileged by
   5  design: it is a reference point, not a competitor.
   6  """
```
- Docstring: the two reference points (TakeFallback = floor, OracleFairSplit = ceiling) are privileged/trivial by design, not competitors.

**Lines 7–15**
```python
   7  from __future__ import annotations
   8  
   9  import numpy as np
  10  
  11  from .. import yardsticks
  12  from ..env import core as C
  13  from ..env.core import F, B
  14  
  15  EPS = 1e-9
```
- Imports; `EPS = 1e-9` is a tolerance for floating-point price comparisons.

**Lines 18–28**
```python
  18  class ScriptedAgent:
  19      """Base class: one decision per open thread, from that thread's observation and mask."""
  20      name = "scripted"
  21  
  22      def __init__(self, role: int, sc, **kw):
  23          self.role, self.sc = role, sc
  24          self.prices = np.linspace(sc.p_lo, sc.p_hi, sc.n_bins)
  25          self.K = sc.n_bins
  26          self.A_ACCEPT, self.A_WALK, self.A_WAIT = self.K, self.K + 1, self.K + 2
  27          self.ttl = sc.ttl if sc.ttl is not None else sc.T
  28          self.start = sc.p_hi if role == F else sc.p_lo
```
- `ScriptedAgent` base class. Stores role and scenario, builds the same price grid and action ids as the env, `ttl` (offer lifetime), and `start` = opening price: a farmer starts at the top of the grid (p_hi), a buyer at the bottom (p_lo).

**Lines 30–32**
```python
  30      def reset(self, seed: int) -> None:
  31          self.rng = np.random.default_rng([seed, self.role, 7919])
  32          self.mem: dict = {}
```
- `reset`: seeded RNG from `[seed, role, 7919]` (a distinct stream per role) and an empty per-thread memory dict.

**Lines 34–40**
```python
  34      def act(self, env, obs, mask, active) -> np.ndarray:
  35          acts = np.full(active.shape, self.A_WAIT, dtype=int)
  36          for i, j in zip(*np.nonzero(active)):
  37              acts[i, j] = self.act_thread((i, j), obs[i, j], mask[i, j])
  38          return acts
  39  
  40      # helpers
```
- `act`: default WAIT everywhere; for each active thread calls `act_thread` with that thread's observation and mask.

**Lines 41–50**
```python
  41      def _bin_for(self, target: float, m: np.ndarray):
  42          """Valid bin closest to the target without being worse for me than the target."""
  43          valid = np.nonzero(m[:self.K])[0]
  44          if valid.size == 0:
  45              return None
  46          if self.role == F:
  47              ok = valid[self.prices[valid] >= target - EPS]
  48              return int(ok.min()) if ok.size else int(valid.max())
  49          ok = valid[self.prices[valid] <= target + EPS]
  50          return int(ok.max()) if ok.size else int(valid.min())
```
- `_bin_for(target, mask)`: choose the legal price bin nearest the target *without being worse for me*. Farmer: smallest legal price ≥ target (else the highest legal). Buyer: largest legal price ≤ target (else the lowest legal).

**Lines 52–53**
```python
  52      def _beats(self, cp_price: float, mine: float) -> bool:
  53          return cp_price >= mine - EPS if self.role == F else cp_price <= mine + EPS
```
- `_beats(cp_price, mine)`: is the counterparty's offer at least as good as what I'd offer? Farmer: cp ≥ mine; buyer: cp ≤ mine.

**Lines 55–61**
```python
  55      def _offer_or_wait(self, b: int, o: np.ndarray) -> int:
  56          """Send bin b unless the same offer is already live and not about to expire."""
  57          same = o[C.O_OWN_LIVE] > 0.5 and abs(o[C.O_OWN_PRICE] - self.prices[b]) < 1e-6
  58          age = round(float(o[C.O_OWN_AGE]) * self.sc.T)
  59          if same and age < self.ttl - 1:
  60              return self.A_WAIT
  61          return b
```
- `_offer_or_wait`: re-sending an identical live offer wastes nothing but spams; if my current offer is identical and not about to expire (age < ttl−1), WAIT; otherwise send bin b.

**Lines 63–68**
```python
  63      def _decide(self, b, o, m) -> int:
  64          if b is None:
  65              return self.A_WAIT
  66          if m[self.A_ACCEPT] and self._beats(float(o[C.O_CP_PRICE]), float(self.prices[b])):
  67              return self.A_ACCEPT
  68          return self._offer_or_wait(b, o)
```
- `_decide(b, obs, mask)`: the common decision rule. No valid bin → WAIT. If ACCEPT is legal and the counterparty's offer is at least as good as my target → ACCEPT. Else offer-or-wait.

**Lines 70–71**
```python
  70      def act_thread(self, key, o, m) -> int:
  71          raise NotImplementedError
```
- `act_thread` is abstract: each strategy implements it.

**Lines 74–75**
```python
  74  class TimeDependent(ScriptedAgent):
  75      """Faratin et al. time-dependent tactic. beta < 1 Boulware, 1 linear, > 1 conceder."""
```
- `TimeDependent`: Faratin et al.'s time-dependent concession tactic. β<1 Boulware (holds firm, concedes late), β=1 linear, β>1 conceder (concedes early).

**Lines 77–82**
```python
  77      def __init__(self, role, sc, beta: float = 1.0, start: float | None = None, name: str | None = None, **kw):
  78          super().__init__(role, sc)
  79          self.beta = beta
  80          if start is not None:
  81              self.start = start
  82          self.name = name or f"TD(beta={beta:g})"
```
- Constructor: stores β, an optional custom starting price, and a name (default `TD(beta=…)`).

**Lines 84–93**
```python
  84      def target(self, o) -> float:
  85          t = round(float(o[C.O_TIME]) * self.sc.T)
  86          left = round(float(o[C.O_DEADLINE]) * self.sc.T)
  87          horizon = max(1, t + left - 2)             # two steps are needed to close after an offer
  88          alpha = min(1.0, t / horizon) ** (1.0 / self.beta)
  89          reserve = float(o[C.O_WALKAWAY])
  90          return self.start + alpha * (reserve - self.start)
  91  
  92      def act_thread(self, key, o, m) -> int:
  93          return self._decide(self._bin_for(self.target(o), m), o, m)
```
- `target`: recover current time and steps-left from the normalised observation (multiply by T). `horizon = t + left − 2` (two steps are needed to close after an offer). Concession fraction α = min(1, t/horizon)^(1/β). Target = start + α·(reserve − start): move from the opening price towards the walk-away price as the deadline nears. `act_thread` chooses a bin for that target and applies the shared decision rule.

**Lines 96–96**
```python
  96  TACTIC_BETA = {"Boulware": 0.2, "Linear": 1.0, "Conceder": 5.0}     # <1 holds firm, 1 steady, >1 concedes early
```
- `TACTIC_BETA`: the three standard benchmark settings as a table — Boulware β=0.2, Linear β=1, Conceder β=5. They are no longer three near-identical classes; `make` builds them from this table.

**Lines 99–113**
```python
  99  class ZIC(ScriptedAgent):
 100      """Zero-intelligence with constraint: random individually rational offers (Gode and Sunder)."""
 101      name = "ZI-C"
 102  
 103      def act_thread(self, key, o, m) -> int:
 104          valid = np.nonzero(m[:self.K])[0]
 105          if valid.size == 0:
 106              return self.A_WAIT
 107          b = int(self.rng.choice(valid))
 108          if m[self.A_ACCEPT] and self._beats(float(o[C.O_CP_PRICE]), float(self.prices[b])):
 109              return self.A_ACCEPT
 110          # hold each random offer for three steps so the counterpart has time to accept it
 111          if o[C.O_OWN_LIVE] > 0.5 and round(float(o[C.O_OWN_AGE]) * self.sc.T) < 3:
 112              return self.A_WAIT
 113          return b
```
- `ZIC` (Gode & Sunder zero-intelligence with constraint): picks a random *legal* price (never irrational because of masking). Accepts if the counterparty's offer beats it. Holds each random offer three steps so the other side can respond. A noise baseline.

**Lines 116–121**
```python
 116  class RandomValid(ScriptedAgent):
 117      """Uniform over all currently valid actions. Used only for stress tests."""
 118      name = "RandomValid"
 119  
 120      def act_thread(self, key, o, m) -> int:
 121          return int(self.rng.choice(np.nonzero(m)[0]))
```
- `RandomValid`: uniform over all legal actions; only for stress tests of the env.

**Lines 124–130**
```python
 124  class TitForTat(ScriptedAgent):
 125      """Matches the counterpart's last concession; never concedes less than Boulware would."""
 126      name = "TitForTat"
 127  
 128      def __init__(self, role, sc, **kw):
 129          super().__init__(role, sc)
 130          self._floor_tactic = TimeDependent(role, sc, beta=0.2)
```
- `TitForTat`: mirrors the counterparty's last concession but never concedes less than Boulware would. Keeps a Boulware instance as the floor tactic.

**Lines 132–149**
```python
 132      def act_thread(self, key, o, m) -> int:
 133          sign = -1.0 if self.role == F else 1.0      # direction in which I concede
 134          boul = self._floor_tactic.target(o)
 135          has_cp = o[C.O_CP_HAS] > 0.5
 136          cp = float(o[C.O_CP_PRICE])
 137          prev_cp, mine = self.mem.get(key, (None, None))
 138          if mine is None:
 139              target = boul
 140          else:
 141              step = abs(cp - prev_cp) if (has_cp and prev_cp is not None) else 0.0
 142              target = mine + sign * step
 143              target = min(target, boul) if self.role == F else max(target, boul)
 144          reserve = float(o[C.O_WALKAWAY])
 145          target = max(target, reserve) if self.role == F else min(target, reserve)
 146          b = self._bin_for(target, m)
 147          if b is not None:
 148              self.mem[key] = (cp if has_cp else prev_cp, float(self.prices[b]))
 149          return self._decide(b, o, m)
```
- `act_thread`: `sign` is the direction in which I concede (−1 for farmer = lower price, +1 for buyer = higher). `boul` = Boulware target. If I haven't offered yet, use Boulware. Otherwise the step = how much the counterparty moved since last time; target = my last price + sign·step, bounded by Boulware's target (so I never concede faster than Boulware... farmer: min; buyer: max) and by my reserve. Store (their last price, my price) in memory, then decide.

**Lines 152–163**
```python
 152  class HardballBuyer(ScriptedAgent):
 153      """Exploitation probe: bids one bin above its estimate of the farmer's fallback and never moves."""
 154      name = "Hardball"
 155  
 156      def __init__(self, role, sc, **kw):
 157          assert role == B, "the hardball probe is a buyer"
 158          super().__init__(role, sc)
 159  
 160      def act_thread(self, key, o, m) -> int:
 161          est = self.sc.theta * float(o[C.O_SIGNAL]) * float(o[C.O_FRESH])
 162          idx = min(self.K - 1, int(np.searchsorted(self.prices, est - EPS)) + 1)
 163          return self._decide(self._bin_for(float(self.prices[idx]), m), o, m)
```
- `HardballBuyer`: an exploitation probe (buyer only). Estimates the farmer's fallback as θ·signal·freshness, bids one bin above, and never moves. Tests whether learned farmers can be exploited by an aggressive low-balling buyer.

**Lines 166–171**
```python
 166  class TakeFallback(ScriptedAgent):
 167      """Reference floor: everyone walks at step 0."""
 168      name = "TakeFallback"
 169  
 170      def act_thread(self, key, o, m) -> int:
 171          return self.A_WALK
```
- `TakeFallback`: everyone walks at step 0 → the *floor* reference (what happens with no trade at all).

**Lines 174–194**
```python
 174  class OracleFairSplit(ScriptedAgent):
 175      """Reference ceiling (privileged): optimal matching, Nash price, fastest possible close."""
 176      name = "OracleFair"
 177  
 178      def act(self, env, obs, mask, active) -> np.ndarray:
 179          acts = np.full(active.shape, self.A_WAIT, dtype=int)
 180          if "pairs" not in self.mem:
 181              self.mem["pairs"] = set(yardsticks.max_gains(env.v, env.o_f, env.o_b, env.cap)[1])
 182          for i, j in zip(*np.nonzero(active)):
 183              m = mask[i, j]
 184              if (i, j) not in self.mem["pairs"]:
 185                  acts[i, j] = self.A_WALK
 186              elif self.role == F:
 187                  p = yardsticks.nash_price(env.v[j], env.o_f[i], env.o_b[j], env.sc.rho, env.t + 2)
 188                  valid = np.nonzero(m[:self.K])[0]
 189                  stale = env.sent_id[F, i, j] < 0 or env.t >= env.sent_t[F, i, j] + self.ttl - 1
 190                  if p is not None and valid.size and stale and not env.pending_accept[B, i, j]:
 191                      acts[i, j] = int(valid[np.argmin(np.abs(self.prices[valid] - p))])
 192              elif m[self.A_ACCEPT]:
 193                  acts[i, j] = self.A_ACCEPT
 194          return acts
```
- `OracleFairSplit` (privileged *ceiling*): overrides `act` to read the true env state. Computes the optimal matching once (Hungarian) and caches it. Threads not in the optimal matching WALK. For matched farmers: propose the **Nash price** (equal split of gains) for t+2, rounded to the nearest legal bin, only when the previous offer is stale and the buyer hasn't accepted yet. Matched buyers just ACCEPT when legal. This gives the maximum efficiency with a perfectly fair split — the upper bound every method is compared to.

**Lines 196–198**
```python
 196  
 197  REGISTRY = {c.__name__: c for c in
 198              [ZIC, RandomValid, TitForTat, HardballBuyer, TakeFallback, OracleFairSplit]}
```
- `REGISTRY`: name → class lookup for the agents that are their own class (the three time-dependent tactics are handled in `make`).

**Lines 201–207**
```python
 201  def make(name: str, role: int, sc, **kw) -> ScriptedAgent:
 202      if name.startswith("TD:"):                     # e.g. "TD:0.5:1.0" -> beta, start
 203          parts = name.split(":")
 204          return TimeDependent(role, sc, beta=float(parts[1]),
 205                               start=float(parts[2]) if len(parts) > 2 else None)
 206      if name in TACTIC_BETA:
 207          return TimeDependent(role, sc, beta=TACTIC_BETA[name], name=name)
```
- `make(name, role, sc)`: factory. `TD:0.5:1.0` builds a TimeDependent with β=0.5 and start price 1.0; the names Boulware/Linear/Conceder build a TimeDependent from `TACTIC_BETA`; anything else is looked up in the registry.

**Lines 208–208**
```python
 208      return REGISTRY[name](role, sc, **kw)
```
- Anything else is looked up in the registry with the given arguments.


## 2.7  `agrinegotiate/agents/networks.py`

Neural network definitions: a feed-forward actor-critic and a recurrent (GRU) actor-critic. Both output a masked categorical policy and two value heads (reward and fee-base cost).

**Lines 1–9**
```python
   1  """Policy networks: 64 tanh units, masked categorical policy.
   2  
   3  ActorCritic           feed-forward, two hidden layers (delay-unaware agents, and the default).
   4  RecurrentActorCritic  one hidden layer followed by a 64-unit GRU (delay-aware agents).
   5  
   6  Both have a second value head that predicts the expected fee base.
   7  `walk_bias` lowers the initial probability of WALK so that early random play does not
   8  teach both sides to leave before they have ever seen a deal (DECISIONS.md #18).
   9  """
```
- Docstring: 64 tanh units; feed-forward for delay-unaware agents (default), GRU for delay-aware agents; a second value head predicts the expected fee base; `walk_bias` lowers WALK's initial probability.

**Lines 10–13**
```python
  10  from __future__ import annotations
  11  
  12  import torch
  13  import torch.nn as nn
```
- Imports torch and `torch.nn`.

**Lines 16–20**
```python
  16  def _layer(i, o, gain):
  17      lin = nn.Linear(i, o)
  18      nn.init.orthogonal_(lin.weight, gain)
  19      nn.init.zeros_(lin.bias)
  20      return lin
```
- `_layer`: a Linear layer with **orthogonal initialisation** (scaled by `gain`) and zero bias — the standard PPO init that keeps activations and gradients well-scaled.

**Lines 23–28**
```python
  23  def _policy_head(hidden, n_actions, walk_bias):
  24      head = _layer(hidden, n_actions, 0.01)
  25      if walk_bias:
  26          with torch.no_grad():
  27              head.bias[n_actions - 2] = walk_bias          # action index K+1 is WALK
  28      return head
```
- `_policy_head`: the output layer with tiny gain 0.01 (so initial policy is nearly uniform). If `walk_bias` is set, the bias of action index `n_actions−2` (WALK) is set to −3, making WALK about 20× less likely at the start.

**Lines 31–32**
```python
  31  class ActorCritic(nn.Module):
  32      recurrent = False
```
- `ActorCritic`: feed-forward model. `recurrent = False` is a class flag the training code checks.

**Lines 34–41**
```python
  34      def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64, walk_bias: float = 0.0):
  35          super().__init__()
  36          g = 2 ** 0.5
  37          self.pi = nn.Sequential(_layer(obs_dim, hidden, g), nn.Tanh(), _layer(hidden, hidden, g), nn.Tanh(),
  38                                  _policy_head(hidden, n_actions, walk_bias))
  39          self.vf = nn.Sequential(_layer(obs_dim, hidden, g), nn.Tanh(), _layer(hidden, hidden, g), nn.Tanh())
  40          self.v_reward = _layer(hidden, 1, 1.0)
  41          self.v_cost = _layer(hidden, 1, 1.0)
```
- Gain √2 (the standard for tanh/ReLU). Policy net: obs → 64 tanh → 64 tanh → logits. Separate value trunk of the same shape (policy and value don't share weights). Two linear heads on the trunk: expected reward and expected cost.

**Lines 43–45**
```python
  43      def dist(self, obs, mask):
  44          logits = self.pi(obs).masked_fill(~mask, -1e9)
  45          return torch.distributions.Categorical(logits=logits)
```
- `dist`: computes logits, replaces illegal actions' logits with −1e9 (≈ probability 0), returns a Categorical distribution. **This is action masking.**

**Lines 47–49**
```python
  47      def values(self, obs):
  48          h = self.vf(obs)
  49          return self.v_reward(h).squeeze(-1), self.v_cost(h).squeeze(-1)
```
- `values`: returns (reward value, cost value), squeezed to vectors.

**Lines 51–53**
```python
  51      def step(self, obs, mask, h=None):
  52          """Same interface as the recurrent net: (dist, reward value, cost value, new hidden state)."""
  53          return self.dist(obs, mask), *self.values(obs), h
```
- `step(obs, mask, h)`: the feed-forward net exposes the **same interface as the recurrent net** — returns (distribution, reward value, cost value, hidden state), with `h` passed through unchanged. Training and acting code can therefore treat both network types identically.

**Lines 56–57**
```python
  56  class RecurrentActorCritic(nn.Module):
  57      recurrent = True
```
- `RecurrentActorCritic`: GRU version; `recurrent = True`.

**Lines 59–66**
```python
  59      def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64, walk_bias: float = 0.0):
  60          super().__init__()
  61          self.hidden = hidden
  62          self.enc = nn.Sequential(_layer(obs_dim, hidden, 2 ** 0.5), nn.Tanh())
  63          self.gru = nn.GRU(hidden, hidden, batch_first=True)
  64          self.pi = _policy_head(hidden, n_actions, walk_bias)
  65          self.v_reward = _layer(hidden, 1, 1.0)
  66          self.v_cost = _layer(hidden, 1, 1.0)
```
- Layers: encoder (obs → 64 tanh), a GRU(64→64, batch_first), a policy head on the GRU output, and the two value heads. GRU lets the agent remember what happened under delay.

**Lines 68–69**
```python
  68      def initial(self, n: int):
  69          return torch.zeros(n, self.hidden)
```
- `initial`: zero hidden state for n threads.

**Lines 71–76**
```python
  71      def step(self, obs, mask, h):
  72          """One decision step. obs [n, D], mask [n, A], h [n, H] -> dist, value, cost value, new h."""
  73          out, h2 = self.gru(self.enc(obs).unsqueeze(1), h.unsqueeze(0))
  74          z = out.squeeze(1)
  75          d = torch.distributions.Categorical(logits=self.pi(z).masked_fill(~mask, -1e9))
  76          return d, self.v_reward(z).squeeze(-1), self.v_cost(z).squeeze(-1), h2.squeeze(0)
```
- `step`: one decision for many threads. Encode, add a time axis of length 1, run the GRU with the previous hidden state, build the masked distribution, return distribution, values, new hidden state.

**Lines 78–81**
```python
  78      def seq(self, obs):
  79          """Whole trajectories from a zero state. obs [n, T, D] -> logits [n, T, A], v [n, T], vc [n, T]."""
  80          z, _ = self.gru(self.enc(obs))
  81          return self.pi(z), self.v_reward(z).squeeze(-1), self.v_cost(z).squeeze(-1)
```
- `seq`: runs whole padded trajectories from a zero state, for training updates; returns logits and values at every step.

**Lines 84–86**
```python
  84  def make_net(obs_dim, n_actions, hidden=64, recurrent=False, walk_bias=0.0):
  85      cls = RecurrentActorCritic if recurrent else ActorCritic
  86      return cls(obs_dim, n_actions, hidden, walk_bias)
```
- `make_net`: factory choosing the class by the `recurrent` flag.


## 2.8  `agrinegotiate/agents/ppo.py`

The learning algorithm: Independent PPO (IPPO). One network for all farmers, one for all buyers, trained by self-play. Each (agent, thread) pair is one trajectory; reward arrives only at the end (gamma = 1). Includes the dual-ascent fairness constraint.

**Lines 1–6**
```python
   1  """Independent PPO with one shared network per role (IPPO), action masking,
   2  per-thread decisions and the optional adaptive fee.
   3  
   4  Each negotiation thread is one trajectory. The agent's terminal reward is credited
   5  to the last step of each of its threads; gamma = 1.
   6  """
```
- Docstring: IPPO = each role learns independently, treating the other as part of the environment; action masking; per-thread decisions; optional adaptive fee. The agent's single terminal reward is credited to the last step of each of its threads, no discounting (γ=1) because the episode is short and the reward is final.

**Lines 7–21**
```python
   7  from __future__ import annotations
   8  
   9  import json
  10  import pathlib
  11  import time
  12  from dataclasses import asdict, dataclass
  13  
  14  import numpy as np
  15  import torch
  16  
  17  from ..env.core import Market, OBS_DIM, F, B
  18  from ..eval.evaluate import EVAL_SEED_START, evaluate
  19  from ..eval.stats import KEYS
  20  from . import scripted as S
  21  from .networks import make_net
```
- Imports: json/pathlib/time for logging and files, dataclasses for the config, NumPy + PyTorch, the `Market` env and constants, `evaluate` (periodic evaluation), `KEYS` (the metric names summarised per run, shared with `eval/stats.py`), the scripted agents as `S` (used as training opponents), and `make_net`.

**Lines 23–23**
```python
  23  TRAIN_EPISODE_SEED_BASE = 10_000_000      # training episodes never reuse tuning or evaluation seeds
```
- Training episodes use seeds starting at 10,000,000 so they never overlap tuning or evaluation seeds — prevents train/test leakage.

**Lines 26–48**
```python
  26  @dataclass
  27  class PPOConfig:
  28      iters: int = 200
  29      episodes_per_iter: int = 256
  30      lr: float = 3e-4
  31      gae_lambda: float = 0.95
  32      clip: float = 0.2
  33      ent_start: float = 0.01
  34      ent_end: float = 0.001
  35      vf_coef: float = 0.5
  36      max_grad_norm: float = 0.5
  37      epochs: int = 4
  38      minibatches: int = 4
  39      hidden: int = 64
  40      reward_scale: float = 5.0
  41      mix_scripted: float = 0.3             # share of training episodes with one scripted side
  42      mix_pool: tuple = ("Boulware", "Linear", "Conceder", "TitForTat")
  43      eval_every: int = 10
  44      eval_episodes: int = 300
  45      final_eval_episodes: int = 1000
  46      delay_aware: bool = True
  47      recurrent: bool = False
  48      walk_bias: float = -3.0               # initial logit offset on WALK (DECISIONS.md #18)
```
- `PPOConfig`: all hyperparameters. `iters` 200 × `episodes_per_iter` 256 = the training budget. `lr` 3e-4 (Adam). `gae_lambda` 0.95 (bias/variance trade-off of advantage estimates). `clip` 0.2 (PPO trust region). `ent_start/ent_end` entropy bonus decays 0.01→0.001 (explore early, commit later). `vf_coef` 0.5 weights the value loss. `max_grad_norm` 0.5 clips gradients. `epochs` 4 passes × `minibatches` 4 per batch. `hidden` 64 units. `reward_scale` 5 rescales rewards to a friendlier magnitude. `mix_scripted` 0.3: 30% of episodes have one side replaced by a scripted bargainer from `mix_pool` (Boulware, Linear, Conceder, TitForTat) so learners don't only co-adapt to each other. `eval_every` 10 iterations, 300 eval episodes; 1000 for the final eval. `delay_aware` and `recurrent` pick the agent type. `walk_bias` −3 pushes WALK's initial logit down (fixed early collapse where agents learned to leave immediately).

**Lines 49–49**
```python
  49      target_agent_steps: int = 0           # > 0: train until this many agent-steps instead of `iters`
```
- `target_agent_steps`: when above 0, training runs until this many agent-steps (decisions by both roles) have been collected, instead of a fixed number of iterations. The 3M budget check sets it to 3,000,000.

**Lines 52–58**
```python
  52  def _step(net, store, keys, obs, mask):
  53      """One decision for a batch of threads. Recurrent nets keep a hidden state per thread key in `store`."""
  54      h = torch.stack([store.get(k, torch.zeros(net.hidden)) for k in keys]) if net.recurrent else None
  55      dist, v, cv, h2 = net.step(obs, mask, h)
  56      if net.recurrent:
  57          store.update(zip(keys, h2))
  58      return dist, v, cv
```
- `_step(net, store, keys, obs, mask)`: **one decision for a batch of threads, for either network type.** Recurrent nets get a hidden state per thread key from `store` (zeros for a new thread) and write the new states back; feed-forward nets ignore this. Returns the action distribution and both values. Used by `PolicyAgent` and `collect`, so the recurrent bookkeeping exists in exactly one place.

**Lines 61–68**
```python
  61  class PolicyAgent:
  62      """Wraps a trained network so it can be used anywhere a scripted agent can."""
  63  
  64      def __init__(self, net, name: str = "IPPO"):
  65          self.net, self.name = net, name
  66  
  67      def reset(self, seed: int) -> None:
  68          self.gen = torch.Generator().manual_seed(int(seed) + 12345)
```
- `PolicyAgent`: wraps a network so it exposes the same `reset/act` interface as scripted agents. `reset` makes a seeded torch generator (seed+12345) for reproducible sampling and clears the hidden-state store. `act` (under `@torch.no_grad()`): start with every thread set to WAIT, find the active threads, run `_step`, and sample one action per thread with `torch.multinomial`.

**Lines 69–79**
```python
  69          self.h: dict = {}
  70  
  71      @torch.no_grad()
  72      def act(self, env, obs, mask, active) -> np.ndarray:
  73          acts = np.full(active.shape, env.A_WAIT, dtype=int)
  74          idx = np.nonzero(active)
  75          if idx[0].size:
  76              keys = list(zip(idx[0].tolist(), idx[1].tolist()))
  77              d, _, _ = _step(self.net, self.h, keys, torch.as_tensor(obs[idx]), torch.as_tensor(mask[idx]))
  78              acts[idx] = torch.multinomial(d.probs, 1, generator=self.gen).squeeze(-1).numpy()
  79          return acts
```
- If there are active threads, `_step` returns the masked distribution; `torch.multinomial` samples one action per thread (stochastic policy), written into the action grid and returned.

**Lines 82–93**
```python
  82  class Buffer:
  83      def __init__(self):
  84          self.obs, self.mask, self.act, self.logp, self.val, self.cval, self.traj = [], [], [], [], [], [], []
  85          self.traj_ids: dict = {}
  86          self.final_r: dict = {}
  87          self.final_c: dict = {}
  88  
  89      def tid(self, key):
  90          return self.traj_ids.setdefault(key, len(self.traj_ids))
  91  
  92      def __len__(self):
  93          return sum(len(o) for o in self.obs)
```
- `Buffer`: lists of per-step tensors (obs, mask, action, log-prob, reward value, cost value, trajectory id). `traj_ids` maps (episode, i, j) → integer id; `final_r/final_c` store each trajectory's terminal reward and fee-base. `tid` assigns ids lazily with `setdefault`; `__len__` counts steps.

**Lines 96–97**
```python
  96  def collect(sc, nets, cfg: PPOConfig, mech, seeds, rng):
  97      """Play len(seeds) episodes in lock-step. Returns per-role buffers and the finished results."""
```
- `collect`: plays many episodes in lock-step (batched) so network calls are vectorised, and returns buffers plus finished results.

**Lines 98–107**
```python
  98      envs = [Market(sc, mechanism=mech, delay_aware=cfg.delay_aware) for _ in seeds]
  99      scripted = []
 100      for e, (env, seed) in enumerate(zip(envs, seeds)):
 101          env.reset(int(seed))
 102          ag = {}
 103          if rng.random() < cfg.mix_scripted:
 104              side = int(rng.integers(2))
 105              ag[side] = S.make(str(rng.choice(cfg.mix_pool)), side, sc)
 106              ag[side].reset(int(seed))
 107          scripted.append(ag)
```
- Builds one `Market` per seed and resets it. With probability `mix_scripted`, one random side gets a randomly chosen scripted agent (reset with the seed). `scripted[e]` holds those overrides.

**Lines 108–110**
```python
 108      bufs = {F: Buffer(), B: Buffer()}
 109      hid = {F: {}, B: {}}
 110      live = list(range(len(envs)))
```
- Separate buffers per role; `hid` holds recurrent states; `live` is the list of unfinished episodes.

**Lines 111–113**
```python
 111      while live:
 112          views = {e: envs[e].observe() for e in live}
 113          acts = {e: np.full((2, sc.n_farmers, sc.n_buyers), envs[e].A_WAIT, dtype=int) for e in live}
```
- Main loop until all episodes finish. Get every live env's observation; default all actions to WAIT.

**Lines 114–124**
```python
 114          for s in (F, B):
 115              rows, where = [], []
 116              for e in live:
 117                  obs, mask, active = views[e]
 118                  if s in scripted[e]:
 119                      acts[e][s] = scripted[e][s].act(envs[e], obs[s], mask[s], active[s])
 120                      continue
 121                  ii, jj = np.nonzero(active[s])
 122                  for i, j in zip(ii, jj):
 123                      rows.append((obs[s, i, j], mask[s, i, j]))
 124                      where.append((e, i, j))
```
- For each side: if that side is scripted in an episode, take its action directly. Otherwise gather observation/mask rows of all active threads and remember where each came from (`where`).

**Lines 125–128**
```python
 125              if not rows:
 126                  continue
 127              o = torch.as_tensor(np.stack([r[0] for r in rows]))
 128              m = torch.as_tensor(np.stack([r[1] for r in rows]))
```
- Skip if there is nothing to decide; else stack rows into batched tensors.

**Lines 129–132**
```python
 129              with torch.no_grad():
 130                  d, v, cv = _step(nets[s], hid[s], where, o, m)
 131                  a = d.sample()
 132                  lp = d.log_prob(a)
```
- One batched forward pass without gradients through `_step` (the same helper `PolicyAgent` uses; it handles the GRU hidden states for recurrent nets and works unchanged for feed-forward ones), returning the distribution and both value heads. Sample actions and record their log-probs (needed later for the PPO ratio).

**Lines 133–138**
```python
 133              buf = bufs[s]
 134              buf.obs.append(o); buf.mask.append(m); buf.act.append(a); buf.logp.append(lp)
 135              buf.val.append(v); buf.cval.append(cv)
 136              buf.traj.append(torch.tensor([buf.tid(w) for w in where]))
 137              for (e, i, j), ai in zip(where, a.tolist()):
 138                  acts[e][s, i, j] = ai
```
- Store the step in the role's buffer, with trajectory ids, and write sampled actions back into each env's action array.

**Lines 139–141**
```python
 139          for e in live:
 140              envs[e].step(acts[e], views[e][1])
 141          live = [e for e in live if not envs[e].done]
```
- Step every live env (passing the mask so illegal actions raise assertions); drop finished episodes.

**Lines 142–148**
```python
 142      # credit each thread with its agent's terminal reward (farmer i, or buyer j and its fee base)
 143      rewards = [env.rewards() for env in envs]
 144      for (e, i, j), tid in bufs[F].traj_ids.items():
 145          bufs[F].final_r[tid], bufs[F].final_c[tid] = rewards[e][0][i], 0.0
 146      for (e, i, j), tid in bufs[B].traj_ids.items():
 147          bufs[B].final_r[tid], bufs[B].final_c[tid] = rewards[e][1][j], rewards[e][2][j]
 148      return bufs, [env.result() for env in envs]
```
- After all episodes end, credit each thread with its agent's terminal reward. `rewards[e]` is `(r_f, r_b, c_b)` for episode *e*; every trajectory key is `(e, i, j)`, so one pass over each role's trajectories is enough: farmer threads get `r_f[i]` with zero cost, buyer threads get `r_b[j]` and the fee base `c_b[j]`. Returns the buffers and the `EpisodeResult`s.

**Lines 151–161**
```python
 151  def _gae(vals, final, lam):
 152      """Advantages and returns for one trajectory with a single terminal reward, gamma = 1."""
 153      n = len(vals)
 154      adv = np.zeros(n, dtype=np.float32)
 155      last = 0.0
 156      for k in reversed(range(n)):
 157          nxt = vals[k + 1] if k + 1 < n else 0.0
 158          delta = (final if k == n - 1 else 0.0) + nxt - vals[k]
 159          last = delta + lam * last
 160          adv[k] = last
 161      return adv, adv + vals
```
- `_gae`: Generalised Advantage Estimation for one trajectory with a single terminal reward. Walk backwards: δ_k = r_k + V(k+1) − V(k), where r is 0 except the last step (`final`) and V after the end is 0. Running `last = δ + λ·last` (γ=1). Returns advantages and returns (= adv + value) used as value targets.

**Lines 164–175**
```python
 164  def _advantages(buf: Buffer, cfg: PPOConfig, lam_fair: float):
 165      """Per-trajectory GAE for the reward critic and the cost critic.
 166      Returns the trajectories (as index arrays), the Lagrangian advantage A_reward - lam * A_cost, and both returns."""
 167      val, cval, traj = torch.cat(buf.val).numpy(), torch.cat(buf.cval).numpy(), torch.cat(buf.traj).numpy()
 168      adv_r, ret_r, adv_c, ret_c = (np.zeros_like(val) for _ in range(4))
 169      order = np.argsort(traj, kind="stable")
 170      segs = np.split(order, np.flatnonzero(np.diff(traj[order])) + 1)
 171      for seg in segs:
 172          tid = int(traj[seg[0]])
 173          adv_r[seg], ret_r[seg] = _gae(val[seg], cfg.reward_scale * buf.final_r[tid], cfg.gae_lambda)
 174          adv_c[seg], ret_c[seg] = _gae(cval[seg], cfg.reward_scale * buf.final_c[tid], cfg.gae_lambda)
 175      return segs, torch.as_tensor(adv_r - lam_fair * adv_c), torch.as_tensor(ret_r), torch.as_tensor(ret_c)
```
- `_advantages`: for every trajectory (steps sorted by trajectory id and split where the id changes) run `_gae` for the reward critic and for the cost critic (terminal rewards scaled by `reward_scale`). Returns the trajectories, the **Lagrangian advantage `A_reward − λ·A_cost`** (for farmers λ is 0; for buyers it is the current fee multiplier — this is how the fairness constraint enters the policy gradient), and both sets of returns used as value targets.

**Lines 178–186**
```python
 178  def update(net, opt, buf: Buffer, cfg: PPOConfig, ent_coef: float, lam_fair: float, rng) -> dict:
 179      if not buf.obs:
 180          return {}
 181      obs, mask = torch.cat(buf.obs), torch.cat(buf.mask)
 182      act, logp_old = torch.cat(buf.act), torch.cat(buf.logp)
 183      segs, adv, ret_r, ret_c = _advantages(buf, cfg, lam_fair)
 184      n = obs.shape[0]
 185      stats = {}
 186      if net.recurrent:                      # padded whole trajectories, one row per thread
```
- `update`: one PPO update for one role. Empty buffer → nothing. Concatenate the stored lists into tensors, get the advantages from `_advantages`, and prepare for the epochs below.

**Lines 187–193**
```python
 187          tmax = max(len(g) for g in segs)
 188          idx_pad = torch.full((len(segs), tmax), -1, dtype=torch.long)
 189          for r, g in enumerate(segs):
 190              idx_pad[r, :len(g)] = torch.as_tensor(g)
 191          obs_pad = torch.zeros(len(segs), tmax, obs.shape[1])
 192          valid_all = idx_pad >= 0
 193          obs_pad[valid_all] = obs[idx_pad[valid_all]]
```
- `n` = number of steps. For recurrent nets the trajectories are padded into a `[threads, tmax]` index grid and a padded observation tensor with a validity mask, because the GRU must see whole trajectories.

**Lines 194–197**
```python
 194      for _ in range(cfg.epochs):
 195          units = len(segs) if net.recurrent else n
 196          perm = torch.as_tensor(rng.permutation(units))
 197          for mb in torch.tensor_split(perm, cfg.minibatches):
```
- PPO loop: for each epoch, shuffle units (steps, or trajectories for recurrent) and split into minibatches.

**Lines 198–203**
```python
 198              if net.recurrent:
 199                  logits, v, cv = net.seq(obs_pad[mb])
 200                  valid = idx_pad[mb] >= 0
 201                  flat = idx_pad[mb][valid]
 202                  logits, v, cv = logits[valid], v[valid], cv[valid]
 203                  d = torch.distributions.Categorical(logits=logits.masked_fill(~mask[flat], -1e9))
```
- Recurrent: run the GRU over padded sequences, keep only valid positions, mask illegal actions with −1e9 before the softmax.

**Lines 204–207**
```python
 204              else:
 205                  flat = mb
 206                  d = net.dist(obs[flat], mask[flat])
 207                  v, cv = net.values(obs[flat])
```
- Feed-forward: masked distribution and both values straight from the minibatch.

**Lines 208–210**
```python
 208              lp = d.log_prob(act[flat])
 209              a = adv[flat]
 210              a = (a - a.mean()) / (a.std() + 1e-8)
```
- New log-prob of the taken actions; advantages are normalised per minibatch (mean 0, std 1) for stability.

**Lines 211–212**
```python
 211              ratio = torch.exp(lp - logp_old[flat])
 212              pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
```
- **PPO clipped objective**: ratio = π_new/π_old; loss = −mean(min(ratio·A, clip(ratio, 0.8, 1.2)·A)). The clip stops the policy moving too far in one update.

**Lines 213–215**
```python
 213              vloss = 0.5 * ((v - ret_r[flat]) ** 2).mean() + 0.5 * ((cv - ret_c[flat]) ** 2).mean()
 214              ent = d.entropy().mean()
 215              loss = pg + cfg.vf_coef * vloss - ent_coef * ent
```
- Value loss = ½ MSE of both critics (reward and cost). Entropy bonus encourages exploration. Total loss = policy + 0.5·value − ent_coef·entropy.

**Lines 216–220**
```python
 216              opt.zero_grad()
 217              loss.backward()
 218              torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
 219              opt.step()
 220              stats = dict(pg=pg.item(), vloss=vloss.item(), entropy=ent.item())
```
- Standard step: zero grads, backprop, clip gradient norm to 0.5, optimiser step; record last stats.

**Lines 221–221**
```python
 221      return stats
```
- Returns the last minibatch's stats for logging.

**Lines 224–225**
```python
 224  def train(sc, cfg: PPOConfig, seed: int, out_dir, mech=None, adaptive_fee: bool = False, log=print) -> dict:
 225      """Train farmer and buyer networks by self-play. Writes config, curve, checkpoints, metrics."""
```
- `train`: the full loop for one scenario/seed.

**Lines 226–230**
```python
 226      out = pathlib.Path(out_dir)
 227      out.mkdir(parents=True, exist_ok=True)
 228      torch.manual_seed(seed)
 229      torch.set_num_threads(1)
 230      rng = np.random.default_rng(seed)
```
- Create the output folder, seed torch and NumPy, use a single torch thread (several runs in parallel on few CPUs).

**Lines 231–233**
```python
 231      n_actions = sc.n_bins + 3
 232      nets = {s: make_net(OBS_DIM, n_actions, cfg.hidden, cfg.recurrent, cfg.walk_bias) for s in (F, B)}
 233      opts = {s: torch.optim.Adam(nets[s].parameters(), lr=cfg.lr, eps=1e-5) for s in (F, B)}
```
- Build one network and one Adam optimiser per role. 24 actions = bins+3.

**Lines 234–236**
```python
 234      (out / "config.json").write_text(json.dumps(dict(
 235          scenario=sc.to_dict(), ppo=asdict(cfg), seed=seed, adaptive_fee=adaptive_fee,
 236          mechanism=None if mech is None else {k: v for k, v in vars(mech).items()}), indent=1, default=str))
```
- Save `config.json` (scenario, PPO config, seed, mechanism parameters) so every run is reproducible and auditable.

**Lines 237–238**
```python
 237      agents = {s: PolicyAgent(nets[s]) for s in (F, B)}
 238      curve, j_c_ema, t0, env_steps = [], None, time.time(), 0
```
- Wrap networks as agents for evaluation; initialise the curve, the EMA of the constraint cost, a timer and a step counter.

**Lines 239–245**
```python
 239      budget, it = cfg.target_agent_steps, 0
 240      while (env_steps < budget) if budget else (it < cfg.iters):
 241          frac = min(1.0, env_steps / budget) if budget else it / max(1, cfg.iters - 1)
 242          for s in (F, B):
 243              opts[s].param_groups[0]["lr"] = cfg.lr * (1 - frac) + 1e-6
 244          ent = cfg.ent_start + frac * (cfg.ent_end - cfg.ent_start)
 245          seeds = TRAIN_EPISODE_SEED_BASE + seed * 1_000_000 + it * cfg.episodes_per_iter + np.arange(cfg.episodes_per_iter)
```
- The loop runs either for `cfg.iters` iterations or, when a step budget is set, until `env_steps` reaches it. `frac` = training progress (share of the budget used, or of the iterations done); linearly decay the learning rate (+1e-6 floor); linearly interpolate the entropy coefficient; build unique training seeds from base + seed·1e6 + iteration·episodes.

**Lines 246–247**
```python
 246          bufs, results = collect(sc, nets, cfg, mech, seeds, rng)
 247          env_steps += len(bufs[F]) + len(bufs[B])
```
- Collect experience; count agent steps.

**Lines 248–249**
```python
 248          lam = getattr(mech, "lam", 0.0)            # 0 without a fee mechanism
 249          stats = {s: update(nets[s], opts[s], bufs[s], cfg, ent, lam if s == B else 0.0, rng) for s in (F, B)}
```
- Current λ from the mechanism (`getattr` returns 0 when there is no fee mechanism); update both roles (λ is applied only to buyers).

**Lines 250–255**
```python
 250          if adaptive_fee:
 251              deals = [d for r in results for d in r.deals if d["g_t"] > 1e-9]
 252              if deals:
 253                  j_c = float(np.mean([d["shortfall"] / d["g_t"] for d in deals]))
 254                  j_c_ema = j_c if j_c_ema is None else 0.8 * j_c_ema + 0.2 * j_c
 255                  mech.dual_update(j_c_ema)
```
- **Dual ascent** (adaptive fee): measure the constraint cost J_c = mean(shortfall / gains-from-trade) over deals with positive gains; smooth with an EMA (0.8/0.2); call `mech.dual_update` which raises λ if J_c > ε and lowers it otherwise.

**Lines 256–265**
```python
 256          last = (env_steps >= budget) if budget else (it == cfg.iters - 1)
 257          if it % cfg.eval_every == 0 or last:
 258              m = evaluate(sc, agents[F], agents[B], cfg.eval_episodes, EVAL_SEED_START, mech, cfg.delay_aware)
 259              row = dict(iter=it, agent_steps=env_steps, seconds=round(time.time() - t0, 1),
 260                         lam_fair=lam, j_c_ema=j_c_ema,
 261                         entropy_f=stats[F].get("entropy"), entropy_b=stats[B].get("entropy"),
 262                         **{k: m.get(k) for k in ("E", "FGC", "BGC", "S_F", "agreement_rate", "phi_mean",
 263                                                  "time_to_agreement", "price_share", "j_c", "fees_over_gstar",
 264                                                  "fallback_rate", "nash_distance")})
 265              curve.append(row)
```
- `last` marks the final iteration (budget reached, or the last of `cfg.iters`). Every `eval_every` iterations, and on the last one, evaluate on fixed evaluation seeds (300 episodes) and log a row of metrics, λ, entropies and the smoothed constraint.

**Lines 266–269**
```python
 266              log(f"[{sc.name} seed {seed}] it {it:4d} steps {env_steps:8d} E {row['E']:.3f} FGC {row['FGC']:.3f} "
 267                  f"S_F {row['S_F'] if row['S_F'] is None else round(row['S_F'], 3)} agree {row['agreement_rate']:.3f} "
 268                  f"t_deal {row['time_to_agreement']} lam {lam:.3f}")
 269              (out / "curve.json").write_text(json.dumps(curve, indent=1))
```
- Print a progress line and write `curve.json` incrementally (so partial runs are usable).

**Lines 270–270**
```python
 270          it += 1
```
- Next iteration.

**Lines 271–273**
```python
 271      torch.save({"farmer": nets[F].state_dict(), "buyer": nets[B].state_dict(), "hidden": cfg.hidden,
 272                  "recurrent": cfg.recurrent, "delay_aware": cfg.delay_aware,
 273                  "lam_fair": getattr(mech, "lam", 0.0)}, out / "checkpoint.pt")
```
- Save the checkpoint: both state dicts and the settings needed to rebuild them (hidden size, recurrent, delay-aware, λ).

**Lines 274–282**
```python
 274      final = evaluate(sc, agents[F], agents[B], cfg.final_eval_episodes, EVAL_SEED_START, mech, cfg.delay_aware)
 275      last10 = curve[-10:]
 276      summary = dict(
 277          seed=seed, scenario=sc.name, agent_steps=env_steps, seconds=round(time.time() - t0, 1),
 278          final_checkpoint_1000_episodes=final,
 279          last10_checkpoint_mean={k: _mean_or_none([r[k] for r in last10]) for k in KEYS},
 280          last10_checkpoints_used=len(last10))
 281      (out / "metrics.json").write_text(json.dumps(summary, indent=1))
 282      return summary
```
- Final evaluation with 1000 episodes. Summary has the final-checkpoint metrics and the mean of the last 10 checkpoints (a less noisy estimate) for every metric in `KEYS`. Written to `metrics.json` and returned.

**Lines 285–287**
```python
 285  def _mean_or_none(xs):
 286      xs = [x for x in xs if x is not None]
 287      return float(np.mean(xs)) if xs else None
```
- `_mean_or_none`: mean ignoring None (S_F can be undefined).

**Lines 290–297**
```python
 290  def load_agents(path, sc):
 291      ck = torch.load(path, weights_only=True)
 292      out = {}
 293      for s, key in ((F, "farmer"), (B, "buyer")):
 294          net = make_net(OBS_DIM, sc.n_bins + 3, ck["hidden"], ck.get("recurrent", False))
 295          net.load_state_dict(ck[key])
 296          out[s] = PolicyAgent(net)
 297      return out
```
- `load_agents`: loads a checkpoint (`weights_only=True` for safety), rebuilds the networks and wraps them as agents so any experiment can reuse trained policies.


## 2.9  `agrinegotiate/mechanisms/fee.py`

The fairness mechanisms: the Lagrangian *fee* (our method) and the *price floor* (baseline).

**Lines 1–9**
```python
   1  """Fairness mechanisms.
   2  
   3  FeeMechanism: fee = lam * shortfall, shortfall = max(0, target price - deal price).
   4    oracle target      p_tau = o_F*rho^t + tau*G_t           (uses private values; simulation only)
   5    observable target  p_tau = kappa * P_ref * rho^t         (public quantities only)
   6  `lam` is fixed here; the trainer adapts it by dual ascent via `dual_update`.
   7  
   8  FloorMechanism: the platform rejects any offer below kappa * P_ref * rho^t.
   9  """
```
- Docstring: fee = λ · shortfall, shortfall = max(0, target price − deal price). Oracle target uses private values (simulation only); observable target uses only public quantities (κ·P_ref·ρ^t). λ is adapted by dual ascent in the trainer. FloorMechanism rejects any offer below κ·P_ref·ρ^t.

**Lines 10–12**
```python
  10  from __future__ import annotations
  11  
  12  from .. import yardsticks
```
- Future-annotations import and `yardsticks` (for the target price).

**Lines 15–16**
```python
  15  class FeeMechanism:
  16      floor_kappa = None
```
- `FeeMechanism`; `floor_kappa = None` means it imposes no floor (the env checks this attribute).

**Lines 18–23**
```python
  18      def __init__(self, tau: float = 0.4, lam: float = 0.0, observable: bool = False,
  19                   kappa: float | None = None, eps_c: float = 0.02, eta: float = 0.05, lam_max: float = 5.0):
  20          if observable and kappa is None:
  21              raise ValueError("observable mechanism needs kappa (target price as a share of P_ref)")
  22          self.tau, self.lam, self.observable, self.kappa = tau, lam, observable, kappa
  23          self.eps_c, self.eta, self.lam_max = eps_c, eta, lam_max
```
- Parameters: τ (target farmer share of gains, 0.4), λ (initial multiplier), whether the target is observable, κ (for the observable target), ε_c (allowed constraint cost 0.02), η (dual step 0.05), `lam_max` 5. An observable mechanism without κ raises an error; all parameters are stored.

**Lines 25–28**
```python
  25      def target(self, env, i, j, t):
  26          if self.observable:
  27              return self.kappa * env.level * env.sc.rho ** t
  28          return yardsticks.target_price(env.v[j], env.o_f[i], env.o_b[j], self.tau, env.sc.rho, t)
```
- `target`: observable → κ·level·ρ^t (public price index decayed by spoilage); oracle → `yardsticks.target_price` = farmer reservation (decayed) + τ·gains.

**Lines 30–34**
```python
  30      def shortfall(self, env, i, j, p, t) -> float:
  31          target = self.target(env, i, j, t)
  32          if target is None:
  33              return 0.0
  34          return max(0.0, target - p)
```
- `shortfall`: how far below the target the deal price fell, floored at zero; 0 if the target is undefined (no gains from trade).

**Lines 36–39**
```python
  36      def dual_update(self, j_c_hat: float) -> float:
  37          """lam <- clip(lam + eta * (J_c - eps_c), 0, lam_max). Returns the new value."""
  38          self.lam = min(self.lam_max, max(0.0, self.lam + self.eta * (j_c_hat - self.eps_c)))
  39          return self.lam
```
- `dual_update(J_c_hat)`: **dual ascent on the Lagrange multiplier**: λ ← clip(λ + η·(Ĵ_c − ε_c), 0, λ_max). If the measured constraint cost exceeds ε, the fee grows; if the constraint is slack it shrinks. Clipping keeps λ in [0, 5].

**Lines 42–51**
```python
  42  class FloorMechanism:
  43      lam = 0.0
  44      lam_max = 1.0
  45      tau = 0.0
  46  
  47      def __init__(self, kappa: float):
  48          self.floor_kappa = kappa
  49  
  50      def shortfall(self, env, i, j, p, t) -> float:
  51          return 0.0
```
- `FloorMechanism`: the baseline. λ=0 and τ=0 so the buyer sees no fee; `floor_kappa=κ` makes the env mask buyer prices below κ·level·ρ^t (`_floor` in core.py); `shortfall` is always 0 (no fee).


## 2.10  `agrinegotiate/eval/evaluate.py`

Runs episodes and evaluates a pair of agents on fixed seeds.

**Lines 1–1**
```python
   1  """Episode runner and fixed-seed evaluation."""
```
- Docstring.

**Lines 2–7**
```python
   2  from __future__ import annotations
   3  
   4  import numpy as np
   5  
   6  from .. import metrics
   7  from ..env.core import Market, F, B
```
- Imports NumPy, the metrics module and the env.

**Lines 9–11**
```python
   9  EVAL_SEED_START = 50_000          # evaluation episodes: 50000..50999
  10  TUNE_SEED_START = 0               # tuning: 0..99
  11  TRAIN_SEED_START = 1000           # training seeds: 1000..1009
```
- **Seed discipline** (prevents leakage): tuning seeds 0–99, training-run seeds 1000–1009, evaluation episodes 50,000–50,999. Training episodes use 10,000,000+ (see ppo.py).

**Lines 14–25**
```python
  14  def run_episode(sc, farmer, buyer, seed: int, mechanism=None, log_events: bool = False,
  15                  delay_aware: bool = True):
  16      env = Market(sc, mechanism=mechanism, delay_aware=delay_aware, log_events=log_events)
  17      env.reset(seed)
  18      farmer.reset(seed)
  19      buyer.reset(seed)
  20      while not env.done:
  21          obs, mask, active = env.observe()
  22          acts = np.stack([farmer.act(env, obs[F], mask[F], active[F]),
  23                           buyer.act(env, obs[B], mask[B], active[B])])
  24          env.step(acts, mask)
  25      return env.result()
```
- `run_episode`: build and reset a Market and both agents with the same seed; loop `observe → act (both sides) → step` until done; return the `EpisodeResult`. The mask is passed to `step` so illegal actions fail loudly.

**Lines 28–32**
```python
  28  def evaluate(sc, farmer, buyer, n_episodes: int = 1000, seed_start: int = EVAL_SEED_START,
  29               mechanism=None, delay_aware: bool = True) -> dict:
  30      results = [run_episode(sc, farmer, buyer, seed_start + k, mechanism, delay_aware=delay_aware)
  31                 for k in range(n_episodes)]
  32      return metrics.aggregate(results)
```
- `evaluate`: run `n_episodes` consecutive seeds starting from `seed_start` and aggregate into the metrics dict. Because the seeds are fixed, every method is compared on **exactly the same markets** (paired comparison).


## 2.11  `agrinegotiate/eval/tuning.py`

Tunes the 'tuned heuristic' baseline fairly: only on tuning seeds 0–99, never the evaluation seeds.

**Lines 1–2**
```python
   1  """Tuned heuristic: pick the time-dependent beta that maximises the role's own
   2  gain capture against a pool of standard opponents, on tuning seeds only (0..99)."""
```
- Docstring: pick the time-dependent β that maximises a role's own gain capture against a pool of standard opponents.

**Lines 3–7**
```python
   3  from __future__ import annotations
   4  
   5  from ..agents import scripted as S
   6  from ..env.core import F, B
   7  from .evaluate import TUNE_SEED_START, evaluate
```
- Imports.

**Lines 9–10**
```python
   9  BETAS = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
  10  POOL = ["Boulware", "Linear", "Conceder", "TitForTat"]
```
- Candidate β values (from very Boulware 0.2 to very conceding 30) and the opponent pool.

**Lines 13–25**
```python
  13  def tune_role(sc, role: int, n_episodes: int = 100) -> tuple[float, dict]:
  14      scores = {}
  15      for beta in BETAS:
  16          total = 0.0
  17          for opp in POOL:
  18              me = S.make(f"TD:{beta}", role, sc)
  19              other = S.make(opp, 1 - role, sc)
  20              farmer, buyer = (me, other) if role == F else (other, me)
  21              r = evaluate(sc, farmer, buyer, n_episodes, seed_start=TUNE_SEED_START)
  22              total += r["FGC"] if role == F else r["BGC"]
  23          scores[beta] = total / len(POOL)
  24      best = max(scores, key=scores.get)
  25      return best, scores
```
- `tune_role`: for each β, play the TD(β) agent against each opponent in the pool for 100 tuning episodes, score = the role's own captured gains (FGC for farmers, BGC for buyers), averaged over opponents. Return the best β and all scores.

**Lines 28–31**
```python
  28  def tune(sc, n_episodes: int = 100) -> dict:
  29      bf, sf = tune_role(sc, F, n_episodes)
  30      bb, sb = tune_role(sc, B, n_episodes)
  31      return {"farmer_beta": bf, "buyer_beta": bb, "farmer_scores": sf, "buyer_scores": sb}
```
- `tune`: tune farmer and buyer separately and return both betas and score tables.


## 2.12  `agrinegotiate/eval/stats.py`

Statistics across training seeds: interquartile mean (IQM) with bootstrap confidence intervals — the reporting standard from 'Deep RL at the Edge of the Statistical Precipice' (rliable).

**Lines 1–2**
```python
   1  """Aggregate per-seed scores: interquartile mean with a 95% stratified bootstrap
   2  confidence interval (rliable's statistic), plus mean and standard deviation and a failed-run count."""
```
- Docstring: IQM + 95% bootstrap CI, plus mean/std and a failed-run count.

**Lines 3–8**
```python
   3  from __future__ import annotations
   4  
   5  import json
   6  import pathlib
   7  
   8  import numpy as np
```
- Imports.

**Lines 10–12**
```python
  10  KEYS = ("E", "FGC", "BGC", "S_F", "agreement_rate", "price_share", "fees_over_gstar", "lam_fair", "j_c",
  11          "time_to_agreement", "phi_mean")
  12  FAILED_E = 0.1
```
- `KEYS`: the metrics summarised. `FAILED_E = 0.1`: a run with efficiency below 0.1 counts as failed (collapsed).

**Lines 15–19**
```python
  15  def load_runs(folder) -> list[dict]:
  16      runs = []
  17      for f in sorted(pathlib.Path(folder).glob("seed_*/metrics.json")):
  18          runs.append(json.loads(f.read_text()))
  19      return runs
```
- `load_runs`: reads `metrics.json` from every `seed_*` folder, sorted.

**Lines 22–25**
```python
  22  def iqm(x) -> float:
  23      """Interquartile mean: mean of the middle 50% of runs (same statistic as rliable)."""
  24      from scipy.stats import trim_mean
  25      return float(trim_mean(np.asarray(x, float), 0.25))
```
- `iqm`: mean of the middle 50% of values via `trim_mean(x, 0.25)` — robust to outlier seeds (collapsed runs).

**Lines 28–36**
```python
  28  def iqm_ci(values, reps: int = 5000, seed: int = 0):
  29      """IQM with a 95% percentile-bootstrap interval, resampling runs with replacement.
  30  
  31      For a single task this is the stratified bootstrap rliable uses. Implemented here because
  32      rliable 1.2.0 fails to import with pandas 3 (its `arch` dependency); see DECISIONS.md #16.
  33      """
  34      x = np.asarray(values, float)
  35      boots = _bootstrap_iqms(x, np.random.default_rng(seed), reps)
  36      return iqm(x), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
```
- `iqm_ci`: percentile bootstrap. Resample runs with replacement 5000 times and take the IQM of each resample (`_bootstrap_iqms`); the 2.5th and 97.5th percentiles are the 95% CI. Implemented locally because the rliable package broke with pandas 3 (DECISIONS #16).

**Lines 39–44**
```python
  39  def iqm_diff_ci(a, b, reps: int = 5000, seed: int = 0):
  40      """IQM(a) - IQM(b) with a 95% bootstrap interval; the two groups are resampled independently."""
  41      a, b = np.asarray(a, float), np.asarray(b, float)
  42      rng = np.random.default_rng(seed)
  43      d = _bootstrap_iqms(a, rng, reps) - _bootstrap_iqms(b, rng, reps)
  44      return iqm(a) - iqm(b), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
```
- `iqm_diff_ci(a, b)`: difference of IQMs with a 95% bootstrap interval. The two groups are different runs, so each is resampled *independently* (same generator, `a` first, then `b`); the interval is the percentiles of the resampled differences. `make_report.py` uses it for every hypothesis test.

**Lines 47–51**
```python
  47  def _bootstrap_iqms(x, rng, reps):
  48      """IQM of each of `reps` resamples of x drawn with replacement."""
  49      samp = np.sort(rng.choice(x, size=(reps, x.size), replace=True), axis=1)
  50      cut = int(0.25 * x.size)                               # same trimming rule as scipy.stats.trim_mean
  51      return samp[:, cut:x.size - cut].mean(axis=1)
```
- `_bootstrap_iqms`: draws `reps` resamples of `x`, sorts each, trims 25% from each side (the same rule as `scipy.stats.trim_mean`) and returns the mean of what is left, one IQM per resample.

**Lines 54–66**
```python
  54  def summarise(folder) -> dict:
  55      runs = load_runs(folder)
  56      out = {"n_seeds": len(runs), "seeds": [r["seed"] for r in runs],
  57             "failed_runs": sum(1 for r in runs if (r["last10_checkpoint_mean"]["E"] or 0) < FAILED_E)}
  58      for k in KEYS:
  59          vals = [r["last10_checkpoint_mean"].get(k) for r in runs]
  60          defined = [v for v in vals if v is not None]
  61          out[k] = {"per_seed": vals, "undefined": len(vals) - len(defined)}
  62          if len(defined) >= 3:
  63              iqm, lo, hi = iqm_ci(defined)
  64              out[k].update(iqm=iqm, ci95=[lo, hi], mean=float(np.mean(defined)), std=float(np.std(defined, ddof=1)),
  65                            min=float(np.min(defined)), max=float(np.max(defined)))
  66      return out
```
- `summarise(folder)`: number of seeds, which seeds, failed-run count (last-10-checkpoint E < 0.1). For each metric collect per-seed values, count undefined ones (e.g. S_F undefined when E tiny), and when ≥3 are defined compute IQM, CI, mean, sample std (ddof=1), min and max.

**Lines 69–78**
```python
  69  if __name__ == "__main__":
  70      import sys
  71      for folder in sys.argv[1:]:
  72          s = summarise(folder)
  73          pathlib.Path(folder, "summary.json").write_text(json.dumps(s, indent=1))
  74          print(folder, "seeds:", s["n_seeds"], "failed:", s["failed_runs"])
  75          for k in KEYS[:5]:
  76              if "iqm" in s[k]:
  77                  print(f"  {k:15s} IQM {s[k]['iqm']:.3f}  95% CI [{s[k]['ci95'][0]:.3f}, {s[k]['ci95'][1]:.3f}]  "
  78                        f"mean {s[k]['mean']:.3f} sd {s[k]['std']:.3f}  range [{s[k]['min']:.3f}, {s[k]['max']:.3f}]")
```
- Command-line use: for each folder, write `summary.json` and print IQM, CI, mean, sd, range for the first five metrics.


## 2.13  `agrinegotiate/eval/replay.py`

Records one episode as a JSON-friendly dict (parameters, reference price lines, the full event log, outcome). It feeds the dashboard and the API.

**Lines 1–2**
```python
   1  """Record one episode as a JSON-friendly dict: public and private parameters, the reference
   2  price lines, the full event log, and the outcome. Used by the dashboard and its API."""
```
- Docstring.

**Lines 3–13**
```python
   3  from __future__ import annotations
   4  
   5  import json
   6  import pathlib
   7  
   8  import numpy as np
   9  
  10  from .. import metrics, scenarios, yardsticks
  11  from ..agents import scripted as S
  12  from ..env.core import Market, F, B
  13  from ..mechanisms.fee import FeeMechanism, FloorMechanism
```
- Imports: metrics, scenarios, yardsticks, scripted agents, env, and the two mechanisms.

**Lines 15–16**
```python
  15  ROOT = pathlib.Path(__file__).resolve().parents[2]
  16  SCRIPTED = ["ZIC", "Boulware", "Linear", "Conceder", "TitForTat", "OracleFairSplit", "TakeFallback"]
```
- `ROOT` = project root (two folders up). `SCRIPTED` lists scripted names the dashboard offers.

**Lines 19–26**
```python
  19  def make_mechanism(spec: dict | None):
  20      """spec: None | {"kind": "fee", "tau": .., "lam": .., ["observable": bool, "kappa": ..]} | {"kind": "floor", "kappa": ..}"""
  21      if not spec or spec.get("kind") in (None, "none"):
  22          return None
  23      if spec["kind"] == "floor":
  24          return FloorMechanism(float(spec["kappa"]))
  25      return FeeMechanism(tau=float(spec.get("tau", 0.4)), lam=float(spec.get("lam", 1.0)),
  26                          observable=bool(spec.get("observable", False)), kappa=spec.get("kappa"))
```
- `make_mechanism(spec)`: turns a dict into a mechanism object — None, a FloorMechanism(κ), or a FeeMechanism with τ, λ (default 1.0), observable flag and κ.

**Lines 29–45**
```python
  29  def make_agents(sc, farmer: str, buyer: str):
  30      """Agent names: a scripted class name, or 'run:<path under results/>' for a trained checkpoint
  31      (the same run supplies whichever side asks for it). Returns (farmer, buyer, delay_aware, lam_fair)."""
  32      out, delay_aware, lam = {}, True, None
  33      for role, name in ((F, farmer), (B, buyer)):
  34          if name.startswith("run:"):
  35              import torch
  36              from ..agents.ppo import load_agents
  37              path = ROOT / "results" / name[4:] / "checkpoint.pt"
  38              ck = torch.load(path, weights_only=True)
  39              out[role] = load_agents(path, sc)[role]
  40              delay_aware = bool(ck.get("delay_aware", True))
  41              if role == B:
  42                  lam = float(ck.get("lam_fair", 0.0))
  43          else:
  44              out[role] = S.make(name, role, sc)
  45      return out[F], out[B], delay_aware, lam
```
- `make_agents`: for each role, a name starting `run:` loads a trained checkpoint from `results/<path>/checkpoint.pt` (and picks up its delay-awareness and trained fee λ for the buyer); anything else is a scripted agent from the factory. Returns both agents plus flags.

**Lines 48–81**
```python
  48  def record_episode(sc, farmer, buyer, seed: int, mechanism=None, delay_aware: bool = True) -> dict:
  49      env = Market(sc, mechanism=mechanism, delay_aware=delay_aware, log_events=True)
  50      env.reset(seed)
  51      farmer.reset(seed)
  52      buyer.reset(seed)
  53      while not env.done:
  54          obs, mask, active = env.observe()
  55          env.step(np.stack([farmer.act(env, obs[F], mask[F], active[F]),
  56                             buyer.act(env, obs[B], mask[B], active[B])]), mask)
  57      res = env.result()
  58      T, rho = sc.T, sc.rho
  59      v, o_f, o_b = float(env.v[0]), float(env.o_f[0]), float(env.o_b[0])
  60      ts = list(range(T + 1))
  61      lines = {
  62          "nash": [yardsticks.nash_price(v, o_f, o_b, rho, t) for t in ts],
  63          "farmer_fallback": [o_f * rho ** t for t in ts],
  64          "buyer_walkaway": [v * rho ** t - o_b for t in ts],
  65      }
  66      if isinstance(mechanism, FeeMechanism):
  67          lines["target"] = [mechanism.target(env, 0, 0, t) for t in ts]
  68      if getattr(mechanism, "floor_kappa", None) is not None:
  69          lines["floor"] = [mechanism.floor_kappa * env.level * rho ** t for t in ts]
  70      m = metrics.aggregate([res])
  71      return {
  72          "scenario": sc.name, "seed": int(seed), "n_farmers": sc.n_farmers, "n_buyers": sc.n_buyers,
  73          "T": T, "rho": rho, "ttl": sc.ttl, "duty": sc.duty, "level": env.level,
  74          "v": env.v.tolist(), "o_f": env.o_f.tolist(), "o_b": env.o_b.tolist(),
  75          "deadline": env.deadline.tolist(), "p_lo": sc.p_lo, "p_hi": sc.p_hi,
  76          "lines": lines, "events": res.events, "deals": res.deals,
  77          "farmer_gain": res.farmer_gain.tolist(), "buyer_gain": res.buyer_gain.tolist(),
  78          "price_received": res.price_received.tolist(), "fees": res.buyer_fees.tolist(),
  79          "g_star": res.g_star, "E": m.get("E"), "FGC": m.get("FGC"),
  80          "farmer": getattr(farmer, "name", "?"), "buyer": getattr(buyer, "name", "?"),
  81      }
```
- `record_episode`: runs one episode with event logging on. Then builds the reference lines over t = 0..T for thread (0,0): the **Nash price**, the farmer's decaying fallback, and the buyer's decaying walk-away; plus the fee target and the floor if a mechanism exists. Returns a big dict: scenario info, private values, the lines, the event list, deals, per-agent gains, price received, fees, G*, E, FGC and the agent names.

**Lines 84–92**
```python
  84  def record(scenario: str, farmer: str, buyer: str, seed: int, mechanism: dict | None = None,
  85             overrides: dict | None = None) -> dict:
  86      sc = scenarios.get(scenario).with_(**(overrides or {}))
  87      f, b, delay_aware, lam = make_agents(sc, farmer, buyer)
  88      if mechanism and mechanism.get("kind") == "fee" and "lam" not in mechanism and lam is not None:
  89          mechanism = dict(mechanism, lam=lam)                 # use the fee rate the buyer was trained with
  90      ep = record_episode(sc, f, b, seed, make_mechanism(mechanism), delay_aware)
  91      ep["farmer"], ep["buyer"], ep["mechanism"] = farmer, buyer, mechanism
  92      return json.loads(json.dumps(ep, default=float))
```
- `record`: loads the scenario (with optional overrides), builds agents, uses the trained λ if the request didn't give one, records, and round-trips through JSON (`default=float`) so NumPy types become plain numbers.


## 2.14  `agrinegotiate/data/fetch.py`

Fetches real mandi (market) prices from India's open-data portal. Written but NOT run: the sandbox has no network access to data.gov.in. This is the 'real data' roadmap item.

**Lines 1–8**
```python
   1  """Fetch current mandi prices from data.gov.in.
   2  
   3  UNVERIFIED AND NOT RUN: the resource id below comes from the literature review and must be
   4  confirmed on data.gov.in. The API appears to serve current-day prices only; for history use a
   5  Kaggle mirror and pass its CSV to `agrinegotiate.data.calibrate.run`. Check the licence first.
   6  
   7  Usage: DATA_GOV_IN_KEY=... python -m agrinegotiate.data.fetch out.csv --commodity Tomato
   8  """
```
- Docstring: honest disclaimer — the resource id is unverified; the API serves current-day prices only; history needs a Kaggle mirror; check the licence. Usage needs an API key in an environment variable.

**Lines 9–10**
```python
   9  import argparse
  10  import os
```
- Imports `argparse` (CLI) and `os` (read the key).

**Lines 12–13**
```python
  12  RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"   # Verify before use
  13  URL = "https://api.data.gov.in/resource/" + RESOURCE_ID
```
- Resource id and the API URL built from it.

**Lines 16–19**
```python
  16  def fetch(commodity: str | None = None, limit: int = 1000, max_pages: int = 50):
  17      import pandas as pd
  18      import requests
  19      key = os.environ["DATA_GOV_IN_KEY"]
```
- `fetch`: imports pandas/requests lazily (so the package works without them) and reads the API key from the environment (raises KeyError if missing).

**Lines 20–32**
```python
  20      rows, offset = [], 0
  21      for _ in range(max_pages):
  22          params = {"api-key": key, "format": "json", "limit": limit, "offset": offset}
  23          if commodity:
  24              params["filters[commodity]"] = commodity
  25          r = requests.get(URL, params=params, timeout=60)
  26          r.raise_for_status()
  27          recs = r.json().get("records", [])
  28          if not recs:
  29              break
  30          rows += recs
  31          offset += limit
  32      return pd.DataFrame(rows)
```
- Pagination loop: request `limit` rows at a time with an increasing offset, optionally filtered by commodity; `raise_for_status` surfaces HTTP errors; stop when a page returns no records or after `max_pages`. Returns a DataFrame.

**Lines 35–40**
```python
  35  if __name__ == "__main__":
  36      ap = argparse.ArgumentParser()
  37      ap.add_argument("out")
  38      ap.add_argument("--commodity")
  39      a = ap.parse_args()
  40      fetch(a.commodity).to_csv(a.out, index=False)
```
- CLI entry: output path and optional commodity; saves the CSV.


## 2.15  `agrinegotiate/data/clean.py`

Cleaning rules for raw mandi price rows. Not yet run on real data.

**Lines 1–7**
```python
   1  """Cleaning rules for AGMARKNET mandi price rows.
   2  
   3  Expected columns: State, District, Market, Commodity, Variety, Arrival_Date,
   4  Min_Price, Max_Price, Modal_Price (rupees per quintal). Column names differ between
   5  the data.gov.in API and the Kaggle mirrors; `normalise_columns` maps the variants we know of.
   6  NOT YET RUN ON REAL DATA: this sandbox cannot reach data.gov.in or Kaggle.
   7  """
```
- Docstring: expected columns (rupees per quintal), and the note that column names differ between sources and that it has not been run on real data.

**Lines 8–10**
```python
   8  from __future__ import annotations
   9  
  10  import pandas as pd
```
- Future-annotations and pandas imports.

**Lines 12–17**
```python
  12  ALIASES = {
  13      "state": "state", "district": "district", "market": "market", "commodity": "commodity",
  14      "variety": "variety", "arrival_date": "date", "date": "date",
  15      "min_price": "min", "max_price": "max", "modal_price": "modal",
  16      "min price": "min", "max price": "max", "modal price": "modal",
  17  }
```
- `ALIASES`: maps known column-name variants (case/spacing/underscore) to canonical names: state, district, market, commodity, variety, date, min, max, modal.

**Lines 20–26**
```python
  20  def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
  21      df = df.rename(columns={c: ALIASES.get(c.strip().lower().replace("_x0020_", "_"), c) for c in df.columns})
  22      need = ["market", "commodity", "date", "min", "max", "modal"]
  23      missing = [c for c in need if c not in df.columns]
  24      if missing:
  25          raise ValueError(f"missing columns after renaming: {missing}; have {list(df.columns)}")
  26      return df[need].copy()
```
- `normalise_columns`: rename using the alias map after stripping/lowercasing (and fixing '_x0020_', an XML-escaped space); require the six needed columns or raise a clear error; return only those columns.

**Lines 29–41**
```python
  29  def clean(df: pd.DataFrame) -> pd.DataFrame:
  30      df = normalise_columns(df)
  31      df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
  32      for c in ("min", "max", "modal"):
  33          df[c] = pd.to_numeric(df[c], errors="coerce")
  34      df = df.dropna(subset=["date", "min", "max", "modal"])
  35      df = df[(df["modal"] > 0) & (df["min"] > 0) & (df["max"] > 0)]
  36      df = df[(df["modal"] >= df["min"]) & (df["modal"] <= df["max"])]
  37      df = df.drop_duplicates(subset=["market", "commodity", "date"])
  38      lo = df.groupby(["market", "commodity"])["modal"].transform(lambda s: s.quantile(0.01))
  39      hi = df.groupby(["market", "commodity"])["modal"].transform(lambda s: s.quantile(0.99))
  40      df["modal"] = df["modal"].clip(lo, hi)
  41      return df.sort_values(["commodity", "market", "date"]).reset_index(drop=True)
```
- `clean`: (1) normalise columns; (2) parse dates day-first, bad → NaT; (3) convert prices to numbers, bad → NaN; (4) drop rows missing any of them; (5) keep positive prices only; (6) keep rows where min ≤ modal ≤ max; (7) drop duplicate (market, commodity, date); (8) **winsorise** the modal price to its 1st–99th percentile per market-commodity (tames typos/outliers); (9) sort and reset the index.


## 2.16  `agrinegotiate/data/calibrate.py`

Turns cleaned prices into scenario parameters (how noisy farmers' price information is, the typical spread). Not yet run on real data.

**Lines 1–7**
```python
   1  """Calibration statistics from cleaned prices.
   2  
   3  Per commodity: median modal price, standard deviation of lagged log returns (the farmer's
   4  stale-information noise sigma_info at lags 1, 3 and 7 days) and the relative daily spread.
   5  Gaps longer than 3 days are never bridged.
   6  NOT YET RUN ON REAL DATA.
   7  """
```
- Docstring: per commodity — median modal price, standard deviation of lagged log-returns at 1, 3 and 7 days (the farmer's stale-information noise σ_info), and the relative spread. Gaps longer than 3 days are never bridged.

**Lines 8–13**
```python
   8  from __future__ import annotations
   9  
  10  import json
  11  
  12  import numpy as np
  13  import pandas as pd
```
- Imports.

**Lines 15–16**
```python
  15  LAGS = (1, 3, 7)
  16  MAX_GAP_DAYS = 3
```
- `LAGS = (1, 3, 7)` days and `MAX_GAP_DAYS = 3`.

**Lines 19–27**
```python
  19  def calibrate(df: pd.DataFrame) -> dict:
  20      out = {}
  21      for commodity, g in df.groupby("commodity"):
  22          rets = {lag: [] for lag in LAGS}
  23          for _, s in g.groupby("market"):
  24              s = s.set_index("date")["modal"].asfreq("D")
  25              s = s.interpolate(limit=MAX_GAP_DAYS, limit_area="inside")
  26              for lag in LAGS:
  27                  rets[lag].append(np.log(s / s.shift(lag)).dropna().to_numpy())
```
- `calibrate`: per commodity and per market, make a daily series (`asfreq('D')`), interpolate only short interior gaps (≤3 days), then collect log-returns `log(s_t / s_{t−lag})` for each lag.

**Lines 28–36**
```python
  28          out[str(commodity)] = {
  29              "rows": int(len(g)),
  30              "markets": int(g["market"].nunique()),
  31              "median_modal_rs_per_quintal": float(g["modal"].median()),
  32              "sigma_info_by_lag_days": {str(lag): float(np.concatenate(rets[lag]).std()) if rets[lag] and
  33                                         sum(len(r) for r in rets[lag]) > 1 else None for lag in LAGS},
  34              "relative_spread": float(((g["max"] - g["min"]) / g["modal"]).median()),
  35          }
  36      return out
```
- Output per commodity: row count, number of markets, median modal price, σ_info for each lag (standard deviation of the pooled log-returns, None if fewer than 2 values), and median relative spread `(max−min)/modal`.

**Lines 39–46**
```python
  39  def run(csv_in: str, parquet_out: str, json_out: str) -> dict:
  40      from .clean import clean
  41      df = clean(pd.read_csv(csv_in))
  42      df.to_parquet(parquet_out, index=False)
  43      cal = calibrate(df)
  44      with open(json_out, "w") as f:
  45          json.dump(cal, f, indent=1)
  46      return cal
```
- `run`: clean the raw CSV, save a parquet, calibrate, write the JSON, return it.


## 2.17  `experiments/run.py`

The generic experiment runner: a *job* is a dict; each run writes `results/<experiment>/<label>/seed_<n>/`. `python experiments/run.py <batch> --workers 2`. Finished runs are skipped, so a batch can be safely restarted.

**Lines 1–7**
```python
   1  """Generic experiment runner. A job is a dict; each run writes results/<exp>/<label>/seed_<n>/.
   2  
   3      python experiments/run.py <batch> [--workers 2]
   4  
   5  Batches are defined in BATCHES below. Finished runs (metrics.json present) are skipped,
   6  so a batch can be restarted safely.
   7  """
```
- Docstring: usage and the resume-safe design (a run with `metrics.json` is skipped).

**Lines 8–9**
```python
   8  import argparse, pathlib, sys
   9  from multiprocessing import Pool
```
- Imports; `Pool` for parallel workers.

**Lines 11–12**
```python
  11  ROOT = pathlib.Path(__file__).resolve().parents[1]
  12  sys.path.insert(0, str(ROOT))
```
- `ROOT` = project folder; inserting it in `sys.path` lets the script import the `agrinegotiate` package when run directly.

**Lines 14–16**
```python
  14  from agrinegotiate import scenarios                                   # noqa: E402
  15  from agrinegotiate.agents.ppo import PPOConfig, train                 # noqa: E402
  16  from agrinegotiate.mechanisms.fee import FeeMechanism, FloorMechanism  # noqa: E402
```
- Imports of scenarios, PPO train/config and the two mechanisms (the `noqa: E402` comments silence the 'import not at top' lint warning, which is intentional here).

**Lines 18–21**
```python
  18  S10 = list(range(1000, 1010))
  19  S5 = list(range(1000, 1005))
  20  S3 = list(range(1000, 1003))
  21  KAPPA_TAU04_PERISH = 0.606     # average oracle target price / (P_ref * rho^t) at tau = 0.4
```
- Seed lists: 10 seeds (1000–1009), 5, 3. `KAPPA_TAU04_PERISH = 0.606` is the observable-target κ equal to the average oracle target at τ=0.4 in PERISH (spec 7.4).

**Lines 22–22**
```python
  22  STEPS_3M = 3_000_000           # planned per-run budget in agent-steps
```
- `STEPS_3M`: the per-run budget planned in the proposal, in agent-steps; used by the `long3m` batches.

**Lines 25–26**
```python
  25  def J(exp, label, scenario, seeds, **kw):
  26      return [dict(exp=exp, label=label, scenario=scenario, seed=s, **kw) for s in seeds]
```
- `J(...)`: one job per seed with the shared settings.

**Lines 29–38**
```python
  29  def mech_jobs(exp, scn, tau_main_seeds=S10):
  30      jobs = J(exp, "fee_oracle_tau0.4", scn, tau_main_seeds, mech="fee_adaptive", tau=0.4, iters=150)
  31      for tau in (0.2, 0.3, 0.5):
  32          jobs += J(exp, f"fee_oracle_tau{tau}", scn, S3, mech="fee_adaptive", tau=tau, iters=150)
  33      for kappa in (0.50, 0.55, 0.60, 0.65):
  34          jobs += J(exp, f"floor_kappa{kappa:.2f}", scn, S3, mech="floor", kappa=kappa, iters=150)
  35      jobs += J(exp, "fee_fixed_lam1.0_tau0.4", scn, S3, mech="fee_fixed", tau=0.4, lam=1.0, iters=150)
  36      jobs += J(exp, "fee_observable_tau0.4", scn, S3, mech="fee_adaptive", tau=0.4, observable=True,
  37                kappa=KAPPA_TAU04_PERISH, iters=150)
  38      return jobs
```
- `mech_jobs`: the mechanism experiment grid: adaptive fee at τ=0.4 with 10 seeds; τ = 0.2/0.3/0.5 with 3 seeds; floors κ = 0.50–0.65; a fixed λ=1 fee; and the observable-target fee.

**Lines 41–50**
```python
  41  def delay_jobs(exp):
  42      jobs = []
  43      cells = {"unaware": dict(recurrent=False, delay_aware=False), "aware": dict(recurrent=True, delay_aware=True)}
  44      for duty in (1.0, 0.6, 0.4):
  45          for name, kw in cells.items():
  46              jobs += J(exp, f"{name}_duty{duty}", "RURAL1-DTN", S3, sc=dict(duty=duty), **kw)
  47      # 2 x 2 ablation at duty 0.6: memory only, features only
  48      jobs += J(exp, "memory_only_duty0.6", "RURAL1-DTN", S3, sc=dict(duty=0.6), recurrent=True, delay_aware=False)
  49      jobs += J(exp, "features_only_duty0.6", "RURAL1-DTN", S3, sc=dict(duty=0.6), recurrent=False, delay_aware=True)
  50      return jobs
```
- `delay_jobs`: delay experiment: delay-unaware (feed-forward, delay features hidden) vs delay-aware (GRU + features) at duty 1.0/0.6/0.4; plus the 2×2 ablation at 0.6 (memory only; features only).

**Lines 53–80**
```python
  53  BATCHES = {
  54      "gate_core": J("x2_gate", "PERISH", "PERISH", S10) + J("x2_gate", "SYM", "SYM", S10),
  55      "mechanism": mech_jobs("x3_mechanism", "PERISH"),
  56      "delay": delay_jobs("x4_delay"),
  57      "gate_more": J("x2_gate", "DEADLINE", "DEADLINE", S5) + J("x2_gate", "INFO", "INFO", S5)
  58                   + J("x2_gate", "THIN3", "THIN3", S5),
  59      # Added after seeing that the fee rate was still rising at the end of the main mechanism runs:
  60      # the same adaptive fee with a five times larger dual step size. Post hoc, so exploratory.
  61      "mechanism_fast": J("x3_mechanism", "fee_oracle_tau0.4_eta0.25", "PERISH", S5, mech="fee_adaptive", tau=0.4,
  62                          eta=0.25, iters=150),
  63      # Heterogeneous buyer values: does the floor still win when it must exclude low-value buyers? Post hoc.
  64      "hetero": J("x6_hetero", "unconstrained", "PERISH-HET", S5)
  65                + J("x6_hetero", "fee_oracle_tau0.4_eta0.25", "PERISH-HET", S5, mech="fee_adaptive", tau=0.4, eta=0.25, iters=150)
  66                + J("x6_hetero", "fee_fixed_lam1.0_tau0.4", "PERISH-HET", S3, mech="fee_fixed", tau=0.4, lam=1.0, iters=150)
  67                + sum((J("x6_hetero", f"floor_kappa{k:.2f}", "PERISH-HET", S3, mech="floor", kappa=k, iters=150)
  68                       for k in (0.50, 0.55, 0.60)), []),
  69      # Confirmation of the PERISH-HET lead at 10 seeds (fee vs floors vs no rule). Same settings as "hetero".
  70      "hetero10": J("x6_hetero", "unconstrained", "PERISH-HET", S10)
  71                  + J("x6_hetero", "fee_oracle_tau0.4_eta0.25", "PERISH-HET", S10, mech="fee_adaptive", tau=0.4, eta=0.25, iters=150)
  72                  + sum((J("x6_hetero", f"floor_kappa{k:.2f}", "PERISH-HET", S10, mech="floor", kappa=k, iters=150)
  73                         for k in (0.50, 0.55, 0.60)), []),
  74      # Scenarios calibrated on real Agmarknet lemon prices (data/real): price level, information noise and mandi-to-mandi spread.
  75      "real": J("x7_real", "unconstrained", "PERISH-REAL", S10)
  76              + J("x7_real", "fee_oracle_tau0.4_eta0.25", "PERISH-REAL", S10, mech="fee_adaptive", tau=0.4, eta=0.25, iters=150)
  77              + J("x7_real", "floor_kappa0.55", "PERISH-REAL", S5, mech="floor", kappa=0.55, iters=150)
  78              + J("x7_real", "aware_duty0.6", "RURAL1-REAL-DTN", S5, recurrent=True, delay_aware=True)
  79              + J("x7_real", "unaware_duty0.6", "RURAL1-REAL-DTN", S5, recurrent=False, delay_aware=False),
  80      "diagnosis": J("x5_diagnosis", "SYM_mix0", "SYM", S3, mix=0.0) + J("x5_diagnosis", "SYMX", "SYMX", S3),
```
- `BATCHES`: names → job lists: gate_core (PERISH, SYM), mechanism, delay, gate_more (DEADLINE, INFO, THIN3), mechanism_fast (η=0.25, **post hoc, exploratory**), hetero and hetero10 (heterogeneous buyers, 5 then 10 seeds), real (scenarios calibrated on real prices, below), diagnosis (SYM without scripted mixing, SYMX). The comments openly mark which ones were added after seeing results.

**Lines 74–79**
```python
  74      # Scenarios calibrated on real Agmarknet lemon prices (data/real): price level, information noise and mandi-to-mandi spread.
  75      "real": J("x7_real", "unconstrained", "PERISH-REAL", S10)
  76              + J("x7_real", "fee_oracle_tau0.4_eta0.25", "PERISH-REAL", S10, mech="fee_adaptive", tau=0.4, eta=0.25, iters=150)
  77              + J("x7_real", "floor_kappa0.55", "PERISH-REAL", S5, mech="floor", kappa=0.55, iters=150)
  78              + J("x7_real", "aware_duty0.6", "RURAL1-REAL-DTN", S5, recurrent=True, delay_aware=True)
  79              + J("x7_real", "unaware_duty0.6", "RURAL1-REAL-DTN", S5, recurrent=False, delay_aware=False),
```
- Line 73: comment for the new batch.
- Lines 74–78: batch `real` (results folder `x7_real`). Five configurations on the real-data scenarios: unconstrained (10 seeds), the adaptive fee with fast dual step η = 0.25 (10 seeds), a hard price floor κ = 0.55 (5 seeds), and the delay-aware GRU agent against the delay-unaware agent on `RURAL1-REAL-DTN` (5 seeds each). These feed hypotheses H3_real and H5_real.

**Lines 81–97**
```python
  81      # The main comparisons rerun at the planned budget of 3 million agent-steps per run (the runs above
  82      # used 0.3 to 0.7 million). Same scenarios, settings and seeds; written to x8_3m_<exp>/.
  83      "long3m": [dict(j, exp="x8_3m_" + j["exp"].split("_", 1)[1], steps=STEPS_3M) for j in
  84                 J("x2_gate", "SYM", "SYM", S10) + J("x2_gate", "PERISH", "PERISH", S10)
  85                 + J("x2_gate", "THIN3", "THIN3", S5)
  86                 + J("x3_mechanism", "fee_oracle_tau0.4", "PERISH", S10, mech="fee_adaptive", tau=0.4)
  87                 + J("x3_mechanism", "fee_fixed_lam1.0_tau0.4", "PERISH", S3, mech="fee_fixed", tau=0.4, lam=1.0)
  88                 + sum((J("x3_mechanism", f"floor_kappa{k:.2f}", "PERISH", S3, mech="floor", kappa=k)
  89                        for k in (0.50, 0.55, 0.60, 0.65)), [])
  90                 + J("x6_hetero", "unconstrained", "PERISH-HET", S10)
  91                 + J("x6_hetero", "fee_oracle_tau0.4_eta0.25", "PERISH-HET", S10, mech="fee_adaptive", tau=0.4, eta=0.25)
  92                 + sum((J("x6_hetero", f"floor_kappa{k:.2f}", "PERISH-HET", S10, mech="floor", kappa=k)
  93                        for k in (0.50, 0.55, 0.60)), [])
  94                 + J("x7_real", "unconstrained", "PERISH-REAL", S10)
  95                 + J("x7_real", "fee_oracle_tau0.4_eta0.25", "PERISH-REAL", S10, mech="fee_adaptive", tau=0.4, eta=0.25)
  96                 + J("x7_real", "floor_kappa0.55", "PERISH-REAL", S5, mech="floor", kappa=0.55)],
  97  }
```
- `long3m`: the main comparisons rerun at 3 million agent-steps per run. It lists the same jobs as the gate, mechanism, hetero and real batches (same labels, settings and seeds, without `iters`), then rewrites each one to write into `x8_3m_<experiment>/` (e.g. `x2_gate` → `x8_3m_gate`) and to train until `STEPS_3M` agent-steps.

**Lines 98–105**
```python
  98  # The full long3m set needs about 10 hours here (40-50 minutes per run, 12 at a time). The core subset keeps
  99  # every main comparison: no fixed fee or outer floors in PERISH, 5 seeds in PERISH-HET and PERISH-REAL.
 100  BATCHES["long3m_core"] = [
 101      j for j in BATCHES["long3m"]
 102      if j["label"] not in ("fee_fixed_lam1.0_tau0.4",)
 103      and not (j["exp"] == "x8_3m_mechanism" and j["label"] in ("floor_kappa0.50", "floor_kappa0.65"))
 104      and not (j["exp"] == "x8_3m_hetero" and j["label"] == "floor_kappa0.55")
 105      and not (j["exp"] in ("x8_3m_hetero", "x8_3m_real") and j["seed"] not in S5)]
```
- `long3m_core`: the subset actually run (the full set needs about 10 hours on this machine). It drops the fixed fee, the outer floors (κ 0.50 and 0.65) in PERISH and floor κ 0.55 in PERISH-HET, and keeps 5 seeds in PERISH-HET and PERISH-REAL. Every main comparison is still there.

**Lines 108–112**
```python
 108  def run_job(job):
 109      out = ROOT / "results" / job["exp"] / job["label"] / f"seed_{job['seed']}"
 110      if (out / "metrics.json").exists():
 111          return job, None
 112      sc = scenarios.get(job["scenario"]).with_(**job.get("sc", {}))
```
- `run_job`: output folder; skip if done; build the scenario with optional overrides (e.g. `duty`).

**Lines 113–115**
```python
 113      cfg = PPOConfig(iters=job.get("iters", 120), mix_scripted=job.get("mix", 0.3),
 114                      recurrent=job.get("recurrent", False), delay_aware=job.get("delay_aware", True),
 115                      target_agent_steps=job.get("steps", 0))
```
- PPO config from job fields: default 120 iterations; `steps` (if given) switches to a fixed agent-step budget.

**Lines 116–123**
```python
 116      mech, adaptive = None, False
 117      kind = job.get("mech")
 118      if kind == "floor":
 119          mech = FloorMechanism(job["kappa"])
 120      elif kind in ("fee_adaptive", "fee_fixed"):
 121          mech = FeeMechanism(tau=job["tau"], lam=job.get("lam", 0.0), observable=job.get("observable", False),
 122                              kappa=job.get("kappa"), eta=job.get("eta", 0.05))
 123          adaptive = kind == "fee_adaptive"
```
- Build the mechanism: floor with κ, or fee (adaptive or fixed) with τ, λ, observable flag, κ and η.

**Lines 124–125**
```python
 124      summary = train(sc, cfg, job["seed"], out, mech=mech, adaptive_fee=adaptive, log=lambda *a: None)
 125      return job, summary
```
- Train quietly (log suppressed) and return the summary.

**Lines 128–139**
```python
 128  if __name__ == "__main__":
 129      ap = argparse.ArgumentParser()
 130      ap.add_argument("batches", nargs="+")
 131      ap.add_argument("--workers", type=int, default=2)
 132      a = ap.parse_args()
 133      jobs = [j for b in a.batches for j in BATCHES[b]]
 134      print(len(jobs), "jobs", flush=True)
 135      with Pool(a.workers) as pool:
 136          for job, s in pool.imap_unordered(run_job, jobs, chunksize=1):
 137              m = "skipped" if s is None else {k: (v and round(v, 3)) for k, v in s["last10_checkpoint_mean"].items()
 138                                                if k in ("E", "FGC", "S_F", "agreement_rate", "lam_fair")}
 139              print("done", job["exp"], job["label"], job["seed"], m, flush=True)
```
- Entry point: parse batches and workers, flatten jobs, run them in a process pool with `imap_unordered` (print as each finishes), reporting key metrics rounded.


## 2.18  `experiments/calibrate_real.py`

**Role of this file.** It turns real Agmarknet lemon prices into the three numbers the simulator needs and writes them to `data/real/calibration.json`. Run once; the scenarios read the json. It keeps the data real for prices and market conditions while the negotiations themselves stay simulated, because no public record of farmer–buyer negotiations exists.

**Likely questions.** Why winsorise? (A few mistyped prices would inflate every spread.) Why a 61-day local median? (A 30-step episode sees a short price shock, not the slow seasonal swing.) Why only lemon? (It is the crop with enough clean daily data found; tomato and onion are future work.)

**Lines 1–10**
```python
   1  """Calibrate the scenarios from real Agmarknet mandi prices (data/real/lemon_daily_agmarknet.csv.gz).
   2  
   3      python experiments/calibrate_real.py
   4  
   5  Writes data/real/calibration.json with three things the simulator needs:
   6    level_samples   quantiles of the national daily price relative to its median -> the market "price level" of an episode
   7    sigma_info      std of log price changes over 3 days at the same mandi -> how stale a farmer's price signal is
   8    theta_spread    same-day price spread between mandis of one state -> how different farmers' fallback options are
   9  Prices are winsorised per mandi and the spreads use clipped or robust statistics, so a few mistyped prices cannot inflate them.
  10  """
```
- Lines 1–10: docstring. It names the input file, the command, and the three outputs: `level_samples`, `sigma_info`, `theta_spread`, plus the robustness choice.

**Lines 11–15**
```python
  11  import json, pathlib
  12  import numpy as np, pandas as pd
  13  
  14  ROOT = pathlib.Path(__file__).resolve().parents[1]
  15  SRC, OUT = ROOT / "data/real/lemon_daily_agmarknet.csv.gz", ROOT / "data/real/calibration.json"
```
- Lines 11–15: imports; `ROOT` is the project folder; `SRC` and `OUT` are the price file and the json that gets written.

**Lines 18–20**
```python
  18  def robust_std(x) -> float:
  19      x = np.asarray(x, float)
  20      return float(1.4826 * np.median(np.abs(x - np.median(x))))
```
- Lines 18–20: `robust_std` = 1.4826 × median absolute deviation, which equals the standard deviation for normal data but ignores outliers.

**Lines 23–27**
```python
  23  d = pd.read_csv(SRC, parse_dates=["date"])
  24  d["mandi"] = d["state"] + "/" + d["market"]          # market names repeat across states
  25  lo, hi = d.groupby("mandi")["modal"].transform(lambda s: s.quantile(.01)), d.groupby("mandi")["modal"].transform(lambda s: s.quantile(.99))
  26  d["modal"] = d["modal"].clip(lo, hi)                       # winsorise each mandi at its 1st-99th percentile
  27  md = d.groupby(["state", "mandi", "date"], as_index=False)["modal"].median()   # one price per mandi and day
```
- Line 23: read the daily price table.
- Line 24: make a unique mandi key `state/market` because market names repeat across states.
- Lines 25–26: work out each mandi's 1st and 99th percentile price and clip its prices to that range (winsorising), so mistyped prices cannot dominate.
- Line 27: one price per mandi and day (median).

**Lines 29–34**
```python
  29  # 1. price level of an episode = the national price relative to its own local (61-day) median, i.e. the short-run
  30  #    price shock a negotiation takes place in (the slow seasonal swing is not part of one 30-step episode)
  31  nat = md.groupby("date")["modal"].median().asfreq("D").interpolate(limit=3)
  32  ratio = (nat / nat.rolling(61, center=True, min_periods=30).median()).dropna()
  33  qs = np.quantile(ratio, np.linspace(0.02, 0.98, 25))
  34  level_samples = [round(float(np.clip(0.9 * q, 0.6, 1.15)), 4) for q in qs]   # scaled so the typical level is 0.9 and stays on the grid
```
- Lines 29–30: explain the goal, the price level of an episode.
- Line 31: national daily price (median across mandis), filling gaps of up to 3 days.
- Line 32: divide by its own centred 61-day rolling median, giving the short-run shock around the local trend.
- Lines 33–34: take 25 quantiles of that ratio from the 2nd to the 98th percentile, scale by 0.9 so the typical level is 0.9, and clip to [0.6, 1.15] so it stays on the price grid. These are `level_samples`.

**Lines 36–43**
```python
  36  # 2. information staleness: log price change at the same mandi after k days (extreme 2.5% on each side clipped: typos)
  37  wide = md.pivot(index="date", columns="mandi", values="modal").asfreq("D")
  38  sigma = {}
  39  for k in (1, 3, 7):
  40      r = np.log(wide / wide.shift(k)).to_numpy().ravel()
  41      r = r[np.isfinite(r)]
  42      c = np.clip(r, *np.quantile(r, [0.025, 0.975]))
  43      sigma[str(k)] = dict(std_clipped=round(float(c.std()), 4), share_unchanged=round(float((r == 0).mean()), 3), pairs=int(r.size))
```
- Lines 36–43: information staleness. For gaps of 1, 3 and 7 days, take the log price change at the same mandi, clip the extreme 2.5% on each side, and record the standard deviation, the share of unchanged prices and the pair count. The 3-day value becomes `sigma_info`.

**Lines 45–52**
```python
  45  # 3. same-day spread between mandis of a state (>= 5 mandis reporting)
  46  g = md.groupby(["state", "date"])["modal"]
  47  n, med = g.transform("size"), g.transform("median")
  48  sub = md[n >= 5].assign(rel=lambda x: x["modal"] / med[n >= 5])
  49  spread = sub.groupby(["state", "date"])["rel"].apply(robust_std)
  50  rel_spread = float(spread.median())
  51  theta = 0.45
  52  theta_spread = round(float(min(0.20, rel_spread * np.sqrt(3) * theta)), 3)   # U(theta +- s) has relative std s / (sqrt(3) theta)
```
- Lines 45–50: for each state and day with at least 5 mandis reporting, measure how far each mandi's price sits from that day's state median, using `robust_std`; take the median over all days.
- Lines 51–52: convert that relative spread into `theta_spread`. A uniform spread of ± s around θ has relative standard deviation s / (√3 θ), so s = spread × √3 × θ, capped at 0.20.

**Lines 54–61**
```python
  54  cal = dict(source="Agmarknet daily lemon prices, Government Open Data Licence India", rows=int(len(d)), mandis=int(d.mandi.nunique()),
  55             states=int(d.state.nunique()), days=int(d.date.nunique()), first=str(d.date.min().date()), last=str(d.date.max().date()),
  56             price_level_ratio_quantiles={f"p{q}": round(float(ratio.quantile(q / 100)), 3) for q in (2, 10, 50, 90, 98)},
  57             level_samples=level_samples, sigma_info=sigma, rel_spread_between_mandis=round(rel_spread, 4),
  58             theta_spread=theta_spread, sigma_info_used=sigma["3"]["std_clipped"])
  59  OUT.write_text(json.dumps(cal, indent=1))
  60  print(json.dumps({k: v for k, v in cal.items() if k != "level_samples"}, indent=1))
  61  print("level_samples range", min(level_samples), max(level_samples))
```
- Lines 54–61: assemble the calibration record (source, row, mandi, state and day counts, date range, key quantiles, `level_samples`, all σ values, spread), write it as json, and print a summary without the long list.


## 2.19  `experiments/x1_scripted.py`

Experiment X1 (scripted part): baseline table for every named scenario — the answer to 'where is your benchmark?'.

**Lines 1–5**
```python
   1  """Experiment X1 (scripted part): baseline table for every named scenario.
   2  
   3  Usage: python experiments/x1_scripted.py [--episodes 1000] [--workers 2]
   4  Writes results/x1_scripted/{metrics.json, tuning.json, table.md}.
   5  """
```
- Docstring: usage; outputs `metrics.json`, `tuning.json`, `table.md`.

**Lines 6–7**
```python
   6  import argparse, json, pathlib, subprocess, sys, time
   7  from multiprocessing import Pool
```
- Imports of argparse/json/pathlib/subprocess/sys/time and multiprocessing `Pool`.

**Lines 9–10**
```python
   9  ROOT = pathlib.Path(__file__).resolve().parents[1]
  10  sys.path.insert(0, str(ROOT))
```
- Project root on `sys.path` so the package imports work when run as a script.

**Lines 12–16**
```python
  12  from agrinegotiate import scenarios                      # noqa: E402
  13  from agrinegotiate.agents import scripted as S           # noqa: E402
  14  from agrinegotiate.env.core import F, B                  # noqa: E402
  15  from agrinegotiate.eval.evaluate import evaluate, EVAL_SEED_START   # noqa: E402
  16  from agrinegotiate.eval.tuning import tune               # noqa: E402
```
- Package imports: scenarios, scripted agents, side constants, evaluator + eval seed start, tuner.

**Lines 18–22**
```python
  18  NAMED = ["SYM", "PERISH", "DEADLINE", "INFO", "THIN", "RURAL", "RURAL-DTN"]
  19  PAIRS = [("TakeFallback", "TakeFallback"), ("OracleFairSplit", "OracleFairSplit"), ("ZIC", "ZIC"),
  20           ("Boulware", "Boulware"), ("Linear", "Linear"), ("Conceder", "Conceder"),
  21           ("TitForTat", "TitForTat"), ("Tuned", "Tuned"), ("Boulware", "HardballBuyer"),
  22           ("Tuned", "HardballBuyer")]
```
- `NAMED` scenarios and `PAIRS` of (farmer, buyer) strategies: the floor (TakeFallback), ceiling (OracleFair), ZIC, the three time-dependent tactics, TitForTat, the Tuned heuristic, and Hardball buyer probes.

**Lines 25–26**
```python
  25  def tune_job(name):
  26      return name, tune(scenarios.get(name))
```
- `tune_job`: tune β for a scenario (runs in parallel).

**Lines 29–35**
```python
  29  def eval_job(args):
  30      name, fn, bn, tuned, n = args
  31      sc = scenarios.get(name)
  32      mk = lambda n_, role: S.make(f"TD:{tuned['farmer_beta' if role == F else 'buyer_beta']}", role, sc) \
  33          if n_ == "Tuned" else S.make(n_, role, sc)
  34      r = evaluate(sc, mk(fn, F), mk(bn, B), n, seed_start=EVAL_SEED_START)
  35      return dict(scenario=name, farmer=fn, buyer=bn, **r)
```
- `eval_job`: unpack the job; a lambda swaps the string 'Tuned' for a time-dependent agent with the tuned β, otherwise builds a named agent; run the evaluation on fixed seeds; return a result row.

**Lines 38–39**
```python
  38  def fmt(x, d=3):
  39      return "n/a" if x is None else f"{x:.{d}f}"
```
- `fmt`: prints 'n/a' for None, else fixed decimals.

**Lines 42–56**
```python
  42  def render(out, rows, tuned, episodes):
  43      lines = [f"# X1 scripted baselines ({episodes} fixed evaluation episodes per cell, seeds {EVAL_SEED_START}+)", "",
  44               "Scripted agents are deterministic given the seed, so there are no training seeds and no",
  45               "confidence intervals here. E can be negative: waiting and then failing is worse than walking away.", ""]
  46      for n in NAMED:
  47          lines += [f"## {n}  (tuned beta: farmer {tuned[n]['farmer_beta']:g}, buyer {tuned[n]['buyer_beta']:g})", "",
  48                    "| Farmer | Buyer | E | FGC | S_F | Agreement | Price share | Mean phi | Steps to deal |",
  49                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
  50          for r in rows:
  51              if r["scenario"] == n:
  52                  lines.append(f"| {r['farmer']} | {r['buyer']} | {fmt(r['E'])} | {fmt(r['FGC'])} | {fmt(r['S_F'])} | "
  53                               f"{fmt(r['agreement_rate'])} | {fmt(r['price_share'])} | {fmt(r['phi_mean'])} | "
  54                               f"{fmt(r['time_to_agreement'], 1)} |")
  55          lines.append("")
  56      (out / "table.md").write_text("\n".join(lines))
```
- `render`: writes a markdown table per scenario (E, FGC, S_F, agreement, price share, mean φ, steps to deal), with a note that scripted agents are deterministic so there are no CIs and E can be negative (waiting then failing is worse than walking away).

**Lines 59–77**
```python
  59  if __name__ == "__main__":
  60      ap = argparse.ArgumentParser()
  61      ap.add_argument("--episodes", type=int, default=1000)
  62      ap.add_argument("--workers", type=int, default=2)
  63      a = ap.parse_args()
  64      out = ROOT / "results" / "x1_scripted"
  65      out.mkdir(parents=True, exist_ok=True)
  66      t0 = time.time()
  67      with Pool(a.workers) as pool:
  68          tuned = dict(pool.map(tune_job, NAMED))
  69          (out / "tuning.json").write_text(json.dumps(tuned, indent=1))
  70          jobs = [(n, f, b, tuned[n], a.episodes) for n in NAMED for f, b in PAIRS]
  71          rows = pool.map(eval_job, jobs, chunksize=1)
  72      commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
  73      meta = dict(episodes=a.episodes, eval_seed_start=EVAL_SEED_START, commit=commit or None,
  74                  seconds=round(time.time() - t0, 1), scenarios={n: scenarios.get(n).to_dict() for n in NAMED})
  75      (out / "metrics.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))
  76      render(out, rows, tuned, a.episodes)
  77      print("done in", meta["seconds"], "s ->", out)
```
- Main: parse arguments; tune all scenarios in parallel and save `tuning.json`; evaluate every scenario × pair; record the git commit hash, settings and timing for reproducibility; save metrics and the table.


## 2.20  `experiments/x1_learned.py`

Trains IPPO by self-play in one scenario for several seeds.

**Lines 1–5**
```python
   1  """Train IPPO by self-play in one scenario for several seeds.
   2  
   3  Usage: python experiments/x1_learned.py --scenario SYM --seeds 1000 1001 1002 --iters 150
   4  Each run writes results/x1_learned/<scenario>/seed_<n>/{config.json,curve.json,checkpoint.pt,metrics.json}.
   5  """
```
- Docstring: usage and the output files (config, curve, checkpoint, metrics).

**Lines 6–7**
```python
   6  import argparse, pathlib, sys
   7  from multiprocessing import Pool
```
- Imports.

**Lines 9–10**
```python
   9  ROOT = pathlib.Path(__file__).resolve().parents[1]
  10  sys.path.insert(0, str(ROOT))
```
- Path setup.

**Lines 12–13**
```python
  12  from agrinegotiate import scenarios                       # noqa: E402
  13  from agrinegotiate.agents.ppo import PPOConfig, train     # noqa: E402
```
- Package imports.

**Lines 16–19**
```python
  16  def job(args):
  17      scn, seed, iters, eps, tag = args
  18      out = ROOT / "results" / tag / scn / f"seed_{seed}"
  19      return train(scenarios.get(scn), PPOConfig(iters=iters, episodes_per_iter=eps), seed, out)
```
- `job`: unpack arguments, build the output folder, train and return the summary.

**Lines 22–33**
```python
  22  if __name__ == "__main__":
  23      ap = argparse.ArgumentParser()
  24      ap.add_argument("--scenario", default="SYM")
  25      ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
  26      ap.add_argument("--iters", type=int, default=150)
  27      ap.add_argument("--episodes", type=int, default=256)
  28      ap.add_argument("--workers", type=int, default=2)
  29      ap.add_argument("--tag", default="x1_learned")
  30      a = ap.parse_args()
  31      with Pool(a.workers) as pool:
  32          for s in pool.imap_unordered(job, [(a.scenario, sd, a.iters, a.episodes, a.tag) for sd in a.seeds]):
  33              print("finished", s["scenario"], s["seed"], s["last10_checkpoint_mean"], flush=True)
```
- CLI: scenario, seeds, iterations, episodes per iteration, workers, tag; runs the seeds in parallel and prints the final metrics of each.


## 2.21  `experiments/crossplay_scripted.py`

Evaluates trained policies against scripted opponents (robustness / exploitability check, part of X6).

**Lines 1–6**
```python
   1  """Evaluate trained policies against scripted opponents (part of X6).
   2  
   3  Usage: python experiments/crossplay_scripted.py SYM
   4  For every seed folder in results/x1_learned/<scenario>, plays the learned farmer against each scripted
   5  buyer and the learned buyer against each scripted farmer on the first 300 evaluation episodes.
   6  """
```
- Docstring: for every seed folder, learned farmer vs each scripted buyer and learned buyer vs each scripted farmer, on 300 evaluation episodes.

**Lines 7–7**
```python
   7  import json, pathlib, sys
```
- Imports.

**Lines 9–10**
```python
   9  ROOT = pathlib.Path(__file__).resolve().parents[1]
  10  sys.path.insert(0, str(ROOT))
```
- Path setup.

**Lines 12–16**
```python
  12  from agrinegotiate import scenarios                         # noqa: E402
  13  from agrinegotiate.agents import scripted as S              # noqa: E402
  14  from agrinegotiate.agents.ppo import load_agents            # noqa: E402
  15  from agrinegotiate.env.core import F, B                     # noqa: E402
  16  from agrinegotiate.eval.evaluate import evaluate            # noqa: E402
```
- Package imports (scenarios, scripted agents, checkpoint loader, side constants, evaluator).

**Lines 18–18**
```python
  18  OPP = ["ZIC", "Boulware", "Linear", "Conceder", "TitForTat"]
```
- Opponent list.

**Lines 20–23**
```python
  20  if __name__ == "__main__":
  21      scn = sys.argv[1]
  22      sc = scenarios.get(scn)
  23      base = ROOT / "results" / "x1_learned" / scn
```
- Entry point: read the scenario name from the command line and find the results folder.

**Lines 24–30**
```python
  24      rows = []
  25      for d in sorted(base.glob("seed_*")):
  26          ag = load_agents(d / "checkpoint.pt", sc)
  27          for o in OPP + ["HardballBuyer"]:
  28              r = evaluate(sc, ag[F], S.make(o, B, sc), 300)
  29              rows.append(dict(seed=d.name, learned="farmer", opponent=o, E=r["E"], FGC=r["FGC"], BGC=r["BGC"],
  30                               agreement_rate=r["agreement_rate"]))
```
- For each seed: load the trained agents; learned farmer vs ZIC/Boulware/Linear/Conceder/TitForTat and the Hardball buyer, recording E, FGC, BGC, agreement.

**Lines 31–34**
```python
  31          for o in OPP:
  32              r = evaluate(sc, S.make(o, F, sc), ag[B], 300)
  33              rows.append(dict(seed=d.name, learned="buyer", opponent=o, E=r["E"], FGC=r["FGC"], BGC=r["BGC"],
  34                               agreement_rate=r["agreement_rate"]))
```
- Same for the learned buyer vs scripted farmers.

**Lines 35–35**
```python
  35      (base / "crossplay_scripted.json").write_text(json.dumps(rows, indent=1))
```
- Save all rows as JSON.

**Lines 36–44**
```python
  36      import collections
  37      agg = collections.defaultdict(list)
  38      for r in rows:
  39          agg[(r["learned"], r["opponent"])].append(r)
  40      print(f"{scn}: mean over {len(list(base.glob('seed_*')))} seeds, 300 episodes each")
  41      for (who, opp), rs in agg.items():
  42          own = "FGC" if who == "farmer" else "BGC"
  43          print(f"  learned {who:6s} vs {opp:13s} own gain capture {sum(r[own] for r in rs)/len(rs):.3f}  "
  44                f"E {sum(r['E'] for r in rs)/len(rs):.3f}  agreement {sum(r['agreement_rate'] for r in rs)/len(rs):.3f}")
```
- Aggregate by (role, opponent) and print own-gain capture, efficiency and agreement averaged over seeds.


## 2.22  `experiments/make_report.py`

Builds `results/summary.md` and `summary.json` from the run folders: every table as IQM + 95% bootstrap CI, and a *programmatic* verdict on each hypothesis (H1–H5). The pass/fail rule is written in code *before* looking at numbers, which is the point.

**Lines 1–5**
```python
   1  """Build results/summary.md and results/summary.json from the run folders: every table with
   2  interquartile means and 95% bootstrap intervals, and a programmatic verdict on each hypothesis.
   3  
   4      python experiments/make_report.py
   5  """
```
- Docstring: what it builds and how to run it.

**Lines 6–15**
```python
   6  import json, pathlib, sys
   7  
   8  ROOT = pathlib.Path(__file__).resolve().parents[1]
   9  sys.path.insert(0, str(ROOT))
  10  
  11  from scipy.stats import mannwhitneyu                 # noqa: E402
  12  from agrinegotiate.eval import stats                 # noqa: E402
  13  
  14  RES = ROOT / "results"
  15  MECH_COLS = ("E", "FGC", "S_F", "agreement_rate", "lam_fair", "fees_over_gstar")
```
- Imports (json, SciPy's Mann–Whitney test, the stats module), the `results` folder path, and the column set used by the mechanism tables.

**Lines 18–20**
```python
  18  def load(exp, label):
  19      f = RES / exp / label
  20      return stats.summarise(f) if list(f.glob("seed_*/metrics.json")) else None
```
- `load`: summarise one (experiment, label) folder, or None if no finished seeds exist.

**Lines 23–24**
```python
  23  def subdirs(exp):
  24      return sorted(p.name for p in (RES / exp).glob("*") if p.is_dir()) if (RES / exp).exists() else []
```
- `subdirs`: the setting folders of one experiment, sorted (empty if the experiment was not run).

**Lines 27–34**
```python
  27  def cell(s, k):
  28      if s is None or "iqm" not in s[k]:
  29          return "n/a"
  30      return f"{s[k]['iqm']:.3f} [{s[k]['ci95'][0]:.3f}, {s[k]['ci95'][1]:.3f}]"
  31  
  32  
  33  def point(s, k):
  34      return "n/a" if s is None or "iqm" not in s[k] else f"{s[k]['iqm']:.3f}"
```
- `cell`: formats 'IQM [lo, hi]' or 'n/a'. `point`: the IQM alone (used for the short-run columns of the 3M table).

**Lines 37–46**
```python
  37  def defined(s, k):
  38      return [v for v in s[k]["per_seed"] if v is not None]
  39  
  40  
  41  def per_seed(s, k):
  42      return [round(v, 2) for v in defined(s, k)]
  43  
  44  
  45  def span(d):
  46      return f"{d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]"
```
- `defined`: the per-seed values that are not None (S_F can be undefined). `per_seed`: the same, rounded for printing. `span`: formats a difference and its interval as 'd [lo, hi]'.

**Lines 49–57**
```python
  49  def table(title, rows, cols=("E", "FGC", "S_F", "agreement_rate")):
  50      names = {"E": "Efficiency", "FGC": "Farmer gain capture", "S_F": "Farmer share", "agreement_rate": "Agreement",
  51               "lam_fair": "Fee rate", "fees_over_gstar": "Fees / G*", "price_share": "Price share"}
  52      out = [f"### {title}", "", "| Setting | Seeds | Failed | " + " | ".join(names[c] for c in cols) + " |",
  53             "| --- | --- | --- | " + " | ".join("---" for _ in cols) + " |"]
  54      for name, s in rows:
  55          if s is not None:
  56              out.append(f"| {name} | {s['n_seeds']} | {s['failed_runs']} | " + " | ".join(cell(s, c) for c in cols) + " |")
  57      return out + [""]
```
- `table`: renders a markdown table with seeds, failed-run count and chosen metric columns (skipping missing settings).

**Lines 60–69**
```python
  60  # Hypothesis checks. Each one writes its verdicts into V. The same checks run on the short runs and
  61  # on the 3M-step rerun: `sfx` is "_3m" for the rerun and `pre` starts the evidence text.
  62  
  63  def check_h1(V, sym, sfx="", pre=""):
  64      """H1: in SYM learners are efficient (E >= 0.85) and split the gains fairly (farmer share in [0.40, 0.60])."""
  65      lo, hi = sym["S_F"]["ci95"]
  66      V["H1_efficiency" + sfx] = dict(passed=bool(sym["E"]["ci95"][0] >= 0.85),
  67                                      detail=f"{pre}SYM E {cell(sym, 'E')}; needs the whole interval at or above 0.85")
  68      V["H1_fair_split" + sfx] = dict(passed=bool(lo >= 0.40 and hi <= 0.60),
  69                                      detail=f"{pre}SYM farmer share {cell(sym, 'S_F')}; per seed {per_seed(sym, 'S_F')}")
```
- The hypothesis checks are functions so the short runs and the 3M rerun use exactly the same rule: `sfx` is `_3m` for the rerun and `pre` starts the evidence text.
- **H1** (SYM): *efficiency* passes only if the lower CI ≥ 0.85; *fair split* passes only if the whole CI of farmer share lies in [0.40, 0.60].

**Lines 72–81**
```python
  72  def check_h2(V, k, g, sym, sfx="", pre=""):
  73      """H2: the farmer's share in scenario k is lower than in SYM."""
  74      if g is None or not defined(g, "S_F"):
  75          return
  76      a, b = defined(g, "S_F"), defined(sym, "S_F")
  77      d = stats.iqm_diff_ci(a, b)
  78      p = mannwhitneyu(a, b, alternative="less").pvalue
  79      V[f"H2_{k}{sfx}"] = dict(passed=bool(d[2] < 0), seeds=len(a),
  80                               detail=f"{pre}farmer share {k} minus SYM = {span(d)}; Mann-Whitney one-sided p = {p:.4f}; "
  81                                      f"{g['S_F']['undefined']} of {g['n_seeds']} {k} runs undefined")
```
- **H2**: farmer share in a harder scenario minus SYM's, with a bootstrap CI (`stats.iqm_diff_ci`); passes if the upper bound < 0 (farmers do worse). Also a one-sided Mann–Whitney p-value and the number of undefined runs. Skipped if the scenario has no defined shares.

**Lines 84–91**
```python
  84  def check_fee(V, fee, base, name="", sfx="", pre=""):
  85      """H3: the fee raises farmer gain capture by at least 0.10 and costs at most 0.05 efficiency."""
  86      dF = stats.iqm_diff_ci(fee["FGC"]["per_seed"], base["FGC"]["per_seed"])
  87      dE = stats.iqm_diff_ci(fee["E"]["per_seed"], base["E"]["per_seed"])
  88      V[f"H3_{name}gain{sfx}"] = dict(
  89          passed=bool(dF[1] >= 0.10), detail=f"{pre}FGC with fee minus without = {span(dF)}; needs the whole interval at or above 0.10")
  90      V[f"H3_{name}efficiency_cost{sfx}"] = dict(
  91          passed=bool(dE[1] >= -0.05), detail=f"{pre}E with fee minus without = {span(dE)}; needs the whole interval at or above -0.05")
```
- **H3**, gain and cost: the fee must raise farmer gain capture by ≥ 0.10 (whole CI) and cost no more than 0.05 efficiency (whole CI). `name` is `real_` for the real-price scenario.

**Lines 94–99**
```python
  94  def check_floors(V, fee, floors, sfx="", pre=""):
  95      """H3: no price floor matches the fee on both efficiency and farmer gain (point estimates)."""
  96      dom = [m for m, s in floors if s and s["FGC"]["iqm"] >= fee["FGC"]["iqm"] and s["E"]["iqm"] >= fee["E"]["iqm"]]
  97      V["H3_beats_floor" + sfx] = dict(passed=bool(not dom), detail=pre + (
  98          "no price floor setting matches the fee on both E and FGC" if not dom else
  99          f"price floor settings that match or beat the fee on both E and FGC (point estimates): {dom}") + "; floors have 3 seeds each")
```
- **H3**, floors: passes if no price floor setting matches or beats the fee on both E and FGC (point estimates). Notes floors have only 3 seeds.

**Lines 102–105**
```python
 102  def check_h5(V, key, aware, unaware, seeds, what):
 103      """H5: delay-aware learners are more efficient than delay-unaware ones."""
 104      d = stats.iqm_diff_ci(aware["E"]["per_seed"], unaware["E"]["per_seed"])
 105      V[key] = dict(passed=bool(d[1] > 0), seeds=seeds, detail=f"{what} = {span(d)}")
```
- **H5**: delay-aware minus delay-unaware efficiency must have a CI entirely above 0.

**Lines 108–112**
```python
 108  if __name__ == "__main__":
 109      L, V = ["# Results summary", "",
 110              "Interquartile mean over training seeds with a 95% bootstrap interval. Each seed's score is the mean of",
 111              "its last 10 evaluation checkpoints (300 fixed evaluation episodes each). A failed run ends with E below 0.1.",
 112              "Results with fewer than 10 seeds are exploratory.", ""], {}
```
- Main: starts the report text list `L` and the verdict dict `V`; states the aggregation rule and that <10 seeds is exploratory.

**Lines 114–124**
```python
 114      # gate
 115      gate = {k: load("x2_gate", k) for k in ("SYM", "PERISH", "DEADLINE", "INFO", "THIN3")}
 116      L += table("Unconstrained learners by scenario (X2)", gate.items())
 117      sym = gate["SYM"]
 118      if sym:
 119          check_h1(V, sym)
 120          for k in ("PERISH", "DEADLINE", "INFO", "THIN3"):
 121              check_h2(V, k, gate[k], sym)
 122      diag = [(k, load("x5_diagnosis", k)) for k in ("SYM_mix0", "SYMX")]
 123      L += table("SYM diagnosis (X5, exploratory)", diag)
 124      L += [f"- {k} farmer share per seed: {per_seed(s, 'S_F')}" for k, s in diag if s] + [""]
```
- X2 gate: unconstrained learners across five scenarios, then H1 on SYM and H2 for each harder scenario.
- SYM diagnosis table and per-seed shares (to inspect collapsed or lopsided runs).

**Lines 126–136**
```python
 126      # mechanism
 127      base = gate["PERISH"]
 128      mech = subdirs("x3_mechanism")
 129      L += table("Mechanisms in PERISH (X3)", [("Unconstrained", base)] + [(m, load("x3_mechanism", m)) for m in mech], MECH_COLS)
 130      fee = load("x3_mechanism", "fee_oracle_tau0.4")
 131      if fee and base:
 132          check_fee(V, fee, base)
 133          check_floors(V, fee, [(m, load("x3_mechanism", m)) for m in mech if m.startswith("floor")])
 134  
 135      L += table("Mechanisms when buyer values vary widely, PERISH-HET (X6, post hoc, exploratory)",
 136                 [(m, load("x6_hetero", m)) for m in subdirs("x6_hetero")], MECH_COLS)
```
- Mechanism table in PERISH (unconstrained baseline + every mechanism folder), then H3 with the adaptive fee (τ = 0.4) against no rule and against the floors.
- PERISH-HET table (post hoc, exploratory).

**Lines 138–145**
```python
 138      # scenarios calibrated on real mandi prices
 139      real = {k: load("x7_real", k) for k in ("unconstrained", "fee_oracle_tau0.4_eta0.25", "floor_kappa0.55", "unaware_duty0.6", "aware_duty0.6")}
 140      L += table("Scenarios calibrated on real Agmarknet lemon prices (X7)", real.items(), ("E", "FGC", "S_F", "agreement_rate", "lam_fair"))
 141      if real["unconstrained"] and real["fee_oracle_tau0.4_eta0.25"]:
 142          check_fee(V, real["fee_oracle_tau0.4_eta0.25"], real["unconstrained"], "real_", pre="real-data PERISH-REAL: ")
 143      if real["aware_duty0.6"] and real["unaware_duty0.6"]:
 144          check_h5(V, "H5_real_delay_aware", real["aware_duty0.6"], real["unaware_duty0.6"], 5,
 145                   "real-data RURAL1-REAL-DTN: E delay-aware minus delay-unaware")
```
- Load the five `x7_real` result sets and print them as table X7 (E, FGC, S_F, agreement rate, final λ). H3 on real prices (`H3_real_gain`, `H3_real_efficiency_cost`) and H5 on the real-price delay scenario (`H5_real_delay_aware`, 5 seeds).

**Lines 147–157**
```python
 147      # delay
 148      dl = {k: load("x4_delay", k) for k in ("unaware_duty1.0", "unaware_duty0.6", "unaware_duty0.4", "aware_duty1.0", "aware_duty0.6",
 149                                            "aware_duty0.4", "memory_only_duty0.6", "features_only_duty0.6")}
 150      L += table("Delay in RURAL1-DTN (X4, 3 seeds per cell, exploratory)", dl.items())
 151      u = [dl[f"unaware_duty{d}"] for d in ("1.0", "0.6", "0.4")]
 152      if all(u):
 153          es, ags = [s["E"]["iqm"] for s in u], [s["agreement_rate"]["iqm"] for s in u]
 154          V["H4_delay_cost"] = dict(passed=bool(es[0] > es[1] > es[2] and ags[0] > ags[1] > ags[2]), seeds=3,
 155                                    detail=f"delay-unaware E at link uptime 100/60/40%: {[round(v, 3) for v in es]}; agreement {[round(v, 3) for v in ags]}")
 156      if dl["aware_duty0.6"] and dl["unaware_duty0.6"]:
 157          check_h5(V, "H5_delay_aware", dl["aware_duty0.6"], dl["unaware_duty0.6"], 3, "E delay-aware minus delay-unaware at 60% uptime")
```
- Delay table. **H4**: delay-unaware efficiency and agreement must fall strictly as link uptime goes 100% → 60% → 40%. **H5**: delay-aware vs delay-unaware at 60% uptime (3 seeds).

**Lines 159–166**
```python
 159      # budget check: the main comparisons rerun at 3 million agent-steps per run (X8)
 160      pairs = [("gate", "x2_gate", k) for k in ("SYM", "PERISH", "THIN3")] \
 161          + [("mechanism", "x3_mechanism", k) for k in ("fee_oracle_tau0.4", "fee_fixed_lam1.0_tau0.4", "floor_kappa0.50",
 162                                                        "floor_kappa0.55", "floor_kappa0.60", "floor_kappa0.65")] \
 163          + [("hetero", "x6_hetero", k) for k in ("unconstrained", "fee_oracle_tau0.4_eta0.25", "floor_kappa0.50",
 164                                                  "floor_kappa0.55", "floor_kappa0.60")] \
 165          + [("real", "x7_real", k) for k in ("unconstrained", "fee_oracle_tau0.4_eta0.25", "floor_kappa0.55")]
 166      long = {(g, k): load(f"x8_3m_{g}", k) for g, _, k in pairs}
```
- Budget check (X8): the (experiment, setting) pairs that were rerun at 3 million agent-steps, and their 3M summaries from the `x8_3m_*` folders.

**Lines 167–179**
```python
 167      if any(long.values()):
 168          steps = [json.loads(f.read_text())["agent_steps"] for f in RES.glob("x8_3m_*/*/seed_*/metrics.json")]
 169          L += ["### Budget check: main comparisons at 3 million agent-steps per run (X8)", "",
 170                f"Same scenarios, settings and seeds as above; {len(steps)} runs, "
 171                f"{min(steps) / 1e6:.2f} to {max(steps) / 1e6:.2f} million agent-steps each. Short = the runs above (0.3 to 0.7 million).", "",
 172                "| Experiment | Setting | Seeds | E short | E 3M | Farmer gain short | Farmer gain 3M | Farmer share short | Farmer share 3M | Agreement 3M | Failed 3M |",
 173                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
 174          for g, exp, k in pairs:
 175              s, l = load(exp, k), long[(g, k)]
 176              if l is not None:
 177                  L.append(f"| {g} | {k} | {l['n_seeds']} | {point(s, 'E')} | {cell(l, 'E')} | {point(s, 'FGC')} | {cell(l, 'FGC')} | "
 178                           f"{point(s, 'S_F')} | {cell(l, 'S_F')} | {cell(l, 'agreement_rate')} | {l['failed_runs']} |")
 179          L += [""]
```
- If any 3M runs exist: count the runs and their agent-steps, then one table row per setting with the short-run IQM next to the 3M IQM and interval.

**Lines 180–193**
```python
 180          sym3, per3 = long[("gate", "SYM")], long[("gate", "PERISH")]
 181          if sym3:
 182              L += [f"- SYM farmer share per seed at 3M: {per_seed(sym3, 'S_F')}", ""]
 183              check_h1(V, sym3, "_3m", "3M budget: ")
 184              for k in ("PERISH", "THIN3"):
 185                  check_h2(V, k, long[("gate", k)], sym3, "_3m", "3M budget: ")
 186          fee3 = long[("mechanism", "fee_oracle_tau0.4")]
 187          if fee3 and per3:
 188              check_fee(V, fee3, per3, "", "_3m", "3M budget: ")
 189              check_floors(V, fee3, [(k, s) for (g, k), s in long.items() if g == "mechanism" and k.startswith("floor")],
 190                           "_3m", "3M budget: ")
 191          u3, f3 = long[("real", "unconstrained")], long[("real", "fee_oracle_tau0.4_eta0.25")]
 192          if u3 and f3:
 193              check_fee(V, f3, u3, "real_", "_3m", "3M budget, PERISH-REAL: ")
```
- The same checks again on the 3M runs, with `_3m` added to each name: H1 on SYM, H2 for PERISH and THIN3, H3 in PERISH (fee vs no rule, fee vs floors) and H3 on real prices.

**Lines 195–204**
```python
 195      L += ["## Hypothesis verdicts", "", "Rows ending in _3m repeat a test at the 3 million agent-step budget (X8).", "",
 196            "| Test | Result | Evidence |", "| --- | --- | --- |"]
 197      for k, v in V.items():
 198          tag = "PASS" if v["passed"] else "FAIL"
 199          if v.get("seeds", 10) < 10:
 200              tag += f" (exploratory, {v['seeds']} seeds)"
 201          L.append(f"| {k} | {tag} | {v['detail']} |")
 202      (RES / "summary.md").write_text("\n".join(L) + "\n")
 203      (RES / "summary.json").write_text(json.dumps(V, indent=1))
 204      print("\n".join(L))
```
- Verdict table: PASS/FAIL, tagged 'exploratory' if fewer than 10 seeds; writes `summary.md` and `summary.json` and prints.


## 2.23  `experiments/make_figures.py`

Generates the paper's PNG figures with matplotlib from the results folders.

**Lines 1–4**
```python
   1  """Paper figures from the results folders. Writes PNG files to paper/figures/.
   2  
   3      python experiments/make_figures.py
   4  """
```
- Docstring.

**Lines 5–16**
```python
   5  import json, pathlib, sys
   6  
   7  ROOT = pathlib.Path(__file__).resolve().parents[1]
   8  sys.path.insert(0, str(ROOT))
   9  
  10  import matplotlib                                    # noqa: E402
  11  matplotlib.use("Agg")
  12  import matplotlib.pyplot as plt                      # noqa: E402
  13  import numpy as np                                   # noqa: E402
  14  from agrinegotiate.eval import stats                 # noqa: E402
  15  
  16  RES, OUT = ROOT / "results", ROOT / "paper" / "figures"
```
- Imports. `matplotlib.use('Agg')` selects a non-interactive backend (works on a server). Output folder `paper/figures`.

**Lines 17–22**
```python
  17  INK, INK2, RULE, SURF = "#14201b", "#4b5650", "#d9ded9", "#fcfcfb"
  18  S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]          # fixed categorical order
  19  plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
  20                       "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
  21                       "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": RULE,
  22                       "grid.linewidth": 0.8, "savefig.dpi": 200, "savefig.bbox": "tight"})
```
- Palette (ink, grey, rule, surface colours; a fixed categorical order of five hues) and global matplotlib style: top/right spines off, light grid, 200 dpi, tight bounding box.

**Lines 25–27**
```python
  25  def summ(exp, label):
  26      f = RES / exp / label
  27      return stats.summarise(f) if list(f.glob("seed_*/metrics.json")) else None
```
- `summ`: summarise a folder or None.

**Lines 30–50**
```python
  30  def fig_shares():
  31      rows = [("x2_gate", "SYM", "Equal conditions"), ("x2_gate", "PERISH", "Perishable crop"),
  32              ("x2_gate", "DEADLINE", "Earlier farmer deadline"), ("x2_gate", "INFO", "Noisy price signal"),
  33              ("x2_gate", "THIN3", "Three farmers, one buyer")]
  34      rows = [(n, summ(e, l)) for e, l, n in rows]
  35      rows = [(n, s) for n, s in rows if s]
  36      fig, ax = plt.subplots(figsize=(7, 0.62 * len(rows) + 1.1))
  37      for y, (name, s) in enumerate(rows[::-1]):
  38          vals = [v for v in s["S_F"]["per_seed"] if v is not None]
  39          jit = (np.arange(len(vals)) % 3 - 1) * 0.12
  40          ax.scatter(vals, y + jit, s=34, color=S[0], edgecolor=SURF, linewidth=1.2, zorder=3)
  41          if "iqm" in s["S_F"]:
  42              ax.plot(s["S_F"]["ci95"], [y, y], color=INK, lw=2, zorder=2)
  43              ax.plot([s["S_F"]["iqm"]] * 2, [y - 0.28, y + 0.28], color=INK, lw=3, zorder=4)
  44      ax.axvline(0.5, color=INK2, ls="--", lw=1)
  45      ax.text(0.5, len(rows) - 0.35, "equal split (Nash)", ha="center", color=INK, fontsize=9)
  46      ax.set_yticks(range(len(rows)), [f"{n}\n({s['n_seeds']} runs)" for n, s in rows[::-1]], color=INK)
  47      ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(rows) - 0.1)
  48      ax.set_xlabel("Farmer's share of the realised gains (one dot per training run; bar = IQM and 95% interval)")
  49      ax.grid(axis="y", visible=False)
  50      fig.savefig(OUT / "fig3_farmer_share_by_scenario.png"); plt.close(fig)
```
- `fig_shares`: dot plot of farmer share per scenario. One dot per training run (jittered), a bar for the 95% CI and a thick tick at the IQM, a dashed line at 0.5 ('equal split (Nash)').

**Lines 53–83**
```python
  53  def fig_frontier():
  54      fams = [("Unconstrained", [("x2_gate", "PERISH", "")]),
  55              ("Adaptive fee, true values", [("x3_mechanism", f"fee_oracle_tau{t}", f"target {t}") for t in (0.2, 0.3, 0.4, 0.5)]
  56               + [("x3_mechanism", "fee_oracle_tau0.4_eta0.25", "target 0.4, fast update")]),
  57              ("Adaptive fee, public prices only", [("x3_mechanism", "fee_observable_tau0.4", "target 0.4")]),
  58              ("Hard price floor", [("x3_mechanism", f"floor_kappa{k:.2f}", f"floor {k:.2f}") for k in (0.50, 0.55, 0.60, 0.65)]),
  59              ("Fixed fee", [("x3_mechanism", "fee_fixed_lam1.0_tau0.4", "rate 1.0")])]
  60      fig, ax = plt.subplots(figsize=(7, 4.6))
  61      drawn = False
  62      for (name, items), col in zip(fams, S):
  63          pts = [(summ(e, l), lab) for e, l, lab in items]
  64          pts = [(s, lab) for s, lab in pts if s and "iqm" in s["E"]]
  65          if not pts:
  66              continue
  67          drawn = True
  68          xs, ys = [s["FGC"]["iqm"] for s, _ in pts], [s["E"]["iqm"] for s, _ in pts]
  69          order = np.argsort(xs)
  70          ax.plot(np.array(xs)[order], np.array(ys)[order], color=col, lw=2, alpha=0.5, zorder=2)
  71          for s, lab in pts:
  72              ax.errorbar(s["FGC"]["iqm"], s["E"]["iqm"],
  73                          xerr=[[s["FGC"]["iqm"] - s["FGC"]["ci95"][0]], [s["FGC"]["ci95"][1] - s["FGC"]["iqm"]]],
  74                          yerr=[[s["E"]["iqm"] - s["E"]["ci95"][0]], [s["E"]["ci95"][1] - s["E"]["iqm"]]],
  75                          color=col, lw=0.8, alpha=0.7, zorder=2)
  76              ax.annotate(lab, (s["FGC"]["iqm"], s["E"]["iqm"]), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK)
  77          ax.scatter(xs, ys, s=55, color=col, edgecolor=SURF, linewidth=1.5, zorder=3, label=name)
  78      if not drawn:
  79          plt.close(fig); return
  80      ax.set_xlabel("Farmer gain capture (share of possible gains reaching farmers)")
  81      ax.set_ylabel("Efficiency (share of possible gains realised)")
  82      ax.legend(frameon=False, fontsize=8.5, loc="lower left")
  83      fig.savefig(OUT / "fig4_efficiency_fairness_frontier.png"); plt.close(fig)
```
- `fig_frontier`: efficiency vs farmer-gain-capture frontier. Families of settings (unconstrained, fee variants, floors, fixed fee) as coloured points with CI error bars and labels, connected in order of FGC.

**Lines 86–106**
```python
  86  def fig_delay():
  87      duties = ["1.0", "0.6", "0.4"]
  88      cells = [("unaware", "Delay-unaware", S[0]), ("aware", "Delay-aware", S[1])]
  89      if not summ("x4_delay", "unaware_duty0.6"):
  90          return
  91      fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
  92      for ax, (k, title) in zip(axes, [("E", "Efficiency"), ("agreement_rate", "Agreement rate"), ("FGC", "Farmer gain capture")]):
  93          for key, name, col in cells:
  94              ss = [summ("x4_delay", f"{key}_duty{d}") for d in duties]
  95              xs = [i for i, s in enumerate(ss) if s and "iqm" in s[k]]
  96              if not xs:
  97                  continue
  98              ys = [ss[i][k]["iqm"] for i in xs]
  99              ax.plot(xs, ys, color=col, lw=2, marker="o", ms=7, mec=SURF, mew=1.5, label=name)
 100              for i in xs:
 101                  ax.plot([i, i], ss[i][k]["ci95"], color=col, lw=0.9, alpha=0.7)
 102              ax.annotate(name, (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, 0), fontsize=8.5, color=INK, va="center")
 103          ax.set_xticks(range(3), ["100%", "60%", "40%"]); ax.set_xlim(-0.2, 2.9)
 104          ax.set_title(title, loc="left", color=INK, fontsize=10); ax.set_xlabel("Farmer link uptime")
 105      axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
 106      fig.savefig(OUT / "fig5_delay.png"); plt.close(fig)
```
- `fig_delay`: three small panels (efficiency, agreement, FGC) vs link uptime for delay-unaware vs delay-aware agents, with CI whiskers.

**Lines 109–128**
```python
 109  def fig_learning():
 110      items = [("x2_gate", "SYM", "Equal conditions", S[0]), ("x2_gate", "PERISH", "Perishable crop", S[1])]
 111      fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
 112      for e, l, name, col in items:
 113          curves = [json.loads(f.read_text()) for f in sorted((RES / e / l).glob("seed_*/curve.json")) if (f.parent / "metrics.json").exists()]
 114          if not curves:
 115              continue
 116          n = min(len(c) for c in curves)
 117          it = [curves[0][k]["iter"] for k in range(n)]
 118          for ax, key in zip(axes, ("E", "S_F")):
 119              for c in curves:
 120                  ax.plot(it, [c[k][key] if c[k][key] is not None else np.nan for k in range(n)], color=col, lw=0.8, alpha=0.35)
 121              mean = [np.nanmean([c[k][key] if c[k][key] is not None else np.nan for c in curves]) for k in range(n)]
 122              ax.plot(it, mean, color=col, lw=2, label=name)
 123      axes[0].set_title("Efficiency", loc="left", color=INK, fontsize=10); axes[1].set_title("Farmer share", loc="left", color=INK, fontsize=10)
 124      axes[1].axhline(0.5, color=INK2, ls="--", lw=1); axes[1].set_ylim(0, 1)
 125      for ax in axes:
 126          ax.set_xlabel("Training iteration (thin lines: single runs; thick: mean)")
 127      axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
 128      fig.savefig(OUT / "fig2_learning_curves.png"); plt.close(fig)
```
- `fig_learning`: learning curves (efficiency and farmer share vs iteration): thin lines per run, thick line the mean across runs (NaN-safe), dashed 0.5 line.

**Lines 131–135**
```python
 131  if __name__ == "__main__":
 132      OUT.mkdir(parents=True, exist_ok=True)
 133      for f in (fig_learning, fig_shares, fig_frontier, fig_delay):
 134          f()
 135      print(sorted(p.name for p in OUT.glob("*.png")))
```
- Main: create the folder, draw all four figures, list the files.


## 2.24  `dashboard/export_data.py`

Collects everything the front end shows into `data.json`/`data.js`. It only *reads* results; nothing is trained here.

**Lines 1–8**
```python
   1  """Collect everything the front end shows into dashboard/web/data.json and data.js.
   2  
   3      python dashboard/export_data.py
   4  
   5  Reads results/ only: per-seed summaries are aggregated with the interquartile mean and a 95%
   6  bootstrap interval, learning curves are averaged over seeds, and a few episodes are replayed
   7  from saved checkpoints with fixed evaluation seeds. Nothing is trained here.
   8  """
```
- Docstring.

**Lines 9–15**
```python
   9  import datetime, json, pathlib, sys
  10  
  11  ROOT = pathlib.Path(__file__).resolve().parents[1]
  12  sys.path.insert(0, str(ROOT))
  13  
  14  import numpy as np                                          # noqa: E402
  15  from agrinegotiate.eval import replay, stats               # noqa: E402
```
- Imports and path setup.

**Lines 17–19**
```python
  17  RES = ROOT / "results"
  18  SHOW = ("E", "FGC", "BGC", "S_F", "agreement_rate", "price_share", "fees_over_gstar", "lam_fair", "time_to_agreement")
  19  CURVE = ("E", "FGC", "S_F", "agreement_rate", "lam_fair")
```
- `SHOW` metrics exported per group; `CURVE` metrics averaged for learning curves.

**Lines 22–40**
```python
  22  def group(folder: pathlib.Path) -> dict | None:
  23      runs = stats.load_runs(folder)
  24      if not runs:
  25          return None
  26      s = stats.summarise(folder)
  27      curves = [json.loads((d / "curve.json").read_text()) for d in sorted(folder.glob("seed_*")) if (d / "metrics.json").exists()]
  28      n = min(len(c) for c in curves)
  29      curve = []
  30      for k in range(n):
  31          row = {"iter": curves[0][k]["iter"]}
  32          for key in CURVE:
  33              vals = [c[k].get(key) for c in curves if c[k].get(key) is not None]
  34              row[key] = float(np.mean(vals)) if vals else None
  35          curve.append(row)
  36      cfg = json.loads((sorted(folder.glob("seed_*"))[0] / "config.json").read_text())
  37      return {"exp": folder.parent.name, "label": folder.name, "scenario": cfg["scenario"]["name"],
  38              "duty": cfg["scenario"]["duty"], "mechanism": cfg.get("mechanism"), "n_seeds": s["n_seeds"],
  39              "failed_runs": s["failed_runs"], "agent_steps": runs[0]["agent_steps"],
  40              "metrics": {k: {kk: vv for kk, vv in s[k].items()} for k in SHOW if k in s}, "curve": curve}
```
- `group(folder)`: summarise a result folder; load every finished seed's curve; truncate to the shortest; average each metric across seeds per iteration (ignoring None); read the first seed's config for scenario/duty/mechanism; return one dict with metrics and the curve.

**Lines 43–45**
```python
  43  def first_seed(exp, label):
  44      d = sorted((RES / exp / label).glob("seed_*/checkpoint.pt"))
  45      return f"run:{exp}/{label}/{d[0].parent.name}" if d else None
```
- `first_seed`: path-like id (`run:exp/label/seed_n`) of a trained checkpoint for the replays.

**Lines 48–76**
```python
  48  def replays():
  49      out = []
  50  
  51      def add(title, note, scenario, farmer, buyer, seed, mechanism=None, overrides=None):
  52          if farmer is None or buyer is None:
  53              return
  54          try:
  55              ep = replay.record(scenario, farmer, buyer, seed, mechanism, overrides)
  56          except FileNotFoundError:
  57              return
  58          out.append({"title": title, "note": note, "episode": ep})
  59  
  60      sym, per = first_seed("x2_gate", "SYM"), first_seed("x2_gate", "PERISH")
  61      fee = first_seed("x3_mechanism", "fee_oracle_tau0.4")
  62      aware, unaware = first_seed("x4_delay", "aware_duty0.6"), first_seed("x4_delay", "unaware_duty0.6")
  63      for seed in (50001, 50007):
  64          add("Equal conditions, scripted Linear agents", "Both sides concede steadily.", "SYM", "Linear", "Linear", seed)
  65          add("Equal conditions, learned agents", "Self-interested PPO agents after self-play.", "SYM", sym, sym, seed)
  66          add("Perishable crop, hardball buyer against a Linear farmer", "The buyer never moves from just above the farmer's fallback.",
  67              "PERISH", "Linear", "HardballBuyer", seed)
  68          add("Perishable crop, learned agents, no mechanism", "What unconstrained learners do when the crop decays.",
  69              "PERISH", per, per, seed)
  70          add("Perishable crop, learned agents with the adaptive fee", "Buyers trained under the fee (target share 0.4).",
  71              "PERISH", fee, fee, seed, {"kind": "fee", "tau": 0.4})
  72          add("Rural, intermittent link, delay-aware learned agents", "Farmer link is up 60% of the time.",
  73              "RURAL1-DTN", aware, aware, seed)
  74          add("Rural, intermittent link, delay-unaware learned agents", "Same link, agents cannot see message age or link state.",
  75              "RURAL1-DTN", unaware, unaware, seed)
  76      return out
```
- `replays()`: inner `add` records an episode through `replay.record` (skipping if a checkpoint is missing). Then records fixed-seed episodes (50001, 50007) for: scripted Linear vs Linear, learned SYM agents, hardball buyer vs Linear farmer, unconstrained PERISH, PERISH with the fee, and delay-aware vs unaware agents on the intermittent link.

**Lines 79–100**
```python
  79  if __name__ == "__main__":
  80      groups = {}
  81      for folder in sorted(RES.glob("x[2-9]_*/*")):
  82          if folder.is_dir():
  83              g = group(folder)
  84              if g:
  85                  groups[f"{g['exp']}/{g['label']}"] = g
  86      scripted = json.loads((RES / "x1_scripted" / "metrics.json").read_text())
  87      keep = ("scenario", "farmer", "buyer", "E", "FGC", "BGC", "S_F", "agreement_rate", "price_share", "phi_mean",
  88              "time_to_agreement")
  89      data = {
  90          "generated": datetime.date.today().isoformat(),
  91          "groups": groups,
  92          "scripted": [{k: r.get(k) for k in keep} for r in scripted["rows"]],
  93          "scripted_episodes": scripted["meta"]["episodes"],
  94          "replays": replays(),
  95      }
  96      web = ROOT / "dashboard" / "web"
  97      txt = json.dumps(data, separators=(",", ":"), default=float)
  98      (web / "data.json").write_text(txt)
  99      (web / "data.js").write_text("window.AGRI_DATA=" + txt + ";")
 100      print("groups:", len(groups), "replays:", len(data["replays"]), "bytes:", len(txt))
```
- Main: build all groups from `results/x2..x9_*`, load the scripted baselines, assemble the data dict with date and replays, write compact JSON and a `window.AGRI_DATA=...` JS version (so the page works when opened from disk without a server).


## 2.25  `dashboard/api/main.py`

FastAPI backend: serves the front end and runs episodes on request (loads checkpoints; never trains).

**Lines 1–12**
```python
   1  """Dashboard backend. Serves the static front end and runs episodes on request.
   2  
   3      uvicorn dashboard.api.main:app --reload        # then open http://127.0.0.1:8000
   4  
   5  Endpoints
   6      GET  /api/scenarios                    named scenarios and their parameters
   7      GET  /api/methods                      scripted agents and trained checkpoints found under results/
   8      GET  /api/results                      the aggregated results the front end charts (web/data.json)
   9      POST /api/episodes                     run one episode, return the full record
  10      WS   /api/episodes/stream              same request as JSON; events are sent one step at a time
  11  It loads checkpoints on demand and never trains anything.
  12  """
```
- Docstring listing the endpoints: scenarios, methods, results, POST episodes, and a WebSocket that streams events.

**Lines 13–23**
```python
  13  import asyncio
  14  import json
  15  import pathlib
  16  
  17  from fastapi import FastAPI, HTTPException, WebSocket
  18  from fastapi.responses import FileResponse, JSONResponse
  19  from fastapi.staticfiles import StaticFiles
  20  from pydantic import BaseModel
  21  
  22  from agrinegotiate import scenarios
  23  from agrinegotiate.eval import replay
```
- Imports: asyncio, FastAPI, responses, static files, pydantic.

**Lines 25–27**
```python
  25  ROOT = pathlib.Path(__file__).resolve().parents[2]
  26  WEB = ROOT / "dashboard" / "web"
  27  app = FastAPI(title="AgriNegotiate dashboard")
```
- Paths and the app object.

**Lines 30–36**
```python
  30  class EpisodeRequest(BaseModel):
  31      scenario: str = "PERISH"
  32      farmer: str = "Linear"
  33      buyer: str = "HardballBuyer"
  34      seed: int = 50000
  35      mechanism: dict | None = None
  36      overrides: dict | None = None          # e.g. {"duty": 0.6, "rho": 0.97}
```
- `EpisodeRequest` (pydantic model validates input): scenario, farmer/buyer method names, seed, optional mechanism and parameter overrides.

**Lines 39–41**
```python
  39  def _bilateral(name: str) -> bool:
  40      sc = scenarios.get(name)
  41      return sc.n_farmers == 1 and sc.n_buyers == 1
```
- `_bilateral`: true for 1×1 scenarios (the UI replays those).

**Lines 44–46**
```python
  44  @app.get("/api/scenarios")
  45  def get_scenarios():
  46      return {k: dict(v.to_dict(), bilateral=_bilateral(k)) for k, v in scenarios.SCENARIOS.items()}
```
- `GET /api/scenarios`: every scenario's parameters plus the bilateral flag.

**Lines 49–53**
```python
  49  @app.get("/api/methods")
  50  def get_methods():
  51      runs = sorted(str(p.parent.relative_to(ROOT / "results")) for p in (ROOT / "results").glob("*/*/seed_*/checkpoint.pt"))
  52      return {"scripted_farmer": replay.SCRIPTED, "scripted_buyer": replay.SCRIPTED + ["HardballBuyer"],
  53              "trained": ["run:" + r for r in runs]}
```
- `GET /api/methods`: scripted names and every trained checkpoint found under `results/`.

**Lines 56–61**
```python
  56  @app.get("/api/results")
  57  def get_results():
  58      f = WEB / "data.json"
  59      if not f.exists():
  60          raise HTTPException(404, "run `python dashboard/export_data.py` first")
  61      return JSONResponse(json.loads(f.read_text()))
```
- `GET /api/results`: serves `data.json`, or a 404 telling the user to run the export.

**Lines 64–70**
```python
  64  def _run(req: EpisodeRequest) -> dict:
  65      if req.scenario not in scenarios.SCENARIOS:
  66          raise HTTPException(404, f"unknown scenario {req.scenario}")
  67      try:
  68          return replay.record(req.scenario, req.farmer, req.buyer, req.seed, req.mechanism, req.overrides)
  69      except (KeyError, FileNotFoundError, TypeError) as e:
  70          raise HTTPException(400, f"cannot run that episode: {e}")
```
- `_run`: validate the scenario, run `replay.record`, convert expected errors to HTTP 400/404.

**Lines 73–75**
```python
  73  @app.post("/api/episodes")
  74  def post_episode(req: EpisodeRequest):
  75      return _run(req)
```
- `POST /api/episodes`: returns the full episode record.

**Lines 78–96**
```python
  78  @app.websocket("/api/episodes/stream")
  79  async def stream(ws: WebSocket):
  80      await ws.accept()
  81      try:
  82          req = EpisodeRequest(**json.loads(await ws.receive_text()))
  83          ep = _run(req)
  84      except Exception as e:                                   # report the problem to the client and stop
  85          await ws.send_json({"type": "error", "detail": str(e)})
  86          await ws.close()
  87          return
  88      events = ep.pop("events")
  89      await ws.send_json({"type": "meta", "episode": ep})
  90      for t in range(ep["T"] + 1):
  91          step = [e for e in events if e["t"] == t]
  92          if step:
  93              await ws.send_json({"type": "step", "t": t, "events": step})
  94              await asyncio.sleep(0.15)
  95      await ws.send_json({"type": "done"})
  96      await ws.close()
```
- WebSocket stream: accept, read one JSON request, run it (errors are sent to the client). Send metadata, then the events one time-step at a time with a 0.15 s pause (animation), then 'done'.

**Lines 99–101**
```python
  99  @app.get("/")
 100  def index():
 101      return FileResponse(WEB / "index.html")
```
- `GET /`: the dashboard page.

**Lines 104–104**
```python
 104  app.mount("/", StaticFiles(directory=WEB), name="web")
```
- Mounts the web folder as static files (must come last so API routes win).


## 2.26  `tests/test_env.py`

Tests of the protocol, the delay model and the metric identities. They are the evidence that the simulator behaves as specified — the answer to 'how do you know your environment is correct?'

**Lines 1–12**
```python
   1  """Protocol, delay-model and metric-identity tests ."""
   2  import os
   3  
   4  import numpy as np
   5  import pytest
   6  
   7  from agrinegotiate import metrics, scenarios
   8  from agrinegotiate.agents import scripted as S
   9  from agrinegotiate.env.core import Market, F, B
  10  from agrinegotiate.eval.evaluate import evaluate, run_episode
  11  
  12  SYM = scenarios.get("SYM")
```
- Docstring, imports, and the SYM scenario used as the test market.

**Lines 15–21**
```python
  15  def manual_env(**kw):
  16      """1x1 market whose farmer link is controlled by hand."""
  17      env = Market(SYM.with_(duty=0.5, **kw))
  18      env.reset(123)
  19      env.p_fail, env.p_rec = 0.0, 0.0        # link stays wherever the test puts it
  20      env.link[:] = True
  21      return env
```
- `manual_env`: a 1×1 market whose link we control by hand: duty 0.5, then link failure/recovery probabilities set to 0 so the link stays where the test puts it, link forced up.

**Lines 24–28**
```python
  24  def acts(env, f=None, b=None):
  25      a = np.full((2, 1, 1), env.A_WAIT, dtype=int)
  26      if f is not None: a[F, 0, 0] = f
  27      if b is not None: a[B, 0, 0] = b
  28      return a
```
- `acts`: builds an action array (all WAIT) with optional farmer/buyer actions.

**Lines 31–35**
```python
  31  def mid_bin(env):
  32      """A bin both sides may legally offer in the symmetric scenario."""
  33      _, mask, _ = env.observe()
  34      both = np.nonzero(mask[F, 0, 0, :env.K] & mask[B, 0, 0, :env.K])[0]
  35      return int(both[len(both) // 2])
```
- `mid_bin`: a price bin both sides may legally offer (middle of the overlap of masks).

**Lines 38–46**
```python
  38  def test_message_takes_one_step_and_deal_closes_on_delivery():
  39      env = manual_env()
  40      k = mid_bin(env)
  41      env.step(acts(env, b=k))                          # t=0: buyer bids
  42      assert env.seen_id[B, 0, 0] >= 0 and env.t == 1   # delivered at t=1
  43      _, mask, _ = env.observe()
  44      assert mask[F, 0, 0, env.A_ACCEPT]
  45      env.step(acts(env, f=env.A_ACCEPT))               # t=1: farmer accepts
  46      assert env.done and env.deals[0]["t"] == 2 and env.deals[0]["price"] == pytest.approx(env.prices[k])
```
- **Test: one-step delay.** A bid at t=0 is delivered at t=1; the farmer can then accept; the deal closes on delivery at t=2 at exactly the bid price.

**Lines 49–63**
```python
  49  def test_late_accept_fails():
  50      env = manual_env(ttl=3)
  51      k = mid_bin(env)
  52      env.step(acts(env, b=k))                          # bid sent at t=0, valid through t=3
  53      env.step(acts(env, f=env.A_ACCEPT))               # accept sent at t=1 ...
  54      assert env.done                                   # (sanity: with the link up it closes)
  55      env = manual_env(ttl=3)
  56      env.step(acts(env, b=k))
  57      env.link[:] = False                               # ... but now the link drops first
  58      env.step(acts(env, f=env.A_ACCEPT))
  59      for _ in range(4):
  60          env.step(acts(env))
  61      env.link[:] = True
  62      env.step(acts(env))                               # accept finally delivered, too late
  63      assert not env.deals and env.failed_accepts == 1 and not env.pending_accept.any()
```
- **Test: late accept fails.** With link up the accept closes. When the link drops before the accept is sent and stays down past the TTL, the accept arrives too late: no deal, one failed accept, no pending accepts left.

**Lines 66–72**
```python
  66  def test_cancelled_offer_cannot_be_accepted():
  67      env = manual_env()
  68      k = mid_bin(env)
  69      env.step(acts(env, b=k))
  70      env.step(acts(env, f=env.A_ACCEPT, b=k - 1))      # buyer replaces its bid as the farmer accepts
  71      assert not env.deals and env.failed_accepts == 1
  72      assert env.seen_price[B, 0, 0] == pytest.approx(env.prices[k - 1])
```
- **Test: cancelled offer.** If the buyer replaces its bid in the same step the farmer accepts, the accept references a superseded id → fails, and the newer price is what gets seen.

**Lines 75–85**
```python
  75  def test_store_and_forward_keeps_order_and_latest_offer_wins():
  76      env = manual_env()
  77      _, mask, _ = env.observe()
  78      valid = np.nonzero(mask[F, 0, 0, :env.K])[0]
  79      env.link[:] = False
  80      env.step(acts(env, f=int(valid[-1])))             # ask sent while the link is down
  81      env.step(acts(env, f=int(valid[-2])))             # replaced one step later
  82      assert env.seen_id[F, 0, 0] < 0 and len(env.queue[0]) == 2
  83      env.link[:] = True
  84      env.step(acts(env))
  85      assert env.seen_price[F, 0, 0] == pytest.approx(env.prices[valid[-2]]) and not env.queue[0]
```
- **Test: store-and-forward.** Two asks sent while the link is down are both queued; when the link returns the *latest* one is what the counterparty sees and the queue empties.

**Lines 88–96**
```python
  88  def test_drop_mode_loses_messages_sent_while_off():
  89      env = manual_env(drop_mode=True)
  90      _, mask, _ = env.observe()
  91      k = int(np.nonzero(mask[F, 0, 0, :env.K])[0][-1])
  92      env.link[:] = False
  93      env.step(acts(env, f=k))
  94      env.link[:] = True
  95      env.step(acts(env)); env.step(acts(env))
  96      assert env.seen_id[F, 0, 0] < 0
```
- **Test: drop mode.** A message sent while the link is down is lost forever.

**Lines 99–101**
```python
  99  def test_walk_at_step_zero_gives_exactly_the_fallback():
 100      r = evaluate(SYM, S.make("TakeFallback", F, SYM), S.make("TakeFallback", B, SYM), 50)
 101      assert r["E"] == pytest.approx(0.0) and r["agreement_rate"] == 0.0
```
- **Test: floor reference.** Everyone walking at step 0 gives efficiency exactly 0 and no agreements.

**Lines 104–107**
```python
 104  def test_oracle_fair_split_is_the_ceiling():
 105      r = evaluate(SYM, S.make("OracleFairSplit", F, SYM), S.make("OracleFairSplit", B, SYM), 200)
 106      assert r["E"] > 0.95 and r["agreement_rate"] == 1.0
 107      assert abs(r["S_F"] - 0.5) < 0.03          # limited by the 0.05 price grid
```
- **Test: ceiling reference.** Oracle fair split gets E>0.95, agreement 1.0 and a share within 0.03 of 0.5 (limited by the 0.05 price grid).

**Lines 110–114**
```python
 110  def test_same_seed_same_episode():
 111      sc = scenarios.get("RURAL-DTN")
 112      a = run_episode(sc, S.make("Linear", F, sc), S.make("TitForTat", B, sc), 7)
 113      b = run_episode(sc, S.make("Linear", F, sc), S.make("TitForTat", B, sc), 7)
 114      assert np.allclose(a.farmer_gain, b.farmer_gain) and a.deals == b.deals
```
- **Test: determinism.** Same seed, same episode (gains and deals identical).

**Lines 117–129**
```python
 117  @pytest.mark.parametrize("name", list(scenarios.SCENARIOS))
 118  def test_identity_and_rationality_under_random_play(name):
 119      """E = FGC + BGC, and no deal ever closes with negative gains, under random valid actions."""
 120      sc = scenarios.get(name)
 121      n = int(os.environ.get("AGRI_RANDOM_EPISODES", 150))
 122      res = [run_episode(sc, S.make("RandomValid", F, sc), S.make("RandomValid", B, sc), s) for s in range(n)]
 123      m = metrics.aggregate(res)
 124      assert m["E"] == pytest.approx(m["FGC"] + m["BGC"])
 125      for r in res:
 126          for d in r.deals:
 127              assert d["g_t"] >= -1e-9 and (d["phi"] is None or -1e-6 <= d["phi"] <= 1 + 1e-6)
 128          total = r.farmer_gain.sum() + r.buyer_gain.sum()
 129          assert total <= r.g_star + 1e-9          # nobody can beat the efficiency ceiling
```
- **Test across every scenario, with random valid play:** the identity E = FGC + BGC holds; no deal ever has negative gains; φ stays in [0,1]; and total gains never exceed G* (nobody beats the ceiling).

**Lines 132–136**
```python
 132  def test_inequality_indices_reject_negative_input():
 133      with pytest.raises(AssertionError):
 134          metrics.gini([-1.0, 2.0])
 135      assert metrics.jain([1.0, 1.0, 1.0]) == pytest.approx(1.0)
 136      assert metrics.gini([1.0, 1.0]) == pytest.approx(0.0)
```
- **Test: inequality indices.** Gini rejects negatives; Jain of equal values = 1; Gini of equal values = 0.


## 2.27  `tests/test_yardsticks.py`

Unit tests of the yardstick formulas using hand-computed numbers.

**Lines 1–5**
```python
   1  """Unit tests for the closed-form reference points."""
   2  import numpy as np
   3  import pytest
   4  
   5  from agrinegotiate import yardsticks as y
```
- Docstring and imports.

**Lines 8–14**
```python
   8  def test_worked_example():
   9      v, o_f, o_b = 1900.0, 900.0, 200.0
  10      assert y.gains(v, o_f, o_b) == pytest.approx(800)
  11      assert y.nash_price(v, o_f, o_b) == pytest.approx(1300)
  12      assert y.gain_share(1300, v, o_f, o_b) == pytest.approx(0.5)
  13      assert y.gain_share(950, v, o_f, o_b) == pytest.approx(0.0625)
  14      assert y.target_price(v, o_f, o_b, 0.4) == pytest.approx(1220)
```
- Worked example: v=1900, o_f=900, o_b=200 → gains 800, Nash price 1300, share at 1300 = 0.5, share at 950 = 0.0625, target price at τ=0.4 is 1220.

**Lines 17–24**
```python
  17  def test_rubinstein_example_and_delay_table():
  18      assert round(y.rubinstein_share(0.90, 0.98, farmer_first=True), 3) == 0.169
  19      assert round(y.rubinstein_share(0.90, 0.98, farmer_first=False), 3) == 0.153
  20      assert round(y.rubinstein_limit(0.90, 0.98), 3) == 0.161
  21      table = {1: (0.169, 0.153), 3: (0.187, 0.137), 5: (0.206, 0.122)}
  22      for k, (ff, bf) in table.items():
  23          assert round(y.rubinstein_share(0.90, 0.98, True, k), 3) == ff
  24          assert round(y.rubinstein_share(0.90, 0.98, False, k), 3) == bf
```
- Rubinstein worked numbers (δ_f 0.90, δ_b 0.98): shares 0.169/0.153, limit 0.161, and the delay-length table (k = 1, 3, 5).

**Lines 27–28**
```python
  27  def test_symmetric_rubinstein_is_near_half():
  28      assert y.rubinstein_limit(0.95, 0.95) == pytest.approx(0.5)
```
- Symmetric patience gives a share of 0.5.

**Lines 31–36**
```python
  31  def test_max_gains_matches_bruteforce_hand_case():
  32      v, o_b, cap = [0.95, 0.90], [0.10, 0.05], [1, 1]
  33      o_f = [0.45, 0.50, 0.30]
  34      g, pairs = y.max_gains(v, o_f, o_b, cap)
  35      assert g == pytest.approx(y.max_gains_bruteforce(v, o_f, o_b, cap))
  36      assert len(pairs) == 2
```
- Hungarian result equals brute force on a hand case with two matched pairs.

**Lines 39–46**
```python
  39  def test_max_gains_matches_bruteforce_random():
  40      rng = np.random.default_rng(0)
  41      for _ in range(200):
  42          n, m = rng.integers(1, 5), rng.integers(1, 3)
  43          v = rng.uniform(0.4, 1.1, m); o_b = rng.uniform(0.0, 0.2, m)
  44          o_f = rng.uniform(0.2, 0.9, n); cap = rng.integers(1, 3, m)
  45          g, _ = y.max_gains(v, o_f, o_b, cap)
  46          assert g == pytest.approx(y.max_gains_bruteforce(v, o_f, o_b, cap))
```
- Hungarian equals brute force over 200 random markets.

**Lines 49–60**
```python
  49  def test_shares_sum_to_one():
  50      rng = np.random.default_rng(1)
  51      for _ in range(100):
  52          v, o_f, o_b = rng.uniform(0.8, 1.1), rng.uniform(0.2, 0.5), rng.uniform(0, 0.2)
  53          rho, t = rng.uniform(0.95, 1.0), int(rng.integers(0, 10))
  54          g = y.gains(v, o_f, o_b, rho, t)
  55          if g <= 0:
  56              continue
  57          p = rng.uniform(o_f * rho ** t, v * rho ** t - o_b)
  58          phi = y.gain_share(p, v, o_f, o_b, rho, t)
  59          buyer = (v * rho ** t - p - o_b) / g
  60          assert phi + buyer == pytest.approx(1.0)
```
- Farmer share + buyer share = 1 for random deals.

**Lines 63–67**
```python
  63  def test_no_trade_possible_is_reported_not_divided():
  64      assert y.nash_price(0.5, 0.45, 0.10) is None
  65      assert y.gain_share(0.5, 0.5, 0.45, 0.10) is None
  66      assert y.target_price(0.5, 0.45, 0.10, 0.4) is None
  67      assert y.max_gains([0.5], [0.45], [0.10], [1]) == (0.0, [])
```
- No possible trade: Nash price, share, target price return None (not divide-by-zero) and max gains = (0, []).


## 2.28  `tests/test_mechanisms.py`

Tests for the fee and floor mechanisms.

**Lines 1–9**
```python
   1  import pytest
   2  
   3  from agrinegotiate import scenarios
   4  from agrinegotiate.agents import scripted as S
   5  from agrinegotiate.env.core import Market, F, B
   6  from agrinegotiate.eval.evaluate import evaluate
   7  from agrinegotiate.mechanisms.fee import FeeMechanism, FloorMechanism
   8  
   9  SYM = scenarios.get("SYM")
```
- Imports and the SYM scenario.

**Lines 12–19**
```python
  12  def test_oracle_shortfall_matches_worked_example():
  13      mech = FeeMechanism(tau=0.4, lam=1.0)
  14      env = Market(SYM, mechanism=mech)
  15      env.reset(0)
  16      env.v[:], env.o_f[:], env.o_b[:] = 1.9, 0.9, 0.2        # the worked example, scaled by 1/1000
  17      assert mech.target(env, 0, 0, 0) == pytest.approx(1.22)
  18      assert mech.shortfall(env, 0, 0, 0.95, 0) == pytest.approx(0.27)
  19      assert mech.shortfall(env, 0, 0, 1.30, 0) == 0.0
```
- Shortfall matches the worked example: with the spec's numbers the target is 1.22 and a deal at 0.95 has shortfall 0.27; a deal at 1.30 has none.

**Lines 22–26**
```python
  22  def test_dual_update_moves_towards_constraint_and_is_clipped():
  23      mech = FeeMechanism(tau=0.4, lam=0.0)
  24      assert mech.dual_update(0.30) == pytest.approx(0.05 * (0.30 - 0.02))
  25      assert FeeMechanism(lam=0.0).dual_update(0.0) == 0.0             # cannot go negative
  26      assert FeeMechanism(lam=4.99).dual_update(10.0) == 5.0           # capped
```
- Dual update: λ moves by η·(J_c − ε) = 0.05·0.28; cannot go below 0; capped at 5.

**Lines 29–35**
```python
  29  def test_fee_is_charged_to_hardball_buyer_but_not_counted_as_lost_welfare():
  30      plain = evaluate(SYM, S.make("Boulware", F, SYM), S.make("HardballBuyer", B, SYM), 100)
  31      fee = evaluate(SYM, S.make("Boulware", F, SYM), S.make("HardballBuyer", B, SYM), 100,
  32                     mechanism=FeeMechanism(tau=0.4, lam=1.0))
  33      assert fee["E"] == pytest.approx(plain["E"])                     # scripted behaviour is unchanged
  34      assert fee["fees_over_gstar"] > 0 and fee["BGC_net_of_fees"] < fee["BGC"]
  35      assert fee["j_c"] > 0.02
```
- A fee is charged to a hardball buyer but scripted behaviour (hence efficiency) is unchanged; the fee is a transfer, not lost welfare; the constraint cost exceeds ε.

**Lines 38–46**
```python
  38  def test_price_floor_removes_low_bids():
  39      mech = FloorMechanism(kappa=0.6)
  40      env = Market(SYM, mechanism=mech)
  41      env.reset(1)
  42      _, mask, _ = env.observe()
  43      allowed = env.prices[mask[B, 0, 0, :env.K]]
  44      assert allowed.min() >= 0.6 * env.level - 1e-9
  45      r = evaluate(SYM, S.make("Boulware", F, SYM), S.make("HardballBuyer", B, SYM), 100, mechanism=mech)
  46      assert r["price_share"] >= 0.6 - 1e-9 or r["agreement_rate"] < 1.0
```
- Floor: the buyer's allowed prices are never below κ·level; with a floor, either price share ≥ κ or some deals fail.


## 2.29  `tests/test_dashboard.py`

Tests for the API (skipped if FastAPI is not installed).

**Lines 1–9**
```python
   1  """Backend endpoints and the episode recorder."""
   2  import pytest
   3  
   4  fastapi = pytest.importorskip("fastapi")
   5  from fastapi.testclient import TestClient      # noqa: E402
   6  
   7  from dashboard.api.main import app             # noqa: E402
   8  
   9  client = TestClient(app)
```
- Docstring, `importorskip`, test client.

**Lines 12–16**
```python
  12  def test_lists():
  13      sc = client.get("/api/scenarios").json()
  14      assert sc["SYM"]["bilateral"] and not sc["RURAL"]["bilateral"]
  15      me = client.get("/api/methods").json()
  16      assert "HardballBuyer" in me["scripted_buyer"] and "HardballBuyer" not in me["scripted_farmer"]
```
- Scenario list flags SYM as bilateral and RURAL as not; the hardball buyer is available to buyers only.

**Lines 19–29**
```python
  19  def test_episode_record_has_reference_lines_and_events():
  20      r = client.post("/api/episodes", json={"scenario": "PERISH", "farmer": "Conceder", "buyer": "HardballBuyer",
  21                                             "seed": 50001, "mechanism": {"kind": "fee", "tau": 0.4, "lam": 1.0}})
  22      assert r.status_code == 200
  23      ep = r.json()
  24      assert {"nash", "target", "farmer_fallback", "buyer_walkaway"} <= set(ep["lines"])
  25      assert ep["events"][0]["kind"] == "reset" and len(ep["lines"]["nash"]) == ep["T"] + 1
  26      assert ep["lines"]["target"][0] < ep["lines"]["nash"][0]            # tau 0.4 sits below the 0.5 split
  27      again = client.post("/api/episodes", json={"scenario": "PERISH", "farmer": "Conceder", "buyer": "HardballBuyer",
  28                                                 "seed": 50001, "mechanism": {"kind": "fee", "tau": 0.4, "lam": 1.0}}).json()
  29      assert again["deals"] == ep["deals"]                                 # same seed, same episode
```
- An episode record contains the four reference lines and a reset event first; the fee target lies below the Nash price (τ=0.4 < 0.5); the same request returns identical deals.

**Lines 32–34**
```python
  32  def test_bad_requests_are_reported():
  33      assert client.post("/api/episodes", json={"scenario": "NOPE"}).status_code == 404
  34      assert client.post("/api/episodes", json={"scenario": "SYM", "farmer": "NoSuchAgent"}).status_code == 400
```
- Unknown scenario → 404; unknown agent → 400.

**Lines 37–46**
```python
  37  def test_stream_sends_steps_then_done():
  38      with client.websocket_connect("/api/episodes/stream") as ws:
  39          ws.send_text('{"scenario": "SYM", "farmer": "Conceder", "buyer": "Conceder", "seed": 50002}')
  40          kinds = []
  41          while True:
  42              m = ws.receive_json()
  43              kinds.append(m["type"])
  44              if m["type"] in ("done", "error"):
  45                  break
  46      assert kinds[0] == "meta" and kinds[-1] == "done" and "step" in kinds
```
- WebSocket stream sends `meta` first, `step` messages, then `done`.


## 2.30  `tests/test_pettingzoo.py`

Rollout test of the PettingZoo wrapper.

**Lines 1–4**
```python
   1  import numpy as np
   2  
   3  from agrinegotiate import scenarios
   4  from agrinegotiate.env.pettingzoo_env import AgriNegotiateEnv
```
- Imports.

**Lines 7–23**
```python
   7  def test_parallel_env_random_rollout():
   8      for name in ["SYM", "RURAL-DTN"]:
   9          env = AgriNegotiateEnv(scenarios.get(name))
  10          rng = np.random.default_rng(0)
  11          for ep in range(5):
  12              obs, _ = env.reset(seed=ep)
  13              total = dict.fromkeys(env.possible_agents, 0.0)
  14              while env.agents:
  15                  acts = {}
  16                  for a in env.agents:
  17                      assert env.observation_space(a)["observation"].shape == obs[a]["observation"].shape
  18                      m = obs[a]["action_mask"].astype(bool)
  19                      acts[a] = np.array([rng.choice(np.nonzero(row)[0]) for row in m])
  20                  obs, rew, term, trunc, _ = env.step(acts)
  21                  for a, r in rew.items():
  22                      total[a] += r
  23              assert all(np.isfinite(v) for v in total.values())
```
- For SYM and RURAL-DTN, run 5 episodes with random legal actions (sampled from each row of the mask); check observation shapes match the declared space and that total rewards are finite.


## 2.31  `tests/test_data.py`

Test of cleaning and calibration on a tiny hand-made table (a fixture, not real data).

**Lines 1–5**
```python
   1  """Cleaning and calibration logic on a tiny hand-made frame (test fixture, not real data)."""
   2  import pandas as pd
   3  
   4  from agrinegotiate.data.calibrate import calibrate
   5  from agrinegotiate.data.clean import clean
```
- Imports.

**Lines 8–20**
```python
   8  def test_clean_drops_bad_rows_and_calibrates():
   9      days = pd.date_range("2025-01-01", periods=20, freq="D").strftime("%d/%m/%Y")
  10      df = pd.DataFrame({"State": "X", "District": "Y", "Market": "M1", "Commodity": "Tomato", "Variety": "Local",
  11                         "Arrival_Date": days, "Min_Price": 900.0, "Max_Price": 1250.0,
  12                         "Modal_Price": [1000.0 + 10 * k for k in range(20)]})
  13      bad = df.iloc[:3].copy()
  14      bad.loc[bad.index[0], "Modal_Price"] = 0            # zero price
  15      bad.loc[bad.index[1], "Modal_Price"] = 5000         # outside min..max
  16      out = clean(pd.concat([df, bad, df.iloc[:2]]))      # plus exact duplicates
  17      assert len(out) == 20 and list(out.columns) == ["market", "commodity", "date", "min", "max", "modal"]
  18      cal = calibrate(out)["Tomato"]
  19      assert cal["markets"] == 1 and cal["sigma_info_by_lag_days"]["1"] >= 0
  20      assert 0.25 < cal["relative_spread"] < 0.40
```
- Build 20 daily rows, add a zero-price row, an out-of-range row and exact duplicates. After `clean`, exactly 20 rows remain with the canonical column names; calibration returns one market, a non-negative σ, and a relative spread between 0.25 and 0.40.


---

# Part 3 — Everything that is not Python

## 3.1 `configs/scenarios/*.yaml`
Eight YAML files, one per main scenario (SYM, PERISH, DEADLINE, INFO, THIN, RURAL, RURAL-DTN, SYMX), each about 25 lines. They are **generated** by the `__main__` block of `scenarios.py` (it dumps each `Scenario.to_dict()`), so they are a readable record of the exact parameters, not a second source of truth. Each key is a field of the `Scenario` dataclass: `n_farmers`, `n_buyers`, `capacity`, `T`, `rho`, `deadline_lo/hi`, `h_farmer`, `base_cost`, `theta`, `theta_spread`, `o_b`, `margin_lo/hi`, `level_lo/hi`, `sigma_info`, `duty`, `outage_len`, `ttl`, `drop_mode`, `n_bins`, `p_lo`, `p_hi`.

## 3.2 `requirements.txt`
Pinned versions of the libraries used: `numpy`, `scipy` (Hungarian algorithm, trimmed mean, Mann–Whitney), `pandas` (data cleaning), `torch` (networks and PPO), `gymnasium` + `pettingzoo` (standard env API), `PyYAML` (configs), `pytest` (tests); unpinned `fastapi`, `uvicorn`, `httpx` (dashboard backend and its test client) and `matplotlib` (paper figures).

## 3.3 `__init__.py` files
Empty files that make each folder a Python package so `from agrinegotiate.env.core import Market` works.

## 3.4 The `results/` folder
`results/<experiment>/<label>/seed_<n>/` holds, for every training run: `config.json` (exact settings), `curve.json` (metrics every 10 iterations), `metrics.json` (final numbers) and `checkpoint.pt` (the trained weights). `summary.md` / `summary.json` at the top are produced by `experiments/make_report.py`. Experiments: `x1_scripted` (baselines), `x2_gate` (unconstrained learners), `x3_mechanism` (fee vs floors), `x4_delay`, `x5_diagnosis`, `x6_hetero` (PERISH-HET).

## 3.5 The documents
- `DECISIONS.md` — numbered design decisions and their reasons (for example #16: our own bootstrap because `rliable` broke with pandas 3; #18: the WALK logit offset).
- `PROGRESS.md` — dated log of what was run and what was found, including the corrections.
- `README.md` — how to install, run tests, run experiments, start the dashboard.
- `FRONTEND_BRIEF.md` — the brief the website was built from.

## 3.6 The website (`site/`)
Plain HTML, CSS and JavaScript with canvas and SVG — no frameworks. `index.html` is the page structure (navbar, sections), `style.css` the earthy theme, `data.js` holds the exported results, and `site.js` (about 540 lines) does four things: (1) draws the swaying wheat field on a canvas; (2) runs a small JavaScript negotiation simulator in the browser (same idea as `Market`, simplified: two agents, concession curves, a delay slider); (3) replays recorded episodes and the training curves from `data.js`; (4) draws the dot plots with confidence intervals as SVG. It does not train anything; every number on the site comes from the `results/` folder through `dashboard/export_data.py`.

## 3.7 If someone points at a line and asks "why?"
Use this checklist: (1) *Is it a modelling choice?* (ρ, TTL, price grid, margins) → it is a scenario parameter and is varied in a scenario, so it is not hidden. (2) *Is it a safety check?* (asserts, masks) → it exists so impossible outcomes crash instead of silently corrupting results. (3) *Is it a statistics choice?* (IQM, last-10 checkpoints, fixed seeds) → it is there to prevent cherry-picking. (4) *Is it a stability trick?* (orthogonal init, gradient clipping, advantage normalisation, WALK bias) → standard PPO practice, each documented in its own block above.
