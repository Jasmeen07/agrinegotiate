# Progress log

## 16 November 2026 (second entry): all phases built, first full set of experiments run

Everything in the build specification now exists in code and has been run at least once.
Full tables with intervals: `results/summary.md`. Figures: `paper/figures/`. Paper draft: `paper/draft.md`.
Design departures: `DECISIONS.md` (23 entries).

### What exists

- Benchmark environment, yardsticks, scripted baselines, metrics, PettingZoo wrapper.
- PPO learners (feed-forward and recurrent), adaptive fee (true-value and public-price targets), fixed fee, hard price floor.
- Experiments X1 to X6: about 140 training runs across 34 settings, plus the scripted table (70 cells of 1,000 episodes).
- Dashboard: FastAPI backend with live episodes and a static front end with replay, charts and tables.
- 39 automated tests pass.

### Results in one table

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| H1: equal conditions give efficient, near-equal deals | **Fails** | Efficiency 0.84 [0.83, 0.86]. Farmer share is not near 0.5: eight runs settle near 0.26, two near 0.65 |
| H2: unequal conditions lower the farmer's share | **Holds for two of four conditions** | Perishable crop 0.16 [0.14, 0.18] (10 runs) and three farmers per buyer 0.14 [0.12, 0.15] (5 runs), both clearly below equal conditions. No effect from an earlier farmer deadline or a noisy price signal (5 runs each) |
| H3: the adaptive fee restores farmer gains at small efficiency cost and beats a price floor | **Mostly fails** | The fee helps: farmer gain capture rises from 0.11 to 0.19, and efficiency rises from 0.67 to 0.72. But the gain is below the 0.10 threshold set in advance, the 0.4 target share is never reached (0.27, or 0.30 with a faster fee update), and a plain price floor beats the fee on both measures (efficiency 0.83, farmer share 0.39 to 0.54) |
| H4: intermittent links reduce agreement and efficiency | Holds (3 runs per point) | Efficiency 0.65 at full uptime, 0.36 at 60%, zero at 40%, where every run stopped trading |
| H5: delay-aware agents lose less | **Fails** (3 runs per point) | 0.38 against 0.36 at 60% uptime; the interval of the difference includes zero |

### What the results mean for the project

1. **Exploitation is real in this benchmark** when the crop is perishable or farmers outnumber buyers. That part of the motivation stands.
2. **The equal-conditions control is not a fair baseline.** Learners pick one of two lopsided conventions. Removing scripted opponents from training, or equalising uncertainty, does not remove the two clusters (3 runs each). The paper must present this as a finding about equilibrium selection, and compare distributions, not means.
3. **The fee is not the winning mechanism as specified.** In the main scenario a price floor is simpler and better. The floor works there because buyer values barely vary, so a floor never blocks a worthwhile trade.
4. **A lead worth confirming:** when buyer values vary widely (PERISH-HET, post hoc, 3 to 5 runs), each floor loses trades (agreement 55%, 41%, 24% as the floor rises) while the fee keeps 68% and the highest efficiency (0.60). This is where a fee could justify itself. It needs 10 seeds and pre-set thresholds before it can be claimed.
5. **Delay hurts badly, and our delay-aware agents do not fix it.** At 40% uptime agents give up entirely.

### Honest limits of this round

- Training budget is 0.3 to 0.7 million agent-steps per run, not the 3 million planned. The fee rate was still rising at the end of the main mechanism runs.
- Sweeps, delay and the follow-up use 3 to 5 seeds and are exploratory.
- The mechanism was tested in the one-farmer, one-buyer perishable scenario and the delay study in a one-to-one rural scenario. The 8-farmer markets have scripted baselines only.
- No real mandi data has been loaded. The cleaning and calibration code is tested on a hand-made fixture only.
- Not built: centralised-critic PPO, best-response buyer test, cross-play between seeds, inequity-averse baseline, colluding buyers, 41-bin ablation.
- The two closest prior papers (Zhang, Xu and Dou; Shea and Yu) are still unread in full.

### Next steps

1. Confirm the PERISH-HET lead with 10 seeds and thresholds fixed in advance.
2. Rerun the main comparisons at 3 million steps to check nothing changes.
3. Load real mandi data and set buyer-value spread from it, since that spread decides whether floor or fee wins.
4. Run the 8-farmer market scenarios with learners.

## 16 November 2026 (first entry): phases 0 and 1

Environment, yardsticks, scripted agents, metrics and tests built. Scripted baseline table in
`results/x1_scripted/table.md`. First 10-seed runs showed three of ten perishable-crop runs collapsing
to no trade; fixed by lowering the initial probability of walking away (DECISIONS.md #18).

## 17 November 2026: project website
- Built `site/` (index.html, style.css, site.js, data.js): six-section editorial site covering the problem, how a negotiation works, a live in-browser simulator (scripted strategies ported to JS, link quality, floor/fee rule, play either side, replay of recorded learned episodes), how we measured, findings H1 to H5 with per-run dots and intervals, and limits.
- Data is a compact export of `dashboard/web/data.json`; no new experiments were run.
- The browser simulator is a simplified single-pair model; scripted agents there ignore the fee.
- Published to the existing artifact; zip rebuilt as `agrinegotiate.zip`.
- The earlier `dashboard/web/` page is kept but superseded.

## 24 November 2026: site v2
- Site rebuilt with a wind-swept wheat field (canvas, reacts to the mouse), a "How the agents learn" section (22-number observation, 24 masked actions, IPPO, dual-ascent fairness fee, GRU variant), a scrubbable training-curve replay, and the earlier simulator and findings.
- Design-tool list from the user noted; built with plain HTML/canvas/SVG (no libraries) so it stays one static folder.

## 24 November 2026: PERISH-HET confirmed at 10 seeds
- Reran PERISH-HET with 10 seeds for no rule, fee (eta 0.25) and floors 0.50, 0.55, 0.60 (`experiments/run.py hetero10`, 31 new runs). Fixed-rate fee stays at 3 seeds.
- Floors lose trades as they tighten: deal rate 0.55, 0.41, 0.24 (tight intervals). Fee deal rate 0.66 [0.34, 0.69], but 2 of 10 fee runs collapsed to no trade. No rule: 5 of 10 collapsed.
- At equal farmer share (about 0.30) the fee and floor 0.50 have the same efficiency (about 0.57), so the earlier "fee has the highest efficiency" lead is not supported. Floors still give higher farmer share (0.46, 0.57) at the cost of trades.
- Site updated (Fig. 8 now shows deal rate and efficiency; roadmap item 1 marked done).

## 2026-11-26 — code cleanup
Behaviour-preserving refactor: vectorised the alt-offer features in `observe`, split `step` into `step`/`_apply`, one `_step` helper for acting in PPO, `_advantages` extracted, scripted tactics built from one table, stale spec references removed. Verified bit-identical on 44 fixed-seed runs (scripted and trained) and 39 tests pass.

## 2026-11-26 — real data
Real Agmarknet daily lemon prices (157,829 rows, 667 mandis, 23 states, 2022-01 to 2024-11; Government Open Data Licence India) were obtained from a public GitHub extract and saved as `data/real/lemon_daily_agmarknet.csv.gz`. `experiments/calibrate_real.py` derives from them: the episode price level (national price relative to its 61-day local median, 25 quantiles), the farmer's price-signal noise (std of 3-day log price change at the same mandi, 0.128; 1 day 0.095, 7 days 0.169) and the spread of farmer fallbacks (same-day spread between mandis of a state, 24%, i.e. theta_spread 0.188). New scenarios PERISH-REAL, RURAL1-REAL, RURAL1-REAL-DTN use them (Scenario.level_samples). Batch `real` (35 runs): see results/summary.md section X7. Prices are real; the negotiations are still simulated, because no public farmer-buyer negotiation data exists.
