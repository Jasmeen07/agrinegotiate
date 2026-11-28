# Do Learning Negotiators Exploit Weaker Sellers? A Benchmark for Fair, Delay-Tolerant Bargaining in Agricultural Markets

*Working draft, 16 November 2026. Every number comes from `results/summary.md`. Simulation only.*

## Abstract

Smallholder farmers often sell under unequal bargaining power. We introduce AgriNegotiate, an open
benchmark in which farmer and buyer agents bargain over price through alternating offers sent over
intermittent links, scored against closed-form reference points (maximum gains from trade, the Nash
bargaining price, the Rubinstein split). Self-interested PPO agents leave farmers 16% of the gains
when the crop is perishable and 14% when three farmers face one buyer. Under nominally equal
conditions they do not converge to an equal split either: independent runs settle on one of two
unequal conventions. A learned platform fee on underpaying buyers raises farmers' gains and
efficiency but is dominated by a simple price floor when buyer values are similar; exploratory runs
suggest the ranking changes when buyer values vary widely. Intermittent connectivity sharply reduces
trade, and our delay-aware agents do not recover it. We report the negative results in full.

## 1 Introduction

- Field evidence: farm-gate prices far below wholesale, and price information alone not raising them (Mitra et al., 2018). **Verify figures against the published paper.**
- Question: do learning negotiators exploit bargaining-power gaps, and which platform rule limits it?
- Contributions: (1) the benchmark with exact yardsticks and baseline scores; (2) fairness as a constraint on the farmer's share of the gains from trade, enforced by an adaptive fee, compared with a price floor; (3) an empirical study with pre-registered hypotheses, three of which fail.

## 2 Related work

Negotiation with RL (ANAC and NegMAS; Bagga et al., 2020; Zhang, Xu and Dou, 2026, **unread in full**),
fairness in negotiation (Shea and Yu, 2024, **unread in full**; FairMarket-RL), fair and constrained MARL
(FEN, SOTO, MAPPO-Lagrangian), delay in MARL (CoDe, DACOM), agricultural markets. Use the feature table
from the build specification, section 4.3. All novelty claims are "to our knowledge".

## 3 Problem

Game, payoffs, protocol, asymmetry knobs and link model as in the build specification, sections 5, 6 and 8,
with the extensions listed in `DECISIONS.md`. Metrics: efficiency E, farmer gain capture FGC, farmer share
S_F, agreement rate, all over every lot including unsold ones.

## 4 Reference points and baselines

Oracle fair split (protocol ceiling: E 0.98 with equal conditions, 0.84 with a perishable crop).
Random traders reach E 0.90 under equal conditions, so efficiency alone does not show skill.
Time-dependent heuristics end worse than walking away when the crop decays (E about -0.7); a tuned
heuristic reaches 0.75. A hardball buyer holds a Boulware farmer to a 0.07 share.

## 5 Results

All learned results: interquartile mean over seeds, 95% bootstrap interval in brackets.

**5.1 Equal conditions are not fair (H1 fails).** E 0.842 [0.826, 0.860]. Farmer share per run:
0.25, 0.65, 0.27, 0.26, 0.66, 0.28, 0.27, 0.26, 0.26, 0.27. Figure 2, Figure 3.

**5.2 Exploitation (H2 partly holds).** Perishable crop: share 0.161 [0.142, 0.175], 10 runs, difference from
equal conditions -0.106 [-0.297, -0.089]. Three farmers, one buyer: 0.136 [0.122, 0.149], 5 runs.
Earlier deadline 0.345 [0.300, 0.525] and noisy price signal 0.273 [0.258, 0.558], 5 runs each: no reduction.

**5.3 Mechanisms, perishable crop (H3 mostly fails).** Figure 4.

| Rule | Runs | E | FGC | S_F |
| --- | --- | --- | --- | --- |
| None | 10 | 0.666 [0.650, 0.681] | 0.108 [0.094, 0.120] | 0.161 [0.142, 0.175] |
| Adaptive fee, target 0.4 | 10 | 0.721 [0.708, 0.732] | 0.194 [0.172, 0.208] | 0.269 [0.241, 0.284] |
| Adaptive fee, target 0.4, faster update | 5 | 0.759 [0.754, 0.772] | 0.233 [0.218, 0.260] | 0.304 [0.284, 0.343] |
| Adaptive fee, public prices only | 3 | 0.720 [0.711, 0.725] | 0.183 [0.135, 0.215] | 0.253 [0.189, 0.297] |
| Price floor 0.55 | 3 | 0.825 [0.824, 0.826] | 0.325 [0.324, 0.327] | 0.394 [0.392, 0.395] |
| Price floor 0.60 | 3 | 0.825 [0.822, 0.828] | 0.448 [0.445, 0.451] | 0.543 [0.541, 0.545] |

The fee improves both measures over no rule but never reaches its 0.4 target, and the floor dominates it.
The floor also removes the haggling: agreement is 100%.

**5.4 Varied buyer values (exploratory, post hoc).** Floors lose trades: agreement 0.550, 0.406, 0.236 at
floors 0.50, 0.55, 0.60, with E 0.575, 0.457, 0.292. The adaptive fee: agreement 0.683, E 0.600 [0.501, 0.626],
FGC 0.177 [0.142, 0.193]. Floor 0.55 gives farmers more (FGC 0.209) at lower efficiency. Two of five
unconstrained runs stopped trading in this scenario.

**5.5 Delay (H4 holds, H5 fails; 3 runs per point).** Figure 5. Delay-unaware E: 0.647, 0.364, 0.000 at
100%, 60%, 40% link uptime. Delay-aware at 60%: 0.378 [0.368, 0.394] against 0.364 [0.320, 0.411].

## 6 Discussion

- Why two conventions under equal conditions: many equilibria; self-play selects one. Open question: why the buyer-favouring one is more common.
- When a floor is enough: buyer values close to a public reference price. When it is not: varied values. Real mandi spread data should decide which case applies.
- Why the fee undershoots: slow dual ascent, short training, and buyers who prefer paying part of the fee.

## 7 Limitations

Simulation only; most parameters are assumptions; no real price data loaded; training budgets below plan;
sweeps and delay results use 3 to 5 seeds; mechanism and delay tested in one-to-one scenarios only; no
best-response robustness test; price grid of 0.05 limits resolution; dual use (the same code trains
exploitative buyers).

## Figures

- Figure 2: `figures/fig2_learning_curves.png`
- Figure 3: `figures/fig3_farmer_share_by_scenario.png`
- Figure 4: `figures/fig4_efficiency_fairness_frontier.png`
- Figure 5: `figures/fig5_delay.png`
