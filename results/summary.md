# Results summary

Interquartile mean over training seeds with a 95% bootstrap interval. Each seed's score is the mean of
its last 10 evaluation checkpoints (300 fixed evaluation episodes each). A failed run ends with E below 0.1.
Results with fewer than 10 seeds are exploratory.

### Unconstrained learners by scenario (X2)

| Setting | Seeds | Failed | Efficiency | Farmer gain capture | Farmer share | Agreement |
| --- | --- | --- | --- | --- | --- | --- |
| SYM | 10 | 0 | 0.842 [0.826, 0.860] | 0.228 [0.218, 0.381] | 0.267 [0.259, 0.459] | 0.934 [0.921, 0.946] |
| PERISH | 10 | 0 | 0.666 [0.650, 0.681] | 0.108 [0.094, 0.120] | 0.161 [0.142, 0.175] | 0.889 [0.862, 0.914] |
| DEADLINE | 5 | 0 | 0.822 [0.789, 0.850] | 0.274 [0.250, 0.441] | 0.345 [0.300, 0.525] | 0.888 [0.863, 0.906] |
| INFO | 5 | 0 | 0.853 [0.828, 0.869] | 0.229 [0.221, 0.472] | 0.273 [0.258, 0.558] | 0.942 [0.927, 0.952] |
| THIN3 | 5 | 0 | 0.832 [0.824, 0.841] | 0.113 [0.100, 0.125] | 0.136 [0.122, 0.149] | 0.985 [0.979, 0.989] |

### SYM diagnosis (X5, exploratory)

| Setting | Seeds | Failed | Efficiency | Farmer gain capture | Farmer share | Agreement |
| --- | --- | --- | --- | --- | --- | --- |
| SYM_mix0 | 3 | 0 | 0.868 [0.864, 0.873] | 0.491 [0.211, 0.636] | 0.566 [0.243, 0.732] | 0.948 [0.947, 0.950] |
| SYMX | 3 | 0 | 0.863 [0.769, 0.913] | 0.301 [0.205, 0.488] | 0.362 [0.226, 0.632] | 0.949 [0.882, 0.983] |

- SYM_mix0 farmer share per seed: [0.72, 0.24, 0.73]
- SYMX farmer share per seed: [0.23, 0.23, 0.63]

### Mechanisms in PERISH (X3)

| Setting | Seeds | Failed | Efficiency | Farmer gain capture | Farmer share | Agreement | Fee rate | Fees / G* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unconstrained | 10 | 0 | 0.666 [0.650, 0.681] | 0.108 [0.094, 0.120] | 0.161 [0.142, 0.175] | 0.889 [0.862, 0.914] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| fee_fixed_lam1.0_tau0.4 | 3 | 0 | 0.763 [0.736, 0.777] | 0.231 [0.191, 0.269] | 0.301 [0.259, 0.346] | 0.944 [0.936, 0.950] | 1.000 [1.000, 1.000] | 0.083 [0.067, 0.095] |
| fee_observable_tau0.4 | 3 | 0 | 0.720 [0.711, 0.725] | 0.183 [0.135, 0.215] | 0.253 [0.189, 0.297] | 0.902 [0.868, 0.923] | 0.578 [0.554, 0.592] | 0.060 [0.044, 0.083] |
| fee_oracle_tau0.2 | 3 | 0 | 0.696 [0.686, 0.717] | 0.164 [0.123, 0.210] | 0.234 [0.179, 0.291] | 0.885 [0.839, 0.920] | 0.056 [0.031, 0.078] | 0.001 [0.000, 0.002] |
| fee_oracle_tau0.3 | 3 | 0 | 0.717 [0.692, 0.747] | 0.177 [0.114, 0.225] | 0.245 [0.164, 0.301] | 0.924 [0.877, 0.960] | 0.216 [0.196, 0.251] | 0.012 [0.008, 0.020] |
| fee_oracle_tau0.4 | 10 | 0 | 0.721 [0.708, 0.732] | 0.194 [0.172, 0.208] | 0.269 [0.241, 0.284] | 0.937 [0.914, 0.950] | 0.454 [0.436, 0.492] | 0.037 [0.034, 0.047] |
| fee_oracle_tau0.4_eta0.25 | 5 | 0 | 0.759 [0.754, 0.772] | 0.233 [0.218, 0.260] | 0.304 [0.284, 0.343] | 0.950 [0.934, 0.971] | 2.012 [1.804, 2.178] | 0.151 [0.110, 0.183] |
| fee_oracle_tau0.5 | 3 | 0 | 0.730 [0.719, 0.739] | 0.195 [0.156, 0.218] | 0.267 [0.217, 0.298] | 0.932 [0.876, 0.968] | 0.796 [0.767, 0.841] | 0.117 [0.097, 0.150] |
| floor_kappa0.50 | 3 | 0 | 0.827 [0.825, 0.828] | 0.201 [0.199, 0.205] | 0.244 [0.241, 0.248] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| floor_kappa0.55 | 3 | 0 | 0.825 [0.824, 0.826] | 0.325 [0.324, 0.327] | 0.394 [0.392, 0.395] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| floor_kappa0.60 | 3 | 0 | 0.825 [0.822, 0.828] | 0.448 [0.445, 0.451] | 0.543 [0.541, 0.545] | 0.999 [0.998, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| floor_kappa0.65 | 3 | 0 | 0.764 [0.759, 0.769] | 0.489 [0.485, 0.492] | 0.640 [0.639, 0.640] | 0.964 [0.959, 0.968] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### Mechanisms when buyer values vary widely, PERISH-HET (X6, post hoc, exploratory)

| Setting | Seeds | Failed | Efficiency | Farmer gain capture | Farmer share | Agreement | Fee rate | Fees / G* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fee_fixed_lam1.0_tau0.4 | 3 | 1 | 0.394 [-0.000, 0.623] | 0.110 [-0.000, 0.193] | n/a | 0.449 [0.000, 0.703] | 1.000 [1.000, 1.000] | 0.039 [0.000, 0.065] |
| fee_oracle_tau0.4_eta0.25 | 10 | 2 | 0.571 [0.297, 0.613] | 0.165 [0.086, 0.186] | 0.292 [0.282, 0.307] | 0.655 [0.339, 0.691] | 2.008 [1.432, 2.132] | 0.103 [0.054, 0.117] |
| floor_kappa0.50 | 10 | 0 | 0.574 [0.572, 0.576] | 0.170 [0.169, 0.172] | 0.297 [0.295, 0.298] | 0.550 [0.549, 0.551] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| floor_kappa0.55 | 10 | 0 | 0.455 [0.452, 0.457] | 0.210 [0.206, 0.212] | 0.460 [0.455, 0.464] | 0.406 [0.405, 0.406] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| floor_kappa0.60 | 10 | 0 | 0.293 [0.291, 0.296] | 0.167 [0.166, 0.169] | 0.571 [0.567, 0.574] | 0.238 [0.236, 0.240] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| unconstrained | 10 | 5 | 0.162 [0.001, 0.350] | 0.037 [0.000, 0.079] | 0.223 [0.209, 0.244] | 0.192 [0.002, 0.417] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### Scenarios calibrated on real Agmarknet lemon prices (X7)

| Setting | Seeds | Failed | Efficiency | Farmer gain capture | Farmer share | Agreement | Fee rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unconstrained | 10 | 0 | 0.636 [0.629, 0.643] | 0.168 [0.165, 0.179] | 0.265 [0.261, 0.278] | 0.722 [0.713, 0.734] | 0.000 [0.000, 0.000] |
| fee_oracle_tau0.4_eta0.25 | 10 | 0 | 0.727 [0.699, 0.738] | 0.257 [0.230, 0.283] | 0.351 [0.325, 0.383] | 0.842 [0.794, 0.870] | 1.783 [1.654, 1.900] |
| floor_kappa0.55 | 5 | 0 | 0.817 [0.814, 0.819] | 0.302 [0.301, 0.304] | 0.370 [0.369, 0.372] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| unaware_duty0.6 | 5 | 0 | 0.399 [0.341, 0.447] | 0.101 [0.071, 0.126] | 0.247 [0.206, 0.286] | 0.544 [0.475, 0.610] | 0.000 [0.000, 0.000] |
| aware_duty0.6 | 5 | 0 | 0.353 [0.332, 0.377] | 0.091 [0.080, 0.107] | 0.247 [0.230, 0.281] | 0.502 [0.453, 0.530] | 0.000 [0.000, 0.000] |

### Delay in RURAL1-DTN (X4, 3 seeds per cell, exploratory)

| Setting | Seeds | Failed | Efficiency | Farmer gain capture | Farmer share | Agreement |
| --- | --- | --- | --- | --- | --- | --- |
| unaware_duty1.0 | 3 | 0 | 0.647 [0.626, 0.663] | 0.123 [0.105, 0.132] | 0.189 [0.168, 0.200] | 0.844 [0.837, 0.854] |
| unaware_duty0.6 | 3 | 0 | 0.364 [0.320, 0.411] | 0.068 [0.042, 0.096] | 0.184 [0.135, 0.231] | 0.589 [0.515, 0.679] |
| unaware_duty0.4 | 3 | 3 | -0.000 [-0.000, -0.000] | -0.000 [-0.000, -0.000] | n/a | 0.000 [0.000, 0.000] |
| aware_duty1.0 | 3 | 0 | 0.650 [0.641, 0.662] | 0.134 [0.125, 0.143] | 0.206 [0.194, 0.216] | 0.884 [0.879, 0.890] |
| aware_duty0.6 | 3 | 0 | 0.378 [0.368, 0.394] | 0.081 [0.078, 0.086] | 0.211 [0.196, 0.227] | 0.614 [0.596, 0.650] |
| aware_duty0.4 | 3 | 3 | 0.002 [0.001, 0.002] | -0.000 [-0.001, -0.000] | n/a | 0.005 [0.004, 0.006] |
| memory_only_duty0.6 | 3 | 0 | 0.396 [0.388, 0.400] | 0.083 [0.069, 0.093] | 0.207 [0.173, 0.225] | 0.656 [0.635, 0.676] |
| features_only_duty0.6 | 3 | 1 | 0.219 [0.002, 0.388] | 0.043 [-0.000, 0.079] | n/a | 0.347 [0.003, 0.615] |

### Budget check: main comparisons at 3 million agent-steps per run (X8)

Same scenarios, settings and seeds as above; 76 runs, 3.00 to 3.01 million agent-steps each. Short = the runs above (0.3 to 0.7 million).

| Experiment | Setting | Seeds | E short | E 3M | Farmer gain short | Farmer gain 3M | Farmer share short | Farmer share 3M | Agreement 3M | Failed 3M |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate | SYM | 10 | 0.842 | 0.617 [0.603, 0.668] | 0.228 | 0.150 [0.139, 0.258] | 0.267 | 0.242 [0.229, 0.353] | 0.906 [0.897, 0.920] | 0 |
| gate | PERISH | 10 | 0.666 | 0.712 [0.710, 0.728] | 0.108 | 0.136 [0.135, 0.163] | 0.161 | 0.190 [0.190, 0.221] | 0.855 [0.853, 0.879] | 0 |
| gate | THIN3 | 5 | 0.832 | 0.827 [0.818, 0.851] | 0.113 | 0.082 [0.076, 0.090] | 0.136 | 0.099 [0.092, 0.109] | 0.983 [0.978, 0.986] | 0 |
| mechanism | fee_oracle_tau0.4 | 10 | 0.721 | 0.822 [0.810, 0.833] | 0.194 | 0.325 [0.298, 0.348] | 0.269 | 0.396 [0.365, 0.420] | 0.999 [0.991, 1.000] | 0 |
| mechanism | floor_kappa0.55 | 3 | 0.825 | 0.829 [0.828, 0.830] | 0.325 | 0.304 [0.302, 0.308] | 0.394 | 0.367 [0.364, 0.371] | 1.000 [1.000, 1.000] | 0 |
| mechanism | floor_kappa0.60 | 3 | 0.825 | 0.826 [0.809, 0.834] | 0.448 | 0.426 [0.407, 0.435] | 0.543 | 0.516 [0.503, 0.522] | 1.000 [1.000, 1.000] | 0 |
| hetero | unconstrained | 5 | 0.162 | 0.423 [0.111, 0.628] | 0.037 | 0.094 [0.024, 0.143] | 0.223 | 0.216 [0.188, 0.249] | 0.473 [0.124, 0.702] | 1 |
| hetero | fee_oracle_tau0.4_eta0.25 | 5 | 0.571 | 0.696 [0.229, 0.712] | 0.165 | 0.281 [0.083, 0.306] | 0.292 | 0.423 [0.362, 0.435] | 0.764 [0.252, 0.783] | 1 |
| hetero | floor_kappa0.50 | 5 | 0.574 | 0.587 [0.585, 0.590] | 0.170 | 0.164 [0.162, 0.166] | 0.297 | 0.280 [0.276, 0.282] | 0.551 [0.550, 0.553] | 0 |
| hetero | floor_kappa0.60 | 5 | 0.293 | 0.297 [0.296, 0.299] | 0.167 | 0.165 [0.164, 0.166] | 0.571 | 0.554 [0.552, 0.558] | 0.232 [0.232, 0.232] | 0 |
| real | unconstrained | 5 | 0.636 | 0.675 [0.667, 0.729] | 0.168 | 0.175 [0.171, 0.243] | 0.265 | 0.260 [0.256, 0.331] | 0.714 [0.709, 0.802] | 0 |
| real | fee_oracle_tau0.4_eta0.25 | 5 | 0.727 | 0.830 [0.828, 0.834] | 0.257 | 0.442 [0.417, 0.479] | 0.351 | 0.532 [0.501, 0.577] | 0.984 [0.983, 0.993] | 0 |
| real | floor_kappa0.55 | 5 | 0.817 | 0.823 [0.823, 0.823] | 0.302 | 0.304 [0.304, 0.304] | 0.370 | 0.369 [0.369, 0.370] | 1.000 [1.000, 1.000] | 0 |

- SYM farmer share per seed at 3M: [0.2, 0.87, 0.23, 0.24, 0.25, 0.24, 0.24, 0.22, 0.25, 0.25]

## Hypothesis verdicts

Rows ending in _3m repeat a test at the 3 million agent-step budget (X8).

| Test | Result | Evidence |
| --- | --- | --- |
| H1_efficiency | FAIL | SYM E 0.842 [0.826, 0.860]; needs the whole interval at or above 0.85 |
| H1_fair_split | FAIL | SYM farmer share 0.267 [0.259, 0.459]; per seed [0.25, 0.65, 0.27, 0.26, 0.66, 0.28, 0.27, 0.26, 0.26, 0.27] |
| H2_PERISH | PASS | farmer share PERISH minus SYM = -0.106 [-0.297, -0.089]; Mann-Whitney one-sided p = 0.0001; 0 of 10 PERISH runs undefined |
| H2_DEADLINE | FAIL (exploratory, 5 seeds) | farmer share DEADLINE minus SYM = 0.078 [-0.096, 0.256]; Mann-Whitney one-sided p = 0.9724; 0 of 5 DEADLINE runs undefined |
| H2_INFO | FAIL (exploratory, 5 seeds) | farmer share INFO minus SYM = 0.006 [-0.139, 0.285]; Mann-Whitney one-sided p = 0.6161; 0 of 5 INFO runs undefined |
| H2_THIN3 | PASS (exploratory, 5 seeds) | farmer share THIN3 minus SYM = -0.131 [-0.320, -0.117]; Mann-Whitney one-sided p = 0.0003; 0 of 5 THIN3 runs undefined |
| H3_gain | FAIL | FGC with fee minus without = 0.086 [0.061, 0.106]; needs the whole interval at or above 0.10 |
| H3_efficiency_cost | PASS | E with fee minus without = 0.055 [0.035, 0.074]; needs the whole interval at or above -0.05 |
| H3_beats_floor | FAIL | price floor settings that match or beat the fee on both E and FGC (point estimates): ['floor_kappa0.50', 'floor_kappa0.55', 'floor_kappa0.60', 'floor_kappa0.65']; floors have 3 seeds each |
| H3_real_gain | FAIL | real-data PERISH-REAL: FGC with fee minus without = 0.089 [0.060, 0.114]; needs the whole interval at or above 0.10 |
| H3_real_efficiency_cost | PASS | real-data PERISH-REAL: E with fee minus without = 0.091 [0.063, 0.104]; needs the whole interval at or above -0.05 |
| H5_real_delay_aware | FAIL (exploratory, 5 seeds) | real-data RURAL1-REAL-DTN: E delay-aware minus delay-unaware = -0.046 [-0.097, 0.013] |
| H4_delay_cost | PASS (exploratory, 3 seeds) | delay-unaware E at link uptime 100/60/40%: [0.647, 0.364, -0.0]; agreement [0.844, 0.589, 0.0] |
| H5_delay_aware | FAIL (exploratory, 3 seeds) | E delay-aware minus delay-unaware at 60% uptime = 0.015 [-0.031, 0.058] |
| H1_efficiency_3m | FAIL | 3M budget: SYM E 0.617 [0.603, 0.668]; needs the whole interval at or above 0.85 |
| H1_fair_split_3m | FAIL | 3M budget: SYM farmer share 0.242 [0.229, 0.353]; per seed [0.2, 0.87, 0.23, 0.24, 0.25, 0.24, 0.24, 0.22, 0.25, 0.25] |
| H2_PERISH_3m | PASS | 3M budget: farmer share PERISH minus SYM = -0.052 [-0.162, -0.017]; Mann-Whitney one-sided p = 0.0011; 0 of 10 PERISH runs undefined |
| H2_THIN3_3m | PASS (exploratory, 5 seeds) | 3M budget: farmer share THIN3 minus SYM = -0.144 [-0.257, -0.127]; Mann-Whitney one-sided p = 0.0003; 0 of 5 THIN3 runs undefined |
| H3_gain_3m | PASS | 3M budget: FGC with fee minus without = 0.190 [0.148, 0.211]; needs the whole interval at or above 0.10 |
| H3_efficiency_cost_3m | PASS | 3M budget: E with fee minus without = 0.110 [0.089, 0.120]; needs the whole interval at or above -0.05 |
| H3_beats_floor_3m | FAIL | 3M budget: price floor settings that match or beat the fee on both E and FGC (point estimates): ['floor_kappa0.60']; floors have 3 seeds each |
| H3_real_gain_3m | PASS | 3M budget, PERISH-REAL: FGC with fee minus without = 0.268 [0.199, 0.304]; needs the whole interval at or above 0.10 |
| H3_real_efficiency_cost_3m | PASS | 3M budget, PERISH-REAL: E with fee minus without = 0.155 [0.102, 0.163]; needs the whole interval at or above -0.05 |
