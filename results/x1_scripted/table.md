# X1 scripted baselines (1000 fixed evaluation episodes per cell, seeds 50000+)

Scripted agents are deterministic given the seed, so there are no training seeds and no
confidence intervals here. E can be negative: waiting and then failing is worse than walking away.

## SYM  (tuned beta: farmer 0.2, buyer 0.2)

| Farmer | Buyer | E | FGC | S_F | Agreement | Price share | Mean phi | Steps to deal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TakeFallback | TakeFallback | 0.000 | 0.000 | n/a | 0.000 | 0.450 | n/a | n/a |
| OracleFairSplit | OracleFairSplit | 0.977 | 0.490 | 0.502 | 1.000 | 0.648 | 0.502 | 2.0 |
| ZIC | ZIC | 0.900 | 0.457 | 0.508 | 0.982 | 0.646 | 0.507 | 7.0 |
| Boulware | Boulware | 0.676 | 0.433 | 0.641 | 1.000 | 0.689 | 0.602 | 28.7 |
| Linear | Linear | 0.736 | 0.441 | 0.599 | 1.000 | 0.679 | 0.579 | 23.3 |
| Conceder | Conceder | 0.895 | 0.521 | 0.582 | 1.000 | 0.679 | 0.579 | 9.3 |
| TitForTat | TitForTat | 0.678 | 0.475 | 0.699 | 1.000 | 0.708 | 0.652 | 28.5 |
| Tuned | Tuned | 0.676 | 0.433 | 0.641 | 1.000 | 0.689 | 0.602 | 28.7 |
| Boulware | HardballBuyer | 0.640 | 0.047 | 0.073 | 0.965 | 0.534 | 0.212 | 29.0 |
| Tuned | HardballBuyer | 0.640 | 0.047 | 0.073 | 0.965 | 0.534 | 0.212 | 29.0 |

## PERISH  (tuned beta: farmer 30, buyer 10)

| Farmer | Buyer | E | FGC | S_F | Agreement | Price share | Mean phi | Steps to deal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TakeFallback | TakeFallback | 0.000 | 0.000 | n/a | 0.000 | 0.450 | n/a | n/a |
| OracleFairSplit | OracleFairSplit | 0.836 | 0.382 | 0.456 | 1.000 | 0.605 | 0.497 | 2.0 |
| ZIC | ZIC | -0.026 | -0.245 | n/a | 0.727 | 0.385 | 0.420 | 9.7 |
| Boulware | Boulware | -0.748 | -0.670 | n/a | 0.944 | 0.249 | 0.647 | 29.1 |
| Linear | Linear | -0.716 | -0.663 | n/a | 0.959 | 0.251 | 0.591 | 28.4 |
| Conceder | Conceder | -0.351 | -0.429 | n/a | 0.987 | 0.323 | 0.493 | 21.2 |
| TitForTat | TitForTat | -0.747 | -0.667 | n/a | 0.959 | 0.251 | 0.656 | 29.2 |
| Tuned | Tuned | 0.751 | 0.329 | 0.438 | 1.000 | 0.587 | 0.508 | 3.1 |
| Boulware | HardballBuyer | -0.747 | -0.667 | n/a | 0.959 | 0.251 | 0.656 | 29.2 |
| Tuned | HardballBuyer | 0.650 | 0.101 | 0.155 | 1.000 | 0.501 | 0.322 | 4.5 |

## DEADLINE  (tuned beta: farmer 2, buyer 2)

| Farmer | Buyer | E | FGC | S_F | Agreement | Price share | Mean phi | Steps to deal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TakeFallback | TakeFallback | 0.000 | 0.000 | n/a | 0.000 | 0.450 | n/a | n/a |
| OracleFairSplit | OracleFairSplit | 0.977 | 0.490 | 0.502 | 1.000 | 0.648 | 0.502 | 2.0 |
| ZIC | ZIC | 0.796 | 0.404 | 0.508 | 0.858 | 0.622 | 0.505 | 5.3 |
| Boulware | Boulware | -0.167 | -0.084 | n/a | 0.000 | 0.450 | n/a | n/a |
| Linear | Linear | 0.543 | 0.034 | 0.063 | 0.684 | 0.495 | 0.160 | 15.3 |
| Conceder | Conceder | 0.927 | 0.425 | 0.458 | 1.000 | 0.634 | 0.463 | 6.4 |
| TitForTat | TitForTat | 0.829 | 0.329 | 0.398 | 0.994 | 0.624 | 0.435 | 14.6 |
| Tuned | Tuned | 0.865 | 0.239 | 0.276 | 0.998 | 0.571 | 0.303 | 11.8 |
| Boulware | HardballBuyer | 0.809 | 0.131 | 0.162 | 0.965 | 0.534 | 0.212 | 14.0 |
| Tuned | HardballBuyer | 0.827 | 0.140 | 0.170 | 0.965 | 0.534 | 0.212 | 12.4 |

## INFO  (tuned beta: farmer 0.2, buyer 0.2)

| Farmer | Buyer | E | FGC | S_F | Agreement | Price share | Mean phi | Steps to deal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TakeFallback | TakeFallback | 0.000 | 0.000 | n/a | 0.000 | 0.450 | n/a | n/a |
| OracleFairSplit | OracleFairSplit | 0.977 | 0.490 | 0.502 | 1.000 | 0.648 | 0.502 | 2.0 |
| ZIC | ZIC | 0.900 | 0.459 | 0.510 | 0.982 | 0.647 | 0.508 | 7.0 |
| Boulware | Boulware | 0.676 | 0.440 | 0.651 | 1.000 | 0.691 | 0.608 | 28.7 |
| Linear | Linear | 0.736 | 0.440 | 0.598 | 1.000 | 0.679 | 0.578 | 23.3 |
| Conceder | Conceder | 0.895 | 0.520 | 0.581 | 1.000 | 0.678 | 0.578 | 9.3 |
| TitForTat | TitForTat | 0.678 | 0.477 | 0.703 | 1.000 | 0.709 | 0.654 | 28.5 |
| Tuned | Tuned | 0.676 | 0.440 | 0.651 | 1.000 | 0.691 | 0.608 | 28.7 |
| Boulware | HardballBuyer | 0.640 | 0.047 | 0.073 | 0.965 | 0.534 | 0.212 | 29.0 |
| Tuned | HardballBuyer | 0.640 | 0.047 | 0.073 | 0.965 | 0.534 | 0.212 | 29.0 |

## THIN  (tuned beta: farmer 0.2, buyer 0.2)

| Farmer | Buyer | E | FGC | S_F | Agreement | Price share | Mean phi | Steps to deal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TakeFallback | TakeFallback | 0.000 | 0.000 | n/a | 0.000 | 0.450 | n/a | n/a |
| OracleFairSplit | OracleFairSplit | 0.979 | 0.490 | 0.500 | 1.000 | 0.558 | 0.500 | 2.0 |
| ZIC | ZIC | 0.882 | 0.437 | 0.496 | 1.000 | 0.552 | 0.506 | 2.6 |
| Boulware | Boulware | 0.509 | 0.273 | 0.537 | 1.000 | 0.574 | 0.605 | 28.5 |
| Linear | Linear | 0.621 | 0.327 | 0.527 | 1.000 | 0.573 | 0.584 | 22.7 |
| Conceder | Conceder | 0.854 | 0.483 | 0.566 | 1.000 | 0.575 | 0.586 | 8.2 |
| TitForTat | TitForTat | 0.517 | 0.326 | 0.631 | 1.000 | 0.587 | 0.666 | 28.2 |
| Tuned | Tuned | 0.509 | 0.273 | 0.537 | 1.000 | 0.574 | 0.605 | 28.5 |
| Boulware | HardballBuyer | 0.501 | -0.075 | n/a | 1.000 | 0.499 | 0.231 | 28.9 |
| Tuned | HardballBuyer | 0.501 | -0.075 | n/a | 1.000 | 0.499 | 0.231 | 28.9 |

## RURAL  (tuned beta: farmer 10, buyer 1)

| Farmer | Buyer | E | FGC | S_F | Agreement | Price share | Mean phi | Steps to deal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TakeFallback | TakeFallback | 0.000 | 0.000 | n/a | 0.000 | 0.450 | n/a | n/a |
| OracleFairSplit | OracleFairSplit | 0.824 | 0.369 | 0.448 | 1.000 | 0.537 | 0.499 | 2.0 |
| ZIC | ZIC | 0.173 | -0.258 | n/a | 0.967 | 0.443 | 0.418 | 4.2 |
| Boulware | Boulware | -1.385 | -1.370 | n/a | 0.210 | 0.280 | 0.106 | 17.1 |
| Linear | Linear | -0.814 | -1.145 | n/a | 0.999 | 0.318 | 0.249 | 14.2 |
| Conceder | Conceder | 0.018 | -0.274 | n/a | 1.000 | 0.448 | 0.559 | 7.0 |
| TitForTat | TitForTat | -0.786 | -0.975 | n/a | 1.000 | 0.351 | 0.502 | 13.5 |
| Tuned | Tuned | -0.588 | -1.021 | n/a | 0.998 | 0.330 | 0.184 | 12.2 |
| Boulware | HardballBuyer | -0.791 | -1.038 | n/a | 1.000 | 0.339 | 0.401 | 13.6 |
| Tuned | HardballBuyer | 0.162 | -0.298 | n/a | 1.000 | 0.436 | 0.387 | 5.8 |

## RURAL-DTN  (tuned beta: farmer 10, buyer 1)

| Farmer | Buyer | E | FGC | S_F | Agreement | Price share | Mean phi | Steps to deal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TakeFallback | TakeFallback | 0.000 | 0.000 | n/a | 0.000 | 0.450 | n/a | n/a |
| OracleFairSplit | OracleFairSplit | 0.687 | 0.314 | 0.457 | 0.995 | 0.532 | 0.553 | 3.6 |
| ZIC | ZIC | -0.329 | -0.647 | n/a | 0.825 | 0.390 | 0.426 | 5.9 |
| Boulware | Boulware | -1.418 | -1.375 | n/a | 0.146 | 0.279 | 0.120 | 17.5 |
| Linear | Linear | -0.969 | -1.203 | n/a | 0.885 | 0.312 | 0.303 | 15.3 |
| Conceder | Conceder | -0.121 | -0.388 | n/a | 1.000 | 0.432 | 0.565 | 7.9 |
| TitForTat | TitForTat | -0.981 | -1.131 | n/a | 0.887 | 0.328 | 0.466 | 15.1 |
| Tuned | Tuned | -0.710 | -1.083 | n/a | 0.974 | 0.324 | 0.215 | 13.1 |
| Boulware | HardballBuyer | -0.946 | -1.135 | n/a | 0.913 | 0.325 | 0.415 | 14.8 |
| Tuned | HardballBuyer | -0.009 | -0.428 | n/a | 1.000 | 0.419 | 0.396 | 7.0 |
