# AgriNegotiate

A benchmark and learning code for fair, delay-tolerant price negotiation between farmer and
buyer agents. Built from the project build specification (kept alongside this repository).
Simulation only: nothing here has been tested with real farmers or traders.

## Layout

    agrinegotiate/yardsticks.py        closed-form reference points: G*, Nash price, Rubinstein shares
    agrinegotiate/scenarios.py         named scenarios (SYM, PERISH, DEADLINE, INFO, THIN, RURAL, RURAL-DTN)
    agrinegotiate/env/core.py          market, negotiation protocol, intermittent links, offer validity
    agrinegotiate/env/pettingzoo_env.py  PettingZoo ParallelEnv wrapper
    agrinegotiate/agents/scripted.py   ZI-C, Boulware, Linear, Conceder, tit-for-tat, hardball buyer, references
    agrinegotiate/agents/ppo.py        independent PPO with action masking and the adaptive fee hook
    agrinegotiate/mechanisms/fee.py    fee (oracle and observable target) and hard price floor
    agrinegotiate/metrics.py           E, FGC, BGC, S_F, agreement rate and secondary metrics
    agrinegotiate/eval/                fixed-seed evaluation, heuristic tuning
    agrinegotiate/data/                mandi price cleaning and calibration (not yet run on real data)
    experiments/                       one script per experiment
    results/                           outputs, one folder per run with its config
    tests/                             unit and protocol tests

## Quick start

    pip install -r requirements.txt
    python -m pytest -q
    python experiments/x1_scripted.py --episodes 1000        # scripted baseline table
    python experiments/x1_learned.py --scenario SYM --seeds 1000 1001 1002 --iters 150

## Experiments, report, figures, dashboard

    python experiments/run.py gate_core mechanism delay gate_more diagnosis mechanism_fast hetero   # resumable
    python experiments/make_report.py          # results/summary.md with hypothesis verdicts
    python experiments/make_figures.py         # paper/figures/*.png
    python dashboard/export_data.py            # dashboard/web/data.json and data.js
    uvicorn dashboard.api.main:app             # then open http://127.0.0.1:8000 (live episodes)

The dashboard also opens straight from `dashboard/web/index.html` with recorded episodes only.
The backend needs `pip install fastapi uvicorn`; figures need matplotlib.

## Seeds

Tuning episodes 0 to 99, training seeds 1000 to 1009, evaluation episodes 50000 to 50999.
Training episodes are drawn from 10,000,000 upward and never overlap the other ranges.

See PROGRESS.md for what is built and measured so far, and DECISIONS.md for every place
the code departs from or extends the specification.
