"""Experiment X1 (scripted part): baseline table for every named scenario.

Usage: python experiments/x1_scripted.py [--episodes 1000] [--workers 2]
Writes results/x1_scripted/{metrics.json, tuning.json, table.md}.
"""
import argparse, json, pathlib, subprocess, sys, time
from multiprocessing import Pool

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agrinegotiate import scenarios                      # noqa: E402
from agrinegotiate.agents import scripted as S           # noqa: E402
from agrinegotiate.env.core import F, B                  # noqa: E402
from agrinegotiate.eval.evaluate import evaluate, EVAL_SEED_START   # noqa: E402
from agrinegotiate.eval.tuning import tune               # noqa: E402

NAMED = ["SYM", "PERISH", "DEADLINE", "INFO", "THIN", "RURAL", "RURAL-DTN"]
PAIRS = [("TakeFallback", "TakeFallback"), ("OracleFairSplit", "OracleFairSplit"), ("ZIC", "ZIC"),
         ("Boulware", "Boulware"), ("Linear", "Linear"), ("Conceder", "Conceder"),
         ("TitForTat", "TitForTat"), ("Tuned", "Tuned"), ("Boulware", "HardballBuyer"),
         ("Tuned", "HardballBuyer")]


def tune_job(name):
    return name, tune(scenarios.get(name))


def eval_job(args):
    name, fn, bn, tuned, n = args
    sc = scenarios.get(name)
    mk = lambda n_, role: S.make(f"TD:{tuned['farmer_beta' if role == F else 'buyer_beta']}", role, sc) \
        if n_ == "Tuned" else S.make(n_, role, sc)
    r = evaluate(sc, mk(fn, F), mk(bn, B), n, seed_start=EVAL_SEED_START)
    return dict(scenario=name, farmer=fn, buyer=bn, **r)


def fmt(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def render(out, rows, tuned, episodes):
    lines = [f"# X1 scripted baselines ({episodes} fixed evaluation episodes per cell, seeds {EVAL_SEED_START}+)", "",
             "Scripted agents are deterministic given the seed, so there are no training seeds and no",
             "confidence intervals here. E can be negative: waiting and then failing is worse than walking away.", ""]
    for n in NAMED:
        lines += [f"## {n}  (tuned beta: farmer {tuned[n]['farmer_beta']:g}, buyer {tuned[n]['buyer_beta']:g})", "",
                  "| Farmer | Buyer | E | FGC | S_F | Agreement | Price share | Mean phi | Steps to deal |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            if r["scenario"] == n:
                lines.append(f"| {r['farmer']} | {r['buyer']} | {fmt(r['E'])} | {fmt(r['FGC'])} | {fmt(r['S_F'])} | "
                             f"{fmt(r['agreement_rate'])} | {fmt(r['price_share'])} | {fmt(r['phi_mean'])} | "
                             f"{fmt(r['time_to_agreement'], 1)} |")
        lines.append("")
    (out / "table.md").write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = ROOT / "results" / "x1_scripted"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        tuned = dict(pool.map(tune_job, NAMED))
        (out / "tuning.json").write_text(json.dumps(tuned, indent=1))
        jobs = [(n, f, b, tuned[n], a.episodes) for n in NAMED for f, b in PAIRS]
        rows = pool.map(eval_job, jobs, chunksize=1)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    meta = dict(episodes=a.episodes, eval_seed_start=EVAL_SEED_START, commit=commit or None,
                seconds=round(time.time() - t0, 1), scenarios={n: scenarios.get(n).to_dict() for n in NAMED})
    (out / "metrics.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))
    render(out, rows, tuned, a.episodes)
    print("done in", meta["seconds"], "s ->", out)
