"""Evaluate trained policies against scripted opponents (part of X6).

Usage: python experiments/crossplay_scripted.py SYM
For every seed folder in results/x1_learned/<scenario>, plays the learned farmer against each scripted
buyer and the learned buyer against each scripted farmer on the first 300 evaluation episodes.
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agrinegotiate import scenarios                         # noqa: E402
from agrinegotiate.agents import scripted as S              # noqa: E402
from agrinegotiate.agents.ppo import load_agents            # noqa: E402
from agrinegotiate.env.core import F, B                     # noqa: E402
from agrinegotiate.eval.evaluate import evaluate            # noqa: E402

OPP = ["ZIC", "Boulware", "Linear", "Conceder", "TitForTat"]

if __name__ == "__main__":
    scn = sys.argv[1]
    sc = scenarios.get(scn)
    base = ROOT / "results" / "x1_learned" / scn
    rows = []
    for d in sorted(base.glob("seed_*")):
        ag = load_agents(d / "checkpoint.pt", sc)
        for o in OPP + ["HardballBuyer"]:
            r = evaluate(sc, ag[F], S.make(o, B, sc), 300)
            rows.append(dict(seed=d.name, learned="farmer", opponent=o, E=r["E"], FGC=r["FGC"], BGC=r["BGC"],
                             agreement_rate=r["agreement_rate"]))
        for o in OPP:
            r = evaluate(sc, S.make(o, F, sc), ag[B], 300)
            rows.append(dict(seed=d.name, learned="buyer", opponent=o, E=r["E"], FGC=r["FGC"], BGC=r["BGC"],
                             agreement_rate=r["agreement_rate"]))
    (base / "crossplay_scripted.json").write_text(json.dumps(rows, indent=1))
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["learned"], r["opponent"])].append(r)
    print(f"{scn}: mean over {len(list(base.glob('seed_*')))} seeds, 300 episodes each")
    for (who, opp), rs in agg.items():
        own = "FGC" if who == "farmer" else "BGC"
        print(f"  learned {who:6s} vs {opp:13s} own gain capture {sum(r[own] for r in rs)/len(rs):.3f}  "
              f"E {sum(r['E'] for r in rs)/len(rs):.3f}  agreement {sum(r['agreement_rate'] for r in rs)/len(rs):.3f}")
