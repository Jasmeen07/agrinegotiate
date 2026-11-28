"""Train IPPO by self-play in one scenario for several seeds.

Usage: python experiments/x1_learned.py --scenario SYM --seeds 1000 1001 1002 --iters 150
Each run writes results/x1_learned/<scenario>/seed_<n>/{config.json,curve.json,checkpoint.pt,metrics.json}.
"""
import argparse, pathlib, sys
from multiprocessing import Pool

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agrinegotiate import scenarios                       # noqa: E402
from agrinegotiate.agents.ppo import PPOConfig, train     # noqa: E402


def job(args):
    scn, seed, iters, eps, tag = args
    out = ROOT / "results" / tag / scn / f"seed_{seed}"
    return train(scenarios.get(scn), PPOConfig(iters=iters, episodes_per_iter=eps), seed, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="SYM")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="x1_learned")
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        for s in pool.imap_unordered(job, [(a.scenario, sd, a.iters, a.episodes, a.tag) for sd in a.seeds]):
            print("finished", s["scenario"], s["seed"], s["last10_checkpoint_mean"], flush=True)
