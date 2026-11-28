"""Generic experiment runner. A job is a dict; each run writes results/<exp>/<label>/seed_<n>/.

    python experiments/run.py <batch> [--workers 2]

Batches are defined in BATCHES below. Finished runs (metrics.json present) are skipped,
so a batch can be restarted safely.
"""
import argparse, pathlib, sys
from multiprocessing import Pool

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agrinegotiate import scenarios                                   # noqa: E402
from agrinegotiate.agents.ppo import PPOConfig, train                 # noqa: E402
from agrinegotiate.mechanisms.fee import FeeMechanism, FloorMechanism  # noqa: E402

S10 = list(range(1000, 1010))
S5 = list(range(1000, 1005))
S3 = list(range(1000, 1003))
KAPPA_TAU04_PERISH = 0.606     # average oracle target price / (P_ref * rho^t) at tau = 0.4
STEPS_3M = 3_000_000           # planned per-run budget in agent-steps


def J(exp, label, scenario, seeds, **kw):
    return [dict(exp=exp, label=label, scenario=scenario, seed=s, **kw) for s in seeds]


def mech_jobs(exp, scn, tau_main_seeds=S10):
    jobs = J(exp, "fee_oracle_tau0.4", scn, tau_main_seeds, mech="fee_adaptive", tau=0.4, iters=150)
    for tau in (0.2, 0.3, 0.5):
        jobs += J(exp, f"fee_oracle_tau{tau}", scn, S3, mech="fee_adaptive", tau=tau, iters=150)
    for kappa in (0.50, 0.55, 0.60, 0.65):
        jobs += J(exp, f"floor_kappa{kappa:.2f}", scn, S3, mech="floor", kappa=kappa, iters=150)
    jobs += J(exp, "fee_fixed_lam1.0_tau0.4", scn, S3, mech="fee_fixed", tau=0.4, lam=1.0, iters=150)
    jobs += J(exp, "fee_observable_tau0.4", scn, S3, mech="fee_adaptive", tau=0.4, observable=True,
              kappa=KAPPA_TAU04_PERISH, iters=150)
    return jobs


def delay_jobs(exp):
    jobs = []
    cells = {"unaware": dict(recurrent=False, delay_aware=False), "aware": dict(recurrent=True, delay_aware=True)}
    for duty in (1.0, 0.6, 0.4):
        for name, kw in cells.items():
            jobs += J(exp, f"{name}_duty{duty}", "RURAL1-DTN", S3, sc=dict(duty=duty), **kw)
    # 2 x 2 ablation at duty 0.6: memory only, features only
    jobs += J(exp, "memory_only_duty0.6", "RURAL1-DTN", S3, sc=dict(duty=0.6), recurrent=True, delay_aware=False)
    jobs += J(exp, "features_only_duty0.6", "RURAL1-DTN", S3, sc=dict(duty=0.6), recurrent=False, delay_aware=True)
    return jobs


BATCHES = {
    "gate_core": J("x2_gate", "PERISH", "PERISH", S10) + J("x2_gate", "SYM", "SYM", S10),
    "mechanism": mech_jobs("x3_mechanism", "PERISH"),
    "delay": delay_jobs("x4_delay"),
    "gate_more": J("x2_gate", "DEADLINE", "DEADLINE", S5) + J("x2_gate", "INFO", "INFO", S5)
                 + J("x2_gate", "THIN3", "THIN3", S5),
    # Added after seeing that the fee rate was still rising at the end of the main mechanism runs:
    # the same adaptive fee with a five times larger dual step size. Post hoc, so exploratory.
    "mechanism_fast": J("x3_mechanism", "fee_oracle_tau0.4_eta0.25", "PERISH", S5, mech="fee_adaptive", tau=0.4,
                        eta=0.25, iters=150),
    # Heterogeneous buyer values: does the floor still win when it must exclude low-value buyers? Post hoc.
    "hetero": J("x6_hetero", "unconstrained", "PERISH-HET", S5)
              + J("x6_hetero", "fee_oracle_tau0.4_eta0.25", "PERISH-HET", S5, mech="fee_adaptive", tau=0.4, eta=0.25, iters=150)
              + J("x6_hetero", "fee_fixed_lam1.0_tau0.4", "PERISH-HET", S3, mech="fee_fixed", tau=0.4, lam=1.0, iters=150)
              + sum((J("x6_hetero", f"floor_kappa{k:.2f}", "PERISH-HET", S3, mech="floor", kappa=k, iters=150)
                     for k in (0.50, 0.55, 0.60)), []),
    # Confirmation of the PERISH-HET lead at 10 seeds (fee vs floors vs no rule). Same settings as "hetero".
    "hetero10": J("x6_hetero", "unconstrained", "PERISH-HET", S10)
                + J("x6_hetero", "fee_oracle_tau0.4_eta0.25", "PERISH-HET", S10, mech="fee_adaptive", tau=0.4, eta=0.25, iters=150)
                + sum((J("x6_hetero", f"floor_kappa{k:.2f}", "PERISH-HET", S10, mech="floor", kappa=k, iters=150)
                       for k in (0.50, 0.55, 0.60)), []),
    # Scenarios calibrated on real Agmarknet lemon prices (data/real): price level, information noise and mandi-to-mandi spread.
    "real": J("x7_real", "unconstrained", "PERISH-REAL", S10)
            + J("x7_real", "fee_oracle_tau0.4_eta0.25", "PERISH-REAL", S10, mech="fee_adaptive", tau=0.4, eta=0.25, iters=150)
            + J("x7_real", "floor_kappa0.55", "PERISH-REAL", S5, mech="floor", kappa=0.55, iters=150)
            + J("x7_real", "aware_duty0.6", "RURAL1-REAL-DTN", S5, recurrent=True, delay_aware=True)
            + J("x7_real", "unaware_duty0.6", "RURAL1-REAL-DTN", S5, recurrent=False, delay_aware=False),
    "diagnosis": J("x5_diagnosis", "SYM_mix0", "SYM", S3, mix=0.0) + J("x5_diagnosis", "SYMX", "SYMX", S3),
    # The main comparisons rerun at the planned budget of 3 million agent-steps per run (the runs above
    # used 0.3 to 0.7 million). Same scenarios, settings and seeds; written to x8_3m_<exp>/.
    "long3m": [dict(j, exp="x8_3m_" + j["exp"].split("_", 1)[1], steps=STEPS_3M) for j in
               J("x2_gate", "SYM", "SYM", S10) + J("x2_gate", "PERISH", "PERISH", S10)
               + J("x2_gate", "THIN3", "THIN3", S5)
               + J("x3_mechanism", "fee_oracle_tau0.4", "PERISH", S10, mech="fee_adaptive", tau=0.4)
               + J("x3_mechanism", "fee_fixed_lam1.0_tau0.4", "PERISH", S3, mech="fee_fixed", tau=0.4, lam=1.0)
               + sum((J("x3_mechanism", f"floor_kappa{k:.2f}", "PERISH", S3, mech="floor", kappa=k)
                      for k in (0.50, 0.55, 0.60, 0.65)), [])
               + J("x6_hetero", "unconstrained", "PERISH-HET", S10)
               + J("x6_hetero", "fee_oracle_tau0.4_eta0.25", "PERISH-HET", S10, mech="fee_adaptive", tau=0.4, eta=0.25)
               + sum((J("x6_hetero", f"floor_kappa{k:.2f}", "PERISH-HET", S10, mech="floor", kappa=k)
                      for k in (0.50, 0.55, 0.60)), [])
               + J("x7_real", "unconstrained", "PERISH-REAL", S10)
               + J("x7_real", "fee_oracle_tau0.4_eta0.25", "PERISH-REAL", S10, mech="fee_adaptive", tau=0.4, eta=0.25)
               + J("x7_real", "floor_kappa0.55", "PERISH-REAL", S5, mech="floor", kappa=0.55)],
}
# The full long3m set needs about 10 hours here (40-50 minutes per run, 12 at a time). The core subset keeps
# every main comparison: no fixed fee or outer floors in PERISH, 5 seeds in PERISH-HET and PERISH-REAL.
BATCHES["long3m_core"] = [
    j for j in BATCHES["long3m"]
    if j["label"] not in ("fee_fixed_lam1.0_tau0.4",)
    and not (j["exp"] == "x8_3m_mechanism" and j["label"] in ("floor_kappa0.50", "floor_kappa0.65"))
    and not (j["exp"] == "x8_3m_hetero" and j["label"] == "floor_kappa0.55")
    and not (j["exp"] in ("x8_3m_hetero", "x8_3m_real") and j["seed"] not in S5)]


def run_job(job):
    out = ROOT / "results" / job["exp"] / job["label"] / f"seed_{job['seed']}"
    if (out / "metrics.json").exists():
        return job, None
    sc = scenarios.get(job["scenario"]).with_(**job.get("sc", {}))
    cfg = PPOConfig(iters=job.get("iters", 120), mix_scripted=job.get("mix", 0.3),
                    recurrent=job.get("recurrent", False), delay_aware=job.get("delay_aware", True),
                    target_agent_steps=job.get("steps", 0))
    mech, adaptive = None, False
    kind = job.get("mech")
    if kind == "floor":
        mech = FloorMechanism(job["kappa"])
    elif kind in ("fee_adaptive", "fee_fixed"):
        mech = FeeMechanism(tau=job["tau"], lam=job.get("lam", 0.0), observable=job.get("observable", False),
                            kappa=job.get("kappa"), eta=job.get("eta", 0.05))
        adaptive = kind == "fee_adaptive"
    summary = train(sc, cfg, job["seed"], out, mech=mech, adaptive_fee=adaptive, log=lambda *a: None)
    return job, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    jobs = [j for b in a.batches for j in BATCHES[b]]
    print(len(jobs), "jobs", flush=True)
    with Pool(a.workers) as pool:
        for job, s in pool.imap_unordered(run_job, jobs, chunksize=1):
            m = "skipped" if s is None else {k: (v and round(v, 3)) for k, v in s["last10_checkpoint_mean"].items()
                                              if k in ("E", "FGC", "S_F", "agreement_rate", "lam_fair")}
            print("done", job["exp"], job["label"], job["seed"], m, flush=True)
