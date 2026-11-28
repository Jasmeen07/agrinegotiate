"""Collect everything the front end shows into dashboard/web/data.json and data.js.

    python dashboard/export_data.py

Reads results/ only: per-seed summaries are aggregated with the interquartile mean and a 95%
bootstrap interval, learning curves are averaged over seeds, and a few episodes are replayed
from saved checkpoints with fixed evaluation seeds. Nothing is trained here.
"""
import datetime, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                          # noqa: E402
from agrinegotiate.eval import replay, stats               # noqa: E402

RES = ROOT / "results"
SHOW = ("E", "FGC", "BGC", "S_F", "agreement_rate", "price_share", "fees_over_gstar", "lam_fair", "time_to_agreement")
CURVE = ("E", "FGC", "S_F", "agreement_rate", "lam_fair")


def group(folder: pathlib.Path) -> dict | None:
    runs = stats.load_runs(folder)
    if not runs:
        return None
    s = stats.summarise(folder)
    curves = [json.loads((d / "curve.json").read_text()) for d in sorted(folder.glob("seed_*")) if (d / "metrics.json").exists()]
    n = min(len(c) for c in curves)
    curve = []
    for k in range(n):
        row = {"iter": curves[0][k]["iter"]}
        for key in CURVE:
            vals = [c[k].get(key) for c in curves if c[k].get(key) is not None]
            row[key] = float(np.mean(vals)) if vals else None
        curve.append(row)
    cfg = json.loads((sorted(folder.glob("seed_*"))[0] / "config.json").read_text())
    return {"exp": folder.parent.name, "label": folder.name, "scenario": cfg["scenario"]["name"],
            "duty": cfg["scenario"]["duty"], "mechanism": cfg.get("mechanism"), "n_seeds": s["n_seeds"],
            "failed_runs": s["failed_runs"], "agent_steps": runs[0]["agent_steps"],
            "metrics": {k: {kk: vv for kk, vv in s[k].items()} for k in SHOW if k in s}, "curve": curve}


def first_seed(exp, label):
    d = sorted((RES / exp / label).glob("seed_*/checkpoint.pt"))
    return f"run:{exp}/{label}/{d[0].parent.name}" if d else None


def replays():
    out = []

    def add(title, note, scenario, farmer, buyer, seed, mechanism=None, overrides=None):
        if farmer is None or buyer is None:
            return
        try:
            ep = replay.record(scenario, farmer, buyer, seed, mechanism, overrides)
        except FileNotFoundError:
            return
        out.append({"title": title, "note": note, "episode": ep})

    sym, per = first_seed("x2_gate", "SYM"), first_seed("x2_gate", "PERISH")
    fee = first_seed("x3_mechanism", "fee_oracle_tau0.4")
    aware, unaware = first_seed("x4_delay", "aware_duty0.6"), first_seed("x4_delay", "unaware_duty0.6")
    for seed in (50001, 50007):
        add("Equal conditions, scripted Linear agents", "Both sides concede steadily.", "SYM", "Linear", "Linear", seed)
        add("Equal conditions, learned agents", "Self-interested PPO agents after self-play.", "SYM", sym, sym, seed)
        add("Perishable crop, hardball buyer against a Linear farmer", "The buyer never moves from just above the farmer's fallback.",
            "PERISH", "Linear", "HardballBuyer", seed)
        add("Perishable crop, learned agents, no mechanism", "What unconstrained learners do when the crop decays.",
            "PERISH", per, per, seed)
        add("Perishable crop, learned agents with the adaptive fee", "Buyers trained under the fee (target share 0.4).",
            "PERISH", fee, fee, seed, {"kind": "fee", "tau": 0.4})
        add("Rural, intermittent link, delay-aware learned agents", "Farmer link is up 60% of the time.",
            "RURAL1-DTN", aware, aware, seed)
        add("Rural, intermittent link, delay-unaware learned agents", "Same link, agents cannot see message age or link state.",
            "RURAL1-DTN", unaware, unaware, seed)
    return out


if __name__ == "__main__":
    groups = {}
    for folder in sorted(RES.glob("x[2-9]_*/*")):
        if folder.is_dir():
            g = group(folder)
            if g:
                groups[f"{g['exp']}/{g['label']}"] = g
    scripted = json.loads((RES / "x1_scripted" / "metrics.json").read_text())
    keep = ("scenario", "farmer", "buyer", "E", "FGC", "BGC", "S_F", "agreement_rate", "price_share", "phi_mean",
            "time_to_agreement")
    data = {
        "generated": datetime.date.today().isoformat(),
        "groups": groups,
        "scripted": [{k: r.get(k) for k in keep} for r in scripted["rows"]],
        "scripted_episodes": scripted["meta"]["episodes"],
        "replays": replays(),
    }
    web = ROOT / "dashboard" / "web"
    txt = json.dumps(data, separators=(",", ":"), default=float)
    (web / "data.json").write_text(txt)
    (web / "data.js").write_text("window.AGRI_DATA=" + txt + ";")
    print("groups:", len(groups), "replays:", len(data["replays"]), "bytes:", len(txt))
