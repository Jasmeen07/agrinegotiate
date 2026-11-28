"""Build results/summary.md and results/summary.json from the run folders: every table with
interquartile means and 95% bootstrap intervals, and a programmatic verdict on each hypothesis.

    python experiments/make_report.py
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy.stats import mannwhitneyu                 # noqa: E402
from agrinegotiate.eval import stats                 # noqa: E402

RES = ROOT / "results"
MECH_COLS = ("E", "FGC", "S_F", "agreement_rate", "lam_fair", "fees_over_gstar")


def load(exp, label):
    f = RES / exp / label
    return stats.summarise(f) if list(f.glob("seed_*/metrics.json")) else None


def subdirs(exp):
    return sorted(p.name for p in (RES / exp).glob("*") if p.is_dir()) if (RES / exp).exists() else []


def cell(s, k):
    if s is None or "iqm" not in s[k]:
        return "n/a"
    return f"{s[k]['iqm']:.3f} [{s[k]['ci95'][0]:.3f}, {s[k]['ci95'][1]:.3f}]"


def point(s, k):
    return "n/a" if s is None or "iqm" not in s[k] else f"{s[k]['iqm']:.3f}"


def defined(s, k):
    return [v for v in s[k]["per_seed"] if v is not None]


def per_seed(s, k):
    return [round(v, 2) for v in defined(s, k)]


def span(d):
    return f"{d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]"


def table(title, rows, cols=("E", "FGC", "S_F", "agreement_rate")):
    names = {"E": "Efficiency", "FGC": "Farmer gain capture", "S_F": "Farmer share", "agreement_rate": "Agreement",
             "lam_fair": "Fee rate", "fees_over_gstar": "Fees / G*", "price_share": "Price share"}
    out = [f"### {title}", "", "| Setting | Seeds | Failed | " + " | ".join(names[c] for c in cols) + " |",
           "| --- | --- | --- | " + " | ".join("---" for _ in cols) + " |"]
    for name, s in rows:
        if s is not None:
            out.append(f"| {name} | {s['n_seeds']} | {s['failed_runs']} | " + " | ".join(cell(s, c) for c in cols) + " |")
    return out + [""]


# Hypothesis checks. Each one writes its verdicts into V. The same checks run on the short runs and
# on the 3M-step rerun: `sfx` is "_3m" for the rerun and `pre` starts the evidence text.

def check_h1(V, sym, sfx="", pre=""):
    """H1: in SYM learners are efficient (E >= 0.85) and split the gains fairly (farmer share in [0.40, 0.60])."""
    lo, hi = sym["S_F"]["ci95"]
    V["H1_efficiency" + sfx] = dict(passed=bool(sym["E"]["ci95"][0] >= 0.85),
                                    detail=f"{pre}SYM E {cell(sym, 'E')}; needs the whole interval at or above 0.85")
    V["H1_fair_split" + sfx] = dict(passed=bool(lo >= 0.40 and hi <= 0.60),
                                    detail=f"{pre}SYM farmer share {cell(sym, 'S_F')}; per seed {per_seed(sym, 'S_F')}")


def check_h2(V, k, g, sym, sfx="", pre=""):
    """H2: the farmer's share in scenario k is lower than in SYM."""
    if g is None or not defined(g, "S_F"):
        return
    a, b = defined(g, "S_F"), defined(sym, "S_F")
    d = stats.iqm_diff_ci(a, b)
    p = mannwhitneyu(a, b, alternative="less").pvalue
    V[f"H2_{k}{sfx}"] = dict(passed=bool(d[2] < 0), seeds=len(a),
                             detail=f"{pre}farmer share {k} minus SYM = {span(d)}; Mann-Whitney one-sided p = {p:.4f}; "
                                    f"{g['S_F']['undefined']} of {g['n_seeds']} {k} runs undefined")


def check_fee(V, fee, base, name="", sfx="", pre=""):
    """H3: the fee raises farmer gain capture by at least 0.10 and costs at most 0.05 efficiency."""
    dF = stats.iqm_diff_ci(fee["FGC"]["per_seed"], base["FGC"]["per_seed"])
    dE = stats.iqm_diff_ci(fee["E"]["per_seed"], base["E"]["per_seed"])
    V[f"H3_{name}gain{sfx}"] = dict(
        passed=bool(dF[1] >= 0.10), detail=f"{pre}FGC with fee minus without = {span(dF)}; needs the whole interval at or above 0.10")
    V[f"H3_{name}efficiency_cost{sfx}"] = dict(
        passed=bool(dE[1] >= -0.05), detail=f"{pre}E with fee minus without = {span(dE)}; needs the whole interval at or above -0.05")


def check_floors(V, fee, floors, sfx="", pre=""):
    """H3: no price floor matches the fee on both efficiency and farmer gain (point estimates)."""
    dom = [m for m, s in floors if s and s["FGC"]["iqm"] >= fee["FGC"]["iqm"] and s["E"]["iqm"] >= fee["E"]["iqm"]]
    V["H3_beats_floor" + sfx] = dict(passed=bool(not dom), detail=pre + (
        "no price floor setting matches the fee on both E and FGC" if not dom else
        f"price floor settings that match or beat the fee on both E and FGC (point estimates): {dom}") + "; floors have 3 seeds each")


def check_h5(V, key, aware, unaware, seeds, what):
    """H5: delay-aware learners are more efficient than delay-unaware ones."""
    d = stats.iqm_diff_ci(aware["E"]["per_seed"], unaware["E"]["per_seed"])
    V[key] = dict(passed=bool(d[1] > 0), seeds=seeds, detail=f"{what} = {span(d)}")


if __name__ == "__main__":
    L, V = ["# Results summary", "",
            "Interquartile mean over training seeds with a 95% bootstrap interval. Each seed's score is the mean of",
            "its last 10 evaluation checkpoints (300 fixed evaluation episodes each). A failed run ends with E below 0.1.",
            "Results with fewer than 10 seeds are exploratory.", ""], {}

    # gate
    gate = {k: load("x2_gate", k) for k in ("SYM", "PERISH", "DEADLINE", "INFO", "THIN3")}
    L += table("Unconstrained learners by scenario (X2)", gate.items())
    sym = gate["SYM"]
    if sym:
        check_h1(V, sym)
        for k in ("PERISH", "DEADLINE", "INFO", "THIN3"):
            check_h2(V, k, gate[k], sym)
    diag = [(k, load("x5_diagnosis", k)) for k in ("SYM_mix0", "SYMX")]
    L += table("SYM diagnosis (X5, exploratory)", diag)
    L += [f"- {k} farmer share per seed: {per_seed(s, 'S_F')}" for k, s in diag if s] + [""]

    # mechanism
    base = gate["PERISH"]
    mech = subdirs("x3_mechanism")
    L += table("Mechanisms in PERISH (X3)", [("Unconstrained", base)] + [(m, load("x3_mechanism", m)) for m in mech], MECH_COLS)
    fee = load("x3_mechanism", "fee_oracle_tau0.4")
    if fee and base:
        check_fee(V, fee, base)
        check_floors(V, fee, [(m, load("x3_mechanism", m)) for m in mech if m.startswith("floor")])

    L += table("Mechanisms when buyer values vary widely, PERISH-HET (X6, post hoc, exploratory)",
               [(m, load("x6_hetero", m)) for m in subdirs("x6_hetero")], MECH_COLS)

    # scenarios calibrated on real mandi prices
    real = {k: load("x7_real", k) for k in ("unconstrained", "fee_oracle_tau0.4_eta0.25", "floor_kappa0.55", "unaware_duty0.6", "aware_duty0.6")}
    L += table("Scenarios calibrated on real Agmarknet lemon prices (X7)", real.items(), ("E", "FGC", "S_F", "agreement_rate", "lam_fair"))
    if real["unconstrained"] and real["fee_oracle_tau0.4_eta0.25"]:
        check_fee(V, real["fee_oracle_tau0.4_eta0.25"], real["unconstrained"], "real_", pre="real-data PERISH-REAL: ")
    if real["aware_duty0.6"] and real["unaware_duty0.6"]:
        check_h5(V, "H5_real_delay_aware", real["aware_duty0.6"], real["unaware_duty0.6"], 5,
                 "real-data RURAL1-REAL-DTN: E delay-aware minus delay-unaware")

    # delay
    dl = {k: load("x4_delay", k) for k in ("unaware_duty1.0", "unaware_duty0.6", "unaware_duty0.4", "aware_duty1.0", "aware_duty0.6",
                                          "aware_duty0.4", "memory_only_duty0.6", "features_only_duty0.6")}
    L += table("Delay in RURAL1-DTN (X4, 3 seeds per cell, exploratory)", dl.items())
    u = [dl[f"unaware_duty{d}"] for d in ("1.0", "0.6", "0.4")]
    if all(u):
        es, ags = [s["E"]["iqm"] for s in u], [s["agreement_rate"]["iqm"] for s in u]
        V["H4_delay_cost"] = dict(passed=bool(es[0] > es[1] > es[2] and ags[0] > ags[1] > ags[2]), seeds=3,
                                  detail=f"delay-unaware E at link uptime 100/60/40%: {[round(v, 3) for v in es]}; agreement {[round(v, 3) for v in ags]}")
    if dl["aware_duty0.6"] and dl["unaware_duty0.6"]:
        check_h5(V, "H5_delay_aware", dl["aware_duty0.6"], dl["unaware_duty0.6"], 3, "E delay-aware minus delay-unaware at 60% uptime")

    # budget check: the main comparisons rerun at 3 million agent-steps per run (X8)
    pairs = [("gate", "x2_gate", k) for k in ("SYM", "PERISH", "THIN3")] \
        + [("mechanism", "x3_mechanism", k) for k in ("fee_oracle_tau0.4", "fee_fixed_lam1.0_tau0.4", "floor_kappa0.50",
                                                      "floor_kappa0.55", "floor_kappa0.60", "floor_kappa0.65")] \
        + [("hetero", "x6_hetero", k) for k in ("unconstrained", "fee_oracle_tau0.4_eta0.25", "floor_kappa0.50",
                                                "floor_kappa0.55", "floor_kappa0.60")] \
        + [("real", "x7_real", k) for k in ("unconstrained", "fee_oracle_tau0.4_eta0.25", "floor_kappa0.55")]
    long = {(g, k): load(f"x8_3m_{g}", k) for g, _, k in pairs}
    if any(long.values()):
        steps = [json.loads(f.read_text())["agent_steps"] for f in RES.glob("x8_3m_*/*/seed_*/metrics.json")]
        L += ["### Budget check: main comparisons at 3 million agent-steps per run (X8)", "",
              f"Same scenarios, settings and seeds as above; {len(steps)} runs, "
              f"{min(steps) / 1e6:.2f} to {max(steps) / 1e6:.2f} million agent-steps each. Short = the runs above (0.3 to 0.7 million).", "",
              "| Experiment | Setting | Seeds | E short | E 3M | Farmer gain short | Farmer gain 3M | Farmer share short | Farmer share 3M | Agreement 3M | Failed 3M |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for g, exp, k in pairs:
            s, l = load(exp, k), long[(g, k)]
            if l is not None:
                L.append(f"| {g} | {k} | {l['n_seeds']} | {point(s, 'E')} | {cell(l, 'E')} | {point(s, 'FGC')} | {cell(l, 'FGC')} | "
                         f"{point(s, 'S_F')} | {cell(l, 'S_F')} | {cell(l, 'agreement_rate')} | {l['failed_runs']} |")
        L += [""]
        sym3, per3 = long[("gate", "SYM")], long[("gate", "PERISH")]
        if sym3:
            L += [f"- SYM farmer share per seed at 3M: {per_seed(sym3, 'S_F')}", ""]
            check_h1(V, sym3, "_3m", "3M budget: ")
            for k in ("PERISH", "THIN3"):
                check_h2(V, k, long[("gate", k)], sym3, "_3m", "3M budget: ")
        fee3 = long[("mechanism", "fee_oracle_tau0.4")]
        if fee3 and per3:
            check_fee(V, fee3, per3, "", "_3m", "3M budget: ")
            check_floors(V, fee3, [(k, s) for (g, k), s in long.items() if g == "mechanism" and k.startswith("floor")],
                         "_3m", "3M budget: ")
        u3, f3 = long[("real", "unconstrained")], long[("real", "fee_oracle_tau0.4_eta0.25")]
        if u3 and f3:
            check_fee(V, f3, u3, "real_", "_3m", "3M budget, PERISH-REAL: ")

    L += ["## Hypothesis verdicts", "", "Rows ending in _3m repeat a test at the 3 million agent-step budget (X8).", "",
          "| Test | Result | Evidence |", "| --- | --- | --- |"]
    for k, v in V.items():
        tag = "PASS" if v["passed"] else "FAIL"
        if v.get("seeds", 10) < 10:
            tag += f" (exploratory, {v['seeds']} seeds)"
        L.append(f"| {k} | {tag} | {v['detail']} |")
    (RES / "summary.md").write_text("\n".join(L) + "\n")
    (RES / "summary.json").write_text(json.dumps(V, indent=1))
    print("\n".join(L))
