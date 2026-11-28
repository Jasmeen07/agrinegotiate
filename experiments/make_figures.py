"""Paper figures from the results folders. Writes PNG files to paper/figures/.

    python experiments/make_figures.py
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
from agrinegotiate.eval import stats                 # noqa: E402

RES, OUT = ROOT / "results", ROOT / "paper" / "figures"
INK, INK2, RULE, SURF = "#14201b", "#4b5650", "#d9ded9", "#fcfcfb"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]          # fixed categorical order
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": RULE,
                     "grid.linewidth": 0.8, "savefig.dpi": 200, "savefig.bbox": "tight"})


def summ(exp, label):
    f = RES / exp / label
    return stats.summarise(f) if list(f.glob("seed_*/metrics.json")) else None


def fig_shares():
    rows = [("x2_gate", "SYM", "Equal conditions"), ("x2_gate", "PERISH", "Perishable crop"),
            ("x2_gate", "DEADLINE", "Earlier farmer deadline"), ("x2_gate", "INFO", "Noisy price signal"),
            ("x2_gate", "THIN3", "Three farmers, one buyer")]
    rows = [(n, summ(e, l)) for e, l, n in rows]
    rows = [(n, s) for n, s in rows if s]
    fig, ax = plt.subplots(figsize=(7, 0.62 * len(rows) + 1.1))
    for y, (name, s) in enumerate(rows[::-1]):
        vals = [v for v in s["S_F"]["per_seed"] if v is not None]
        jit = (np.arange(len(vals)) % 3 - 1) * 0.12
        ax.scatter(vals, y + jit, s=34, color=S[0], edgecolor=SURF, linewidth=1.2, zorder=3)
        if "iqm" in s["S_F"]:
            ax.plot(s["S_F"]["ci95"], [y, y], color=INK, lw=2, zorder=2)
            ax.plot([s["S_F"]["iqm"]] * 2, [y - 0.28, y + 0.28], color=INK, lw=3, zorder=4)
    ax.axvline(0.5, color=INK2, ls="--", lw=1)
    ax.text(0.5, len(rows) - 0.35, "equal split (Nash)", ha="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(rows)), [f"{n}\n({s['n_seeds']} runs)" for n, s in rows[::-1]], color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("Farmer's share of the realised gains (one dot per training run; bar = IQM and 95% interval)")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig3_farmer_share_by_scenario.png"); plt.close(fig)


def fig_frontier():
    fams = [("Unconstrained", [("x2_gate", "PERISH", "")]),
            ("Adaptive fee, true values", [("x3_mechanism", f"fee_oracle_tau{t}", f"target {t}") for t in (0.2, 0.3, 0.4, 0.5)]
             + [("x3_mechanism", "fee_oracle_tau0.4_eta0.25", "target 0.4, fast update")]),
            ("Adaptive fee, public prices only", [("x3_mechanism", "fee_observable_tau0.4", "target 0.4")]),
            ("Hard price floor", [("x3_mechanism", f"floor_kappa{k:.2f}", f"floor {k:.2f}") for k in (0.50, 0.55, 0.60, 0.65)]),
            ("Fixed fee", [("x3_mechanism", "fee_fixed_lam1.0_tau0.4", "rate 1.0")])]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    drawn = False
    for (name, items), col in zip(fams, S):
        pts = [(summ(e, l), lab) for e, l, lab in items]
        pts = [(s, lab) for s, lab in pts if s and "iqm" in s["E"]]
        if not pts:
            continue
        drawn = True
        xs, ys = [s["FGC"]["iqm"] for s, _ in pts], [s["E"]["iqm"] for s, _ in pts]
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], color=col, lw=2, alpha=0.5, zorder=2)
        for s, lab in pts:
            ax.errorbar(s["FGC"]["iqm"], s["E"]["iqm"],
                        xerr=[[s["FGC"]["iqm"] - s["FGC"]["ci95"][0]], [s["FGC"]["ci95"][1] - s["FGC"]["iqm"]]],
                        yerr=[[s["E"]["iqm"] - s["E"]["ci95"][0]], [s["E"]["ci95"][1] - s["E"]["iqm"]]],
                        color=col, lw=0.8, alpha=0.7, zorder=2)
            ax.annotate(lab, (s["FGC"]["iqm"], s["E"]["iqm"]), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK)
        ax.scatter(xs, ys, s=55, color=col, edgecolor=SURF, linewidth=1.5, zorder=3, label=name)
    if not drawn:
        plt.close(fig); return
    ax.set_xlabel("Farmer gain capture (share of possible gains reaching farmers)")
    ax.set_ylabel("Efficiency (share of possible gains realised)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.savefig(OUT / "fig4_efficiency_fairness_frontier.png"); plt.close(fig)


def fig_delay():
    duties = ["1.0", "0.6", "0.4"]
    cells = [("unaware", "Delay-unaware", S[0]), ("aware", "Delay-aware", S[1])]
    if not summ("x4_delay", "unaware_duty0.6"):
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (k, title) in zip(axes, [("E", "Efficiency"), ("agreement_rate", "Agreement rate"), ("FGC", "Farmer gain capture")]):
        for key, name, col in cells:
            ss = [summ("x4_delay", f"{key}_duty{d}") for d in duties]
            xs = [i for i, s in enumerate(ss) if s and "iqm" in s[k]]
            if not xs:
                continue
            ys = [ss[i][k]["iqm"] for i in xs]
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=7, mec=SURF, mew=1.5, label=name)
            for i in xs:
                ax.plot([i, i], ss[i][k]["ci95"], color=col, lw=0.9, alpha=0.7)
            ax.annotate(name, (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, 0), fontsize=8.5, color=INK, va="center")
        ax.set_xticks(range(3), ["100%", "60%", "40%"]); ax.set_xlim(-0.2, 2.9)
        ax.set_title(title, loc="left", color=INK, fontsize=10); ax.set_xlabel("Farmer link uptime")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.savefig(OUT / "fig5_delay.png"); plt.close(fig)


def fig_learning():
    items = [("x2_gate", "SYM", "Equal conditions", S[0]), ("x2_gate", "PERISH", "Perishable crop", S[1])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for e, l, name, col in items:
        curves = [json.loads(f.read_text()) for f in sorted((RES / e / l).glob("seed_*/curve.json")) if (f.parent / "metrics.json").exists()]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        it = [curves[0][k]["iter"] for k in range(n)]
        for ax, key in zip(axes, ("E", "S_F")):
            for c in curves:
                ax.plot(it, [c[k][key] if c[k][key] is not None else np.nan for k in range(n)], color=col, lw=0.8, alpha=0.35)
            mean = [np.nanmean([c[k][key] if c[k][key] is not None else np.nan for c in curves]) for k in range(n)]
            ax.plot(it, mean, color=col, lw=2, label=name)
    axes[0].set_title("Efficiency", loc="left", color=INK, fontsize=10); axes[1].set_title("Farmer share", loc="left", color=INK, fontsize=10)
    axes[1].axhline(0.5, color=INK2, ls="--", lw=1); axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("Training iteration (thin lines: single runs; thick: mean)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.savefig(OUT / "fig2_learning_curves.png"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_learning, fig_shares, fig_frontier, fig_delay):
        f()
    print(sorted(p.name for p in OUT.glob("*.png")))
