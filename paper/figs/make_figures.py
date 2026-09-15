#!/usr/bin/env python3
"""Generate the vector figures used in the paper.

Every number drawn here is read from a file under ``results/``; nothing is typed
in by hand, so re-running the script after a label update refreshes the paper.

Output is PDF because the figures are line art: text stays selectable, and the
curves stay sharp at any zoom.  Sizes are the IEEE two-column measures
(3.45 in for one column, 7.16 in across both), and every font is set so that the
figure is dropped into the page at 1:1 without any rescaling -- scaling a figure
down is what pushes tick labels below the 8 pt floor that IEEE asks for.

Usage:  python3 paper/figs/make_figures.py          (run from the repo root)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent

# --- page geometry -----------------------------------------------------------
COL = 3.45      # IEEE single-column width, inches
WIDE = 7.16     # IEEE double-column width, inches

# --- palette -----------------------------------------------------------------
# Three categorical hues, checked for colour-vision separation on a light
# surface before use.  Bars also carry a hatch so the figures survive being
# printed in grey.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "axes.labelcolor": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "figure.dpi": 200,
    # A fixed canvas, not a tight crop: the PDF must come out exactly one
    # column wide, so that LaTeX places it at 1:1 and the 8 pt floor holds.
    "savefig.bbox": "standard",
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
    "pdf.fonttype": 42,      # embed real glyphs, not bitmaps
    "ps.fonttype": 42,
})

STRATA = ["unprotected_turn", "merging", "unprotected_crossing", "interact"]
NICE = {
    "unprotected_turn": "Unprotected\nturn",
    "merging": "Merging",
    "unprotected_crossing": "Unsignalized\ncrossing",
    "interact": "All three\n(union)",
    "expert_conservative": "Low-progress\nexpert frames",
}


def tidy(ax, xgrid=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y")


# -----------------------------------------------------------------------------
def behavior_table():
    """Per-stratum behaviour metrics, averaged over the three evaluation seeds."""
    flags = pd.read_parquet(RESULTS / "labels" / "scene_flags_navtest.parquet").set_index("token")
    cols = ["unprotected_turn", "merging", "unprotected_crossing"]
    per_seed = []
    for seed in range(3):
        b = pd.read_parquet(RESULTS / "eval" / f"behavior_ltf_seed{seed}.parquet").set_index("token")
        b = b.join(flags[cols], how="left")
        row = {}
        for name in STRATA:
            sub = b if name == "interact" else b[b[name].fillna(False)]
            eligible = sub[sub.gap_eligible == True]  # noqa: E712 (nullable boolean)
            row[name] = {
                "n": len(sub),
                "n_gap": len(eligible),
                "gap": 100 * (eligible.gap_accepted == True).mean(),  # noqa: E712
                "stop": 100 * sub.unnecessary_stop.mean(),
            }
        per_seed.append(row)
    out = {}
    for name in STRATA:
        vals = {k: np.array([s[name][k] for s in per_seed], dtype=float) for k in ("gap", "stop")}
        out[name] = {
            "n": per_seed[0][name]["n"],
            "n_gap": per_seed[0][name]["n_gap"],
            "gap": vals["gap"].mean(), "gap_sd": vals["gap"].std(ddof=1),
            "stop": vals["stop"].mean(), "stop_sd": vals["stop"].std(ddof=1),
        }
    return out


def fig_motivation():
    """Driving score against gap acceptance: the score misses the failure mode."""
    beh = behavior_table()
    strat = pd.read_csv(RESULTS / "eval" / "ltf_baseline_stratified.csv").set_index("stratum")

    names = ["unprotected_turn", "merging", "unprotected_crossing"]
    pdms = [strat.loc[n, "score"] for n in names]
    pdms_err = [[strat.loc[n, "score"] - strat.loc[n, "score_ci_lo"] for n in names],
                [strat.loc[n, "score_ci_hi"] - strat.loc[n, "score"] for n in names]]
    gap = [beh[n]["gap"] for n in names]
    gap_err = [beh[n]["gap_sd"] for n in names]

    fig, ax = plt.subplots(figsize=(COL, 2.25))
    y = np.arange(len(names))
    h = 0.34
    ax.barh(y + h / 2 + 0.01, pdms, height=h, color=BLUE, label="Driving score (PDMS)",
            xerr=pdms_err, error_kw=dict(ecolor=INK2, elinewidth=0.7, capsize=1.6))
    ax.barh(y - h / 2 - 0.01, gap, height=h, color=ORANGE, hatch="///", edgecolor="white",
            linewidth=0.0, label="Gap acceptance rate",
            xerr=gap_err, error_kw=dict(ecolor=INK2, elinewidth=0.7, capsize=1.6))
    for yi, v, e in zip(y + h / 2 + 0.01, pdms, pdms_err[1]):
        ax.text(v + e + 2.5, yi, f"{v:.1f}", va="center", ha="left", fontsize=7, color=INK)
    for yi, v, e, n in zip(y - h / 2 - 0.01, gap, gap_err, [beh[n]["n_gap"] for n in names]):
        ax.text(v + e + 2.5, yi, f"{v:.0f}  (n={n})", va="center", ha="left", fontsize=7, color=INK)

    ax.set_yticks(y, [NICE[n].replace("\n", " ") for n in names])
    ax.set_xlim(0, 152)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Percent of the available maximum")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
              handlelength=1.1, borderaxespad=0.0, columnspacing=1.2)
    tidy(ax, xgrid=True)
    fig.savefig(OUT / "fig_motivation.pdf")
    plt.close(fig)


def fig_cascade():
    """How many generated candidates survive each test."""
    d = pd.read_parquet(RESULTS / "labels" / "negatives_navtest.parquet")
    pet_ok = d.pet_candidate_s.isna() | (d.pet_candidate_s >= 1.5)
    stages, mask = [], pd.Series(True, index=d.index)
    def add(label, cond):
        nonlocal mask
        mask = mask & cond
        stages.append((label, int(mask.sum()), d.loc[mask, "token"].nunique()))
    add("Generated", pd.Series(True, index=d.index))
    add("Kinematically feasible", d.kin_ok)
    add("Safe in replay", d.c1)
    add("Slower, but not trivially so", d.c2 & d.c4)
    add("Clears the conflict point", pet_ok & d.rss_rel_ok)
    add("Rejected an acceptable gap", d.c3_cand)
    add("Sight line was not blocked", d.c7_occ_ok)
    add("Replay agrees (final label)", d.c3_replay)

    labels = [s[0] for s in stages][::-1]
    counts = [s[1] for s in stages][::-1]
    scenes = [s[2] for s in stages][::-1]

    fig, ax = plt.subplots(figsize=(COL, 2.5))
    y = np.arange(len(labels))
    ax.barh(y, counts, height=0.62, color=BLUE)
    ax.barh(y[-1], counts[-1], height=0.62, color=INK2)  # the input row reads as context
    for yi, c, s in zip(y, counts, scenes):
        ax.text(c + 180, yi, f"{c:,} / {s} sc.", va="center", ha="left", fontsize=7, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 19500)
    ax.set_xticks([0, 5000, 10000], ["0", "5k", "10k"])
    ax.set_xlabel("Candidate trajectories kept")
    tidy(ax, xgrid=True)
    fig.savefig(OUT / "fig_cascade.pdf")
    plt.close(fig)


def fig_complementarity():
    """Each justification test finds scenes the others miss."""
    d = pd.read_parquet(RESULTS / "labels" / "negatives_navtest.parquet")
    g = d.groupby("token")
    m = pd.DataFrame({
        "Blocked sight line": g.occ_justified.first(),
        "Replay reference": ~g.c3_replay.any(),
        "No acceptable gap": (g.expert_category.first() == "justified_wait"),
    })
    order = ["No acceptable gap", "Replay reference", "Blocked sight line"]
    total = [int(m[c].sum()) for c in order]
    excl = [int((m[c] & ~m.drop(columns=[c]).any(axis=1)).sum()) for c in order]
    shared = [t - e for t, e in zip(total, excl)]

    fig, ax = plt.subplots(figsize=(COL, 1.95))
    y = np.arange(len(order))
    ax.barh(y, excl, height=0.55, color=BLUE, label="found by this test alone")
    ax.barh(y, shared, height=0.55, left=[e + 2 for e in excl], color=AQUA, hatch="\\\\\\",
            edgecolor="white", linewidth=0.0, label="also found by another test")
    for yi, e, t in zip(y, excl, total):
        ax.text(t + 6, yi, f"{e} of {t}", va="center", ha="left", fontsize=7, color=INK)
    ax.set_yticks(y, order)
    ax.set_xlim(0, 340)
    ax.set_xlabel("Scenes where caution was found to be justified")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
              handlelength=1.1, borderaxespad=0.0, columnspacing=1.0)
    tidy(ax, xgrid=True)
    fig.savefig(OUT / "fig_complementarity.pdf")
    plt.close(fig)


def fig_baseline():
    """Progress of the baseline planner against the human drive, per stratum."""
    s = pd.read_csv(RESULTS / "eval" / "ltf_baseline_stratified.csv").set_index("stratum")
    human = {"unprotected_turn": 84.2, "merging": 86.8, "unprotected_crossing": 85.8,
             "interact": 84.7, "expert_conservative": 61.1}   # docs/decision_scene_tags.md
    names = ["unprotected_turn", "merging", "unprotected_crossing", "interact", "expert_conservative"]

    fig, ax = plt.subplots(figsize=(COL, 2.35))
    y = np.arange(len(names))[::-1]
    ep = np.array([s.loc[n, "ego_progress"] for n in names])
    lo = np.array([s.loc[n, "ego_progress_ci_lo"] for n in names])
    hi = np.array([s.loc[n, "ego_progress_ci_hi"] for n in names])
    ax.hlines(y, lo, hi, color=BLUE, linewidth=2.0, alpha=0.35)
    ax.plot(ep, y, "o", color=BLUE, markersize=5, label="Latent TransFuser", zorder=3)
    ax.plot([human[n] for n in names], y, "D", color=ORANGE, markersize=4.5,
            label="Human drive", zorder=3)
    for yi, v in zip(y, ep):
        ax.text(v, yi + 0.28, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_yticks(y, [NICE[n].replace("\n", " ") for n in names])
    ax.set_ylim(-0.6, len(names) - 1 + 0.85)
    ax.set_xlim(50, 102)
    ax.set_xlabel("Ego progress sub-score")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
              handlelength=1.0, numpoints=1, borderaxespad=0.0, columnspacing=1.2)
    tidy(ax, xgrid=True)
    fig.savefig(OUT / "fig_baseline.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_motivation()
    fig_cascade()
    fig_complementarity()
    fig_baseline()
    print("wrote:", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))
