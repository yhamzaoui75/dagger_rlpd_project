"""Figures and statistics of the DAgger vs RLPD study, from the saved runs.

Reads ``results/dagger/seed*/{curve.csv,summary.json}`` and
``results/rlpd/<dataset>/seed*/{curve.csv,summary.json}`` and writes, in
``figures/``:

- ``noise_comparison.pdf``: RLPD with the white, pink and OU datasets;
- ``main_comparison.pdf``: the best of them (highest finish rate, then lowest
  IQM) against DAgger and BC;
- ``stats.md``: the summary table and the statistical tests.

Each figure shows the lap time and the finish rate along training, and the
final lap time of every seed.

A lap that is not finished within the step budget counts as 1000 steps
(``MAX_EPISODE_STEPS``) in the learning curves and in the rank tests, and as
"did not finish" (DNF) elsewhere. Run from ``my_project/src``:
``python analysis.py --results ../results_cloud``.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

matplotlib.use("Agg")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "0.3",
    "legend.fancybox": False,
    "lines.linewidth": 1.2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

PROJECT_DIR = Path(__file__).resolve().parents[1]
MAX_EPISODE_STEPS = 1000
EXPERT_LAP_STEPS = 570
N_BOOTSTRAP = 10_000
ALPHA = 0.05

RLPD_DATASETS = {
    "RLPD (white)": "expert_Agent1_n30_noise0.1_seed0",
    "RLPD (pink)": "expert_Agent1_n30_pink0.1_seed0",
    "RLPD (OU)": "expert_Agent1_n30_ou0.1_seed0",
}
COLORS = {  # one fixed color per method (matplotlib's tab10)
    "BC": "0.45",
    "DAgger": "tab:blue",
    "RLPD (white)": "tab:orange",
    "RLPD (pink)": "tab:green",
    "RLPD (OU)": "tab:red",
}
MARKERS = {"BC": "s", "DAgger": "o", "RLPD (white)": "^", "RLPD (pink)": "D", "RLPD (OU)": "v"}


def read_curve(path):
    """Rows of a ``curve.csv`` as (env_steps, lap_steps, finished) arrays."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    steps = np.array([int(r["env_steps"]) for r in rows])
    laps = np.array([int(r["lap_steps"]) for r in rows])
    finished = np.array([r["lap_finished"] == "True" for r in rows])
    return steps, np.where(finished, laps, MAX_EPISODE_STEPS), finished


def read_final(path):
    """Final lap time of a run (``nan`` if the evaluation laps were not finished)."""
    final = json.loads(Path(path).read_text())["final"]
    return final["mean"] if final["finished"] > 0 else math.nan


def load_runs(results):
    """Curves and final lap times of every method, keyed by method name."""
    runs = {}
    seeds = sorted((results / "dagger").glob("seed*"), key=lambda p: int(p.name[4:]))
    curves = [read_curve(s / "curve.csv") for s in seeds]
    # BC is the first DAgger iteration: a policy trained on the expert's lap only.
    runs["BC"] = {"final": np.array([float(c[1][0]) if c[2][0] else math.nan for c in curves])}
    runs["DAgger"] = {"curves": curves, "final": np.array([read_final(s / "summary.json") for s in seeds])}
    for name, dataset in RLPD_DATASETS.items():
        seeds = sorted((results / "rlpd" / dataset).glob("seed*"), key=lambda p: int(p.name[4:]))
        runs[name] = {
            "curves": [read_curve(s / "curve.csv") for s in seeds],
            "final": np.array([read_final(s / "summary.json") for s in seeds]),
        }
    return runs


def iqm(x, axis=-1):
    """Interquartile mean (25% trimmed on each side), as recommended in the course."""
    return stats.trim_mean(x, 0.25, axis=axis)


def bootstrap_ci(x, statistic=iqm, seed=0):
    """Percentile bootstrap 95% confidence interval of a statistic over seeds."""
    x = np.asarray(x, dtype=float)
    boot = np.random.default_rng(seed).choice(x, size=(N_BOOTSTRAP, x.size))
    return tuple(np.percentile(statistic(boot, axis=1), [2.5, 97.5]))


def welch_test(a, b):
    """Two-sided Welch's t-test p-value (exact 1 or 0 when both samples are constant)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    se2 = a.var(ddof=1) / a.size + b.var(ddof=1) / b.size
    diff = a.mean() - b.mean()
    if se2 == 0.0:
        return 1.0 if diff == 0.0 else 0.0
    df = se2**2 / ((a.var(ddof=1) / a.size) ** 2 / (a.size - 1) + (b.var(ddof=1) / b.size) ** 2 / (b.size - 1))
    return float(2.0 * stats.t.sf(abs(diff) / math.sqrt(se2), df))


def panel_label(ax, letter):
    """Bold panel letter in the top-left corner, outside the axes."""
    ax.text(-0.02, 1.03, f"({letter})", transform=ax.transAxes, fontweight="bold", ha="right", va="bottom")


def thousands(ax):
    """Write the environment steps in thousands on the x axis."""
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}"))
    ax.set_xlabel(r"Environment steps ($\times 10^3$)")


def curve_matrix(curves):
    """Stack the per-seed curves on a common x axis (mean env steps at each evaluation)."""
    n = min(len(c[0]) for c in curves)
    steps = np.mean([c[0][:n] for c in curves], axis=0)
    laps = np.array([c[1][:n] for c in curves], dtype=float)
    finished = np.array([c[2][:n] for c in curves])
    return steps, laps, finished


def best_noise(runs):
    """RLPD variant with the highest finish rate, then the lowest IQM of finished laps."""
    def key(name):
        final = runs[name]["final"]
        done = final[~np.isnan(final)]
        return (-done.size / final.size, iqm(done))
    return min(RLPD_DATASETS, key=key)


def plot_curves(ax_lap, ax_fin, runs, names):
    """IQM lap time with its 95% CI (left) and finish rate (right) against environment steps."""
    for name in names:
        steps, laps, finished = curve_matrix(runs[name]["curves"])
        low, high = np.array([bootstrap_ci(laps[:, j]) for j in range(laps.shape[1])]).T
        kw = dict(color=COLORS[name], marker=MARKERS[name], markersize=3, label=name)
        ax_lap.fill_between(steps, low, high, color=COLORS[name], alpha=0.2, linewidth=0)
        ax_lap.plot(steps, iqm(laps, axis=0), **kw)
        ax_fin.plot(steps, 100 * finished.mean(axis=0), **kw)
    ax_lap.axhline(EXPERT_LAP_STEPS, color="black", linestyle="--", linewidth=0.8, label="Expert")
    ax_lap.set_ylim(540, 1020)
    ax_lap.set_ylabel("Lap time (steps)")
    ax_fin.set_ylim(-4, 104)
    ax_fin.set_ylabel("Seeds finishing the lap (%)")
    for ax in (ax_lap, ax_fin):
        ax.set_xlim(0, 31_000)
        thousands(ax)


def plot_final(ax, runs, names):
    """Box plot of the final lap time over seeds, with every seed as a dot; unfinished laps on a DNF row."""
    finished_max = max(np.nanmax(runs[n]["final"]) for n in names)
    has_dnf = any(np.isnan(runs[n]["final"]).any() for n in names)
    dnf_y = finished_max + 0.2 * (finished_max - 550)
    top = dnf_y if has_dnf else finished_max
    rng = np.random.default_rng(0)
    finals = [runs[n]["final"] for n in names]
    ax.boxplot([f[~np.isnan(f)] for f in finals], positions=range(len(names)), widths=0.5, showfliers=False,
               medianprops=dict(color="black", linewidth=1), boxprops=dict(linewidth=0.8),
               whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8))
    for i, (name, final) in enumerate(zip(names, finals)):
        x = i + rng.uniform(-0.12, 0.12, size=final.size)
        ax.scatter(x, np.where(np.isnan(final), dnf_y, final), s=9, color=COLORS[name], marker=MARKERS[name],
                   zorder=3, linewidths=0)
    ax.axhline(EXPERT_LAP_STEPS, color="black", linestyle="--", linewidth=0.8)
    ticks = list(matplotlib.ticker.MaxNLocator(5).tick_values(550, finished_max))
    if has_dnf:
        separator = (finished_max + dnf_y) / 2
        ax.axhline(separator, color="black", linewidth=0.5, linestyle=":")
        ticks = [t for t in ticks if 550 <= t < separator]
        ax.set_yticks([*ticks, dnf_y], [*(f"{t:g}" for t in ticks), "DNF"])
    else:
        ax.set_yticks([t for t in ticks if 550 <= t <= top + 1e-9])
    ax.set_ylim(550 - 0.04 * (top - 550), top + 0.06 * (top - 550))
    labels = [f"{n.replace('RLPD ', 'RLPD\n')}\n{np.sum(~np.isnan(f))}/{f.size}" for n, f in zip(names, finals)]
    ax.set_xticks(range(len(names)), labels)
    ax.tick_params(axis="x", top=False, length=0)
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_ylabel("Final lap time (steps)")


def plot_comparison(runs, curve_names, final_names, out, filename, bc_line=False):
    """One figure: learning curves (a, b) and final lap time per seed (c)."""
    fig, (ax_lap, ax_fin, ax_final) = plt.subplots(
        1, 3, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.1, 1.0, 1.0], "wspace": 0.38})
    plot_curves(ax_lap, ax_fin, runs, curve_names)
    if bc_line:
        ax_lap.axhline(iqm(runs["BC"]["final"]), color="0.45", linestyle=":", linewidth=1, label="BC")
    plot_final(ax_final, runs, final_names)
    handles, labels = ax_lap.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(labels), frameon=False,
               handlelength=2.0, columnspacing=1.5)
    for ax, letter in zip((ax_lap, ax_fin, ax_final), "abc"):
        panel_label(ax, letter)
    fig.savefig(out / f"{filename}.pdf")
    fig.savefig(out / f"{filename}.png", dpi=300)
    plt.close(fig)


def compare(runs, a, b):
    """Fisher (finish rate), Welch (finished laps) and Mann-Whitney (DNF = 1000) p-values."""
    fa, fb = runs[a]["final"], runs[b]["final"]
    da, db = fa[~np.isnan(fa)], fb[~np.isnan(fb)]
    table = [[da.size, fa.size - da.size], [db.size, fb.size - db.size]]
    capped_a, capped_b = np.nan_to_num(fa, nan=MAX_EPISODE_STEPS), np.nan_to_num(fb, nan=MAX_EPISODE_STEPS)
    return {
        "fisher": stats.fisher_exact(table)[1],
        "welch": welch_test(da, db),
        "mannwhitney": stats.mannwhitneyu(capped_a, capped_b).pvalue,
    }


def write_stats(runs, out, best):
    """Summary table and Bonferroni-corrected tests, as a Markdown file (also printed)."""
    lines = ["| Method | Finish | IQM [95% CI] | Median | Mean ± std |", "|---|---|---|---|---|"]
    for name in ("BC", "DAgger", *RLPD_DATASETS):
        final = runs[name]["final"]
        done = final[~np.isnan(final)]
        low, high = bootstrap_ci(done)
        std = done.std(ddof=1) if done.size > 1 else 0.0
        lines.append(f"| {name} | {done.size}/{final.size} | {iqm(done):.1f} [{low:.1f}, {high:.1f}] "
                     f"| {np.median(done):.0f} | {done.mean():.1f} ± {std:.1f} |")
    families = {
        "Noise comparison (RLPD)": [("RLPD (white)", "RLPD (pink)"), ("RLPD (white)", "RLPD (OU)"),
                                    ("RLPD (pink)", "RLPD (OU)")],
        f"Main comparison ({best})": [(best, "DAgger"), (best, "BC"), ("DAgger", "BC")],
    }
    for family, pairs in families.items():
        threshold = ALPHA / len(pairs)
        lines += ["", f"**{family}** (Bonferroni: significant if p < {ALPHA}/{len(pairs)} = {threshold:.4f})", "",
                  "| Comparison | Fisher (finish rate) | Welch (finished laps) | Mann-Whitney (DNF = 1000) |",
                  "|---|---|---|---|"]
        for a, b in pairs:
            p = compare(runs, a, b)
            cells = [f"{p[k]:.3g}{' *' if p[k] < threshold else ''}" for k in ("fisher", "welch", "mannwhitney")]
            lines.append(f"| {a} vs {b} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (out / "stats.md").write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description="Figures and statistics of the DAgger vs RLPD study.")
    parser.add_argument("--results", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--out", type=Path, default=PROJECT_DIR / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results)
    best = best_noise(runs)
    print(f"Best noise for RLPD: {best}")
    plot_comparison(runs, list(RLPD_DATASETS), list(RLPD_DATASETS), args.out, "noise_comparison")
    plot_comparison(runs, ["DAgger", best], ["BC", "DAgger", best], args.out, "main_comparison", bc_line=True)
    write_stats(runs, args.out, best)
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
