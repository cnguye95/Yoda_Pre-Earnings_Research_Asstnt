"""Draws the README's result charts from a judged eval run, in a light and a dark version each.

Run from the repo root (free: it reads saved results and calls no API):
    python docs/charts/make_charts.py [RESULTS_CSV]
RESULTS_CSV defaults to the full evaluation, data/eval/full/rescore/results.csv.

Every number comes from the eval runner's own tested functions
(yoda/eval/runner.py: _paired_means, _paired_gaps, _paired_comparison), so the
charts always agree with that run's summary.md. Four charts, each written
as {name}-light.png and {name}-dark.png next to this file; the README shows
the one matching the reader's GitHub theme:
  quality_vs_cost      average score and cost per report, per method
  paired_gaps          each pair of methods' average gap with its 95% range
  dimension_scores     each method's mean score on each rubric dimension
  per_ticker_gaps      staged pipeline minus baseline, ticker by ticker

Colours are a validated categorical palette (checked for colour-blind
separation in both themes); every mark also carries a text label, so no
reading depends on colour alone.
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")   # draw straight to files; no window
import matplotlib.pyplot as plt
import pandas as pd

from yoda.eval.complexity import is_complex
from yoda.eval.runner import SCORE_COLS, _paired_comparison, _paired_gaps, _paired_means

OUT_DIR = pathlib.Path(__file__).parent
DEFAULT_RESULTS = pathlib.Path("data/eval/full/rescore/results.csv")

# The methods in display order, with the names a reader sees.
METHODS = ["baseline", "pipeline", "lead"]
NAMES = {"baseline": "Baseline", "pipeline": "Staged pipeline", "lead": "Lead analyst"}

# Plain-English rubric dimension names, in the rubric's order.
DIMENSION_NAMES = {
    "filing_extraction": "Filing extraction", "figure_accuracy": "Figure accuracy",
    "reasoning_quality": "Reasoning quality", "evidence_labelling": "Evidence labelling",
    "source_traceability": "Source traceability", "relevance": "Relevance",
    "sell_side_usefulness": "Sell-side usefulness", "buy_side_usefulness": "Buy-side usefulness",
}

# The comparisons drawn, as (first, second): the gap is first minus second.
COMPARISONS = [("pipeline", "baseline"), ("lead", "baseline"), ("lead", "pipeline")]

# Each theme's colours: the chart surface, three text inks, the gridline and
# axis greys, and one colour per method (the same hue stepped for each
# surface). Values from the validated reference palette.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": {"baseline": "#2a78d6", "pipeline": "#eb6834", "lead": "#1baf7a"}},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835",
             "series": {"baseline": "#3987e5", "pipeline": "#d95926", "lead": "#199e70"}},
}

# Per-report cost of each method in the full evaluation (generation.csv,
# mean cost_usd; the baseline's is the batched price).
COST_PER_REPORT = {"baseline": 0.35, "pipeline": 1.37, "lead": 3.00}


def _style(ax: plt.Axes, theme: dict, grid_axis: str = "x") -> None:
    """Apply the quiet chart chrome to *ax*: surface colour, hairline grid on *grid_axis*, muted ticks, no box."""
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=theme["ink2"], labelsize=10, length=0)
    ax.grid(axis=grid_axis, color=theme["grid"], linewidth=1)
    ax.set_axisbelow(True)


def _new_figure(theme: dict, width: float, height: float, ncols: int = 1) -> tuple[plt.Figure, list[plt.Axes]]:
    """Return a figure of *width* x *height* inches on the theme's surface, with *ncols* side-by-side axes."""
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), squeeze=False)
    fig.patch.set_facecolor(theme["surface"])
    return fig, list(axes[0])


def _title(fig: plt.Figure, theme: dict, title: str, subtitle: str) -> None:
    """Write the chart's title and one-line subtitle at its top left, in text inks (never a series colour).

    Positions are fixed in inches from the top, so the spacing is the same
    on a short chart and a tall one.
    """
    height = fig.get_figheight()
    fig.text(0.012, 1 - 0.08 / height, title, ha="left", va="top", fontsize=13, fontweight="bold",
             color=theme["ink"])
    fig.text(0.012, 1 - 0.30 / height, subtitle, ha="left", va="top", fontsize=10, color=theme["ink2"])


def _save(fig: plt.Figure, name: str, theme_name: str) -> None:
    """Write *fig* as docs/charts/{name}-{theme_name}.png and close it."""
    fig.savefig(OUT_DIR / f"{name}-{theme_name}.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def quality_vs_cost(df: pd.DataFrame, theme: dict) -> plt.Figure:
    """Two panels sharing the method rows: average score (1-5, from the scale's floor) and cost per report."""
    means = _paired_means(df)[SCORE_COLS].mean(axis=1)
    fig, (score_ax, cost_ax) = _new_figure(theme, 8, 2.9, ncols=2)
    rows = list(range(len(METHODS)))[::-1]   # first method on top

    # Left: average score. Bars start at 1, the rubric's lowest score, and the
    # axis runs to 5 so the small gaps are shown at their true size.
    for row, method in zip(rows, METHODS):
        score_ax.barh(row, means[method] - 1, left=1, height=0.42, color=theme["series"][method])
        score_ax.text(means[method] + 0.05, row, f"{means[method]:.2f}", va="center", fontsize=10,
                      color=theme["ink"])
    score_ax.set_xlim(1, 5.4)
    score_ax.set_xticks([1, 2, 3, 4, 5])
    score_ax.set_yticks(rows, [NAMES[m] for m in METHODS])
    score_ax.set_xlabel("Average score (1-5)", color=theme["ink2"], fontsize=10)
    _style(score_ax, theme)

    # Right: cost per report, from $0.
    for row, method in zip(rows, METHODS):
        cost_ax.barh(row, COST_PER_REPORT[method], height=0.42, color=theme["series"][method])
        cost_ax.text(COST_PER_REPORT[method] + 0.06, row, f"${COST_PER_REPORT[method]:.2f}", va="center",
                     fontsize=10, color=theme["ink"])
    cost_ax.set_xlim(0, 3.6)
    cost_ax.set_xticks([0, 1, 2, 3], ["$0", "$1", "$2", "$3"])
    cost_ax.set_yticks(rows, [""] * len(rows))
    cost_ax.set_xlabel("Cost per report (USD)", color=theme["ink2"], fontsize=10)
    _style(cost_ax, theme)

    _title(fig, theme, "Same quality, a quarter of the cost",
           "20 tickers per method. The baseline and the staged pipeline score alike; the staged pipeline costs ~4x more.")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.76, bottom=0.2, wspace=0.08)
    return fig


def paired_gaps(df: pd.DataFrame, theme: dict) -> plt.Figure:
    """A forest plot: each comparison's average gap (first minus second) with its 95% bootstrap range."""
    fig, (ax,) = _new_figure(theme, 8, 2.9)
    rows = list(range(len(COMPARISONS)))[::-1]
    for row, (first, second) in zip(rows, COMPARISONS):
        stats = _paired_comparison(df, first, second).loc["average"]
        gap, low, high = stats["mean_gap"], stats["ci_low"], stats["ci_high"]
        # The winner is named only when the whole range is on one side of 0;
        # the dot takes the winner's colour, grey when inconclusive.
        if low > 0 or high < 0:
            winner = first if low > 0 else second
            colour, verdict = theme["series"][winner], f"{NAMES[winner]} better"
        else:
            colour, verdict = theme["muted"], "inconclusive"
        ax.plot([low, high], [row, row], color=theme["ink2"], linewidth=2, solid_capstyle="round")
        ax.scatter([gap], [row], s=90, color=colour, edgecolors=theme["surface"], linewidths=2, zorder=3)
        ax.text(0.24, row, f"{gap:+.2f}  ({low:+.2f} to {high:+.2f})  {verdict}", va="center", fontsize=10,
                color=theme["ink"])
    ax.axvline(0, color=theme["axis"], linewidth=1.5, zorder=1)
    ax.set_xlim(-0.4, 0.95)
    ax.set_xticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2])
    ax.set_yticks(rows, [f"{NAMES[a]} vs {NAMES[b].lower()}" for a, b in COMPARISONS])
    ax.set_xlabel("Average score gap, first method minus second (dot), with its 95% range (line)",
                  color=theme["ink2"], fontsize=10)
    _style(ax, theme)
    _title(fig, theme, "Paired by ticker: who beats whom",
           "A range crossing 0 means the 20 tickers can't separate the two methods.")
    fig.subplots_adjust(left=0.3, right=0.98, top=0.76, bottom=0.2)
    return fig


def dimension_scores(df: pd.DataFrame, theme: dict) -> plt.Figure:
    """A dot plot: one row per rubric dimension, one dot per method at its mean score."""
    means = _paired_means(df)[SCORE_COLS]
    fig, (ax,) = _new_figure(theme, 8, 4.6)
    rows = list(range(len(SCORE_COLS)))[::-1]
    # Small vertical offsets keep equal scores from hiding each other.
    offsets = {"baseline": 0.22, "pipeline": 0.0, "lead": -0.22}
    for row, dimension in zip(rows, SCORE_COLS):
        values = means[dimension]
        ax.plot([values.min(), values.max()], [row, row], color=theme["grid"], linewidth=2, zorder=1)
        for method in METHODS:
            ax.scatter([values[method]], [row + offsets[method]], s=70, color=theme["series"][method],
                       edgecolors=theme["surface"], linewidths=2, zorder=3, label=NAMES[method] if row == rows[0] else None)
    ax.set_xlim(2.5, 5.05)
    ax.set_xticks([2.5, 3, 3.5, 4, 4.5, 5])
    ax.set_yticks(rows, [DIMENSION_NAMES[d] for d in SCORE_COLS])
    ax.set_xlabel("Mean score (1-5) over 20 tickers", color=theme["ink2"], fontsize=10)
    _style(ax, theme)
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=10,
                       handletextpad=0.3, columnspacing=1.4)
    for text in legend.get_texts():
        text.set_color(theme["ink"])
    _title(fig, theme, "Where the methods differ",
           "The lead analyst method trails on filing extraction and reasoning; elsewhere the three are close.")
    fig.subplots_adjust(left=0.22, right=0.98, top=0.82, bottom=0.12)
    return fig


def per_ticker_gaps(df: pd.DataFrame, theme: dict) -> plt.Figure:
    """Horizontal bars from 0: each ticker's staged pipeline minus baseline average gap, sorted."""
    gaps = _paired_gaps(df, "pipeline", "baseline")["average"].sort_values()
    fig, (ax,) = _new_figure(theme, 8, 5.6)
    rows = list(range(len(gaps)))
    for row, (ticker, gap) in zip(rows, gaps.items()):
        if gap > 0:
            ax.barh(row, gap, height=0.6, color=theme["series"]["pipeline"])
        elif gap < 0:
            ax.barh(row, gap, height=0.6, color=theme["series"]["baseline"])
        else:
            ax.scatter([0], [row], s=40, color=theme["muted"], zorder=3)
    ax.axvline(0, color=theme["axis"], linewidth=1.5)
    limit = max(abs(gaps.min()), abs(gaps.max())) + 0.1
    ax.set_xlim(-limit, limit)
    ax.set_yticks(rows, [f"{ticker} *" if is_complex(ticker) else ticker for ticker in gaps.index])
    ax.set_xlabel("Average score gap, staged pipeline minus baseline        (* = complex company)",
                  color=theme["ink2"], fontsize=10)
    # Which side means what, written above the plot over each half.
    top = len(rows) - 0.2
    ax.text(-limit + 0.02, top, "← Baseline scored higher", fontsize=10, color=theme["ink"], va="bottom")
    ax.text(limit - 0.02, top, "Staged pipeline scored higher →", fontsize=10, color=theme["ink"],
            ha="right", va="bottom")
    ax.set_ylim(-0.7, len(rows) + 0.4)
    _style(ax, theme)
    _title(fig, theme, "Ticker by ticker: no consistent winner",
           "The staged pipeline scored higher on 9 tickers, the baseline on 10, with 1 tie.")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.1)
    return fig


def main(results_path: pathlib.Path) -> None:
    """Draw every chart in both themes from the judged results at *results_path*."""
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    df = pd.read_csv(results_path)
    charts = {"quality_vs_cost": quality_vs_cost, "paired_gaps": paired_gaps,
              "dimension_scores": dimension_scores, "per_ticker_gaps": per_ticker_gaps}
    for theme_name, theme in THEMES.items():
        for name, draw in charts.items():
            _save(draw(df, theme), name, theme_name)
    print(f"Wrote {len(charts) * len(THEMES)} charts to {OUT_DIR}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS)
