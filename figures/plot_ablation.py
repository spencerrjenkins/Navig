#!/usr/bin/env python3
"""Generate ablation study visualization figures for NAVIG.

Produces comprehensive ablation visualizations comparing pipeline component
contributions across models and geographic regions:

  COMPONENT ABLATION ANALYSIS
  ablation_fig1_geoscore_impact.pdf     — GeoScore delta for each ablation
  ablation_fig2_component_ranking.pdf   — Ranked importance of components
  ablation_fig3_error_distribution.pdf  — Error distributions by condition
  ablation_fig4_geographic_impact.pdf   — Ablation impact by continent
  ablation_fig5_failure_modes.pdf       — Prediction failure breakdown
  ablation_fig6_accuracy_thresholds.pdf — Accuracy @ distance thresholds

Usage::

    python figures/plot_ablation.py \\
        --base_dir output/im2gps3k_rgb_images \\
        --output_dir figures/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

from metrics import geoscore, haversine_distance, parse_coord, THRESHOLDS

# ── Model registry (same as plot_comparison.py) ────────────────────────────────
MODEL_REGISTRY = [
    ("llama32vision",  "LLaMA-3.2-11B",         "LLaMA",  False),
    ("deepseek",       "DeepSeek-7B",             "Other",  False),
    ("falcon",         "Falcon-11B",              "Other",  False),
    ("llava",          "LLaVA-1.6",               "LLaVA",  False),
    ("llava_sft",      "LLaVA-1.6\n(SFT)",        "LLaVA",  True),
    ("cpm",            "MiniCPM-V-2.6",           "CPM",    False),
    ("cpm_sft",        "MiniCPM-V-2.6\n(SFT)",   "CPM",    True),
    ("qwen",           "Qwen2.5-7B",              "Qwen",   False),
    ("qwen_sft",       "Qwen2.5-7B\n(SFT)",      "Qwen",   True),
]

FAMILY_COLORS = {
    "LLaMA":  "#4C72B0",
    "Other":  "#8C8C8C",
    "LLaVA":  "#DD8452",
    "CPM":    "#55A868",
    "Qwen":   "#C44E52",
}

ABLATION_TYPES = [
    "base_reasoning",    # Base model with reasoning (no SFT)
    "without_reasoning", # Skips reasoning stage
    "without_tools",     # Skips tool usage (OSM)
    "direct_guess",      # Direct coordinate guess
]

ABLATION_LABELS = {
    "base_reasoning":    "Base + reasoning",
    "without_reasoning": "Without reasoning",
    "without_tools":     "Without tools",
    "direct_guess":      "Direct guess",
}

PENALTY_KM = 10_000.0


# ── Geo utilities ─────────────────────────────────────────────────────────────

from geo_utils import lat_to_continent  # noqa: E402


# ── Data loading ──────────────────────────────────────────────────────────────

def _ablation_path(base_dir: Path, key: str, abl_type: str) -> Path:
    return base_dir / f"cmp_shard_{key}_merged" / f"ablation_{abl_type}.jsonl"


def load_results(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["ID"]] = r
    return rows if rows else None


def get_dist(row: dict) -> tuple[float, bool]:
    correct = [float(row["LAT"]), float(row["LON"])]
    try:
        pred = [
            parse_coord(row["answer"]["latitude"]),
            parse_coord(row["answer"]["longitude"]),
        ]
        return haversine_distance(pred, correct), False
    except Exception:
        return PENALTY_KM, True


def compute_stats(rows: dict) -> dict:
    dist_fail_pairs = [get_dist(r) for r in rows.values()]
    dists = [d for d, _ in dist_fail_pairs]
    fails = sum(1 for _, f in dist_fail_pairs if f)
    success_dists = [d for d, f in dist_fail_pairs if not f]
    n = len(dists)
    gs = float(np.mean([geoscore(d) for d in dists]))
    accs = [sum(d <= t for d in dists) / n for t in THRESHOLDS]
    return {
        "geoscore": gs, "accs": accs, "dists": dists, "n": n,
        "fails": fails, "success_dists": success_dists,
        "p25": float(np.percentile(dists, 25)),
        "p50": float(np.percentile(dists, 50)),
        "p75": float(np.percentile(dists, 75)),
    }


def compute_geographic(rows: dict) -> dict:
    geo = defaultdict(list)
    for row in rows.values():
        d, _ = get_dist(row)
        cont = lat_to_continent(float(row["LAT"]), float(row["LON"]))
        geo[cont].append(d)
    return {cont: float(np.mean([geoscore(d) for d in dists]))
            for cont, dists in geo.items()}


def compute_failure_modes(rows: dict) -> dict:
    cats = {"≤25km": 0, "25–200km": 0, "200–2500km": 0, ">2500km": 0, "parse_fail": 0}
    for row in rows.values():
        d, fail = get_dist(row)
        if fail:
            cats["parse_fail"] += 1
        elif d <= 25:
            cats["≤25km"] += 1
        elif d <= 200:
            cats["25–200km"] += 1
        elif d <= 2500:
            cats["200–2500km"] += 1
        else:
            cats[">2500km"] += 1
    return cats


# ── Utilities ─────────────────────────────────────────────────────────────────

def _lighten(hex_color: str, amount: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _model_color(key: str) -> str:
    for k, label, family, is_sft in MODEL_REGISTRY:
        if k == key:
            c = FAMILY_COLORS[family]
            return _lighten(c, 0.4) if is_sft else c
    return "#888888"


def _model_label(key: str) -> str:
    for k, label, *_ in MODEL_REGISTRY:
        if k == key:
            return label.replace("\n", " ")
    return key


# ── Figure ablation_fig1: GeoScore impact per ablation ─────────────────────────

def ablation_fig1_geoscore_impact(models_data: dict, output_dir: Path) -> None:
    """Grouped bar: GeoScore delta (ablation − baseline) per component.
    Shows which components hurt performance most when removed."""
    if not models_data:
        print("  ablation_fig1_geoscore_impact.pdf: no data, skipping")
        return

    # Baseline: full pipeline (base_reasoning)
    baselines = {key: models_data[key]["base_reasoning"]["geoscore"]
                 for key in models_data if "base_reasoning" in models_data[key]}

    abl_types_to_plot = [
        ("without_reasoning", "−Reasoning"),
        ("without_tools", "−Tools"),
        ("direct_guess", "Direct"),
    ]

    available_keys = list(baselines.keys())
    n_m = len(available_keys)
    n_a = len(abl_types_to_plot)
    x = np.arange(n_a)
    bar_w = 0.75 / n_m

    fig, ax = plt.subplots(figsize=(8, 4))

    for j, key in enumerate(available_keys):
        baseline_gs = baselines[key]
        deltas = []
        for abl_type, _ in abl_types_to_plot:
            if abl_type in models_data[key]:
                delta = models_data[key][abl_type]["geoscore"] - baseline_gs
                deltas.append(delta)
            else:
                deltas.append(0)

        offsets = x + (j - (n_m - 1) / 2) * bar_w
        color = _model_color(key)
        ax.bar(offsets, deltas, bar_w * 0.9, color=color, alpha=0.82,
              label=_model_label(key), zorder=3)

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in abl_types_to_plot], fontsize=10)
    ax.set_ylabel("ΔGeoScore vs. baseline", fontsize=10)
    ax.set_xlabel("Ablation condition")
    ax.set_title("Impact of removing pipeline components", fontsize=11, fontweight="bold")
    ax.axhspan(-500, 0, alpha=0.04, color="red", zorder=0)
    ax.axhspan(0, 500, alpha=0.04, color="green", zorder=0)
    ax.legend(fontsize=8.5, ncol=2, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = output_dir / "ablation_fig1_geoscore_impact.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure ablation_fig2: Component importance ranking ──────────────────────────

def ablation_fig2_component_ranking(models_data: dict, output_dir: Path) -> None:
    """Heatmap of normalized importance scores (how much each component contributes)."""
    if not models_data:
        print("  ablation_fig2_component_ranking.pdf: no data, skipping")
        return

    available_keys = list(models_data.keys())
    if not available_keys:
        print("  ablation_fig2_component_ranking.pdf: no data, skipping")
        return

    # Compute importance as absolute delta
    components = ["Reasoning", "Tools", "Direct"]
    importance_matrix = np.zeros((len(available_keys), len(components)))

    for i, key in enumerate(available_keys):
        baseline = models_data[key].get("base_reasoning", {}).get("geoscore", 0)
        importance_matrix[i, 0] = abs(models_data[key].get("without_reasoning", {}).get("geoscore", 0) - baseline)
        importance_matrix[i, 1] = abs(models_data[key].get("without_tools", {}).get("geoscore", 0) - baseline)
        importance_matrix[i, 2] = abs(models_data[key].get("direct_guess", {}).get("geoscore", 0) - baseline)

    # Normalize per model
    importance_matrix = importance_matrix / (importance_matrix.max(axis=1, keepdims=True) + 1e-6)

    labels = [_model_label(k) for k in available_keys]

    fig, ax = plt.subplots(figsize=(5.5, max(3.5, len(available_keys) * 0.45)))
    im = ax.imshow(importance_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, fontsize=9)
    ax.set_yticks(range(len(available_keys)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Component", fontsize=9)
    ax.set_title("Component importance ranking (normalized)", fontsize=10, fontweight="bold")

    for i in range(len(available_keys)):
        for j in range(len(components)):
            v = importance_matrix[i, j]
            textcolor = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                   fontsize=8, color=textcolor, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Normalized importance", fontsize=8)

    fig.tight_layout()
    out = output_dir / "ablation_fig2_component_ranking.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure ablation_fig3: Error distribution by condition ──────────────────────

def ablation_fig3_error_distribution(models_data: dict, output_dir: Path) -> None:
    """Violin plots: error distributions for baseline vs. ablations."""
    available_keys = [k for k in models_data if "base_reasoning" in models_data[k]]
    if not available_keys or len(available_keys) > 3:
        print("  ablation_fig3_error_distribution.pdf: skipping (too many/few models)")
        return

    abl_types = ["base_reasoning", "without_reasoning", "without_tools", "direct_guess"]
    abl_labels = ["Baseline", "−Reasoning", "−Tools", "Direct"]

    n_m = len(available_keys)
    fig, axes = plt.subplots(1, n_m, figsize=(5.5 * n_m, 4), layout="constrained", sharey=True)
    if n_m == 1:
        axes = [axes]

    for ax, key in zip(axes, available_keys):
        dists_by_abl = [[] for _ in abl_types]
        for i, abl_type in enumerate(abl_types):
            if abl_type in models_data[key]:
                dists = models_data[key][abl_type].get("dists", [])
                dists_by_abl[i] = dists

        valid_dists = [d for d in dists_by_abl if len(d) > 0]
        valid_labels = [lbl for dists, lbl in zip(dists_by_abl, abl_labels) if len(dists) > 0]

        parts = ax.violinplot(valid_dists, positions=range(len(valid_dists)),
                             showmedians=True, showextrema=False, widths=0.6)
        for pc in parts["bodies"]:
            pc.set_facecolor("#DD8452")
            pc.set_alpha(0.6)
            pc.set_edgecolor("#DD8452")
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        ax.set_yscale("log")
        ax.set_xticks(range(len(valid_labels)))
        ax.set_xticklabels(valid_labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylim(0.05, 25000)
        ax.axhline(PENALTY_KM, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(_model_label(key), fontsize=9, fontweight="bold")

    axes[0].set_ylabel("Distance to ground truth (km, log scale)", fontsize=9)
    fig.suptitle("Error distribution by ablation condition", fontsize=11, fontweight="bold")

    out = output_dir / "ablation_fig3_error_distribution.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure ablation_fig4: Geographic impact ────────────────────────────────────

def ablation_fig4_geographic_impact(models_data: dict, output_dir: Path) -> None:
    """Heatmap: GeoScore by continent for baseline vs. key ablations."""
    if not models_data:
        print("  ablation_fig4_geographic_impact.pdf: no data, skipping")
        return

    available_keys = [k for k in models_data if "base_reasoning" in models_data[k]]
    if len(available_keys) == 0:
        print("  ablation_fig4_geographic_impact.pdf: no data, skipping")
        return

    continents = ["Europe", "N. America", "Asia (N)", "Asia (S/SE)", "Africa", "S. America", "Oceania"]

    fig, axes = plt.subplots(1, len(available_keys),
                            figsize=(4.5 * len(available_keys), 3.8),
                            layout="constrained", sharey=True)
    if len(available_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, available_keys):
        abl_types = ["base_reasoning", "without_reasoning", "without_tools"]
        matrix = np.full((len(continents), len(abl_types)), np.nan)

        for j, abl_type in enumerate(abl_types):
            if abl_type in models_data[key]:
                geo_data = models_data[key][abl_type].get("geo", {})
                for i, cont in enumerate(continents):
                    if cont in geo_data:
                        matrix[i, j] = geo_data[cont]

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=800, vmax=3500)
        ax.set_xticks(range(len(abl_types)))
        ax.set_xticklabels(["Baseline", "−Reasoning", "−Tools"], fontsize=8, rotation=15, ha="right")
        ax.set_yticks(range(len(continents)))
        ax.set_yticklabels(continents, fontsize=8)
        ax.set_title(_model_label(key), fontsize=9, fontweight="bold")

        for i in range(len(continents)):
            for j in range(len(abl_types)):
                v = matrix[i, j]
                if not np.isnan(v):
                    textcolor = "white" if v > 2800 else "black"
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                           fontsize=7, color=textcolor, fontweight="bold")

    axes[0].set_ylabel("Continent", fontsize=9)
    fig.suptitle("Geographic impact: GeoScore by continent and ablation", fontsize=11, fontweight="bold")

    out = output_dir / "ablation_fig4_geographic_impact.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure ablation_fig5: Failure modes by condition ────────────────────────────

def ablation_fig5_failure_modes(models_data: dict, output_dir: Path) -> None:
    """Stacked bar: failure mode breakdown for baseline vs. ablations."""
    if not models_data:
        print("  ablation_fig5_failure_modes.pdf: no data, skipping")
        return

    available_keys = [k for k in models_data if "base_reasoning" in models_data[k]]
    if not available_keys or len(available_keys) > 2:
        print("  ablation_fig5_failure_modes.pdf: skipping (too many/few models)")
        return

    abl_types = ["base_reasoning", "without_reasoning", "without_tools"]
    cat_labels = ["≤25 km", "25–200 km", "200–2500 km", ">2500 km", "Parse fail"]
    cat_keys   = ["≤25km", "25–200km", "200–2500km", ">2500km", "parse_fail"]
    cat_colors = ["#009E73", "#56B4E9", "#E69F00", "#D55E00", "#999999"]

    fig, axes = plt.subplots(1, len(available_keys), figsize=(5.5 * len(available_keys), 4),
                            layout="constrained")
    if len(available_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, available_keys):
        x = np.arange(len(abl_types))
        bottoms = np.zeros(len(abl_types))

        for cat_key, color, cat_label in zip(cat_keys, cat_colors, cat_labels):
            fracs = []
            for abl_type in abl_types:
                if abl_type in models_data[key]:
                    fail_modes = models_data[key][abl_type].get("fail_modes", {})
                    n = models_data[key][abl_type].get("n", 1)
                    frac = fail_modes.get(cat_key, 0) / n * 100
                    fracs.append(frac)
                else:
                    fracs.append(0)

            ax.bar(x, fracs, bottom=bottoms, color=color, label=cat_label, zorder=3)
            bottoms = bottoms + np.array(fracs)

        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "−Reasoning", "−Tools"], fontsize=8, rotation=15, ha="right")
        ax.set_ylim(0, 102)
        ax.set_ylabel("% of images" if available_keys.index(key) == 0 else "")
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        ax.set_title(_model_label(key), fontsize=9, fontweight="bold")

    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(cat_colors, cat_labels)]
    axes[-1].legend(handles=handles, loc="upper right", ncol=1, fontsize=8, framealpha=0.9)

    fig.suptitle("Prediction outcome decomposition by ablation condition", fontsize=11, fontweight="bold")

    out = output_dir / "ablation_fig5_failure_modes.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure ablation_fig6: Accuracy at thresholds ───────────────────────────────

def ablation_fig6_accuracy_thresholds(models_data: dict, output_dir: Path) -> None:
    """Line chart: accuracy @ various distance thresholds for baseline vs. ablations."""
    available_keys = [k for k in models_data if "base_reasoning" in models_data[k]]
    if not available_keys or len(available_keys) > 2:
        print("  ablation_fig6_accuracy_thresholds.pdf: skipping")
        return

    thresholds = [1, 25, 200, 750, 2500]
    thr_display = ["1 km", "25 km", "200 km", "750 km", "2500 km"]
    abl_types = ["base_reasoning", "without_reasoning", "without_tools"]
    abl_labels = ["Baseline", "−Reasoning", "−Tools"]
    colors_abl = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, len(available_keys), figsize=(6.5 * len(available_keys), 3.8),
                             layout="constrained", sharey=True)
    if len(available_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, available_keys):
        for abl_type, lbl, color in zip(abl_types, abl_labels, colors_abl):
            if abl_type not in models_data[key]:
                continue
            dists = models_data[key][abl_type].get("dists", [])
            n = len(dists)
            if n == 0:
                continue
            accs = [sum(d <= t for d in dists) / n * 100 for t in thresholds]
            ax.plot(range(len(thresholds)), accs, marker="o", color=color,
                   linestyle="-", linewidth=2, markersize=6, label=lbl, zorder=3)

        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels(thr_display, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("Accuracy (%)" if available_keys.index(key) == 0 else "")
        ax.set_ylim(0, 85)
        ax.set_yticks(range(0, 86, 10))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        ax.set_title(_model_label(key), fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--", zorder=1)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    fig.suptitle("Accuracy at distance thresholds", fontsize=11, fontweight="bold")

    out = output_dir / "ablation_fig6_accuracy_thresholds.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_dir", type=str, default="output/im2gps3k_rgb_images",
                   help="Directory containing cmp_shard_*_merged/ subdirectories")
    p.add_argument("--output_dir", type=str, default="figures",
                   help="Directory to write PDF figures")
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ablation data for all models
    models_data = {}
    for key, *_ in MODEL_REGISTRY:
        models_data[key] = {}
        for abl_type in ABLATION_TYPES:
            path = _ablation_path(base_dir, key, abl_type)
            rows = load_results(path)
            if rows is not None:
                stats = compute_stats(rows)
                geo = compute_geographic(rows)
                fail_modes = compute_failure_modes(rows)
                models_data[key][abl_type] = {
                    "stats": stats, "rows": rows, "geo": geo, "fail_modes": fail_modes,
                    "geoscore": stats["geoscore"], "dists": stats["dists"], "n": stats["n"]
                }
                print(f"  [{key}/{abl_type}] n={stats['n']}  GeoScore={stats['geoscore']:.2f}")

    print(f"\nGenerating ablation figures in {output_dir}/")

    ablation_fig1_geoscore_impact(models_data, output_dir)
    ablation_fig2_component_ranking(models_data, output_dir)
    ablation_fig3_error_distribution(models_data, output_dir)
    ablation_fig4_geographic_impact(models_data, output_dir)
    ablation_fig5_failure_modes(models_data, output_dir)
    ablation_fig6_accuracy_thresholds(models_data, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
