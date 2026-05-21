#!/usr/bin/env python3
"""Generate comparison figures for the 9-model NAVIG evaluation.

Produces four publication-ready PDFs:

  fig_comparison_geoscore.pdf   — GeoScore bar chart for all models, SFT pairs grouped
  fig_comparison_sft_delta.pdf  — GeoScore delta (SFT − base) for each model family
  fig_comparison_accuracy.pdf   — Accuracy at 5 thresholds: base vs. SFT grouped bars
  fig_comparison_cdf.pdf        — Empirical CDF of prediction error (km) for all models

Usage::

    python figures/plot_comparison.py \\
        --base_dir output/im2gps3k_rgb_images \\
        --output_dir figures/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from metrics import geoscore, haversine_distance, parse_coord, THRESHOLDS, THRESHOLD_NAMES

# ── Model registry ────────────────────────────────────────────────────────────
# (key, display label, family, is_sft)
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

# Pairs (base_key, sft_key, family_label) for the delta / accuracy figures
SFT_PAIRS = [
    ("llava",    "llava_sft",  "LLaVA-1.6"),
    ("cpm",      "cpm_sft",   "MiniCPM-V-2.6"),
    ("qwen",     "qwen_sft",  "Qwen2.5-7B"),
]

FAMILY_COLORS = {
    "LLaMA":  "#4C72B0",
    "Other":  "#8C8C8C",
    "LLaVA":  "#DD8452",
    "CPM":    "#55A868",
    "Qwen":   "#C44E52",
}

THR_LABELS = [f"@{t} km" for t in THRESHOLDS]


# ── Data loading ──────────────────────────────────────────────────────────────

def _result_path(base_dir: Path, key: str) -> Path:
    return base_dir / f"cmp_shard_{key}_merged" / f"results_s6_{key}.jsonl"


def load_results(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
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
        return 10_000.0, True


def compute_stats(rows: list[dict]) -> dict:
    dists = [get_dist(r)[0] for r in rows]
    n = len(dists)
    gs = float(np.mean([geoscore(d) for d in dists]))
    accs = [sum(d <= t for d in dists) / n for t in THRESHOLDS]
    return {"geoscore": gs, "accs": accs, "dists": dists, "n": n}


# ── Figure 1: GeoScore bar chart ──────────────────────────────────────────────

def fig_geoscore(models: list[dict], output_dir: Path) -> None:
    available = [m for m in models if m["stats"] is not None]
    if not available:
        print("  fig_comparison_geoscore.pdf: no data, skipping")
        return

    labels = [m["label"].replace("\n", " ") for m in available]
    scores = [m["stats"]["geoscore"] for m in available]
    colors = [
        FAMILY_COLORS[m["family"]]
        if not m["is_sft"]
        else _lighten(FAMILY_COLORS[m["family"]], 0.4)
        for m in available
    ]
    hatches = ["" if not m["is_sft"] else "///" for m in available]

    fig, ax = plt.subplots(figsize=(max(8, len(available) * 0.9), 5))
    bars = ax.bar(range(len(available)), scores, color=colors, edgecolor="white", linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("GeoScore", fontsize=11)
    ax.set_title("Stage-6 GeoScore — All Models", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(scores) * 1.18)

    for i, (bar, s) in enumerate(zip(bars, scores)):
        ax.text(bar.get_x() + bar.get_width() / 2, s + max(scores) * 0.01,
                f"{s:.1f}", ha="center", va="bottom", fontsize=8)

    # Legend for SFT indicator
    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="With NAVIG SFT"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="Base model"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    fig.tight_layout()
    out = output_dir / "fig_comparison_geoscore.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 2: SFT delta chart ─────────────────────────────────────────────────

def fig_sft_delta(models_by_key: dict, output_dir: Path) -> None:
    pairs = [(b, s, lbl) for b, s, lbl in SFT_PAIRS
             if b in models_by_key and s in models_by_key
             and models_by_key[b] is not None and models_by_key[s] is not None]
    if not pairs:
        print("  fig_comparison_sft_delta.pdf: no paired data, skipping")
        return

    labels = [lbl for _, _, lbl in pairs]
    deltas = [models_by_key[s]["geoscore"] - models_by_key[b]["geoscore"]
              for b, s, _ in pairs]
    colors = [FAMILY_COLORS[{"LLaVA-1.6": "LLaVA", "MiniCPM-V-2.6": "CPM",
                               "Qwen2.5-7B": "Qwen"}.get(lbl, "Other")]
              for lbl in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(range(len(pairs)), deltas, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for bar, d in zip(bars, deltas):
        va = "bottom" if d >= 0 else "top"
        offset = max(abs(d) * 0.04, 0.5)
        ax.text(bar.get_x() + bar.get_width() / 2,
                d + (offset if d >= 0 else -offset),
                f"{d:+.1f}", ha="center", va=va, fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("ΔGeoScore (SFT − base)", fontsize=11)
    ax.set_title("Impact of NAVIG SFT on GeoScore", fontsize=13, fontweight="bold")

    fig.tight_layout()
    out = output_dir / "fig_comparison_sft_delta.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 3: Accuracy at 5 thresholds (base vs SFT) ─────────────────────────

def fig_accuracy(models_by_key: dict, output_dir: Path) -> None:
    pairs = [(b, s, lbl) for b, s, lbl in SFT_PAIRS
             if b in models_by_key and s in models_by_key
             and models_by_key[b] is not None and models_by_key[s] is not None]
    if not pairs:
        print("  fig_comparison_accuracy.pdf: no paired data, skipping")
        return

    n_thr = len(THRESHOLDS)
    n_pairs = len(pairs)
    group_width = 0.8
    bar_width = group_width / (2 * n_pairs)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(n_thr)

    for i, (base_key, sft_key, lbl) in enumerate(pairs):
        family = {"LLaVA-1.6": "LLaVA", "MiniCPM-V-2.6": "CPM", "Qwen2.5-7B": "Qwen"}.get(lbl, "Other")
        color = FAMILY_COLORS[family]

        base_accs = models_by_key[base_key]["accs"]
        sft_accs = models_by_key[sft_key]["accs"]

        offset_base = (2 * i - n_pairs + 0.5) * bar_width * 1.05
        offset_sft = offset_base + bar_width * 1.05

        ax.bar(x + offset_base, base_accs, bar_width,
               color=color, alpha=0.55, edgecolor="white", label=f"{lbl} (base)")
        ax.bar(x + offset_sft, sft_accs, bar_width,
               color=color, alpha=1.0, edgecolor="white", hatch="///",
               label=f"{lbl} (SFT)")

    ax.set_xticks(x)
    ax.set_xticklabels(THR_LABELS, fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("Accuracy at Distance Thresholds — Base vs. SFT", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left")

    fig.tight_layout()
    out = output_dir / "fig_comparison_accuracy.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 4: Distance CDF ────────────────────────────────────────────────────

def fig_cdf(models: list[dict], output_dir: Path) -> None:
    available = [m for m in models if m["stats"] is not None]
    if not available:
        print("  fig_comparison_cdf.pdf: no data, skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in available:
        dists = sorted(m["stats"]["dists"])
        y = np.arange(1, len(dists) + 1) / len(dists)
        color = FAMILY_COLORS[m["family"]]
        linestyle = "--" if m["is_sft"] else "-"
        alpha = 1.0 if m["is_sft"] else 0.65
        label = m["label"].replace("\n", " ")
        ax.plot(dists, y, color=color, linestyle=linestyle, linewidth=1.5,
                alpha=alpha, label=label)

    for thr in THRESHOLDS:
        ax.axvline(thr, color="gray", linewidth=0.6, linestyle=":")

    ax.set_xscale("log")
    ax.set_xlabel("Distance error (km, log scale)", fontsize=11)
    ax.set_ylabel("Fraction of images", fontsize=11)
    ax.set_title("Empirical CDF of Prediction Error", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax.set_xlim(left=1)

    fig.tight_layout()
    out = output_dir / "fig_comparison_cdf.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _lighten(hex_color: str, amount: float) -> str:
    """Return *hex_color* blended toward white by *amount* (0–1)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


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

    # Load results for each registered model
    models: list[dict] = []
    models_by_key: dict[str, dict | None] = {}
    for key, label, family, is_sft in MODEL_REGISTRY:
        path = _result_path(base_dir, key)
        rows = load_results(path)
        if rows is None:
            print(f"  [{key}] not found: {path}")
            stats = None
        else:
            stats = compute_stats(rows)
            print(f"  [{key}] n={stats['n']}  GeoScore={stats['geoscore']:.2f}")
        entry = {"key": key, "label": label, "family": family,
                 "is_sft": is_sft, "stats": stats}
        models.append(entry)
        models_by_key[key] = stats

    print(f"\nGenerating figures in {output_dir}/")
    fig_geoscore(models, output_dir)
    fig_sft_delta(models_by_key, output_dir)
    fig_accuracy(models_by_key, output_dir)
    fig_cdf(models, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
