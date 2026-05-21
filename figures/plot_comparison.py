#!/usr/bin/env python3
"""Generate publication figures for the NAVIG evaluation.

Produces ten PDFs (numbered to match model_justification.tex figure order):

  fig1_geoscore.pdf      — GeoScore bar chart for all models, SFT pairs grouped
  fig2_thresholds.pdf    — Accuracy at 5 distance thresholds for all models
  fig3_distribution.pdf  — Distance violin + IQR box chart
  fig4_evidence.pdf      — Evidence-component GeoScore deltas
  fig5_geographic.pdf    — Geographic GeoScore heatmap by continent
  fig6_cdf.pdf           — Empirical CDF of prediction error (km) for all models
  fig7_difficulty.pdf    — Accuracy breakdown by image difficulty tercile
  fig8_agreement.pdf     — Pairwise joint accuracy heatmap
  fig9_failure_modes.pdf — Prediction outcome decomposition (stacked bar)
  fig10_sft_delta.pdf    — GeoScore delta (SFT − base) for each model family

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
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

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

THR_LABELS  = [f"@{t} km" for t in THRESHOLDS]
PENALTY_KM  = 10_000.0


# ── Geo utilities ─────────────────────────────────────────────────────────────

def lat_to_continent(lat, lon):
    if lat > 66.5:
        return "Arctic"
    if lat < -60:
        return "Antarctica"
    if -35 < lat < 37 and -20 < lon < 55:
        return "Africa"
    if lat > 35 and -30 < lon < 60:
        return "Europe"
    if lat > 0 and 60 < lon < 180:
        return "Asia (N)"
    if -10 < lat < 35 and 60 < lon < 180:
        return "Asia (S/SE)"
    if -55 < lat < -10 and 110 < lon < 180:
        return "Oceania"
    if 15 < lat < 75 and -170 < lon < -50:
        return "N. America"
    if -60 < lat < 15 and -90 < lon < -30:
        return "S. America"
    return "Other"


# ── Data loading ──────────────────────────────────────────────────────────────

def _result_path(base_dir: Path, key: str) -> Path:
    return base_dir / f"cmp_shard_{key}_merged" / f"results_s6_{key}.jsonl"


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
    gs_excl = float(np.mean([geoscore(d) for d in success_dists])) if success_dists else float("nan")
    accs = [sum(d <= t for d in dists) / n for t in THRESHOLDS]
    return {
        "geoscore": gs, "accs": accs, "dists": dists, "n": n,
        "fails": fails, "success_dists": success_dists, "gs_excl": gs_excl,
        "p25": float(np.percentile(dists, 25)),
        "p50": float(np.percentile(dists, 50)),
        "p75": float(np.percentile(dists, 75)),
    }


def compute_evidence_deltas(rows: dict) -> dict:
    ev_types = ["reasoning", "osm", "rag", "comment"]
    buckets = {ev: {"present": [], "absent": []} for ev in ev_types}
    for row in rows.values():
        d, _ = get_dist(row)
        usage = row.get("usage", {})
        for ev in ev_types:
            if usage.get(ev, 0):
                buckets[ev]["present"].append(d)
            else:
                buckets[ev]["absent"].append(d)
    result = {}
    for ev in ev_types:
        p = buckets[ev]["present"]
        a = buckets[ev]["absent"]
        gs_p = float(np.mean([geoscore(d) for d in p])) if p else float("nan")
        gs_a = float(np.mean([geoscore(d) for d in a])) if a else float("nan")
        result[ev] = {"gs_present": gs_p, "n_present": len(p),
                      "gs_absent": gs_a, "n_absent": len(a),
                      "delta": gs_p - gs_a}
    return result


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


def compute_predictions(rows: dict) -> list[dict]:
    """Per-image predictions with coordinates; used for scatter/density/bias plots."""
    result = []
    for row in rows.values():
        true_lat = float(row["LAT"])
        true_lon = float(row["LON"])
        dist, fail = get_dist(row)
        try:
            ans = row.get("answer") or {}
            pred_lat = parse_coord(ans["latitude"])
            pred_lon = parse_coord(ans["longitude"])
        except Exception:
            pred_lat = pred_lon = float("nan")
        usage = row.get("usage") or {}
        result.append({
            "true_lat": true_lat, "true_lon": true_lon,
            "pred_lat": pred_lat, "pred_lon": pred_lon,
            "dist": dist, "fail": fail,
            "country_pred": (row.get("answer") or {}).get("country", "Unknown"),
            "usage_count": int(sum(usage.values())),
            "usage": usage,
        })
    return result


def compute_crop_stats(rows: dict) -> dict:
    """Fraction of images where each GroundingDINO crop category was detected."""
    cats = ["road sign", "house", "building sign"]
    counts = {cat: 0 for cat in cats}
    n = len(rows)
    for row in rows.values():
        crop = row.get("crop") or {}
        for cat in cats:
            if crop.get(cat):
                counts[cat] += 1
    return {"n": n, "cats": cats, "counts": counts}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _lighten(hex_color: str, amount: float) -> str:
    """Return *hex_color* blended toward white by *amount* (0–1)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _model_color(m: dict) -> str:
    c = FAMILY_COLORS[m["family"]]
    return _lighten(c, 0.4) if m["is_sft"] else c


# ── Figure 1: GeoScore bar chart ──────────────────────────────────────────────

def fig1_geoscore(models: list[dict], output_dir: Path) -> None:
    available = [m for m in models if m["stats"] is not None]
    if not available:
        print("  fig1_geoscore.pdf: no data, skipping")
        return

    labels = [m["label"].replace("\n", " ") for m in available]
    scores = [m["stats"]["geoscore"] for m in available]
    colors = [_model_color(m) for m in available]
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

    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="With NAVIG SFT"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="Base model"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    fig.tight_layout()
    out = output_dir / "fig1_geoscore.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 2: Accuracy at 5 thresholds (all models) ──────────────────────────

THR_DISPLAY = ["1 km\n(street)", "25 km\n(city)", "200 km\n(region)",
               "750 km\n(country)", "2500 km\n(continent)"]
MARKERS = ["o", "s", "^", "D", "P", "v", "X", "<", ">"]

def fig2_thresholds(models: list[dict], output_dir: Path) -> None:
    available = [m for m in models if m["stats"] is not None]
    if not available:
        print("  fig2_thresholds.pdf: no data, skipping")
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    for m, mk in zip(available, MARKERS):
        accs = [a * 100 for a in m["stats"]["accs"]]
        color = _model_color(m)
        linestyle = "--" if m["is_sft"] else "-"
        ax.plot(range(len(THRESHOLDS)), accs,
                marker=mk, color=color, linestyle=linestyle,
                linewidth=1.8, markersize=6,
                label=m["label"].replace("\n", " "),
                zorder=3)
        ax.annotate(f"{accs[-1]:.1f}%",
                    xy=(len(THRESHOLDS) - 1, accs[-1]),
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=7, color=color, va="center")

    ax.set_xticks(range(len(THRESHOLDS)))
    ax.set_xticklabels(THR_DISPLAY, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 85)
    ax.set_yticks(range(0, 86, 10))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7.5)

    fig.tight_layout()
    out = output_dir / "fig2_thresholds.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 3: Distance distribution (violin + IQR) ───────────────────────────

def fig3_distribution(models: list[dict], output_dir: Path) -> None:
    available = [m for m in models if m["stats"] is not None]
    if not available:
        print("  fig3_distribution.pdf: no data, skipping")
        return

    colors = [_model_color(m) for m in available]
    labels = [m["label"].replace("\n", " ") for m in available]

    fig, axes = plt.subplots(1, 2, figsize=(max(7, len(available) * 0.9), 3.6),
                             layout="constrained", gridspec_kw={"wspace": 0.35})

    ax = axes[0]
    all_dists = [m["stats"]["dists"] for m in available]
    parts = ax.violinplot(all_dists, positions=range(len(available)),
                          showmedians=True, showextrema=False, widths=0.6)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor(color)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    ax.set_yscale("log")
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel("Distance to ground truth (km, log scale)")
    ax.set_ylim(0.05, 25000)
    ax.axhline(PENALTY_KM, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(len(available) - 0.5, PENALTY_KM * 1.2, "parse-fail\npenalty",
            fontsize=6.5, color="red", ha="right", va="bottom")
    ax.set_title("(a) Full distribution (incl. failures)")

    ax2 = axes[1]
    x = np.arange(len(available))
    p25 = [m["stats"]["p25"] for m in available]
    p50 = [m["stats"]["p50"] for m in available]
    p75 = [m["stats"]["p75"] for m in available]
    bar_w = 0.5

    for i, (m, color) in enumerate(zip(available, colors)):
        ax2.bar(i, p75[i], bar_w, bottom=p25[i], color=color, alpha=0.35, zorder=2)
        ax2.plot([i - bar_w / 2, i + bar_w / 2], [p50[i], p50[i]],
                 color=color, linewidth=2.5, zorder=3)
        ax2.text(i, p75[i] + 120, f"{p50[i]:.0f}", ha="center",
                 fontsize=7, color=color, fontweight="bold")

    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.5, rotation=20, ha="right")
    ax2.set_ylabel("Distance (km, log scale)")
    ax2.set_ylim(10, 15000)
    ax2.set_title("(b) IQR of all predictions\n(bar = P25–P75, line = P50)")

    out = output_dir / "fig3_distribution.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 4: Evidence component GeoScore deltas ─────────────────────────────

def fig4_evidence(models: list[dict], output_dir: Path) -> None:
    available = [m for m in models if m["stats"] is not None and m.get("ev") is not None]
    if not available:
        print("  fig4_evidence.pdf: no data, skipping")
        return

    ev_display = {"comment": "COMMENT\n(patch descriptions)",
                  "rag":     "RAG\n(guidebook retrieval)",
                  "osm":     "OSM\n(Nominatim search)"}
    ev_keys = ["comment", "rag", "osm"]

    n_ev = len(ev_keys)
    n_m  = len(available)
    group_w = 0.75
    bar_w   = group_w / n_m
    x       = np.arange(n_ev)

    fig, ax = plt.subplots(figsize=(7, 3.6))

    for j, m in enumerate(available):
        color   = _model_color(m)
        offsets = x + (j - (n_m - 1) / 2) * bar_w
        deltas  = [m["ev"].get(ev, {}).get("delta", float("nan")) for ev in ev_keys]
        n_pres  = [m["ev"].get(ev, {}).get("n_present", 0) for ev in ev_keys]
        bars = ax.bar(offsets, deltas, bar_w * 0.9, color=color, alpha=0.80,
                      label=m["label"].replace("\n", " "), zorder=3)
        for bar, d, n in zip(bars, deltas, n_pres):
            if not np.isnan(d) and abs(d) > 30:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        d + (18 if d > 0 else -18),
                        f"n={n}", ha="center",
                        va="bottom" if d > 0 else "top",
                        fontsize=5.5, color=color)

    ax.axhline(0, color="black", linewidth=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([ev_display[e] for e in ev_keys], fontsize=8.5)
    ax.set_ylabel("ΔGeoScore (present − absent)")
    ax.set_xlabel("Evidence component")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    ax.axhspan(-50, 0,   alpha=0.04, color="red",   zorder=0)
    ax.axhspan(0,   200, alpha=0.04, color="green",  zorder=0)

    fig.tight_layout()
    out = output_dir / "fig4_evidence.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 5: Geographic GeoScore heatmap ────────────────────────────────────

def fig5_geographic(models: list[dict], output_dir: Path) -> None:
    available = [m for m in models
                 if m["stats"] is not None and m.get("geo") is not None and m.get("rows")]
    if not available:
        print("  fig5_geographic.pdf: no data, skipping")
        return

    cont_counts = defaultdict(int)
    for row in available[0]["rows"].values():
        cont_counts[lat_to_continent(float(row["LAT"]), float(row["LON"]))] += 1

    keep = ["Europe", "N. America", "Asia (N)", "Africa",
            "S. America", "Oceania", "Asia (S/SE)"]
    conts = [c for c in keep if cont_counts[c] > 0]

    col_labels = [m["label"].replace("\n", " ") for m in available]
    matrix     = np.full((len(conts), len(available)), np.nan)

    for j, m in enumerate(available):
        for i, cont in enumerate(conts):
            if cont in m["geo"]:
                matrix[i, j] = m["geo"][cont]

    row_labels = [f"{c}  (n={cont_counts[c]})" for c in conts]

    fig, ax = plt.subplots(figsize=(max(7, len(available) * 0.9), 3.8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=800, vmax=3500)

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(col_labels, fontsize=8.5, rotation=20, ha="right")
    ax.set_yticks(range(len(conts)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Region")

    for i in range(len(conts)):
        for j in range(len(available)):
            v = matrix[i, j]
            if not np.isnan(v):
                textcolor = "white" if v > 2800 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color=textcolor, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("GeoScore", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    out = output_dir / "fig5_geographic.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 6: Distance CDF ────────────────────────────────────────────────────

def fig6_cdf(models: list[dict], output_dir: Path) -> None:
    available = [m for m in models if m["stats"] is not None]
    if not available:
        print("  fig6_cdf.pdf: no data, skipping")
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
    out = output_dir / "fig6_cdf.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 7: Accuracy by image difficulty tercile ───────────────────────────

def fig7_difficulty(models: list[dict], output_dir: Path) -> None:
    available = [m for m in models if m["stats"] is not None and m.get("rows")]
    if not available:
        print("  fig7_difficulty.pdf: no data, skipping")
        return

    common_ids = set.intersection(*[set(m["rows"].keys()) for m in available])
    if len(common_ids) < 30:
        print("  fig7_difficulty.pdf: insufficient common IDs, skipping")
        return

    mean_dist = {id_: np.mean([get_dist(m["rows"][id_])[0] for m in available])
                 for id_ in common_ids}
    sorted_ids = sorted(common_ids, key=lambda x: mean_dist[x])
    n = len(sorted_ids)
    buckets = {
        "Easy\n(best ⅓)":  set(sorted_ids[:n // 3]),
        "Medium\n(mid ⅓)": set(sorted_ids[n // 3: 2 * n // 3]),
        "Hard\n(worst ⅓)": set(sorted_ids[2 * n // 3:]),
    }

    THR_SHOW     = [25, 200, 750]
    bucket_keys  = list(buckets.keys())
    n_b          = len(bucket_keys)
    n_m          = len(available)
    x            = np.arange(n_b)
    bar_w        = 0.8 / n_m

    fig, axes = plt.subplots(1, len(THR_SHOW), figsize=(8, 3.5), layout="constrained",
                             sharey=True, gridspec_kw={"wspace": 0.15})

    for ax, thr in zip(axes, THR_SHOW):
        for j, m in enumerate(available):
            color = _model_color(m)
            accs = [
                np.mean([get_dist(m["rows"][id_])[0] <= thr
                         for id_ in buckets[bk] if id_ in m["rows"]]) * 100
                for bk in bucket_keys
            ]
            offsets = x + (j - (n_m - 1) / 2) * bar_w
            ax.bar(offsets, accs, bar_w * 0.9, color=color, alpha=0.82, zorder=3,
                   label=m["label"].replace("\n", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_keys, fontsize=8)
        ax.set_title(f"@{thr} km", fontsize=9)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        ax.set_ylim(0, 95)

    axes[0].set_ylabel("Accuracy (%)")
    axes[-1].legend(ncol=1, fontsize=7, loc="upper right", framealpha=0.9)
    fig.suptitle("Accuracy by image difficulty", fontsize=10)

    out = output_dir / "fig7_difficulty.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 8: Pairwise joint accuracy heatmap ────────────────────────────────

def fig8_agreement(models: list[dict], output_dir: Path) -> None:
    available = [m for m in models if m["stats"] is not None and m.get("rows")]
    if not available:
        print("  fig8_agreement.pdf: no data, skipping")
        return

    n      = len(available)
    labels = [m["label"].replace("\n", " ") for m in available]
    T      = 200

    matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            common = set(available[i]["rows"].keys()) & set(available[j]["rows"].keys())
            if not common:
                continue
            joint = np.mean([
                get_dist(available[i]["rows"][id_])[0] <= T and
                get_dist(available[j]["rows"][id_])[0] <= T
                for id_ in common
            ]) * 100
            matrix[i, j] = joint

    mask         = np.triu(np.ones((n, n), dtype=bool), k=1)
    matrix_disp  = np.where(mask, np.nan, matrix)

    fig, ax = plt.subplots(figsize=(max(5.5, n * 0.7), max(4.5, n * 0.6)))
    im = ax.imshow(matrix_disp, cmap="Blues", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"Joint accuracy @{T} km (%)", fontsize=10)

    for i in range(n):
        for j in range(n):
            if not mask[i, j] and not np.isnan(matrix_disp[i, j]):
                v = matrix_disp[i, j]
                textcolor = "white" if v > 55 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=8, color=textcolor)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label("Joint accuracy (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    out = output_dir / "fig8_agreement.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 9: Prediction outcome decomposition ───────────────────────────────

def fig9_failure_modes(models: list[dict], output_dir: Path) -> None:
    available = [m for m in models
                 if m["stats"] is not None and m.get("fail_modes") is not None]
    if not available:
        print("  fig9_failure_modes.pdf: no data, skipping")
        return

    cat_labels = ["≤25 km\n(city)", "25–200 km\n(region)",
                  "200–2500 km\n(country+)", ">2500 km\n(wrong cont.)", "Parse\nfailure"]
    cat_keys   = ["≤25km", "25–200km", "200–2500km", ">2500km", "parse_fail"]
    cat_colors = ["#009E73", "#56B4E9", "#E69F00", "#D55E00", "#999999"]

    fig, ax = plt.subplots(figsize=(max(7, len(available) * 0.9), 3.6))
    x       = np.arange(len(available))
    bottoms = np.zeros(len(available))

    for key, color, label in zip(cat_keys, cat_colors, cat_labels):
        fracs = [m["fail_modes"][key] / m["stats"]["n"] * 100 for m in available]
        ax.bar(x, fracs, bottom=bottoms, color=color, label=label, zorder=3)
        for xi, (frac, bot) in enumerate(zip(fracs, bottoms)):
            if frac >= 5:
                ax.text(xi, bot + frac / 2, f"{frac:.0f}%",
                        ha="center", va="center", fontsize=7,
                        color="white" if color in ("#D55E00", "#009E73") else "black",
                        fontweight="bold")
        bottoms = bottoms + np.array(fracs)

    ax.set_xticks(x)
    ax.set_xticklabels([m["label"].replace("\n", " ") for m in available],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 102)
    ax.set_ylabel("% of images")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(loc="upper right", ncol=2, fontsize=7, framealpha=0.9)
    ax.set_title("Prediction outcome decomposition")

    fig.tight_layout()
    out = output_dir / "fig9_failure_modes.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 10: SFT delta chart ────────────────────────────────────────────────

def fig10_sft_delta(models_by_key: dict, output_dir: Path) -> None:
    pairs = [(b, s, lbl) for b, s, lbl in SFT_PAIRS
             if b in models_by_key and s in models_by_key
             and models_by_key[b] is not None and models_by_key[s] is not None]
    if not pairs:
        print("  fig10_sft_delta.pdf: no paired data, skipping")
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
    out = output_dir / "fig10_sft_delta.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET FIGURES  (describe Im2GPS3k test set, independent of any model)
# ── Citation: Hays & Efros 2008/2015; Astruc et al. CVPR 2024 (OSV5M) ────────
# ═══════════════════════════════════════════════════════════════════════════════

# ── Figure 11: Ground-truth location scatter map ──────────────────────────────

def fig11_dataset_map(models: list[dict], output_dir: Path) -> None:
    """World scatter of Im2GPS3k ground-truth positions.
    Standard in every geolocalization paper (Im2GPS; OSV5M; IMAGEO-Bench 2025)."""
    ref = next((m for m in models if m.get("rows")), None)
    if ref is None:
        print("  fig11_dataset_map.pdf: no rows, skipping")
        return

    lats = [float(r["LAT"]) for r in ref["rows"].values()]
    lons = [float(r["LON"]) for r in ref["rows"].values()]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_facecolor("#dceef7")
    ax.scatter(lons, lats, s=1.5, alpha=0.35, color="#D55E00", rasterized=True, zorder=2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_xticklabels([f"{x}°" for x in range(-180, 181, 60)], fontsize=7)
    ax.set_yticklabels([f"{y}°" for y in range(-90, 91, 30)], fontsize=7)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title(f"Im2GPS3k ground-truth distribution  (n = {len(lats):,})",
                 fontsize=11, fontweight="bold")
    ax.grid(True, linewidth=0.3, alpha=0.5, linestyle="--", zorder=1)

    fig.tight_layout()
    out = output_dir / "fig11_dataset_map.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 12: Dataset continent distribution ─────────────────────────────────

def fig12_dataset_continent(models: list[dict], output_dir: Path) -> None:
    """Bar chart of test images per continent.
    Standard dataset characterization (OSV5M; IMAGEO-Bench 2025)."""
    ref = next((m for m in models if m.get("rows")), None)
    if ref is None:
        print("  fig12_dataset_continent.pdf: no rows, skipping")
        return

    cont_counts: dict[str, int] = defaultdict(int)
    for row in ref["rows"].values():
        cont_counts[lat_to_continent(float(row["LAT"]), float(row["LON"]))] += 1

    order  = sorted(cont_counts, key=lambda c: -cont_counts[c])
    counts = [cont_counts[c] for c in order]
    total  = sum(counts)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(order)))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.barh(order[::-1], counts[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Number of images", fontsize=9)
    ax.set_title(f"Im2GPS3k images per continent  (n = {total:,})",
                 fontsize=10, fontweight="bold")
    for bar, c in zip(bars, counts[::-1]):
        ax.text(c + 8, bar.get_y() + bar.get_height() / 2,
                f"{c}  ({c / total * 100:.1f}%)", va="center", fontsize=8)
    ax.set_xlim(0, max(counts) * 1.25)

    fig.tight_layout()
    out = output_dir / "fig12_dataset_continent.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE FIGURES  (aggregate model performance metrics)
# ── Citations: Im2GPS; PIGEON CVPR 2024; GeoReasoner ICML 2024; NAVIG 2025 ───
# ═══════════════════════════════════════════════════════════════════════════════

# ── Figure 13: Median geodesic error per model ────────────────────────────────

def fig13_median_error(models: list[dict], output_dir: Path) -> None:
    """Bar chart of median ± IQR prediction error.
    Complements GeoScore; reported in PIGEON Table 1, GeoReasoner Table 1."""
    available = [m for m in models if m["stats"] is not None]
    if not available:
        print("  fig13_median_error.pdf: no data, skipping")
        return

    labels  = [m["label"].replace("\n", " ") for m in available]
    medians = np.array([m["stats"]["p50"] for m in available])
    p25     = np.array([m["stats"]["p25"] for m in available])
    p75     = np.array([m["stats"]["p75"] for m in available])
    colors  = [_model_color(m) for m in available]

    fig, ax = plt.subplots(figsize=(max(7, len(available) * 0.9), 4.2))
    x = np.arange(len(available))
    ax.bar(x, medians, color=colors, alpha=0.85, zorder=3)
    ax.errorbar(x, medians, yerr=[medians - p25, p75 - medians],
                fmt="none", color="black", linewidth=1.3, capsize=5, zorder=4)

    for i, (color, val) in enumerate(zip(colors, medians)):
        ax.text(i, val * 1.08, f"{val:.0f} km", ha="center", va="bottom",
                fontsize=7.5, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Median geodesic error (km)", fontsize=10)
    ax.set_title("Median prediction error  (error bars = P25–P75)", fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    out = output_dir / "fig13_median_error.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 14: GeoScore distribution (violin) ────────────────────────────────

def fig14_geoscore_boxplot(models: list[dict], output_dir: Path) -> None:
    """Violin plot of per-image GeoScore distributions.
    Reveals spread, not just mean; complements fig1 (PIGEON Table 1 reports mean
    GeoGuessr score; showing the full distribution is cited in arXiv 2502.14412)."""
    available = [m for m in models if m["stats"] is not None]
    if not available:
        print("  fig14_geoscore_boxplot.pdf: no data, skipping")
        return

    all_scores = [[geoscore(d) for d in m["stats"]["dists"]] for m in available]
    colors     = [_model_color(m) for m in available]
    labels     = [m["label"].replace("\n", " ") for m in available]

    fig, ax = plt.subplots(figsize=(max(7, len(available) * 0.9), 4.2))
    parts = ax.violinplot(all_scores, positions=range(len(available)),
                          showmedians=True, showextrema=False, widths=0.65)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.55)
        pc.set_edgecolor(color)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.8)

    means = [float(np.mean(s)) for s in all_scores]
    ax.scatter(range(len(available)), means, color=colors, s=30, zorder=4,
               marker="D", label="Mean")

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("GeoScore (0–5000)", fontsize=10)
    ax.set_ylim(0, 5100)
    ax.set_title("Per-image GeoScore distribution  (◆ = mean, bar = median)",
                 fontsize=11, fontweight="bold")

    fig.tight_layout()
    out = output_dir / "fig14_geoscore_boxplot.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 15: Error percentile chart ────────────────────────────────────────

def fig15_error_percentiles(models: list[dict], output_dir: Path) -> None:
    """Grouped bar of P10/P25/P50/P75/P90 distances per model.
    Reveals tail behaviour beyond IQR; see arXiv 2502.14412 Fig 2."""
    available = [m for m in models if m["stats"] is not None]
    if not available:
        print("  fig15_error_percentiles.pdf: no data, skipping")
        return

    pcts     = [10, 25, 50, 75, 90]
    pct_lbls = ["P10", "P25", "P50", "P75", "P90"]
    pct_alphas = [0.4, 0.55, 0.75, 0.9, 1.0]
    n_m = len(available)
    n_p = len(pcts)
    x   = np.arange(n_m)
    bar_w = 0.65 / n_p

    fig, ax = plt.subplots(figsize=(max(8, n_m * 1.0), 4.2))
    for j, (pct, lbl, alpha) in enumerate(zip(pcts, pct_lbls, pct_alphas)):
        vals = [float(np.percentile(m["stats"]["dists"], pct)) for m in available]
        offsets = x + (j - (n_p - 1) / 2) * bar_w
        ax.bar(offsets, vals, bar_w * 0.92,
               label=lbl, alpha=alpha, color="steelblue", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([m["label"].replace("\n", " ") for m in available],
                       rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Geodesic error (km)", fontsize=10)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title("Prediction error percentiles per model", fontsize=11, fontweight="bold")
    ax.legend(title="Percentile", fontsize=8, ncol=5, loc="upper left")

    fig.tight_layout()
    out = output_dir / "fig15_error_percentiles.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR ANALYSIS FIGURES
# ── Citations: IMAGEO-Bench 2025 Fig 3; VLMs as GeoGuessr Masters 2025;
#               arXiv 2502.14412 Fig 7 ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# ── Figure 16: Mean GeoScore by continent (heatmap + grouped bars) ────────────

def fig16_error_by_continent(models: list[dict], output_dir: Path) -> None:
    """Per-continent mean GeoScore per model as a grouped bar chart.
    Shows geographic bias; cited as key analysis in VLMs as GeoGuessr Masters
    (Table 3) and IMAGEO-Bench (2025, Figures 3–4)."""
    available = [m for m in models if m["stats"] is not None and m.get("geo") is not None]
    if not available:
        print("  fig16_error_by_continent.pdf: no data, skipping")
        return

    keep   = ["Europe", "N. America", "Asia (N)", "Asia (S/SE)",
              "Africa", "S. America", "Oceania"]
    conts  = [c for c in keep if any(c in m["geo"] for m in available)]
    n_c    = len(conts)
    n_m    = len(available)
    x      = np.arange(n_c)
    bar_w  = 0.75 / n_m

    fig, ax = plt.subplots(figsize=(max(8, n_c * 1.1), 4.2))
    for j, m in enumerate(available):
        vals    = [m["geo"].get(c, float("nan")) for c in conts]
        offsets = x + (j - (n_m - 1) / 2) * bar_w
        ax.bar(offsets, vals, bar_w * 0.92,
               color=_model_color(m), alpha=0.82, zorder=3,
               label=m["label"].replace("\n", " "))

    ax.set_xticks(x)
    ax.set_xticklabels(conts, fontsize=9)
    ax.set_ylabel("Mean GeoScore", fontsize=10)
    ax.set_title("Mean GeoScore by continent", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = output_dir / "fig16_error_by_continent.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 17: Systematic coordinate bias (predicted − true) ─────────────────

def fig17_coordinate_bias(models: list[dict], output_dir: Path) -> None:
    """Box plots of (predicted_lat − true_lat) and (predicted_lon − true_lon).
    Reveals systematic over/under-shooting biases; cited in IMAGEO-Bench 2025
    Figure 2 (scatter of predicted vs. true latitude, colored by confidence)."""
    available = [m for m in models
                 if m["stats"] is not None and m.get("preds") is not None]
    if not available:
        print("  fig17_coordinate_bias.pdf: no data, skipping")
        return

    lat_biases = [[p["pred_lat"] - p["true_lat"]
                   for p in m["preds"] if not p["fail"] and not np.isnan(p["pred_lat"])]
                  for m in available]
    lon_biases = [[p["pred_lon"] - p["true_lon"]
                   for p in m["preds"] if not p["fail"] and not np.isnan(p["pred_lon"])]
                  for m in available]
    labels = [m["label"].replace("\n", " ") for m in available]
    colors = [_model_color(m) for m in available]

    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(available) * 1.0), 4),
                             layout="constrained")
    for ax, biases, title, refline in zip(
            axes, [lat_biases, lon_biases],
            ["Latitude bias (predicted − true, °)", "Longitude bias (predicted − true, °)"],
            [0, 0]):
        parts = ax.violinplot(biases, positions=range(len(available)),
                              showmedians=True, showextrema=False, widths=0.6)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
            pc.set_edgecolor(color)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.8)
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", zorder=1)
        ax.set_xticks(range(len(available)))
        ax.set_xticklabels(labels, fontsize=7.5, rotation=20, ha="right")
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold")

    fig.suptitle("Systematic coordinate prediction bias", fontsize=11, fontweight="bold")

    out = output_dir / "fig17_coordinate_bias.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 18: Median error by ground-truth latitude band ────────────────────

def fig18_error_vs_latitude(models: list[dict], output_dir: Path) -> None:
    """Line chart of median geodesic error by latitude band of the ground truth.
    Quantifies polar/tropical performance asymmetry; similar to geographic error
    heat maps in arXiv 2502.14412 and VLMs as GeoGuessr Masters."""
    available = [m for m in models
                 if m["stats"] is not None and m.get("preds") is not None]
    if not available:
        print("  fig18_error_vs_latitude.pdf: no data, skipping")
        return

    bands      = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]
    band_lbls  = [f"{lo}° to {hi}°" for lo, hi in bands]
    markers    = MARKERS

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m, mk in zip(available, markers):
        band_medians = []
        for lo, hi in bands:
            ds = [p["dist"] for p in m["preds"] if lo <= p["true_lat"] < hi]
            band_medians.append(float(np.median(ds)) if ds else float("nan"))
        ax.plot(range(len(bands)), band_medians,
                marker=mk, color=_model_color(m), linewidth=1.8, markersize=6,
                label=m["label"].replace("\n", " "), zorder=3)

    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(band_lbls, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Median geodesic error (km)", fontsize=9)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title("Median error by ground-truth latitude band", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = output_dir / "fig18_error_vs_latitude.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE / EVIDENCE ANALYSIS FIGURES  (NAVIG-specific)
# ── Citations: NAVIG 2025 Table 6 ablation; PIGEON 2024 Table 1 ablation ──────
# ═══════════════════════════════════════════════════════════════════════════════

# ── Figure 19: Evidence usage rates per model ─────────────────────────────────

def fig19_evidence_usage(models: list[dict], output_dir: Path) -> None:
    """Grouped bar of the fraction of images that used each evidence type.
    Shows which pipeline components fire most; motivated by NAVIG Table 6 ablation
    and PIGEON Table 1 component ablation."""
    available = [m for m in models
                 if m["stats"] is not None and m.get("preds") is not None]
    if not available:
        print("  fig19_evidence_usage.pdf: no data, skipping")
        return

    ev_keys  = ["reasoning", "rag", "osm", "comment"]
    ev_lbls  = ["Reasoning", "RAG", "OSM", "Comment"]
    n_ev, n_m = len(ev_keys), len(available)
    x, bar_w  = np.arange(n_ev), 0.75 / n_m

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for j, m in enumerate(available):
        rates   = [float(np.mean([p["usage"].get(ev, 0) for p in m["preds"]])) * 100
                   for ev in ev_keys]
        offsets = x + (j - (n_m - 1) / 2) * bar_w
        ax.bar(offsets, rates, bar_w * 0.92,
               color=_model_color(m), alpha=0.82, zorder=3,
               label=m["label"].replace("\n", " "))

    ax.set_xticks(x)
    ax.set_xticklabels(ev_lbls, fontsize=9)
    ax.set_ylabel("Images using evidence (%)", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_ylim(0, 110)
    ax.set_title("Evidence component usage rate per model", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = output_dir / "fig19_evidence_usage.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 20: Evidence count vs. mean GeoScore ───────────────────────────────

def fig20_evidence_count_accuracy(models: list[dict], output_dir: Path) -> None:
    """Line chart of mean GeoScore by number of active evidence components (0–4).
    Tests whether stacking more evidence always helps; motivated by NAVIG ablation
    (Table 6) and PIGEON's cumulative ablation (Table 1)."""
    available = [m for m in models
                 if m["stats"] is not None and m.get("preds") is not None]
    if not available:
        print("  fig20_evidence_count_accuracy.pdf: no data, skipping")
        return

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for m, mk in zip(available, MARKERS):
        buckets: dict[int, list[float]] = defaultdict(list)
        for p in m["preds"]:
            buckets[p["usage_count"]].append(geoscore(p["dist"]))
        counts = sorted(buckets)
        means  = [float(np.mean(buckets[c])) for c in counts]
        ns     = [len(buckets[c]) for c in counts]
        ax.plot(counts, means, marker=mk, color=_model_color(m),
                linewidth=1.8, markersize=6,
                label=m["label"].replace("\n", " "), zorder=3)
        for c, gs, n in zip(counts, means, ns):
            ax.annotate(f"n={n}", xy=(c, gs), xytext=(2, 3),
                        textcoords="offset points", fontsize=5.5,
                        color=_model_color(m))

    ax.set_xlabel("Number of active evidence components (0–4)", fontsize=9)
    ax.set_ylabel("Mean GeoScore", fontsize=9)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_title("GeoScore vs. evidence count", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    out = output_dir / "fig20_evidence_count_accuracy.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 21: OSM geocoding impact on accuracy ───────────────────────────────

def fig21_osm_impact(models: list[dict], output_dir: Path) -> None:
    """Paired bar: mean GeoScore when OSM geocoding fired vs. did not.
    Quantifies the Nominatim search contribution; mirrors the tool-augmented
    comparison in arXiv 2502.14412 Tables 2–4 and NAVIG Table 6 ablation."""
    available = [m for m in models
                 if m["stats"] is not None and m.get("preds") is not None]
    if not available:
        print("  fig21_osm_impact.pdf: no data, skipping")
        return

    labels   = [m["label"].replace("\n", " ") for m in available]
    gs_hit   = []
    gs_miss  = []
    n_hit    = []
    for m in available:
        hit  = [geoscore(p["dist"]) for p in m["preds"] if p["usage"].get("osm", 0)]
        miss = [geoscore(p["dist"]) for p in m["preds"] if not p["usage"].get("osm", 0)]
        gs_hit.append(float(np.mean(hit))  if hit  else float("nan"))
        gs_miss.append(float(np.mean(miss)) if miss else float("nan"))
        n_hit.append(len(hit))

    x, w = np.arange(len(available)), 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(available) * 0.9), 4))
    b1 = ax.bar(x - w / 2, gs_miss, w, label="OSM not used", color="steelblue",  alpha=0.7, zorder=3)
    b2 = ax.bar(x + w / 2, gs_hit,  w, label="OSM used",     color="darkorange", alpha=0.85, zorder=3)

    for xi, (h, n) in enumerate(zip(gs_hit, n_hit)):
        if not np.isnan(h):
            ax.text(xi + w / 2, h + 15, f"n={n}", ha="center", va="bottom",
                    fontsize=6.5, color="darkorange")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Mean GeoScore", fontsize=10)
    ax.set_title("GeoScore: OSM geocoding used vs. not", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    out = output_dir / "fig21_osm_impact.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 22: Crop category detection rate ───────────────────────────────────

def fig22_crop_detection(models: list[dict], output_dir: Path) -> None:
    """Grouped bar: fraction of images where GroundingDINO detected each crop
    category (road sign, house, building sign) per model.
    Motivated by NAVIG pipeline description and PIGEON's structured geocell
    annotation approach."""
    available = [m for m in models
                 if m["stats"] is not None and m.get("crop_stats") is not None]
    if not available:
        print("  fig22_crop_detection.pdf: no data, skipping")
        return

    cats     = ["road sign", "house", "building sign"]
    cat_lbls = ["Road sign", "House / building", "Building sign"]
    cat_cols = ["#4C72B0", "#55A868", "#DD8452"]
    n_c, n_m = len(cats), len(available)
    x, bar_w = np.arange(n_c), 0.75 / n_m

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for j, m in enumerate(available):
        cs      = m["crop_stats"]
        rates   = [cs["counts"].get(cat, 0) / cs["n"] * 100 for cat in cats]
        offsets = x + (j - (n_m - 1) / 2) * bar_w
        ax.bar(offsets, rates, bar_w * 0.92,
               color=_model_color(m), alpha=0.82, zorder=3,
               label=m["label"].replace("\n", " "))

    ax.set_xticks(x)
    ax.set_xticklabels(cat_lbls, fontsize=9)
    ax.set_ylabel("Images with detection (%)", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_ylim(0, 110)
    ax.set_title("GroundingDINO crop detection rate per category", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = output_dir / "fig22_crop_detection.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON FIGURES  (cross-model and SFT analysis)
# ── Citations: VLMs as GeoGuessr Masters 2025 Table 1 / Figs 2–5;
#               StreetCLIP 2023; GeoReasoner ICML 2024 Table 1 ────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# ── Figure 23: Prediction density map ────────────────────────────────────────

def fig23_prediction_density(models: list[dict], output_dir: Path) -> None:
    """2D scatter comparing ground-truth point cloud to each model's predicted
    point cloud; reveals geographic prediction collapse / bias.
    Motivated by VLMs as GeoGuessr Masters Figs 2–5 (most-predicted cities)
    and IMAGEO-Bench Fig 2 (predicted vs. true latitude scatter)."""
    available = [m for m in models
                 if m["stats"] is not None and m.get("preds") is not None]
    if not available:
        print("  fig23_prediction_density.pdf: no data, skipping")
        return

    n_m  = len(available)
    ncols = min(3, n_m)
    nrows = (n_m + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows + 1, ncols,
                             figsize=(ncols * 3.5, (nrows + 1) * 2.8),
                             layout="constrained")
    axes = np.array(axes).reshape(-1, ncols)

    # Row 0: ground-truth reference
    true_lons = [p["true_lon"] for p in available[0]["preds"]]
    true_lats = [p["true_lat"] for p in available[0]["preds"]]
    for col in range(ncols):
        ax = axes[0, col]
        if col == ncols // 2:
            ax.scatter(true_lons, true_lats, s=0.8, alpha=0.25,
                       color="#333333", rasterized=True)
            ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
            ax.set_title("Ground truth", fontsize=9, fontweight="bold")
            ax.set_facecolor("#dceef7")
        else:
            ax.set_visible(False)

    # Remaining rows: one panel per model
    for idx, m in enumerate(available):
        row, col = divmod(idx, ncols)
        ax = axes[row + 1, col]
        pred_lons = [p["pred_lon"] for p in m["preds"] if not np.isnan(p["pred_lon"])]
        pred_lats = [p["pred_lat"] for p in m["preds"] if not np.isnan(p["pred_lat"])]
        ax.scatter(pred_lons, pred_lats, s=0.8, alpha=0.25,
                   color=_model_color(m), rasterized=True)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_title(m["label"].replace("\n", " "), fontsize=8, fontweight="bold")
        ax.set_facecolor("#dceef7")

    # Hide any unused axes
    for idx in range(len(available), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row + 1, col].set_visible(False)

    fig.suptitle("Predicted vs. ground-truth location density", fontsize=12, fontweight="bold")

    out = output_dir / "fig23_prediction_density.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 24: SFT GeoScore gain by continent ────────────────────────────────

def fig24_sft_gain_by_continent(models: list[dict], output_dir: Path) -> None:
    """SFT GeoScore delta (SFT − base) broken down by continent for each SFT pair.
    Shows whether fine-tuning helps uniformly or only in certain regions; extends
    the base vs. SFT analysis in GeoReasoner Table 1 and NAVIG Table 6."""
    pairs = [(b, s, lbl) for b, s, lbl in SFT_PAIRS]
    by_key = {m["key"]: m for m in models if m.get("geo") is not None}
    valid  = [(b, s, lbl) for b, s, lbl in pairs
              if b in by_key and s in by_key]
    if not valid:
        print("  fig24_sft_gain_by_continent.pdf: no paired geo data, skipping")
        return

    keep  = ["Europe", "N. America", "Asia (N)", "Asia (S/SE)",
             "Africa", "S. America", "Oceania"]
    conts = [c for c in keep if any(
        c in by_key[b]["geo"] and c in by_key[s]["geo"]
        for b, s, _ in valid)]

    x, bar_w = np.arange(len(conts)), 0.75 / len(valid)
    fig, ax  = plt.subplots(figsize=(max(7, len(conts) * 1.1), 4.2))

    for j, (b_key, s_key, lbl) in enumerate(valid):
        family  = {"LLaVA-1.6": "LLaVA", "MiniCPM-V-2.6": "CPM",
                   "Qwen2.5-7B": "Qwen"}.get(lbl, "Other")
        color   = FAMILY_COLORS[family]
        deltas  = [by_key[s_key]["geo"].get(c, float("nan")) -
                   by_key[b_key]["geo"].get(c, float("nan"))
                   for c in conts]
        offsets = x + (j - (len(valid) - 1) / 2) * bar_w
        ax.bar(offsets, deltas, bar_w * 0.92, color=color, alpha=0.85,
               zorder=3, label=lbl)

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(conts, fontsize=9)
    ax.set_ylabel("ΔGeoScore (SFT − base)", fontsize=10)
    ax.set_title("SFT improvement by continent", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.axhspan(min(0, ax.get_ylim()[0]), 0, alpha=0.04, color="red",  zorder=0)
    ax.axhspan(0, max(200, ax.get_ylim()[1]), alpha=0.04, color="green", zorder=0)

    fig.tight_layout()
    out = output_dir / "fig24_sft_gain_by_continent.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 25: Top predicted countries per model ─────────────────────────────

def fig25_top_countries(models: list[dict], output_dir: Path) -> None:
    """Horizontal bar charts of the most frequently predicted countries per model.
    Exposes geographic prediction collapse and mode bias; this exact analysis is
    the primary finding in VLMs as GeoGuessr Masters (2025, Figs 2–5 & Table 1),
    where GPT-4o/Gemini collapse predictions onto a handful of countries."""
    available = [m for m in models
                 if m["stats"] is not None and m.get("preds") is not None]
    if not available:
        print("  fig25_top_countries.pdf: no data, skipping")
        return

    TOP_N  = 12
    n_m    = len(available)
    ncols  = min(3, n_m)
    nrows  = (n_m + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.2),
                             layout="constrained")
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, m in enumerate(available):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        counts: dict[str, int] = defaultdict(int)
        for p in m["preds"]:
            c = (p["country_pred"] or "Unknown").strip()
            if c:
                counts[c] += 1
        top = sorted(counts, key=lambda k: -counts[k])[:TOP_N]
        vals = [counts[c] for c in top]
        total = sum(counts.values())

        color = _model_color(m)
        ax.barh(top[::-1], vals[::-1], color=color, alpha=0.8)
        for i, (c, v) in enumerate(zip(top[::-1], vals[::-1])):
            ax.text(v + 1, i, f"{v / total * 100:.1f}%", va="center", fontsize=6.5)
        ax.set_title(m["label"].replace("\n", " "), fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Predictions", fontsize=7.5)
        ax.set_xlim(0, max(vals) * 1.28)

    for idx in range(len(available), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Top predicted countries per model  (from answer[\"country\"] field)",
                 fontsize=11, fontweight="bold")

    out = output_dir / "fig25_top_countries.pdf"
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

    base_dir   = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models: list[dict] = []
    models_by_key: dict[str, dict | None] = {}

    for key, label, family, is_sft in MODEL_REGISTRY:
        path = _result_path(base_dir, key)
        rows = load_results(path)
        if rows is None:
            print(f"  [{key}] not found: {path}")
            stats = ev = geo = fail_modes = preds = crop_stats = None
        else:
            stats      = compute_stats(rows)
            ev         = compute_evidence_deltas(rows)
            geo        = compute_geographic(rows)
            fail_modes = compute_failure_modes(rows)
            preds      = compute_predictions(rows)
            crop_stats = compute_crop_stats(rows)
            print(f"  [{key}] n={stats['n']}  GeoScore={stats['geoscore']:.2f}")
        entry = {"key": key, "label": label, "family": family, "is_sft": is_sft,
                 "stats": stats, "rows": rows, "ev": ev, "geo": geo,
                 "fail_modes": fail_modes, "preds": preds,
                 "crop_stats": crop_stats}
        models.append(entry)
        models_by_key[key] = stats

    print(f"\nGenerating figures in {output_dir}/")
    '''fig1_geoscore(models, output_dir)
    fig2_thresholds(models, output_dir)
    fig3_distribution(models, output_dir)
    fig4_evidence(models, output_dir)
    fig5_geographic(models, output_dir)
    fig6_cdf(models, output_dir)
    fig7_difficulty(models, output_dir)
    fig8_agreement(models, output_dir)
    fig9_failure_modes(models, output_dir)
    fig10_sft_delta(models_by_key, output_dir)
    fig11_dataset_map(models, output_dir)
    fig12_dataset_continent(models, output_dir)
    fig13_median_error(models, output_dir)
    fig14_geoscore_boxplot(models, output_dir)
    fig15_error_percentiles(models, output_dir)
    fig16_error_by_continent(models, output_dir)
    fig17_coordinate_bias(models, output_dir)'''
    fig18_error_vs_latitude(models, output_dir)
    fig19_evidence_usage(models, output_dir)
    fig20_evidence_count_accuracy(models, output_dir)
    fig21_osm_impact(models, output_dir)
    fig22_crop_detection(models, output_dir)
    fig23_prediction_density(models, output_dir)
    fig24_sft_gain_by_continent(models, output_dir)
    fig25_top_countries(models, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
