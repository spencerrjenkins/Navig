#!/usr/bin/env python3
"""
plot_results.py — Publication-quality analysis figures for the NAVIG pipeline.

Produces nine PDF figures for inclusion in model_justification.tex:
  fig1_geoscore.pdf         Overall GeoScore (headline vs. excluding failures)
  fig2_thresholds.pdf       Accuracy at five distance thresholds
  fig3_distribution.pdf     Distance-percentile box chart
  fig4_evidence.pdf         Evidence-component GeoScore deltas
  fig5_geographic.pdf       Geographic GeoScore heatmap
  fig6_cdf.pdf              Cumulative distance CDF (log x-axis)
  fig7_difficulty.pdf       Accuracy breakdown by image difficulty tercile
  fig8_agreement.pdf        Pairwise joint accuracy heatmap
  fig9_failure_modes.pdf    Prediction outcome decomposition (stacked bar)

Usage::

    python figures/plot_results.py [--output_dir figures/]

All canonical result files are discovered relative to the project root.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns
from collections import defaultdict

from metrics import geoscore, haversine_distance, parse_coord, THRESHOLDS

# ── Canonical model registry ─────────────────────────────────────────────────
# (path relative to project root, display label, short label, color, failed?)

OKABE_ITO = {
    "blue":       "#0072B2",
    "orange":     "#E69F00",
    "green":      "#009E73",
    "pink":       "#CC79A7",
    "vermilion":  "#D55E00",
    "skyblue":    "#56B4E9",
    "yellow":     "#F0E442",
}

MODELS = [
    {
        "path":   "output/im2gps3k_rgb_images/llava_merged/results_s6_llava.jsonl",
        "label":  "LLaVA-1.6\n(SFT, full)",
        "short":  "LLaVA-1.6",
        "color":  OKABE_ITO["blue"],
        "failed": False,
    },
    {
        "path":   "output/im2gps3k_rgb_images/cmp_shard_deepseek_merged/results_s6_deepseek.jsonl",
        "label":  "DeepSeek-VL-7B\n(full)",
        "short":  "DeepSeek",
        "color":  OKABE_ITO["orange"],
        "failed": False,
    },
    {
        "path":   "output/im2gps3k_rgb_images/cmp_shard_llama32vision_merged/results_s6_llama32vision.jsonl",
        "label":  "LLaMA-3.2-11B\n(full)",
        "short":  "LLaMA-3.2",
        "color":  OKABE_ITO["green"],
        "failed": False,
    },
    {
        "path":   "output/im2gps3k_rgb_images/cmp_shard_cpm_merged/results_s6_cpm.jsonl",
        "label":  "MiniCPM-V-2.6\n(SFT, full)",
        "short":  "MiniCPM-V",
        "color":  OKABE_ITO["yellow"],
        "failed": False,
    },
    {
        "path":   "output/im2gps3k_rgb_images/guess_merged/results_s6_llama32.jsonl",
        "label":  "LLaMA-3.2-11B\n(stage-6 swap)",
        "short":  "LLaMA-3.2\n(swap)",
        "color":  OKABE_ITO["pink"],
        "failed": False,
    },
    {
        "path":   "output/im2gps3k_rgb_images/cmp_shard_falcon_merged/results_s6_falcon.jsonl",
        "label":  "Falcon-11B-VLM\n(full)",
        "short":  "Falcon-11B",
        "color":  OKABE_ITO["vermilion"],
        "failed": True,
    },
    {
        "path":   "output/im2gps3k_rgb_images/cmp_shard_qwen_merged/results_s6_qwen.jsonl",
        "label":  "Qwen2-VL-7B\n(SFT, full)",
        "short":  "Qwen2-VL",
        "color":  OKABE_ITO["skyblue"],
        "failed": True,
    },
]

THR_LABELS  = ["1 km\n(street)", "25 km\n(city)", "200 km\n(region)",
               "750 km\n(country)", "2500 km\n(continent)"]
PENALTY_KM  = 10_000.0   # distance assigned when parse fails

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

def load_model(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["ID"]] = r
    return rows


def get_dist(row):
    """Return (distance_km, is_failure)."""
    correct = [float(row["LAT"]), float(row["LON"])]
    try:
        ans = row["answer"]
        pred = [parse_coord(ans["latitude"]), parse_coord(ans["longitude"])]
        return haversine_distance(pred, correct), False
    except Exception:
        return PENALTY_KM, True


def compute_stats(rows):
    dists, fails = [], 0
    success_dists = []
    counts = [0] * len(THRESHOLDS)
    for row in rows.values():
        d, f = get_dist(row)
        dists.append(d)
        if f:
            fails += 1
        else:
            success_dists.append(d)
        for i, t in enumerate(THRESHOLDS):
            if d <= t:
                counts[i] += 1
    n = len(dists)
    accs = [c / n for c in counts]
    gs_all  = float(np.mean([geoscore(d) for d in dists]))
    gs_excl = float(np.mean([geoscore(d) for d in success_dists])) if success_dists else float("nan")
    return {
        "n": n, "dists": dists, "fails": fails, "success_dists": success_dists,
        "gs_all": gs_all, "gs_excl": gs_excl, "accs": accs,
        "p25": float(np.percentile(dists, 25)),
        "p50": float(np.percentile(dists, 50)),
        "p75": float(np.percentile(dists, 75)),
        "p90": float(np.percentile(dists, 90)),
        "p95": float(np.percentile(dists, 95)),
    }


def compute_evidence_deltas(rows):
    """Return {ev_type: (gs_present, n_present, gs_absent, n_absent)} for each evidence type."""
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
                      "gs_absent":  gs_a, "n_absent":  len(a),
                      "delta": gs_p - gs_a}
    return result


def compute_geographic(rows):
    """Return {continent: mean_geoscore} for all predictions."""
    geo = defaultdict(list)
    for row in rows.values():
        d, _ = get_dist(row)
        cont = lat_to_continent(float(row["LAT"]), float(row["LON"]))
        geo[cont].append(d)
    return {cont: float(np.mean([geoscore(d) for d in dists]))
            for cont, dists in geo.items()}


# ── Plot helpers ──────────────────────────────────────────────────────────────

RCPARAMS = {
    "font.size":        10,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
}
plt.rcParams.update(RCPARAMS)


def short_label(m):
    return m["short"]


# ── Figure 1: Overall GeoScore ────────────────────────────────────────────────

def fig1_geoscore(data, out_dir):
    working = [m for m in data if not m["cfg"]["failed"]]
    failed  = [m for m in data if     m["cfg"]["failed"]]
    all_m   = working + failed

    labels   = [m["cfg"]["short"] for m in all_m]
    gs_all   = [m["stats"]["gs_all"]  for m in all_m]
    gs_excl  = [m["stats"]["gs_excl"] for m in all_m]
    colors   = [m["cfg"]["color"]     for m in all_m]

    x = np.arange(len(all_m))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7, 3.8))

    bars1 = ax.bar(x - w/2, gs_all,  w, label="GeoScore (all)",
                   color=colors, alpha=0.85, zorder=3)
    bars2 = ax.bar(x + w/2, gs_excl, w, label="GeoScore (excl. failures)",
                   color=colors, alpha=0.45, edgecolor=colors, linewidth=1.2,
                   hatch="///", zorder=3)

    # Mark failed models
    for i, m in enumerate(all_m):
        if m["cfg"]["failed"]:
            ax.text(i, 80, "100%\nfail", ha="center", va="bottom",
                    fontsize=7, color="red", style="italic")

    # Value annotations on working bars
    for i, m in enumerate(working):
        ax.text(i - w/2, m["stats"]["gs_all"] + 30,
                f'{m["stats"]["gs_all"]:.0f}', ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=m["cfg"]["color"])
        if not np.isnan(m["stats"]["gs_excl"]):
            ax.text(i + w/2, m["stats"]["gs_excl"] + 30,
                    f'{m["stats"]["gs_excl"]:.0f}', ha="center", va="bottom",
                    fontsize=7.5, color=m["cfg"]["color"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("GeoScore (0–5000)")
    ax.set_ylim(0, 3800)
    ax.set_yticks(range(0, 3801, 500))
    ax.axhline(y=0, color="black", linewidth=0.5)

    patch_all  = mpatches.Patch(facecolor="grey", alpha=0.85, label="All predictions")
    patch_excl = mpatches.Patch(facecolor="grey", alpha=0.45, hatch="///",
                                label="Excluding parse failures")
    ax.legend(handles=[patch_all, patch_excl], loc="upper right", framealpha=0.9)

    fig.tight_layout()
    path = out_dir / "fig1_geoscore.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 2: Threshold accuracy ──────────────────────────────────────────────

def fig2_thresholds(data, out_dir):
    working = [m for m in data if not m["cfg"]["failed"]]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    markers = ["o", "s", "^", "D", "P", "v"]
    for m, mk in zip(working, markers):
        accs = [a * 100 for a in m["stats"]["accs"]]
        ax.plot(range(len(THRESHOLDS)), accs,
                marker=mk, color=m["cfg"]["color"],
                linewidth=1.8, markersize=6,
                label=m["cfg"]["short"].replace("\n", " "),
                zorder=3)
        ax.annotate(f'{accs[-1]:.1f}%',
                    xy=(len(THRESHOLDS) - 1, accs[-1]),
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=7, color=m["cfg"]["color"], va="center")

    ax.set_xticks(range(len(THRESHOLDS)))
    ax.set_xticklabels(THR_LABELS, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 85)
    ax.set_yticks(range(0, 86, 10))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    path = out_dir / "fig2_thresholds.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 3: Distance-percentile box chart ───────────────────────────────────

def fig3_distribution(data, out_dir):
    working = [m for m in data if not m["cfg"]["failed"]]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6), layout="constrained",
                             gridspec_kw={"wspace": 0.35})

    ax = axes[0]
    all_dists = [m["stats"]["dists"] for m in working]
    parts = ax.violinplot(all_dists, positions=range(len(working)),
                         showmedians=True, showextrema=False, widths=0.6)
    for i, (pc, m) in enumerate(zip(parts["bodies"], working)):
        pc.set_facecolor(m["cfg"]["color"])
        pc.set_alpha(0.6)
        pc.set_edgecolor(m["cfg"]["color"])
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    ax.set_yscale("log")
    ax.set_xticks(range(len(working)))
    ax.set_xticklabels([m["cfg"]["short"].replace("\n", "\n") for m in working],
                       fontsize=7.5)
    ax.set_ylabel("Distance to ground truth (km, log scale)")
    ax.set_ylim(0.05, 25000)
    ax.axhline(10000, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(len(working) - 0.5, 10000 * 1.2, "parse-fail\npenalty",
            fontsize=6.5, color="red", ha="right", va="bottom")
    ax.set_title("(a) Full distribution (incl. failures)")

    ax2 = axes[1]
    labels = [m["cfg"]["short"] for m in working]
    x = np.arange(len(working))
    p25 = [m["stats"]["p25"] for m in working]
    p50 = [m["stats"]["p50"] for m in working]
    p75 = [m["stats"]["p75"] for m in working]

    bar_w = 0.5
    for i, m in enumerate(working):
        ax2.bar(i, p75[i], bar_w, bottom=p25[i],
                color=m["cfg"]["color"], alpha=0.35, zorder=2,
                label=m["cfg"]["short"].replace("\n", " ") if i == 0 else None)
        ax2.plot([i - bar_w/2, i + bar_w/2], [p50[i], p50[i]],
                 color=m["cfg"]["color"], linewidth=2.5, zorder=3)
        ax2.text(i, p75[i] + 120, f'{p50[i]:.0f}', ha="center",
                 fontsize=7, color=m["cfg"]["color"], fontweight="bold")

    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_ylabel("Distance (km, log scale)")
    ax2.set_ylim(10, 15000)
    ax2.set_title("(b) IQR of all predictions\n(bar = P25–P75, line = P50)")

    path = out_dir / "fig3_distribution.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 4: Evidence component analysis ─────────────────────────────────────

def fig4_evidence(data, out_dir):
    working = [m for m in data if not m["cfg"]["failed"]]

    ev_display = {"comment": "COMMENT\n(patch descriptions)",
                  "rag":     "RAG\n(guidebook retrieval)",
                  "osm":     "OSM\n(Nominatim search)"}
    ev_keys    = ["comment", "rag", "osm"]

    fig, ax = plt.subplots(figsize=(7, 3.6))

    n_ev = len(ev_keys)
    n_m  = len(working)
    group_w = 0.75
    bar_w   = group_w / n_m
    x       = np.arange(n_ev)

    for j, m in enumerate(working):
        offsets = x + (j - (n_m - 1) / 2) * bar_w
        deltas  = [m["ev"].get(ev, {}).get("delta", float("nan")) for ev in ev_keys]
        n_pres  = [m["ev"].get(ev, {}).get("n_present", 0) for ev in ev_keys]
        bars = ax.bar(offsets, deltas, bar_w * 0.9,
                      color=m["cfg"]["color"], alpha=0.80,
                      label=m["cfg"]["short"].replace("\n", " "), zorder=3)
        for bar, d, n in zip(bars, deltas, n_pres):
            if not np.isnan(d) and abs(d) > 30:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        d + (18 if d > 0 else -18),
                        f'n={n}', ha="center", va="bottom" if d > 0 else "top",
                        fontsize=5.5, color=m["cfg"]["color"])

    ax.axhline(0, color="black", linewidth=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([ev_display[e] for e in ev_keys], fontsize=8.5)
    ax.set_ylabel("ΔGeoScore (present − absent)")
    ax.set_xlabel("Evidence component")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)

    ax.axhspan(-50, 0,   alpha=0.04, color="red",   zorder=0)
    ax.axhspan(0,   200, alpha=0.04, color="green",  zorder=0)

    fig.tight_layout()
    path = out_dir / "fig4_evidence.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 5: Geographic heatmap ─────────────────────────────────────────────

def fig5_geographic(data, out_dir):
    working = [m for m in data if not m["cfg"]["failed"]]

    cont_counts = defaultdict(int)
    for row in working[0]["rows"].values():
        cont_counts[lat_to_continent(float(row["LAT"]), float(row["LON"]))] += 1

    keep = ["Europe", "N. America", "Asia (N)", "Africa",
            "S. America", "Oceania", "Asia (S/SE)"]
    conts = [c for c in keep if cont_counts[c] > 0]

    col_labels = [m["cfg"]["short"].replace("\n", " ") for m in working]
    matrix     = np.full((len(conts), len(working)), np.nan)

    for j, m in enumerate(working):
        for i, cont in enumerate(conts):
            if cont in m["geo"]:
                matrix[i, j] = m["geo"][cont]

    row_labels = [f"{c}  (n={cont_counts[c]})" for c in conts]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                   vmin=800, vmax=3500)

    ax.set_xticks(range(len(working)))
    ax.set_xticklabels(col_labels, fontsize=8.5)
    ax.set_yticks(range(len(conts)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Region")

    for i in range(len(conts)):
        for j in range(len(working)):
            v = matrix[i, j]
            if not np.isnan(v):
                textcolor = "white" if v > 2800 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color=textcolor, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("GeoScore", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    path = out_dir / "fig5_geographic.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Failure-mode breakdown helper ────────────────────────────────────────────

def compute_failure_modes(rows):
    """Return counts for each prediction-quality bucket."""
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


# ── Figure 6: Cumulative distance CDF ────────────────────────────────────────

def fig6_cdf(data, out_dir):
    """Cumulative fraction of images within X km — log x-axis."""
    working = [m for m in data if not m["cfg"]["failed"]]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    x_grid = np.logspace(np.log10(0.2), np.log10(15000), 600)
    markers = ["o", "s", "^", "D", "P", "v"]

    for m, mk in zip(working, markers):
        dists_sorted = np.sort(m["stats"]["dists"])
        cdf = np.searchsorted(dists_sorted, x_grid, side="right") / len(dists_sorted) * 100
        ax.plot(x_grid, cdf, color=m["cfg"]["color"], linewidth=2,
                label=m["cfg"]["short"].replace("\n", " "))
        for pct in (50, 75):
            d_pct = np.percentile(dists_sorted, pct)
            y_pct = np.interp(d_pct, x_grid, cdf)
            ax.plot(d_pct, y_pct, mk, color=m["cfg"]["color"], markersize=5, zorder=4)

    for t, lbl in zip(THRESHOLDS, THR_LABELS):
        ax.axvline(t, color="gray", linestyle="--", linewidth=0.7, alpha=0.65, zorder=1)
        ax.text(t * 1.12, 3, lbl.split("\n")[0], fontsize=6, color="gray",
                va="bottom", rotation=0)

    ax.set_xscale("log")
    ax.set_xlim(0.2, 15000)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Distance to ground truth (km, log scale)")
    ax.set_ylabel("Cumulative % of images")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7.5)
    ax.set_title("Cumulative distance distribution")

    fig.tight_layout()
    path = out_dir / "fig6_cdf.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 7: Difficulty-bucket accuracy ─────────────────────────────────────

def fig7_difficulty(data, out_dir):
    """Accuracy at 200 km broken down by image difficulty tercile.

    Difficulty = mean prediction distance across all working models — images
    where all models struggle are 'hard'; images all models solve are 'easy'.
    """
    working = [m for m in data if not m["cfg"]["failed"]]

    common_ids = set.intersection(*[set(m["rows"].keys()) for m in working])
    if len(common_ids) < 30:
        print("  fig7: insufficient common IDs, skipping")
        return

    mean_dist = {}
    for id_ in common_ids:
        mean_dist[id_] = np.mean([get_dist(m["rows"][id_])[0] for m in working])

    sorted_ids = sorted(common_ids, key=lambda x: mean_dist[x])
    n = len(sorted_ids)
    buckets = {
        "Easy\n(best ⅓)":   set(sorted_ids[:n // 3]),
        "Medium\n(mid ⅓)":  set(sorted_ids[n // 3: 2 * n // 3]),
        "Hard\n(worst ⅓)":  set(sorted_ids[2 * n // 3:]),
    }

    THR_SHOW = [25, 200, 750]
    bucket_keys = list(buckets.keys())
    n_b = len(bucket_keys)
    n_m = len(working)
    x = np.arange(n_b)
    bar_w = 0.8 / n_m

    fig, axes = plt.subplots(1, len(THR_SHOW), figsize=(8, 3.5), layout="constrained",
                             sharey=True, gridspec_kw={"wspace": 0.15})

    for ax, thr in zip(axes, THR_SHOW):
        for j, m in enumerate(working):
            accs = []
            for bk in bucket_keys:
                ids_b = buckets[bk]
                acc = np.mean([get_dist(m["rows"][id_])[0] <= thr
                               for id_ in ids_b if id_ in m["rows"]]) * 100
                accs.append(acc)
            offsets = x + (j - (n_m - 1) / 2) * bar_w
            ax.bar(offsets, accs, bar_w * 0.9,
                   color=m["cfg"]["color"], alpha=0.82, zorder=3,
                   label=m["cfg"]["short"].replace("\n", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_keys, fontsize=8)
        ax.set_title(f"@{thr} km", fontsize=9)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        ax.set_ylim(0, 95)

    axes[0].set_ylabel("Accuracy (%)")
    axes[-1].legend(ncol=1, fontsize=7, loc="upper right", framealpha=0.9)
    fig.suptitle("Accuracy by image difficulty", fontsize=10)

    path = out_dir / "fig7_difficulty.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 8: Pairwise joint accuracy heatmap ─────────────────────────────────

def fig8_agreement(data, out_dir):
    """Lower-triangular heatmap of joint accuracy: fraction of images where
    BOTH models in a pair predict within 200 km of ground truth."""
    working = [m for m in data if not m["cfg"]["failed"]]
    n = len(working)
    labels = [m["cfg"]["short"].replace("\n", " ") for m in working]
    T = 200  # km

    matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            common = set(working[i]["rows"].keys()) & set(working[j]["rows"].keys())
            if not common:
                continue
            joint = np.mean([
                get_dist(working[i]["rows"][id_])[0] <= T and
                get_dist(working[j]["rows"][id_])[0] <= T
                for id_ in common
            ]) * 100
            matrix[i, j] = joint

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    matrix_disp = np.where(mask, np.nan, matrix)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
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
    path = out_dir / "fig8_agreement.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 9: Failure mode decomposition ──────────────────────────────────────

def fig9_failure_modes(data, out_dir):
    """100% stacked bar showing prediction outcome composition per model."""
    working = [m for m in data if not m["cfg"]["failed"]]

    cat_labels  = ["≤25 km\n(city)", "25–200 km\n(region)",
                   "200–2500 km\n(country+)", ">2500 km\n(wrong cont.)", "Parse\nfailure"]
    cat_keys    = ["≤25km", "25–200km", "200–2500km", ">2500km", "parse_fail"]
    cat_colors  = ["#009E73", "#56B4E9", "#E69F00", "#D55E00", "#999999"]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(working))
    bottoms = np.zeros(len(working))

    for key, color, label in zip(cat_keys, cat_colors, cat_labels):
        fracs = [m["fail_modes"][key] / m["stats"]["n"] * 100 for m in working]
        bars = ax.bar(x, fracs, bottom=bottoms, color=color, label=label, zorder=3)
        for xi, (frac, bot) in enumerate(zip(fracs, bottoms)):
            if frac >= 5:
                ax.text(xi, bot + frac / 2, f"{frac:.0f}%",
                        ha="center", va="center", fontsize=7,
                        color="white" if color in ("#D55E00", "#009E73") else "black",
                        fontweight="bold")
        bottoms = bottoms + np.array(fracs)

    ax.set_xticks(x)
    ax.set_xticklabels([m["cfg"]["short"] for m in working], fontsize=8)
    ax.set_ylim(0, 102)
    ax.set_ylabel("% of images")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(loc="upper right", ncol=2, fontsize=7, framealpha=0.9)
    ax.set_title("Prediction outcome decomposition")

    fig.tight_layout()
    path = out_dir / "fig9_failure_modes.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", type=str, default="figures",
                   help="Directory to write PDF figures (default: figures/)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model results...")
    data = []
    for cfg in MODELS:
        path = Path(cfg["path"])
        if not path.exists():
            print(f"  MISSING: {path} — skipping")
            continue
        print(f"  [{cfg['short'].replace(chr(10), ' ')}]  {path}")
        rows       = load_model(path)
        stats      = compute_stats(rows)
        ev         = compute_evidence_deltas(rows)
        geo        = compute_geographic(rows)
        fail_modes = compute_failure_modes(rows)
        data.append({"cfg": cfg, "rows": rows, "stats": stats,
                     "ev": ev, "geo": geo, "fail_modes": fail_modes})

    print(f"\nGenerating figures → {out_dir}/")
    fig1_geoscore(data, out_dir)
    fig2_thresholds(data, out_dir)
    fig3_distribution(data, out_dir)
    fig4_evidence(data, out_dir)
    fig5_geographic(data, out_dir)
    fig6_cdf(data, out_dir)
    fig7_difficulty(data, out_dir)
    fig8_agreement(data, out_dir)
    fig9_failure_modes(data, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
