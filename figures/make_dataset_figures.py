#!/usr/bin/env python3
"""
Generate Im2GPS3k dataset characterization figures for model_justification.tex.

Produces ten PDFs:

  EXISTING
  fig_dataset_examples.pdf        — 2×3 grid: best/worst predictions
  fig_dataset_geo.pdf             — continental image-count bar chart

  NEW  (all require only lat/lon — no external data)
  fig_dataset_density.pdf         — 2D hexbin density heatmap
                                    [OSV5M CVPR 2024; GeoToken arXiv 2511.01082]
  fig_dataset_latlon_hist.pdf     — Marginal latitude + longitude histograms
                                    [IMAGEO-Bench arXiv 2508.01608 Fig 2]
  fig_dataset_scatter_continent.pdf — World scatter colored by continent
                                    [OSV5M CVPR 2024; NAVIG 2025 Fig 5]
  fig_dataset_nn_dist.pdf         — CDF of nearest-neighbor inter-image distances
                                    [OSV5M CVPR 2024 §3; GWS15k 2023 §3]
  fig_dataset_pairwise_dist.pdf   — Histogram of sampled pairwise distances
                                    [Im2GPS chapter Hays & Efros 2015]
  fig_dataset_geocell.pdf         — 1°×1° geocell occupancy histogram (log scale)
                                    [PIGEON CVPR 2024; GeoToken arXiv 2511.01082]
  fig_dataset_coverage.pdf        — Cumulative unique cells vs. images-per-cell threshold
                                    [OSV5M CVPR 2024 coverage analysis]
  fig_dataset_lat_band_polar.pdf  — Polar bar chart of images by latitude band
                                    [GSV-Cities Neurocomputing 2022 Fig 3]

Usage::

    python figures/make_dataset_figures.py \\
        [--output_dir figures/] \\
        [--img_dir dataset/im2gps3k/images] \\
        [--meta_path dataset/im2gps3k/meta.jsonl]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import math
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
import numpy as np
from scipy.spatial import cKDTree

OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "vermilion": "#D55E00",
    "skyblue": "#56B4E9",
    "yellow": "#F0E442",
}

CONTINENT_COLORS = {
    "Europe": OKABE["blue"],
    "N. America": OKABE["orange"],
    "Asia (N)": OKABE["green"],
    "Africa": OKABE["vermilion"],
    "S. America": OKABE["pink"],
    "Oceania": OKABE["skyblue"],
    "Other": "#aaaaaa",
    "Antarctica": "#cccccc",
}

# ── Selected qualitative examples ──────────────────────────────────────────────
BEST = [
    {
        "id": "523882906_8593329b96_198_51035632501@N01",
        "gt": "Paris, France\n(48.86°N, 2.29°E)",
        "pred": "Paris, France\n(48.86°N, 2.29°E)",
        "err": 0.01,
        "note": "Eiffel Tower",
    },
    {
        "id": "297480819_247ee47039_99_32377115@N00",
        "gt": "London, UK\n(51.51°N, 0.13°W)",
        "pred": "London, UK\n(51.51°N, 0.13°W)",
        "err": 0.10,
        "note": "Trafalgar Square",
    },
    {
        "id": "1413644279_bd885b7f3c_1349_73293249@N00",
        "gt": "London, UK\n(51.51°N, 0.13°W)",
        "pred": "London, UK\n(51.51°N, 0.13°W)",
        "err": 0.10,
        "note": "Sherlock Holmes Pub",
    },
]

WORST = [
    {
        "id": "318855058_b978d151af_138_44067110@N00",
        "gt": "A Coruña, Spain\n(43.30°N, 8.38°W)",
        "pred": "Christchurch, NZ\n(43.53°S, 172.61°E)",
        "err": 19931,
        "note": "Ambiguous parking lot:\nEnglish signs confuse hemisphere",
    },
    {
        "id": "1086319092_34f919b886_1232_33463080@N00",
        "gt": "Bangkok, Thailand\n(13.73°N, 100.50°E)",
        "pred": "Cuba\n(12.00°S, 78.00°W)",
        "err": 19763,
        "note": "Gharial/crocodile:\nanimal ID overrides geography",
    },
    {
        "id": "628184053_403e17efac_1438_7412395@N07",
        "gt": "Iceland\n(64.72°N, 18.06°W)",
        "pred": "Antarctica\n(71.50°S, 167.56°E)",
        "err": 19226,
        "note": "Volcanic highland:\npolar confusion, wrong hemisphere",
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def continent(lat, lon):
    """Heuristic continent assignment from latitude and longitude."""
    if lat is None or lon is None:
        return "Other"
    lon = ((lon + 180) % 360) - 180
    if lat <= -60:
        return "Antarctica"
    if -55 <= lat <= 15 and -82 <= lon <= -34:
        return "S. America"
    if 15 <= lat <= 72 and -168 <= lon <= -50:
        return "N. America"
    if 35 <= lat <= 72 and -25 <= lon <= 40:
        return "Europe"
    if -37 <= lat <= 38 and -20 <= lon <= 55:
        return "Africa"
    if 20 <= lat <= 77 and 40 <= lon <= 180:
        return "Asia (N)"
    if -50 <= lat <= 10 and 110 <= lon <= 180:
        return "Oceania"
    return "Other"


def load_image(img_dir: Path, id_):
    path = img_dir / f"{id_}.jpg"
    return mpimg.imread(str(path)) if path.exists() else None


def fmt_err(km):
    if km < 1:
        return f"{km * 1000:.0f} m"
    elif km < 1000:
        return f"{km:.0f} km"
    else:
        return f"{km / 1000:.1f}×10³ km"


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def _haversine_vec(lats1, lons1, lats2, lons2):
    """Vectorised haversine, inputs in degrees, returns km."""
    R = 6371.0
    phi1 = np.radians(lats1)
    phi2 = np.radians(lats2)
    dphi = np.radians(lats2 - lats1)
    dlam = np.radians(lons2 - lons1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ── Figure 1: Best / worst prediction grid ────────────────────────────────────


def make_example_grid(img_dir: Path, fig_dir: Path):
    """2×3 qualitative grid of best and worst predictions.
    Standard in geolocalization papers (PIGEON CVPR 2024 Fig 7;
    GeoReasoner ICML 2024 Figs 7–9)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    fig.patch.set_facecolor("#f8f8f8")

    row_labels = ["Successful localizations", "Failure cases"]
    row_colors = [OKABE["green"], OKABE["vermilion"]]

    for row_idx, (row_examples, row_label, row_col) in enumerate(
        zip([BEST, WORST], row_labels, row_colors)
    ):
        for col_idx, ex in enumerate(row_examples):
            ax = axes[row_idx][col_idx]
            img = load_image(img_dir, ex["id"])
            if img is not None:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_edgecolor(row_col)
                spine.set_linewidth(3)

            ax.set_title(
                ex["note"], fontsize=9.5, fontweight="bold", pad=4, color="#222222"
            )

            err_color = OKABE["green"] if ex["err"] < 50 else OKABE["vermilion"]
            ax.text(
                0.5,
                -0.01,
                f"GT:   {ex['gt']}\nPred: {ex['pred']}",
                transform=ax.transAxes,
                fontsize=7.2,
                ha="center",
                va="top",
                color="#333333",
                fontfamily="monospace",
            )
            ax.text(
                0.5,
                -0.18,
                f"Error: {fmt_err(ex['err'])}",
                transform=ax.transAxes,
                fontsize=8.5,
                ha="center",
                va="top",
                color=err_color,
                fontweight="bold",
            )

        axes[row_idx][0].set_ylabel(
            row_label, fontsize=11, fontweight="bold", color=row_col, labelpad=8
        )

    fig.suptitle(
        "Im2GPS3k benchmark: representative predictions (LLaVA-1.6 full pipeline)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.subplots_adjust(wspace=0.08, hspace=0.55, bottom=0.08)
    out = fig_dir / "fig_dataset_examples.pdf"
    fig.savefig(str(out), bbox_inches="tight", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 2: Continental distribution bar chart ──────────────────────────────


def make_geo_bar(meta: list[dict], fig_dir: Path):
    """Continental image-count bar chart.
    Standard in every geolocalization paper (Im2GPS; OSV5M CVPR 2024;
    IMAGEO-Bench 2025)."""
    counts = Counter(continent(r["LAT"], r["LON"]) for r in meta)
    order = [
        "Europe",
        "N. America",
        "Asia (N)",
        "Other",
        "Africa",
        "S. America",
        "Oceania",
    ]
    colors = [CONTINENT_COLORS[k] for k in order]

    labels = [f"{k}\n({counts[k]:,})" for k in order]
    vals = [counts[k] for k in order]
    total = sum(vals)
    pcts = [100 * v / total for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        labels, vals, color=colors, edgecolor="white", linewidth=0.8, zorder=3
    )
    ax.set_ylabel("Number of images", fontsize=11)
    ax.set_title(
        f"Im2GPS3k geographic distribution  (N = {total:,})",
        fontsize=12,
        fontweight="bold",
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = fig_dir / "fig_dataset_geo.pdf"
    fig.savefig(str(out), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 3: 2D hexbin density heatmap ───────────────────────────────────────


def make_density_hexbin(meta: list[dict], fig_dir: Path):
    """2D hexbin density map of ground-truth locations.
    Cited in OSV5M (Astruc et al. CVPR 2024) train/test density maps and
    GeoToken (arXiv 2511.01082) Figure 1a — both use density heatmaps to
    reveal Europe/N. America concentration bias."""
    lats = [r["LAT"] for r in meta]
    lons = [r["LON"] for r in meta]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#0d1117")
    hb = ax.hexbin(
        lons, lats, gridsize=80, cmap="YlOrRd", mincnt=1, bins="log", linewidths=0.1
    )
    cb = fig.colorbar(hb, ax=ax, pad=0.01, shrink=0.85)
    cb.set_label("Images per cell (log scale)", fontsize=9, color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_xticklabels(
        [f"{x}°" for x in range(-180, 181, 60)], fontsize=7, color="white"
    )
    ax.set_yticklabels([f"{y}°" for y in range(-90, 91, 30)], fontsize=7, color="white")
    ax.tick_params(colors="white")
    ax.set_xlabel("Longitude", fontsize=9, color="white")
    ax.set_ylabel("Latitude", fontsize=9, color="white")
    ax.set_title(
        f"Im2GPS3k ground-truth density  (n = {len(meta):,})",
        fontsize=11,
        fontweight="bold",
        color="white",
    )
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(True, linewidth=0.2, alpha=0.3, linestyle="--", color="white")

    fig.patch.set_facecolor("#0d1117")
    fig.tight_layout()
    out = fig_dir / "fig_dataset_density.pdf"
    fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 4: Marginal lat / lon histograms ────────────────────────────────────


def make_latlon_hist(meta: list[dict], fig_dir: Path):
    """Marginal latitude and longitude distribution histograms.
    Cited in IMAGEO-Bench (arXiv 2508.01608) Figure 2 which shows scatter plots
    of predicted vs. true latitude colored by confidence score — the marginal
    distributions are the foundation of that analysis."""
    lats = np.array([r["LAT"] for r in meta])
    lons = np.array([r["LON"] for r in meta])

    fig, (ax_lat, ax_lon) = plt.subplots(1, 2, figsize=(10, 3.8))

    # Latitude histogram
    ax_lat.hist(
        lats,
        bins=36,
        color=OKABE["blue"],
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax_lat.axvline(
        np.mean(lats),
        color=OKABE["vermilion"],
        linewidth=1.5,
        linestyle="--",
        label=f"Mean = {np.mean(lats):.1f}°",
    )
    ax_lat.axvline(
        np.median(lats),
        color=OKABE["orange"],
        linewidth=1.5,
        linestyle=":",
        label=f"Median = {np.median(lats):.1f}°",
    )
    ax_lat.set_xlabel("Latitude (°)", fontsize=10)
    ax_lat.set_ylabel("Images", fontsize=10)
    ax_lat.set_title("Latitude distribution", fontsize=11, fontweight="bold")
    ax_lat.legend(fontsize=8)
    ax_lat.set_xlim(-90, 90)
    ax_lat.xaxis.set_major_locator(ticker.MultipleLocator(30))

    # Longitude histogram
    ax_lon.hist(
        lons,
        bins=36,
        color=OKABE["green"],
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax_lon.axvline(
        np.mean(lons),
        color=OKABE["vermilion"],
        linewidth=1.5,
        linestyle="--",
        label=f"Mean = {np.mean(lons):.1f}°",
    )
    ax_lon.axvline(
        np.median(lons),
        color=OKABE["orange"],
        linewidth=1.5,
        linestyle=":",
        label=f"Median = {np.median(lons):.1f}°",
    )
    ax_lon.set_xlabel("Longitude (°)", fontsize=10)
    ax_lon.set_ylabel("Images", fontsize=10)
    ax_lon.set_title("Longitude distribution", fontsize=11, fontweight="bold")
    ax_lon.legend(fontsize=8)
    ax_lon.set_xlim(-180, 180)
    ax_lon.xaxis.set_major_locator(ticker.MultipleLocator(60))

    fig.suptitle(
        "Im2GPS3k marginal coordinate distributions", fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    out = fig_dir / "fig_dataset_latlon_hist.pdf"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 5: World scatter colored by continent ──────────────────────────────


def make_scatter_continent(meta: list[dict], fig_dir: Path):
    """World scatter with points colored by continent.
    Used in OSV5M (CVPR 2024) supplementary maps and NAVIG (2025) Figure 5
    (NaviClues global coverage map)."""
    cont_order = [
        "Europe",
        "N. America",
        "Asia (N)",
        "Africa",
        "S. America",
        "Oceania",
        "Other",
    ]

    by_cont: dict[str, list] = defaultdict(list)
    for r in meta:
        c = continent(r["LAT"], r["LON"])
        by_cont[c].append((r["LON"], r["LAT"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#dceef7")

    for c in cont_order:
        pts = by_cont.get(c, [])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(
            xs,
            ys,
            s=2.5,
            alpha=0.5,
            color=CONTINENT_COLORS[c],
            label=f"{c} ({len(pts):,})",
            rasterized=True,
            zorder=2,
        )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_xticklabels([f"{x}°" for x in range(-180, 181, 60)], fontsize=7)
    ax.set_yticklabels([f"{y}°" for y in range(-90, 91, 30)], fontsize=7)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title(
        f"Im2GPS3k ground-truth locations by continent  (n = {len(meta):,})",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linewidth=0.25, alpha=0.4, linestyle="--", zorder=1)
    ax.legend(markerscale=4, fontsize=8, loc="lower left", framealpha=0.85, ncol=2)

    fig.tight_layout()
    out = fig_dir / "fig_dataset_scatter_continent.pdf"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 6: Nearest-neighbor distance CDF ───────────────────────────────────


def make_nn_distance_cdf(meta: list[dict], fig_dir: Path):
    """CDF of test-image nearest-neighbor (test-to-test) distances.
    Used in OSV5M (Astruc et al. CVPR 2024 §3) to verify ≥1 km train-test
    separation; in GWS15k (Clark et al. 2023 §3) to show 92% of test locations
    are >100 m from any training image. Reveals spatial clustering of the
    benchmark."""
    lats = np.radians([r["LAT"] for r in meta])
    lons = np.radians([r["LON"] for r in meta])
    # Convert to 3D unit-sphere coords for cKDTree (Euclidean ≈ chord distance)
    coords = np.column_stack(
        [
            np.cos(lats) * np.cos(lons),
            np.cos(lats) * np.sin(lons),
            np.sin(lats),
        ]
    )
    tree = cKDTree(coords)
    # k=2: closest non-self neighbor
    chord_dists, _ = tree.query(coords, k=2)
    chord_nn = chord_dists[:, 1]
    # chord → km: d_km ≈ 2R * arcsin(chord / 2)
    nn_km = 2 * 6371.0 * np.arcsin(np.clip(chord_nn / 2, 0, 1))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sorted_km = np.sort(nn_km)
    cdf = np.arange(1, len(sorted_km) + 1) / len(sorted_km)
    ax.plot(sorted_km, cdf * 100, color=OKABE["blue"], linewidth=2)
    ax.fill_between(sorted_km, cdf * 100, alpha=0.12, color=OKABE["blue"])

    for thresh, lbl in [(1, "1 km"), (10, "10 km"), (100, "100 km")]:
        pct = np.searchsorted(sorted_km, thresh) / len(sorted_km) * 100
        ax.axvline(thresh, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(
            thresh * 1.12,
            8,
            f"{pct:.0f}%\n<{lbl}",
            fontsize=7.5,
            color="gray",
            va="bottom",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Distance to nearest test neighbor (km, log scale)", fontsize=9)
    ax.set_ylabel("Cumulative % of images", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_ylim(0, 102)
    ax.set_title(
        "Nearest-neighbor inter-image distance CDF\n"
        "(reveals spatial clustering of test set)",
        fontsize=10,
        fontweight="bold",
    )

    fig.tight_layout()
    out = fig_dir / "fig_dataset_nn_dist.pdf"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 7: Pairwise inter-image distance histogram ─────────────────────────


def make_pairwise_dist_hist(meta: list[dict], fig_dir: Path):
    """Histogram of sampled pairwise distances between test images.
    Used implicitly in Im2GPS (Hays & Efros IJCV 2015 chapter §5) to
    characterise the benchmark's geographic spread. Reveals whether the
    dataset is dominated by a few dense clusters or spread globally."""
    rng = np.random.default_rng(42)
    n = len(meta)
    n_sample = min(20_000, n * (n - 1) // 2)

    lats = np.array([r["LAT"] for r in meta])
    lons = np.array([r["LON"] for r in meta])

    idx_i = rng.integers(0, n, n_sample)
    idx_j = rng.integers(0, n, n_sample)
    same = idx_i == idx_j
    idx_j[same] = (idx_j[same] + 1) % n  # avoid self-pairs

    dists_km = _haversine_vec(lats[idx_i], lons[idx_i], lats[idx_j], lons[idx_j])

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(np.log10(0.5), np.log10(20_000), 60)
    ax.hist(
        dists_km,
        bins=bins,
        color=OKABE["orange"],
        alpha=0.8,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )

    med = float(np.median(dists_km))
    ax.axvline(
        med,
        color=OKABE["vermilion"],
        linewidth=1.8,
        linestyle="--",
        label=f"Median = {med:,.0f} km",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Pairwise distance between test images (km, log scale)", fontsize=9)
    ax.set_ylabel("Pair count", fontsize=9)
    ax.set_title(
        f"Im2GPS3k pairwise inter-image distances  "
        f"(n = {n_sample:,} sampled pairs)",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = fig_dir / "fig_dataset_pairwise_dist.pdf"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 8: 1°×1° geocell occupancy histogram ───────────────────────────────


def make_geocell_balance(meta: list[dict], fig_dir: Path):
    """Histogram of images-per-cell for a 1°×1° geographic grid (log–log scale).
    Directly cited in PIGEON (Haas et al. CVPR 2024) which shows naive geocells
    are extremely imbalanced (Zipfian) and motivated their Voronoi-based semantic
    geocells; also in GeoToken (arXiv 2511.01082)."""
    cell_counts: Counter = Counter()
    for r in meta:
        cell = (int(math.floor(r["LAT"])), int(math.floor(r["LON"])))
        cell_counts[cell] += 1

    counts = sorted(cell_counts.values(), reverse=True)
    n_cells = len(counts)
    n_empty = 180 * 360 - n_cells  # total 1°×1° cells on Earth minus occupied

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

    # Left: histogram of counts per occupied cell
    ax = axes[0]
    max_cnt = max(counts)
    bins = np.arange(1, max_cnt + 2) - 0.5
    if max_cnt > 30:
        # Use log-spaced bins
        bins = np.logspace(0, math.log10(max_cnt + 1), 40)
        ax.set_xscale("log")
    ax.hist(
        counts,
        bins=bins,
        color=OKABE["green"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Images per 1°×1° cell", fontsize=9)
    ax.set_ylabel("Number of cells (log scale)", fontsize=9)
    ax.set_title(
        "Per-cell image count\n(log–log, occupied cells only)",
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        0.97,
        0.97,
        f"{n_cells:,} occupied cells\n{n_empty:,} empty cells",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Right: ranked cell counts (Zipf curve)
    ax2 = axes[1]
    ax2.plot(range(1, n_cells + 1), counts, color=OKABE["blue"], linewidth=1.5)
    ax2.fill_between(range(1, n_cells + 1), counts, alpha=0.15, color=OKABE["blue"])
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Cell rank (most → least populated)", fontsize=9)
    ax2.set_ylabel("Images in cell", fontsize=9)
    ax2.set_title(
        "Zipf-like class imbalance\n(ranked cell counts)", fontsize=9, fontweight="bold"
    )

    fig.suptitle(
        "Im2GPS3k: 1°×1° geocell occupancy distribution", fontsize=11, fontweight="bold"
    )
    out = fig_dir / "fig_dataset_geocell.pdf"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 9: Cumulative unique cells vs. threshold ───────────────────────────


def make_coverage_curve(meta: list[dict], fig_dir: Path):
    """Cumulative fraction of unique 1°×1° geocells containing ≥k images,
    as k varies from 1 to max.
    Inspired by OSV5M (Astruc et al. CVPR 2024) coverage analysis comparing
    the geographic footprint of multiple datasets."""
    cell_counts: Counter = Counter()
    for r in meta:
        cell = (int(math.floor(r["LAT"])), int(math.floor(r["LON"])))
        cell_counts[cell] += 1

    counts = np.array(sorted(cell_counts.values(), reverse=True))
    total_earth_cells = 180 * 360  # ≈ 64 800 cells at 1°×1°

    thresholds = range(1, int(counts.max()) + 1)
    n_cells_at_k = [np.sum(counts >= k) for k in thresholds]
    pct_earth = [c / total_earth_cells * 100 for c in n_cells_at_k]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(
        list(thresholds),
        n_cells_at_k,
        color=OKABE["blue"],
        linewidth=2,
        label="# unique cells",
    )
    ax.set_xlabel("Minimum images per cell (k)", fontsize=9)
    ax.set_ylabel("Unique cells with ≥k images", fontsize=9, color=OKABE["blue"])
    ax.tick_params(axis="y", labelcolor=OKABE["blue"])

    ax2 = ax.twinx()
    ax2.plot(
        list(thresholds),
        pct_earth,
        color=OKABE["orange"],
        linewidth=2,
        linestyle="--",
        label="% of Earth surface",
    )
    ax2.set_ylabel("% of 1°×1° Earth cells covered", fontsize=9, color=OKABE["orange"])
    ax2.tick_params(axis="y", labelcolor=OKABE["orange"])
    ax2.set_ylim(0, max(pct_earth) * 1.25)

    ax.set_title(
        "Im2GPS3k: geographic coverage vs. images-per-cell threshold",
        fontsize=10,
        fontweight="bold",
    )
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")

    fig.tight_layout()
    out = fig_dir / "fig_dataset_coverage.pdf"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 10: Polar latitude-band bar chart ──────────────────────────────────


def make_lat_band_polar(meta: list[dict], fig_dir: Path):
    """Polar bar chart of images by latitude band (10° bins).
    The polar projection is visually distinctive and highlights hemispheric
    bias; used in GSV-Cities (Amaralingam et al. Neurocomputing 2022 Fig 3)
    to show 14-year temporal + spatial coverage."""
    lats = [r["LAT"] for r in meta]
    edges = np.arange(-90, 91, 10)
    cnts, _ = np.histogram(lats, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    n_bands = len(centers)

    # Map latitude -90..+90 to polar angle 0..π (south pole → north pole)
    # We use a half-circle polar plot
    angles = np.linspace(0, np.pi, n_bands, endpoint=False) + np.pi / (2 * n_bands)
    width = np.pi / n_bands * 0.85

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_offset(np.pi / 2)  # 0° at top
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    norm_cnts = cnts / cnts.max()
    cmap = plt.cm.RdYlBu_r
    colors = [cmap(v) for v in norm_cnts]

    bars = ax.bar(
        angles,
        cnts,
        width=width,
        bottom=0,
        color=colors,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
    )

    # Label significant bands
    for ang, cnt, ctr in zip(angles, cnts, centers):
        if cnt > cnts.max() * 0.05:
            ax.text(
                ang,
                cnt + cnts.max() * 0.04,
                f"{ctr:.0f}°\n({cnt:,})",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#222222",
            )

    # Theta labels: south (-90) left, north (+90) right
    ax.set_thetagrids(
        [0, 45, 90, 135, 180], ["90°N", "45°N", "0°", "45°S", "90°S"], fontsize=8
    )
    ax.set_rlabel_position(20)
    ax.set_title(
        "Im2GPS3k images by latitude band\n" "(polar projection, 10° bins)",
        fontsize=10,
        fontweight="bold",
        pad=20,
    )
    ax.set_yticklabels([])

    fig.tight_layout()
    out = fig_dir / "fig_dataset_lat_band_polar.pdf"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Directory for output PDFs (default: figures/)",
    )
    p.add_argument(
        "--img_dir",
        type=str,
        default="dataset/im2gps3k/images",
        help="Directory containing {ID}.jpg images",
    )
    p.add_argument(
        "--meta_path",
        type=str,
        default="dataset/im2gps3k/meta.jsonl",
        help="JSONL file with ID / LAT / LON per image",
    )
    args = p.parse_args()

    fig_dir = Path(args.output_dir)
    img_dir = Path(args.img_dir)
    meta_path = Path(args.meta_path)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata from {meta_path} ...")
    with open(meta_path) as f:
        meta = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(meta):,} images.")

    print(f"\nGenerating dataset figures → {fig_dir}/")
    make_example_grid(img_dir, fig_dir)
    make_geo_bar(meta, fig_dir)
    make_density_hexbin(meta, fig_dir)
    make_latlon_hist(meta, fig_dir)
    make_scatter_continent(meta, fig_dir)
    make_nn_distance_cdf(meta, fig_dir)
    make_pairwise_dist_hist(meta, fig_dir)
    make_geocell_balance(meta, fig_dir)
    make_coverage_curve(meta, fig_dir)
    make_lat_band_polar(meta, fig_dir)
    print("Done.")


if __name__ == "__main__":
    main()
