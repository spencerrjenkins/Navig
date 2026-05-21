#!/usr/bin/env python3
"""
Visualise one image's ground-truth location together with all model predictions.

For a given dataset and image ID the script:
  1. Loads the ground-truth lat/lon from dataset/<dataset>/meta.jsonl
  2. Auto-discovers every results_s6_*.jsonl file under output/<dataset>/
  3. Extracts each model's predicted lat/lon, predicted country/city, and
     geodesic distance to the ground truth
  4. Produces a PDF containing:
       • The image itself (top-left)
       • A world-map scatter with the true location (gold star) and each
         model's prediction (coloured circle + label)
       • Dashed geodesic lines connecting each prediction to the true location
       • An auto-zoomed inset focused on the area of interest
       • A summary table (model | predicted coords | distance)

Usage::

    python figures/plot_image_prediction.py \\
        --dataset im2gps3k_rgb_images \\
        --image_id 523882906_8593329b96_198_51035632501@N01 \\
        [--base_dir output/] \\
        [--dataset_dir dataset/] \\
        [--output_dir figures/]

Outputs::

    figures/pred_map_<sanitised_image_id>.pdf
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

from metrics import geoscore, haversine_distance, parse_coord

# ── Color palette (tab10, distinct for up to 10 models) ───────────────────────
_TAB10 = plt.cm.tab10(np.linspace(0, 0.9, 10))

# ── Pretty model labels ────────────────────────────────────────────────────────
# Maps result-file stem suffix (e.g. "llava") to a display label.
# Anything not listed falls back to the raw key.
LABEL_MAP = {
    "llava":          "LLaVA-1.6",
    "llava_sft":      "LLaVA-1.6 (SFT)",
    "deepseek":       "DeepSeek-7B",
    "falcon":         "Falcon-11B",
    "llama32vision":  "LLaMA-3.2-11B",
    "cpm":            "MiniCPM-V-2.6",
    "cpm_sft":        "MiniCPM-V-2.6 (SFT)",
    "qwen":           "Qwen2.5-7B",
    "qwen_sft":       "Qwen2.5-7B (SFT)",
    "llama32":        "LLaMA-3.2 (swap)",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt_dist(km: float) -> str:
    """Human-readable distance in appropriate metric units."""
    if km < 0.001:
        return f"{km * 1_000_000:.0f} m"     # sub-metre
    if km < 1.0:
        return f"{km * 1000:.0f} m"           # metres
    if km < 1000.0:
        return f"{km:.1f} km"
    if km < 10_000.0:
        return f"{km / 1000:.2f}×10³ km"
    return f"{km / 1000:.1f}×10³ km"


def fmt_coord(lat: float, lon: float) -> str:
    """Format a coordinate pair as e.g. '48.86°N, 2.29°E'."""
    lat_h = "N" if lat >= 0 else "S"
    lon_h = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{lat_h}, {abs(lon):.2f}°{lon_h}"


def _sanitise(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)


def _great_circle_path(lat1, lon1, lat2, lon2, n=60):
    """Return (lons, lats) for an approximate great-circle path between two points."""
    lats = np.linspace(lat1, lat2, n)
    lons = np.linspace(lon1, lon2, n)
    return lons, lats


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ground_truth(dataset_dir: Path, dataset: str, image_id: str) -> dict | None:
    """Load the ground-truth lat/lon for image_id from meta.jsonl."""
    candidates = [
        dataset_dir / dataset / "meta.jsonl",
        dataset_dir / dataset.replace("_rgb_images", "") / "meta.jsonl",
    ]
    for meta_path in candidates:
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("ID") == image_id:
                    return row
    return None


def find_result_files(base_dir: Path, dataset: str) -> list[Path]:
    """Return all results_s6_*.jsonl files for the given dataset."""
    search_roots = [
        base_dir / dataset,
        base_dir / dataset.replace("_rgb_images", ""),
    ]
    found = []
    for root in search_roots:
        if root.exists():
            found.extend(root.rglob("results_s6_*.jsonl"))
    return sorted(set(found))


def extract_key(path: Path) -> str:
    """Extract model key from a results_s6_<key>.jsonl path."""
    m = re.search(r"results_s6_(.+)\.jsonl$", path.name)
    return m.group(1) if m else path.stem


def load_prediction(path: Path, image_id: str) -> dict | None:
    """Return the row for image_id from a results JSONL, or None if missing."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("ID") == image_id:
                return row
    return None


def parse_prediction(row: dict) -> tuple[float, float] | None:
    """Return (pred_lat, pred_lon) from a result row, or None on parse failure."""
    try:
        ans = row.get("answer") or {}
        lat = parse_coord(ans["latitude"])
        lon = parse_coord(ans["longitude"])
        if math.isnan(lat) or math.isnan(lon):
            return None
        return float(lat), float(lon)
    except Exception:
        return None


# ── Map drawing helpers ────────────────────────────────────────────────────────

def _draw_world_background(ax, meta_scatter: np.ndarray | None):
    """Draw ocean + faint continent outline from dataset scatter (if provided)."""
    ax.set_facecolor("#c8dff0")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_xticklabels([f"{x}°" for x in range(-180, 181, 60)], fontsize=6.5)
    ax.set_yticklabels([f"{y}°" for y in range(-90, 91, 30)], fontsize=6.5)
    ax.grid(True, linewidth=0.2, alpha=0.4, linestyle="--", zorder=1)

    if meta_scatter is not None and len(meta_scatter) > 0:
        ax.scatter(meta_scatter[:, 1], meta_scatter[:, 0],
                   s=0.4, alpha=0.12, color="#5a4a3a", rasterized=True, zorder=2)


def _draw_zoomed_inset(fig, parent_ax, true_lat, true_lon, predictions,
                       colors, parent_bbox=(0.57, 0.0, 0.43, 0.45)):
    """Add a zoomed inset around the cluster of predictions + ground truth."""
    all_lats = [true_lat] + [p["pred_lat"] for p in predictions]
    all_lons = [true_lon] + [p["pred_lon"] for p in predictions]

    lat_span = max(all_lats) - min(all_lats)
    lon_span = max(all_lons) - min(all_lons)
    max_span_km = haversine_distance(
        [min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)])

    # Only show inset if predictions are reasonably clustered (< 3000 km span)
    if max_span_km > 3000:
        return False

    pad_lat = max(lat_span * 0.4, 1.5)
    pad_lon = max(lon_span * 0.4, 2.5)
    lat_lo  = max(-90,  min(all_lats) - pad_lat)
    lat_hi  = min(90,   max(all_lats) + pad_lat)
    lon_lo  = max(-180, min(all_lons) - pad_lon)
    lon_hi  = min(180,  max(all_lons) + pad_lon)

    ax_in = fig.add_axes([parent_bbox[0], parent_bbox[1],
                          parent_bbox[2], parent_bbox[3]])
    ax_in.set_facecolor("#c8dff0")
    ax_in.set_xlim(lon_lo, lon_hi)
    ax_in.set_ylim(lat_lo, lat_hi)
    ax_in.tick_params(labelsize=6)
    ax_in.set_title("Zoomed view", fontsize=7.5, pad=2)
    ax_in.grid(True, linewidth=0.3, alpha=0.4, linestyle="--")

    # Draw predictions
    for pred, color in zip(predictions, colors):
        lons_path, lats_path = _great_circle_path(
            true_lat, true_lon, pred["pred_lat"], pred["pred_lon"])
        ax_in.plot(lons_path, lats_path, color=color, linewidth=0.8,
                   linestyle="--", alpha=0.7, zorder=3)
        ax_in.scatter(pred["pred_lon"], pred["pred_lat"], s=40,
                      color=color, zorder=5, edgecolors="white", linewidths=0.6)

    # Ground truth
    ax_in.scatter(true_lon, true_lat, s=140, marker="*",
                  color="#FFD700", zorder=6, edgecolors="#333333", linewidths=0.8)

    # Draw indicator rectangle on parent map
    rect = mpatches.FancyArrowPatch(
        posA=(lon_lo, lat_lo), posB=(lon_hi, lat_hi), arrowstyle="->",
        mutation_scale=0, color="red", linewidth=0)
    rect_patch = mpatches.Rectangle(
        (lon_lo, lat_lo), lon_hi - lon_lo, lat_hi - lat_lo,
        linewidth=1.5, edgecolor="red", facecolor="none",
        transform=parent_ax.transData, zorder=10)
    parent_ax.add_patch(rect_patch)

    return True


# ── Main figure builder ────────────────────────────────────────────────────────

def build_figure(image_id: str, gt: dict, predictions: list[dict],
                 image_path: Path | None, meta_scatter: np.ndarray | None,
                 output_path: Path):
    """
    Layout:
      Row 0 (tall): [image or placeholder (1/3)] | [world map (2/3)]
      Row 1 (short): colspan — prediction table
    """
    has_image = image_path is not None and image_path.exists()

    true_lat = float(gt["LAT"])
    true_lon = float(gt["LON"])

    # Assign colors
    colors = [_TAB10[i % 10] for i in range(len(predictions))]

    fig = plt.figure(figsize=(14, 9))

    # ── Row 0 layout ──
    if has_image:
        ax_img = fig.add_axes([0.01, 0.32, 0.28, 0.63])
        ax_map = fig.add_axes([0.30, 0.32, 0.68, 0.63])
    else:
        ax_map = fig.add_axes([0.01, 0.32, 0.97, 0.63])

    # ── Image panel ──
    if has_image:
        img = mpimg.imread(str(image_path))
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title("Query image", fontsize=9, fontweight="bold", pad=3)
        for spine in ax_img.spines.values():
            spine.set_edgecolor("#888888")
            spine.set_linewidth(1)

    # ── World map panel ──
    _draw_world_background(ax_map, meta_scatter)

    # Great-circle paths + prediction dots
    for pred, color in zip(predictions, colors):
        lons_path, lats_path = _great_circle_path(
            true_lat, true_lon, pred["pred_lat"], pred["pred_lon"])
        ax_map.plot(lons_path, lats_path, color=color, linewidth=1.0,
                    linestyle="--", alpha=0.65, zorder=3)
        ax_map.scatter(pred["pred_lon"], pred["pred_lat"], s=70,
                       color=color, zorder=5,
                       edgecolors="white", linewidths=0.8)

    # Ground-truth star
    ax_map.scatter(true_lon, true_lat, s=280, marker="*",
                   color="#FFD700", zorder=7,
                   edgecolors="#333333", linewidths=1.0,
                   label="Ground truth")

    ax_map.set_xlabel("Longitude", fontsize=8)
    ax_map.set_ylabel("Latitude", fontsize=8)
    ax_map.set_title(f"Model predictions  —  image: {image_id}", fontsize=9,
                     fontweight="bold", pad=4)

    # Legend on map (models)
    legend_handles = [
        mpatches.Patch(color=color, label=pred["label"])
        for pred, color in zip(predictions, colors)
    ] + [plt.scatter([], [], marker="*", s=120, color="#FFD700",
                     edgecolors="#333333", linewidths=0.8, label="Ground truth")]
    ax_map.legend(handles=legend_handles, loc="lower left",
                  fontsize=7, framealpha=0.88, ncol=2)

    # ── Zoomed inset ──
    if predictions:
        _draw_zoomed_inset(fig, ax_map, true_lat, true_lon, predictions,
                           colors, parent_bbox=(0.30, 0.32, 0.25, 0.24))

    # ── Prediction table ──
    ax_tbl = fig.add_axes([0.01, 0.01, 0.97, 0.28])
    ax_tbl.axis("off")

    col_labels = ["Model", "Predicted coordinates", "Predicted country / city",
                  "Distance to GT", "GeoScore"]
    rows_data  = []
    for pred in predictions:
        coord_str   = fmt_coord(pred["pred_lat"], pred["pred_lon"])
        loc_str     = pred.get("loc_str", "—")
        dist_str    = fmt_dist(pred["dist_km"])
        gs          = f"{geoscore(pred['dist_km']):.0f}"
        rows_data.append([pred["label"], coord_str, loc_str, dist_str, gs])

    # Ground truth row
    gt_loc = "—"
    rows_data.insert(0, ["★ Ground truth",
                         fmt_coord(true_lat, true_lon),
                         gt_loc, "—", "5000"])

    col_widths = [0.18, 0.18, 0.22, 0.14, 0.10]
    tbl = ax_tbl.table(
        cellText=rows_data,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    # Style header
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.12)

    # Style ground-truth row
    for j in range(len(col_labels)):
        cell = tbl[1, j]
        cell.set_facecolor("#fff9e0")
        cell.set_height(0.10)

    # Style model rows (alternating + colored left cell)
    for i, (pred, color) in enumerate(zip(predictions, colors)):
        row_idx = i + 2
        bg = "#f7f7f7" if i % 2 == 0 else "#ffffff"
        for j in range(len(col_labels)):
            cell = tbl[row_idx, j]
            cell.set_facecolor(bg)
            cell.set_height(0.10)
        # Color swatch in model-name cell
        tbl[row_idx, 0].set_facecolor(
            tuple(list(color[:3]) + [0.25]))

    ax_tbl.set_title("Prediction summary", fontsize=9, fontweight="bold",
                     loc="left", pad=2)

    # ── Super-title with GT info ──
    fig.suptitle(
        f"Image: {image_id}   |   "
        f"Ground truth: {fmt_coord(true_lat, true_lon)}",
        fontsize=10, fontweight="bold", y=0.995,
    )

    fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="Dataset subdirectory name, e.g. im2gps3k_rgb_images")
    p.add_argument("--image_id", required=True,
                   help="Image ID (as stored in meta.jsonl)")
    p.add_argument("--base_dir", default="output",
                   help="Root output directory containing model results (default: output/)")
    p.add_argument("--dataset_dir", default="dataset",
                   help="Root dataset directory (default: dataset/)")
    p.add_argument("--output_dir", default="figures",
                   help="Where to write the PDF (default: figures/)")
    args = p.parse_args()

    base_dir    = Path(args.base_dir)
    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_id = args.image_id
    dataset  = args.dataset

    # ── Ground truth ──
    print(f"Loading ground truth for '{image_id}' from {dataset_dir / dataset} ...")
    gt = load_ground_truth(dataset_dir, dataset, image_id)
    if gt is None:
        sys.exit(f"ERROR: image ID '{image_id}' not found in {dataset_dir / dataset}/meta.jsonl")
    true_lat = float(gt["LAT"])
    true_lon = float(gt["LON"])
    print(f"  Ground truth: {fmt_coord(true_lat, true_lon)}")

    # ── Background scatter from meta.jsonl ──
    meta_path = dataset_dir / dataset / "meta.jsonl"
    if not meta_path.exists():
        meta_path = dataset_dir / dataset.replace("_rgb_images", "") / "meta.jsonl"
    meta_scatter = None
    if meta_path.exists():
        with open(meta_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        meta_scatter = np.array([[r["LAT"], r["LON"]] for r in rows])
        print(f"  Loaded {len(meta_scatter):,} background points from {meta_path}")

    # ── Discover + load model predictions ──
    result_files = find_result_files(base_dir, dataset)
    if not result_files:
        print(f"WARNING: no results_s6_*.jsonl files found under {base_dir / dataset}")

    predictions = []
    for path in result_files:
        key = extract_key(path)
        row = load_prediction(path, image_id)
        if row is None:
            print(f"  [{key}] image not found in {path.name}")
            continue

        coords = parse_prediction(row)
        if coords is None:
            print(f"  [{key}] parse failure — skipping")
            continue

        pred_lat, pred_lon = coords
        dist_km = haversine_distance([pred_lat, pred_lon], [true_lat, true_lon])
        ans      = row.get("answer") or {}
        country  = ans.get("country", "").strip()
        city     = ans.get("city", "").strip()
        loc_str  = ", ".join(filter(None, [city, country])) or "—"

        label = LABEL_MAP.get(key, key)
        predictions.append({
            "key":      key,
            "label":    label,
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
            "dist_km":  dist_km,
            "loc_str":  loc_str,
        })
        print(f"  [{label}]  {fmt_coord(pred_lat, pred_lon)}  →  {fmt_dist(dist_km)}")

    if not predictions:
        print("WARNING: no predictions found — figure will show only ground truth.")

    # Sort by distance for cleaner table ordering
    predictions.sort(key=lambda x: x["dist_km"])

    # ── Image file ──
    img_candidates = [
        dataset_dir / dataset / "images" / f"{image_id}.jpg",
        dataset_dir / dataset.replace("_rgb_images", "") / "images" / f"{image_id}.jpg",
    ]
    image_path = next((p for p in img_candidates if p.exists()), None)
    if image_path:
        print(f"  Image: {image_path}")
    else:
        print("  Image file not found — map-only layout.")

    # ── Output path ──
    safe_id     = _sanitise(image_id)[:80]
    output_path = output_dir / f"pred_map_{safe_id}.pdf"

    # ── Build figure ──
    print(f"\nBuilding figure → {output_path}")
    build_figure(image_id, gt, predictions, image_path, meta_scatter, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
