#!/usr/bin/env python3
"""
Generate im2gps3k dataset illustration figures for model_justification.tex.

Produces:
  <output_dir>/fig_dataset_examples.pdf  — 2×3 grid: top=best predictions, bottom=worst
  <output_dir>/fig_dataset_geo.pdf       — bar chart of regional image counts

Usage::

    python figures/make_dataset_figures.py [--output_dir figures/] [--img_dir dataset/im2gps3k_rgb_images/images] [--meta_path dataset/im2gps3k_rgb_images/meta.jsonl]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "vermilion": "#D55E00",
    "skyblue": "#56B4E9",
    "yellow": "#F0E442",
}

# ---------------------------------------------------------------------------
# Selected examples  (ID, GT description, pred description, error_km, note)
# ---------------------------------------------------------------------------
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


def load_image(img_dir: Path, id_):
    path = img_dir / f"{id_}.jpg"
    if not path.exists():
        return None
    return mpimg.imread(str(path))


def fmt_err(km):
    if km < 1:
        return f"{km*1000:.0f} m"
    elif km < 1000:
        return f"{km:.0f} km"
    else:
        return f"{km/1000:.1f}×10³ km"


# ---------------------------------------------------------------------------
# Figure 1: best/worst prediction grid
# ---------------------------------------------------------------------------
def make_example_grid(img_dir: Path, fig_dir: Path):
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

            err_str = fmt_err(ex["err"])
            for spine in ax.spines.values():
                spine.set_edgecolor(row_col)
                spine.set_linewidth(3)

            ax.set_title(
                ex["note"], fontsize=9.5, fontweight="bold", pad=4, color="#222222"
            )

            gt_line = f"GT:   {ex['gt']}"
            pred_line = f"Pred: {ex['pred']}"
            err_color = OKABE["green"] if ex["err"] < 50 else OKABE["vermilion"]
            err_line = f"Error: {err_str}"

            ax.text(
                0.5, -0.01,
                f"{gt_line}\n{pred_line}",
                transform=ax.transAxes,
                fontsize=7.2, ha="center", va="top",
                color="#333333", fontfamily="monospace",
            )
            ax.text(
                0.5, -0.18,
                err_line,
                transform=ax.transAxes,
                fontsize=8.5, ha="center", va="top",
                color=err_color, fontweight="bold",
            )

        axes[row_idx][0].set_ylabel(
            row_label, fontsize=11, fontweight="bold",
            color=row_col, labelpad=8,
        )

    fig.suptitle(
        "im2gps3k benchmark: representative predictions (LLaVA-1.6 full pipeline)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.subplots_adjust(wspace=0.08, hspace=0.55, bottom=0.08)
    out = fig_dir / "fig_dataset_examples.pdf"
    fig.savefig(str(out), bbox_inches="tight", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: geographic distribution bar chart
# ---------------------------------------------------------------------------
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


def make_geo_bar(meta_path: Path, fig_dir: Path):
    with open(meta_path) as f:
        meta = [json.loads(l) for l in f]

    counts = Counter(continent(r["LAT"], r["LON"]) for r in meta)
    order = ["Europe", "N. America", "Asia (N)", "Other",
             "Africa", "S. America", "Oceania"]
    colors = [
        OKABE["blue"], OKABE["orange"], OKABE["green"], "#aaaaaa",
        OKABE["vermilion"], OKABE["pink"], OKABE["skyblue"],
    ]

    labels = [f"{k}\n({counts[k]:,})" for k in order]
    vals = [counts[k] for k in order]
    total = sum(vals)
    pcts = [100 * v / total for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_ylabel("Number of images", fontsize=11)
    ax.set_title(
        f"im2gps3k geographic distribution (N = {total:,})", fontsize=12, fontweight="bold"
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=9, color="#333333",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = fig_dir / "fig_dataset_geo.pdf"
    fig.savefig(str(out), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", type=str, default="figures",
                   help="Directory for output PDFs (default: figures/)")
    p.add_argument("--img_dir", type=str,
                   default="dataset/im2gps3k_rgb_images/images",
                   help="Directory containing {ID}.jpg images")
    p.add_argument("--meta_path", type=str,
                   default="dataset/im2gps3k_rgb_images/meta.jsonl",
                   help="JSONL file with LAT/LON metadata")
    args = p.parse_args()

    fig_dir = Path(args.output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.img_dir)
    meta_path = Path(args.meta_path)

    print(f"Generating dataset figures → {fig_dir}/")
    make_example_grid(img_dir, fig_dir)
    make_geo_bar(meta_path, fig_dir)
    print("Done.")


if __name__ == "__main__":
    main()
