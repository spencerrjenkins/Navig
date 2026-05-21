#!/usr/bin/env python3
"""Compare two stage-6 result files side by side.

Usage::

    python analysis/compare_results.py BASELINE EXPERIMENT \\
        [--label-a NAME] [--label-b NAME] [--top-k N]

Output
------
  • Per-metric table: GeoScore, Avg/Median Distance, accuracy at each threshold
  • Parse failure rates
  • Win / loss / tie breakdown per image
  • Distance-delta quartiles
  • Top-K images most improved and most hurt by the experiment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np

from metrics import geoscore, haversine_distance, parse_coord, THRESHOLDS, THRESHOLD_NAMES


def load_results(path: str) -> dict[str, dict]:
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[row["ID"]] = row
    return rows


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


def summarise(label: str, distances: list[float], failures: int, n_total: int) -> None:
    thr_labels = [f"@{t}km" for t in THRESHOLDS]
    print(f"\n--- {label} ---")
    print(f"  N: {n_total}  |  Parse failures: {failures} ({100*failures/n_total:.1f}%)")
    print(f"  Avg GeoScore : {np.mean([geoscore(d) for d in distances]):.2f}")
    print(f"  Avg Distance : {np.mean(distances):.2f} km  "
          f"(median {np.median(distances):.2f} km)")
    for t, name, label_t in zip(THRESHOLDS, THRESHOLD_NAMES, thr_labels):
        acc = sum(d <= t for d in distances) / n_total
        print(f"  Acc@{name:<20}: {acc:.4f}  ({sum(d<=t for d in distances)}/{n_total})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline", help="Baseline result JSONL")
    p.add_argument("experiment", help="Experiment result JSONL")
    p.add_argument("--label-a", default="Baseline")
    p.add_argument("--label-b", default="Experiment")
    p.add_argument("--top-k", type=int, default=10,
                   help="Print top-K images most improved / most hurt")
    args = p.parse_args()

    rows_a = load_results(args.baseline)
    rows_b = load_results(args.experiment)

    common = sorted(set(rows_a) & set(rows_b))
    if not common:
        print("ERROR: no common image IDs between the two files.")
        sys.exit(1)
    if len(common) < len(rows_a) or len(common) < len(rows_b):
        print(f"WARNING: only {len(common)} IDs in common "
              f"({len(rows_a)} in baseline, {len(rows_b)} in experiment)")

    dists_a, fails_a = [], 0
    dists_b, fails_b = [], 0
    deltas = []
    per_image = []

    for img_id in common:
        da, fa = get_dist(rows_a[img_id])
        db, fb = get_dist(rows_b[img_id])
        dists_a.append(da)
        dists_b.append(db)
        fails_a += fa
        fails_b += fb
        deltas.append(db - da)
        per_image.append((img_id, da, db, db - da))

    n = len(common)
    summarise(args.label_a, dists_a, fails_a, n)
    summarise(args.label_b, dists_b, fails_b, n)

    wins   = sum(1 for d in deltas if d < -0.5)
    losses = sum(1 for d in deltas if d >  0.5)
    ties   = n - wins - losses
    print(f"\n--- Head-to-head (experiment vs baseline) ---")
    print(f"  Experiment better (>0.5 km closer) : {wins}/{n}  ({100*wins/n:.1f}%)")
    print(f"  Experiment worse  (>0.5 km farther): {losses}/{n}  ({100*losses/n:.1f}%)")
    print(f"  Roughly tied                        : {ties}/{n}  ({100*ties/n:.1f}%)")

    gs_a = np.mean([geoscore(d) for d in dists_a])
    gs_b = np.mean([geoscore(d) for d in dists_b])
    print(f"\n  GeoScore delta (experiment - baseline): {gs_b - gs_a:+.2f}")
    print(f"  Avg Distance delta                    : {np.mean(deltas):+.2f} km")

    print(f"\n  Distance delta quartiles (km, neg = experiment improved):")
    for q, lbl in [(0, "min"), (25, "Q1"), (50, "median"), (75, "Q3"), (100, "max")]:
        print(f"    {lbl:6s}: {np.percentile(deltas, q):+.1f} km")

    per_image.sort(key=lambda x: x[3])
    print(f"\n--- Top {args.top_k} images most improved by experiment ---")
    print(f"  {'ID':<20} {'Baseline(km)':>14} {'Experiment(km)':>16} {'Delta':>10}")
    for img_id, da, db, delta in per_image[: args.top_k]:
        print(f"  {img_id:<20} {da:>14.1f} {db:>16.1f} {delta:>+10.1f}")

    print(f"\n--- Top {args.top_k} images most hurt by experiment ---")
    for img_id, da, db, delta in per_image[-args.top_k:][::-1]:
        print(f"  {img_id:<20} {da:>14.1f} {db:>16.1f} {delta:>+10.1f}")


if __name__ == "__main__":
    main()
