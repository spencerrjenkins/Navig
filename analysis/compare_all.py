#!/usr/bin/env python3
"""Compare stage-6 results across any number of models in a single table.

Usage::

    python analysis/compare_all.py results_s6_llava.jsonl results_s6_qwen.jsonl ... \\
        [--labels LLaVA Qwen2-VL ...] [--sort geoscore]

The first file (or top-ranked by sort key) is treated as the baseline for the
Δvs1st column.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np

from metrics import geoscore, haversine_distance, parse_coord, THRESHOLDS


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


def score_file(path: str) -> dict:
    rows = load_results(path)
    dists, failures = [], 0
    for row in rows.values():
        d, fail = get_dist(row)
        dists.append(d)
        failures += fail
    n = len(dists)
    accs = [sum(d <= t for d in dists) / n for t in THRESHOLDS]
    return {
        "n": n,
        "geoscore": float(np.mean([geoscore(d) for d in dists])),
        "avg_dist": float(np.mean(dists)),
        "median_dist": float(np.median(dists)),
        "accs": accs,
        "failures": failures,
        "fail_pct": 100.0 * failures / n,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="Result JSONL files (first = baseline)")
    p.add_argument("--labels", nargs="*",
                   help="Display names for each file (default: filename stems)")
    p.add_argument("--sort", choices=["geoscore", "avg_dist", "none"],
                   default="geoscore")
    args = p.parse_args()

    if args.labels and len(args.labels) != len(args.files):
        print("ERROR: --labels count must match number of files.", file=sys.stderr)
        sys.exit(1)

    labels = args.labels or [
        Path(f).stem.replace("results_s6_", "") for f in args.files
    ]

    rows = []
    for path, label in zip(args.files, labels):
        try:
            s = score_file(path)
        except FileNotFoundError:
            print(f"WARNING: {path} not found, skipping.", file=sys.stderr)
            continue
        rows.append((label, s))

    if not rows:
        print("No valid files found.")
        sys.exit(1)

    if args.sort == "geoscore":
        rows.sort(key=lambda x: x[1]["geoscore"], reverse=True)
    elif args.sort == "avg_dist":
        rows.sort(key=lambda x: x[1]["avg_dist"])

    baseline_gs = rows[0][1]["geoscore"]
    thr_names = [f"@{t}km" for t in THRESHOLDS]
    col_w = max(len(r[0]) for r in rows) + 2

    print()
    header = (
        f"{'Model':<{col_w}} {'N':>5}  {'GeoScore':>9}  {'Δvs1st':>7}  "
        f"{'AvgDist':>9}  {'MedDist':>9}  "
        + "  ".join(f"{t:>7}" for t in thr_names)
        + f"  {'Fail%':>6}"
    )
    print(header)
    print("-" * len(header))

    for label, s in rows:
        delta = s["geoscore"] - baseline_gs
        delta_str = f"{delta:+.1f}" if delta != 0.0 else "  base"
        acc_str = "  ".join(f"{a:>7.4f}" for a in s["accs"])
        print(
            f"{label:<{col_w}} {s['n']:>5}  {s['geoscore']:>9.2f}  {delta_str:>7}  "
            f"{s['avg_dist']:>9.1f}  {s['median_dist']:>9.1f}  "
            f"{acc_str}  {s['fail_pct']:>5.1f}%"
        )
    print()


if __name__ == "__main__":
    main()
