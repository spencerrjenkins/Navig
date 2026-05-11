#!/usr/bin/env python3
"""Compare two stage-6 result files side by side.

Usage:
    python compare_results.py BASELINE_FILE EXPERIMENT_FILE [--label-a NAME] [--label-b NAME]

Example:
    python compare_results.py \
        output/im2gps3k_rgb_images/results_s6_llava.jsonl \
        output/im2gps3k_rgb_images/results_s6_llama32.jsonl \
        --label-a "LLaVA-7B (baseline)" \
        --label-b "Llama-3.2-11B (experiment)"

Output
------
  • Per-metric table: GeoScore, Avg Distance, and accuracy at each threshold
  • Parse failure rates (how often the model output was unparseable JSON)
  • Win / loss / tie breakdown per image (which model was closer)
  • Distance delta distribution (experiment - baseline) in quartiles
"""

import argparse
import json
import sys
import numpy as np
from math import radians, sin, cos, sqrt, atan2


def haversine(c1, c2):
    lat1, lon1 = map(radians, c1)
    lat2, lon2 = map(radians, c2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * atan2(sqrt(a), sqrt(1 - a))


def geoscore(d):
    return 5000 * np.exp(-d / 1492.7)


def load_results(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row['ID']] = row
    return rows


def get_dist(row):
    correct = [float(row['LAT']), float(row['LON'])]
    try:
        pred = [float(row['answer']['latitude']), float(row['answer']['longitude'])]
        return haversine(pred, correct), False
    except Exception:
        return 10000.0, True   # parse failure → penalise as max distance


def summarise(label, distances, failures, n_total):
    thresholds = [1, 25, 200, 750, 2500]
    names = ['Street(1km)', 'City(25km)', 'Region(200km)', 'Country(750km)', 'Continent(2500km)']
    print(f'\n--- {label} ---')
    print(f'  N: {n_total}  |  Parse failures: {failures} ({100*failures/n_total:.1f}%)')
    print(f'  Avg GeoScore:  {np.mean([geoscore(d) for d in distances]):.2f}')
    print(f'  Avg Distance:  {np.mean(distances):.2f} km  '
          f'(median {np.median(distances):.2f} km)')
    for t, name in zip(thresholds, names):
        acc = sum(d <= t for d in distances) / n_total
        print(f'  Acc@{name:<20}: {acc:.4f}  ({sum(d<=t for d in distances)}/{n_total})')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('baseline', help='Baseline result JSONL (e.g. results_s6_llava.jsonl)')
    p.add_argument('experiment', help='Experiment result JSONL (e.g. results_s6_llama32.jsonl)')
    p.add_argument('--label-a', default='Baseline')
    p.add_argument('--label-b', default='Experiment')
    p.add_argument('--top-k', type=int, default=10,
                   help='Print the top-K images where experiment most improves/hurts vs baseline')
    args = p.parse_args()

    rows_a = load_results(args.baseline)
    rows_b = load_results(args.experiment)

    common = sorted(set(rows_a) & set(rows_b))
    if not common:
        print('ERROR: no common image IDs between the two files.')
        sys.exit(1)
    if len(common) < len(rows_a) or len(common) < len(rows_b):
        print(f'WARNING: only {len(common)} IDs in common '
              f'({len(rows_a)} in baseline, {len(rows_b)} in experiment)')

    dists_a, fails_a = [], 0
    dists_b, fails_b = [], 0
    deltas = []   # dist_b - dist_a  (negative = experiment improved)

    per_image = []
    for img_id in common:
        ra, rb = rows_a[img_id], rows_b[img_id]
        da, fa = get_dist(ra)
        db, fb = get_dist(rb)
        dists_a.append(da)
        dists_b.append(db)
        fails_a += fa
        fails_b += fb
        deltas.append(db - da)
        per_image.append((img_id, da, db, db - da))

    n = len(common)
    summarise(args.label_a, dists_a, fails_a, n)
    summarise(args.label_b, dists_b, fails_b, n)

    # Win/loss/tie
    wins   = sum(1 for d in deltas if d < -0.5)
    losses = sum(1 for d in deltas if d >  0.5)
    ties   = n - wins - losses
    print(f'\n--- Head-to-head (experiment vs baseline) ---')
    print(f'  Experiment better (>0.5 km closer): {wins}/{n}  ({100*wins/n:.1f}%)')
    print(f'  Experiment worse  (>0.5 km farther): {losses}/{n}  ({100*losses/n:.1f}%)')
    print(f'  Roughly tied:                        {ties}/{n}  ({100*ties/n:.1f}%)')

    # GeoScore delta
    gs_a = np.mean([geoscore(d) for d in dists_a])
    gs_b = np.mean([geoscore(d) for d in dists_b])
    print(f'\n  GeoScore delta (experiment - baseline): {gs_b - gs_a:+.2f}')
    print(f'  Avg Distance delta:                     {np.mean(deltas):+.2f} km')

    # Distance-delta quartiles
    print(f'\n  Distance delta quartiles (km, neg = experiment improved):')
    for q, label in [(0, 'min'), (25, 'Q1'), (50, 'median'), (75, 'Q3'), (100, 'max')]:
        print(f'    {label:6s}: {np.percentile(deltas, q):+.1f} km')

    # Top-K improvements and regressions
    per_image.sort(key=lambda x: x[3])   # most improved first
    print(f'\n--- Top {args.top_k} images most improved by experiment ---')
    print(f'  {"ID":<20} {"Baseline(km)":>14} {"Experiment(km)":>16} {"Delta":>10}')
    for img_id, da, db, delta in per_image[:args.top_k]:
        print(f'  {img_id:<20} {da:>14.1f} {db:>16.1f} {delta:>+10.1f}')

    print(f'\n--- Top {args.top_k} images most hurt by experiment ---')
    for img_id, da, db, delta in per_image[-args.top_k:][::-1]:
        print(f'  {img_id:<20} {da:>14.1f} {db:>16.1f} {delta:>+10.1f}')


if __name__ == '__main__':
    main()
