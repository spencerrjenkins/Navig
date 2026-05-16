#!/usr/bin/env python3
"""Compare stage-6 results across any number of models in a single table.

Usage:
    python compare_all.py results_s6_llava.jsonl results_s6_qwen25.jsonl ... \
        [--labels LLaVA Qwen2.5 ...] [--sort geoscore]

The first file is treated as the baseline for the delta column.
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
            if line:
                row = json.loads(line)
                rows[row['ID']] = row
    return rows


def get_dist(row):
    correct = [float(row['LAT']), float(row['LON'])]
    try:
        pred = [float(row['answer']['latitude']), float(row['answer']['longitude'])]
        return haversine(pred, correct), False
    except Exception:
        return 10000.0, True


def score_file(path):
    rows = load_results(path)
    dists, failures = [], 0
    for row in rows.values():
        d, fail = get_dist(row)
        dists.append(d)
        failures += fail
    n = len(dists)
    thresholds = [1, 25, 200, 750, 2500]
    accs = [sum(d <= t for d in dists) / n for t in thresholds]
    return {
        'n': n,
        'geoscore': float(np.mean([geoscore(d) for d in dists])),
        'avg_dist': float(np.mean(dists)),
        'median_dist': float(np.median(dists)),
        'accs': accs,
        'failures': failures,
        'fail_pct': 100.0 * failures / n,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', nargs='+', help='result JSONL files (first = baseline)')
    p.add_argument('--labels', nargs='*',
                   help='display names for each file (default: filename stems)')
    p.add_argument('--sort', choices=['geoscore', 'avg_dist', 'none'], default='geoscore',
                   help='sort rows by this metric (default: geoscore descending)')
    args = p.parse_args()

    if args.labels and len(args.labels) != len(args.files):
        print('ERROR: --labels count must match number of files.', file=sys.stderr)
        sys.exit(1)

    labels = args.labels or [f.split('/')[-1].replace('.jsonl', '') for f in args.files]

    rows = []
    for path, label in zip(args.files, labels):
        try:
            s = score_file(path)
        except FileNotFoundError:
            print(f'WARNING: {path} not found, skipping.', file=sys.stderr)
            continue
        rows.append((label, s))

    if not rows:
        print('No valid files found.')
        sys.exit(1)

    if args.sort == 'geoscore':
        rows.sort(key=lambda x: x[1]['geoscore'], reverse=True)
    elif args.sort == 'avg_dist':
        rows.sort(key=lambda x: x[1]['avg_dist'])

    baseline_gs = rows[0][1]['geoscore'] if args.sort != 'none' else score_file(args.files[0])['geoscore']

    # Header
    thr_names = ['@1km', '@25km', '@200km', '@750km', '@2500km']
    col_w = max(len(r[0]) for r in rows) + 2
    print()
    header = (f"{'Model':<{col_w}} {'N':>5}  {'GeoScore':>9}  {'Δvs1st':>7}  "
              f"{'AvgDist':>9}  {'MedDist':>9}  "
              + '  '.join(f'{t:>7}' for t in thr_names)
              + f"  {'Fail%':>6}")
    print(header)
    print('-' * len(header))

    for label, s in rows:
        delta = s['geoscore'] - baseline_gs
        delta_str = f'{delta:+.1f}' if delta != 0.0 else '  base'
        acc_str = '  '.join(f'{a:>7.4f}' for a in s['accs'])
        print(f"{label:<{col_w}} {s['n']:>5}  {s['geoscore']:>9.2f}  {delta_str:>7}  "
              f"{s['avg_dist']:>9.1f}  {s['median_dist']:>9.1f}  "
              f"{acc_str}  {s['fail_pct']:>5.1f}%")

    print()


if __name__ == '__main__':
    main()
