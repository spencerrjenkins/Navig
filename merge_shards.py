#!/usr/bin/env python3
"""Merge per-shard result files and compute final scores.

Usage:
    python merge_shards.py \
        --base_dir output/im2gps3k_rgb_images \
        --num_shards 4 \
        --results_file results_s6_llava.jsonl \
        --output merged_results.jsonl
"""
import argparse
import json
import os
import numpy as np


def haversine_distance(coord1, coord2):
    from math import radians, sin, cos, sqrt, atan2
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * atan2(sqrt(a), sqrt(1 - a))


def Geoscore(distance):
    return 5000 * np.exp(-distance / 1492.7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--num_shards', type=int, required=True)
    parser.add_argument('--results_file', type=str, default='results_s6_llava.jsonl')
    parser.add_argument('--output', type=str, default='merged_results.jsonl')
    parser.add_argument('--shard_prefix', type=str, default='shard',
                        help='Directory prefix for shard folders (default: "shard" → shard_0_of_4)')
    args = parser.parse_args()

    rows = []
    for shard_id in range(args.num_shards):
        shard_path = os.path.join(
            args.base_dir,
            f"{args.shard_prefix}_{shard_id}_of_{args.num_shards}",
            args.results_file,
        )
        if not os.path.exists(shard_path):
            print(f"WARNING: missing shard file {shard_path}")
            continue
        with open(shard_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    output_path = os.path.join(args.base_dir, args.output)
    with open(output_path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    print(f"Merged {len(rows)} rows → {output_path}")

    # Score
    counts = [0, 0, 0, 0, 0]
    total_points = 0
    total_dist = 0
    thresholds = [1, 25, 200, 750, 2500]
    for row in rows:
        correct = [float(row['LAT']), float(row['LON'])]
        try:
            pred = [float(row['answer']['latitude']), float(row['answer']['longitude'])]
            dist = haversine_distance(pred, correct)
        except Exception:
            dist = 10000
        total_points += Geoscore(dist)
        total_dist += dist
        for i, t in enumerate(thresholds):
            if dist <= t:
                counts[i] += 1

    n = len(rows)
    score = [c / n for c in counts]
    print(f"Five Level Street→Continent: {score}")
    print(f"Avg GeoScore: {total_points / n:.2f}")
    print(f"Avg Distance: {total_dist / n:.2f} km")


if __name__ == '__main__':
    main()
