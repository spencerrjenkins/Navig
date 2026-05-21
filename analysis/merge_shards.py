#!/usr/bin/env python3
"""Merge per-shard result files and compute final scores.

Usage::

    python analysis/merge_shards.py \\
        --base_dir output/im2gps3k \\
        --num_shards 4 \\
        --results_file results_s6_qwen.jsonl \\
        --output merged_results_qwen.jsonl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import os

from metrics import score_results, print_score_summary


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Directory containing shard subdirectories",
    )
    p.add_argument("--num_shards", type=int, required=True)
    p.add_argument(
        "--results_file",
        type=str,
        default="results_s6_llava.jsonl",
        help="Filename to merge from each shard directory",
    )
    p.add_argument(
        "--output",
        type=str,
        default="merged_results.jsonl",
        help="Output filename (written inside --base_dir)",
    )
    p.add_argument(
        "--shard_prefix",
        type=str,
        default="shard",
        help='Directory prefix for shard folders (default: "shard" → shard_0_of_4)',
    )
    args = p.parse_args()

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
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Merged {len(rows)} rows → {output_path}")

    stats = score_results(rows)
    print_score_summary(stats, label=output_path)


if __name__ == "__main__":
    main()
