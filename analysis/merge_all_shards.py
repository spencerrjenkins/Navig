#!/usr/bin/env python3
"""Merge all shard output directories within a given directory.

Shard dirs may follow any of the naming conventions used in this project::

    {model}_shard_{N}_of_{M}         e.g. qwen_shard_0_of_4
    cmp_shard_{model}_{N}_of_{M}     e.g. cmp_shard_deepseek_0_of_4
    guess_shard_{N}_of_{M}           e.g. guess_shard_0_of_4
    shard_{N}_of_{M}                 e.g. shard_0_of_1

The unifying rule: any directory whose name ends with ``_{N}_of_{M}`` (both
integers) is treated as a shard.  Everything before that suffix is the group
key.  All shards with the same key are concatenated (in shard-index order)
into a ``{key}_merged/`` directory alongside the originals.

Usage::

    python analysis/merge_all_shards.py output/im2gps3k_rgb_images [--dry-run]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import re
from collections import defaultdict


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", type=Path, help="Directory containing shard subdirs")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without writing files")
    args = p.parse_args()

    root = args.directory.resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    shard_pattern = re.compile(r"^(.+)_(\d+)_of_(\d+)$")
    groups: dict[str, dict[int, Path]] = defaultdict(dict)
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = shard_pattern.match(child.name)
        if not m:
            continue
        prefix, shard_idx = m.group(1), int(m.group(2))
        groups[prefix][shard_idx] = child

    if not groups:
        print("No shard directories found.", file=sys.stderr)
        sys.exit(1)

    for prefix, shards in sorted(groups.items()):
        shard_dirs = [shards[i] for i in sorted(shards)]
        filenames: set[str] = set()
        for shard_dir in shard_dirs:
            for f in shard_dir.iterdir():
                if f.is_file():
                    filenames.add(f.name)

        merged_dir = root / f"{prefix}_merged"
        print(f"\n[{prefix}] {len(shard_dirs)} shards → {merged_dir}")

        if not args.dry_run:
            merged_dir.mkdir(exist_ok=True)

        for filename in sorted(filenames):
            sources = [
                shard_dir / filename
                for shard_dir in shard_dirs
                if (shard_dir / filename).exists()
            ]
            print(f"  {filename}: merging {len(sources)} shard(s)")
            if args.dry_run:
                continue
            out_path = merged_dir / filename
            total_lines = 0
            with out_path.open("w") as out_f:
                for src in sources:
                    with src.open("r") as in_f:
                        for line in in_f:
                            out_f.write(line)
                            total_lines += 1
            print(f"    → {total_lines} lines → {out_path}")

    if args.dry_run:
        print("\n(dry run — no files written)")


if __name__ == "__main__":
    main()
