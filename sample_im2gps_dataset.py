#!/usr/bin/env python3
"""Create a sampled Im2GPS dataset with matching images and meta.jsonl.

Example:
    python sample_im2gps_dataset.py \
        --source_dir dataset/im2gps3k \
        --output_dir dataset/im2gps200 \
        --num_samples 200 \
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from utils import dump_jsonl, load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a smaller Im2GPS dataset and copy the matching images."
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("dataset/im2gps3k"),
        help="Source dataset directory containing meta.jsonl and images/",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset/im2gps200"),
        help="Destination directory for the sampled dataset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of rows/images to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".jpg",
        help="Image file extension used in the source dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_meta_path = args.source_dir / "meta.jsonl"
    source_images_dir = args.source_dir / "images"
    output_images_dir = args.output_dir / "images"
    output_meta_path = args.output_dir / "meta.jsonl"

    rows = load_data(str(source_meta_path))
    if args.num_samples > len(rows):
        raise ValueError(
            f"Requested {args.num_samples} samples, but {source_meta_path} contains only {len(rows)} rows"
        )

    rng = random.Random(args.seed)
    sampled_indices = sorted(rng.sample(range(len(rows)), args.num_samples))
    sampled_rows = [rows[index] for index in sampled_indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    for row in sampled_rows:
        image_id = row["ID"]
        source_image_path = source_images_dir / f"{image_id}{args.image_ext}"
        if not source_image_path.exists():
            raise FileNotFoundError(
                f"Missing image for {image_id}: {source_image_path}"
            )

        shutil.copy2(source_image_path, output_images_dir / source_image_path.name)

    dump_jsonl(sampled_rows, str(output_meta_path))

    print(
        f"Created sampled dataset at {args.output_dir} with {len(sampled_rows)} rows and images."
    )


if __name__ == "__main__":
    main()
