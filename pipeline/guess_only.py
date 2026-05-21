#!/usr/bin/env python3
"""Run only stage 6 (coordinate guessing) on an existing results_s5.jsonl.

This script is designed for the stage-6 swap experiment: keep all upstream
evidence from the original NAVIG pipeline and test whether a stronger guesser
model improves final accuracy, without re-running the expensive stages 1–5.

Typical workflow
----------------
1.  Run the full pipeline (stages 1–5) with the original model::

        sbatch slurm/evaluate.sh

2.  Merge the sharded s5 files::

        python analysis/merge_shards.py \\
            --base_dir output/im2gps3k_rgb_images \\
            --num_shards 4 --results_file results_s5.jsonl \\
            --output merged_s5.jsonl

3.  Run stage 6 with a new guesser (submit via slurm/guess_only.sh)::

        python pipeline/guess_only.py \\
            --s5_path output/im2gps3k_rgb_images/merged_s5.jsonl \\
            --dataset_path dataset/im2gps3k_rgb_images \\
            --model llama32vision \\
            --model_path /fs/nexus-scratch/$USER/llama-3.2-11b-vision-instruct \\
            --output output/im2gps3k_rgb_images/results_s6_llama32.jsonl

4.  Compare results::

        python analysis/compare_results.py \\
            output/.../results_s6_llava.jsonl \\
            output/.../results_s6_llama32.jsonl

Supported --model choices
--------------------------
  llava          LLaVA-1.6-Vicuna-7B (baseline)
  qwen           Qwen2-VL-7B-Instruct
  cpm            MiniCPM-V-2.6 (base, no fine-tuning)
  cpm_sft        MiniCPM-V-2.6 with NAVIG LoRA adapter (requires --ckpt_dir)
  llama32vision  Llama-3.2-11B-Vision-Instruct  [recommended experiment]
  internvl2      InternVL2-8B
  deepseek       DeepSeek-VL-7B-Chat
  falcon         Falcon-11B-VLM
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import os

from tqdm import tqdm

from llm import load_model
from metrics import score_results, print_score_summary
from utils import load_data, dump_jsonl, parse_json, build_guess_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_guess(
    s5_path: str,
    dataset_path: str,
    model,
    output_path: str,
    shard_id: int = 0,
    num_shards: int = 1,
    rag_threshold: float = 30.0,
) -> None:
    data = load_data(s5_path)
    if num_shards > 1:
        data = data[shard_id::num_shards]
        logger.info("Shard %d/%d: processing %d samples", shard_id, num_shards, len(data))

    def _generate():
        for row in tqdm(data, desc="Stage 6"):
            image = os.path.join(dataset_path, "images", row["ID"] + ".jpg")
            query, usage = build_guess_query(row, rag_threshold=rag_threshold)
            raw = model.base_inference(query, image)
            answer = parse_json(raw)
            logger.debug("raw: %r  |  parsed: %s  |  gt: %s, %s",
                         raw, answer, row["LAT"], row["LON"])
            row["answer"] = answer
            row["usage"] = usage
            yield row

    dump_jsonl(_generate(), output_path)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--s5_path", type=str, required=True,
                   help="Path to results_s5.jsonl (merged or single shard)")
    p.add_argument("--dataset_path", type=str, required=True,
                   help="Dataset root containing images/ subdirectory")
    p.add_argument("--model", type=str, default="llama32vision",
                   choices=["llava", "qwen", "cpm", "cpm_sft", "llama32vision",
                            "internvl2", "deepseek", "falcon"])
    p.add_argument("--model_path", type=str, required=True,
                   help="Local path or HuggingFace model ID for the guesser model")
    p.add_argument("--ckpt_dir", type=str, default=None,
                   help="LoRA checkpoint directory (required for cpm_sft)")
    p.add_argument("--output", type=str, required=True,
                   help="Output JSONL path for stage-6 results")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--score_only", action="store_true",
                   help="Skip inference, just score an existing --output file")
    p.add_argument("--rag_threshold", type=float, default=30.0,
                   help="Max distance (km) for a RAG hit to be included in the stage-6 prompt")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if not args.score_only:
        model = load_model(args.model, args.model_path, ckpt_dir=args.ckpt_dir)
        run_guess(
            args.s5_path, args.dataset_path, model,
            args.output, args.shard_id, args.num_shards,
            rag_threshold=args.rag_threshold,
        )

    data = load_data(args.output)
    stats = score_results(data)
    print_score_summary(stats, label=args.output)
