#!/usr/bin/env python3
"""Ablation study runner.

Measures the contribution of individual pipeline components by disabling
them selectively and comparing the resulting GeoScores.

Ablation modes (mutually exclusive)
-------------------------------------
--without_reasoning  Disable SFT reasoning (stages 2–5 evidence only)
--without_tools      Disable OSM, RAG, comment (SFT reasoning only)
--base_reasoning     Replace SFT reasoning with base-model reasoning
--direct_guess       Disable everything; raw image → coordinates

Usage::

    python pipeline/ablation.py \\
        --dataset_path dataset/im2gps3k_rgb_images \\
        --model qwen \\
        --output_path output/ablation \\
        --results_filename ablation_wo_reasoning.jsonl \\
        --model_path /fs/nexus-scratch/$USER/Qwen2-VL-7B-Instruct \\
        --ckpt_dir vlms/NAVIG/qwen2-vl-7b-instruct \\
        --without_reasoning

Prerequisites: run pipeline/evaluation.py first to produce results_s1.jsonl
and results_s5.jsonl in --output_path.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import glob
import logging
import os

from tqdm import tqdm

import prompts
from llm import load_model, load_sft_model
from metrics import score_results, print_score_summary
from utils import load_data, dump_jsonl, parse_json, build_guess_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _find_file(pattern: str, folder: str) -> str | None:
    results = glob.glob(os.path.join(folder, pattern))
    return results[0] if results else None


class Ablation:
    """Runs targeted ablation experiments on a pre-computed pipeline output."""

    def __init__(
        self,
        dataset_path: str,
        model_type: str,
        output_path: str,
        results_filename: str,
        model_path: str,
        ckpt_dir: str,
    ):
        self.dataset_path = dataset_path
        self.model_type = model_type
        self.output_path = output_path
        self.results_filename = results_filename
        self.model_path = model_path
        self.ckpt_dir = ckpt_dir
        self.base_model = load_model(model_type, model_path)

    def _image_path(self, row: dict) -> str:
        return os.path.join(self.dataset_path, "images", row["ID"] + ".jpg")

    def get_reasoning(self, use_sft: bool = True):
        """Yield rows with 'image_reason' (stage 1 output).

        If *use_sft* is False, the base model is used instead of the SFT adapter.
        """
        logger.info("Stage 1 — Reasoning (sft=%s)", use_sft)
        reasoning_model = (
            load_sft_model(self.model_type, self.model_path, self.ckpt_dir)
            if use_sft
            else self.base_model
        )
        data = load_data(os.path.join(self.dataset_path, "meta.jsonl"))
        for row in tqdm(data, desc="Stage 1"):
            row["image_reason"] = reasoning_model.base_inference(
                prompts.reasoning_prompt, self._image_path(row)
            )
            yield row

    def guess_coordinates(
        self,
        load_file: str,
        include_reasoning: bool = True,
        include_osm: bool = True,
        include_rag: bool = True,
        include_comment: bool = True,
    ):
        """Yield rows with 'answer' and 'usage', with optional evidence components
        disabled for ablation."""
        logger.info(
            "Stage 6 — Guessing (reasoning=%s, osm=%s, rag=%s, comment=%s)",
            include_reasoning, include_osm, include_rag, include_comment,
        )
        data = load_data(load_file)
        for row in tqdm(data, desc="Stage 6"):
            query, usage = build_guess_query(
                row,
                include_reasoning=include_reasoning,
                include_osm=include_osm,
                include_rag=include_rag,
                include_comment=include_comment,
            )
            raw = self.base_model.base_inference(query, self._image_path(row))
            row["answer"] = parse_json(raw)
            row["usage"] = usage
            yield row

    def _run_guess(self, load_file: str, **kwargs) -> None:
        out_file = os.path.join(self.output_path, self.results_filename)
        dump_jsonl(self.guess_coordinates(load_file, **kwargs), out_file)

    def calculate_score(self) -> None:
        data = load_data(os.path.join(self.output_path, self.results_filename))
        stats = score_results(data)
        print_score_summary(stats, label=self.results_filename)

    def calculate_score_cc(self) -> None:
        data = load_data(os.path.join(self.output_path, self.results_filename))
        country_correct = city_correct = total = 0
        for row in data:
            try:
                if row["country"] in row["answer"]["country"]:
                    country_correct += 1
                if row["city"] in row["answer"]["city"]:
                    city_correct += 1
                total += 1
            except Exception:
                continue
        if total:
            print(f"Country match: {country_correct/total:.4f} ({country_correct}/{total})")
            print(f"City match   : {city_correct/total:.4f} ({city_correct}/{total})")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--model", type=str, default="qwen",
                   choices=["qwen", "llava", "cpm", "llama32vision", "deepseek", "falcon"])
    p.add_argument("--output_path", type=str, default=".")
    p.add_argument("--results_filename", type=str, default="ablation_results.jsonl")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--ckpt_dir", type=str, default=None)

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--without_reasoning", action="store_true",
                       help="Disable SFT reasoning; use OSM, RAG, comment only")
    group.add_argument("--without_tools", action="store_true",
                       help="Disable OSM, RAG, comment; use SFT reasoning only")
    group.add_argument("--base_reasoning", action="store_true",
                       help="Replace SFT reasoning with base-model reasoning")
    group.add_argument("--direct_guess", action="store_true",
                       help="No evidence at all; raw image → coordinates")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    ablation = Ablation(
        dataset_path=args.dataset_path,
        model_type=args.model,
        output_path=args.output_path,
        results_filename=args.results_filename,
        model_path=args.model_path,
        ckpt_dir=args.ckpt_dir or "",
    )

    if args.without_reasoning:
        s5 = _find_file("results_s5.jsonl", args.output_path)
        if not s5:
            print("ERROR: results_s5.jsonl not found. Run pipeline/evaluation.py first.")
            sys.exit(1)
        ablation._run_guess(s5, include_reasoning=False)

    elif args.without_tools:
        s1 = _find_file("results_s1.jsonl", args.output_path)
        if not s1:
            print("ERROR: results_s1.jsonl not found. Run pipeline/evaluation.py first.")
            sys.exit(1)
        ablation._run_guess(s1, include_osm=False, include_rag=False, include_comment=False)

    elif args.base_reasoning:
        s1_base = os.path.join(args.output_path, "results_s1_base.jsonl")
        if not os.path.exists(s1_base):
            dump_jsonl(ablation.get_reasoning(use_sft=False), s1_base)
        ablation._run_guess(
            s1_base, include_osm=False, include_rag=False, include_comment=False
        )

    elif args.direct_guess:
        meta = os.path.join(args.dataset_path, "meta.jsonl")
        ablation._run_guess(
            meta,
            include_reasoning=False,
            include_osm=False,
            include_rag=False,
            include_comment=False,
        )

    ablation.calculate_score()
    ablation.calculate_score_cc()
