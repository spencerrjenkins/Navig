#!/usr/bin/env python3
"""Reasoning-chain quality evaluation using ROUGE scores and GPT-4o as judge.

REQUIREMENTS
------------
* Set the OPENAI_API_KEY environment variable before running::

      export OPENAI_API_KEY="sk-..."

* Provide a test JSONL file (--test_path) with fields:
      ID, LAT, LON, country, city, images (list with at least one path)

* To compute ROUGE against human references, provide a reference JSONL
  (--ref_path) with fields: ID, response

Usage::

    python analysis/rouge.py \\
        --model qwen_sft \\
        --test_path /path/to/test_set.jsonl \\
        --ref_path /path/to/reference_responses.jsonl \\
        --output_path output/rouge_eval
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import os

from tqdm import tqdm
from rouge_score import rouge_scorer
from openai import OpenAI

from configuration import Config
import prompts
from utils import load_data, dump_jsonl, parse_json


def _load_model(model_name: str, model_path: str, ckpt_dir: str | None = None):
    from llm import load_model, load_sft_model
    if ckpt_dir:
        return load_sft_model(model_name, model_path, ckpt_dir)
    return load_model(model_name, model_path)


class RougeEvaluator:

    def __init__(self, model_name: str, output_path: str):
        self.model_name = model_name
        self.output_path = output_path

    def get_reasoning(self, test_path: str, model_path: str, ckpt_dir: str | None = None):
        """Run model inference on *test_path* and yield rows with 'model_response'."""
        data = load_data(test_path)
        model = _load_model(self.model_name, model_path, ckpt_dir)
        for row in tqdm(data, desc="Reasoning inference"):
            image = row["images"][0]
            row["model_response"] = model.base_inference(prompts.reasoning_prompt, image)
            yield row

    def get_quality_eval(self, results_path: str):
        """Score reasoning quality with GPT-4o as judge.

        *results_path* should be a JSONL produced by get_reasoning(), with
        fields LAT, LON, country, city, image_reason (or model_response).
        """
        if not Config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before running quality eval."
            )
        data = load_data(results_path)
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        prompt_template = (
            "Please generate a JSON object based on the given reasoning text, "
            "correct country name, city name, and coordinates. Format:\n"
            "{\n"
            '    "country": <0 or 1>,\n'
            '    "country_correct": <0 or 1>,\n'
            '    "others": <0 or 1>,\n'
            '    "others_correct": <0 or 1>\n'
            "}\n\n"
            "Guidelines:\n"
            "country: 1 if reasoning predicts a country, else 0.\n"
            "country_correct: 1 if the prediction matches the ground truth, else 0.\n"
            "others: 1 if reasoning predicts city/town/etc., else 0.\n"
            "others_correct: 1 if that finer prediction is correct, else 0.\n\n"
        )
        for row in tqdm(data, desc="GPT-4o quality eval"):
            text = row.get("image_reason") or row.get("model_response", "")
            special = (
                f"Reasoning text: {text}\n"
                f"Correct country: {row['country']}\n"
                f"Correct city: {row['city']}\n"
                f"Coordinates: Latitude {row['LAT']}, Longitude {row['LON']}"
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt_template + special}
                ]}],
            )
            row["quality"] = parse_json(response.choices[0].message.content)
            yield row

    def get_rouge(self, candidate_path: str, ref_path: str):
        """Score model responses against human references using ROUGE.

        *candidate_path*: JSONL with 'model_response' field.
        *ref_path*: JSONL with 'response' field (human reference).
        """
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        candidates = {row["ID"]: row for row in load_data(candidate_path)}
        references = {row["ID"]: row for row in load_data(ref_path)}
        common_ids = set(candidates) & set(references)
        for img_id in common_ids:
            cand_row = candidates[img_id]
            ref_text = references[img_id]["response"]
            cand_text = cand_row.get("model_response") or cand_row.get("image_reason", "")
        scores = scorer.score(ref_text, cand_text)
            cand_row["rouge_scores"] = {
                k: {"precision": v.precision, "recall": v.recall, "fmeasure": v.fmeasure}
                for k, v in scores.items()
            }
            yield cand_row

    @staticmethod
    def print_rouge_summary(rouge_path: str) -> None:
        """Print average ROUGE scores across all rows in *rouge_path*."""
        data = load_data(rouge_path)
        n = len(data)
        if n == 0:
            print("No rows found.")
            return
        for metric in ["rouge1", "rouge2", "rougeL"]:
            p = sum(row["rouge_scores"][metric]["precision"] for row in data) / n
            r = sum(row["rouge_scores"][metric]["recall"] for row in data) / n
            f = sum(row["rouge_scores"][metric]["fmeasure"] for row in data) / n
            print(f"{metric}: precision={p:.4f}  recall={r:.4f}  fmeasure={f:.4f}  (n={n})")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, default="qwen",
                   choices=["qwen", "llava", "cpm", "llama32vision", "deepseek", "falcon"])
    p.add_argument("--model_path", type=str, default=None,
                   help="Local path to model weights (required unless --results_s1_path is set)")
    p.add_argument("--ckpt_dir", type=str, default=None,
                   help="LoRA SFT adapter directory (empty → zero-shot base model)")
    p.add_argument("--test_path", type=str, default=None,
                   help="JSONL test set for inference (required unless --results_s1_path is set)")
    p.add_argument("--results_s1_path", type=str, default=None,
                   help="Path to existing stage-1 JSONL (image_reason or model_response field). "
                        "If provided, skips model inference entirely.")
    p.add_argument("--ref_path", type=str, default=None,
                   help="JSONL human reference responses (for ROUGE; optional)")
    p.add_argument("--output_path", type=str, default="output/rouge_eval",
                   help="Directory for output files")
    p.add_argument("--quality_eval", action="store_true",
                   help="Run GPT-4o quality evaluation (requires OPENAI_API_KEY)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    evaluator = RougeEvaluator(model_name=args.model, output_path=args.output_path)

    if args.results_s1_path:
        reasoning_path = args.results_s1_path
        print(f"Using existing stage-1 outputs: {reasoning_path}")
    else:
        if not args.model_path:
            print("ERROR: --model_path is required when --results_s1_path is not provided.",
                  file=__import__("sys").stderr)
            __import__("sys").exit(1)
        if not args.test_path:
            print("ERROR: --test_path is required when --results_s1_path is not provided.",
                  file=__import__("sys").stderr)
            __import__("sys").exit(1)
        reasoning_path = os.path.join(args.output_path, "reasoning_output.jsonl")
        dump_jsonl(
            evaluator.get_reasoning(args.test_path, args.model_path, args.ckpt_dir),
            reasoning_path,
        )
        print(f"Reasoning output written to {reasoning_path}")

    if args.quality_eval:
        quality_path = os.path.join(args.output_path, "quality_eval.jsonl")
        dump_jsonl(evaluator.get_quality_eval(reasoning_path), quality_path)
        print(f"Quality eval output written to {quality_path}")

    if args.ref_path:
        rouge_path = os.path.join(args.output_path, "rouge_scores.jsonl")
        dump_jsonl(evaluator.get_rouge(reasoning_path, args.ref_path), rouge_path)
        print(f"\nROUGE scores written to {rouge_path}")
        print("\n--- Average ROUGE ---")
        RougeEvaluator.print_rouge_summary(rouge_path)
