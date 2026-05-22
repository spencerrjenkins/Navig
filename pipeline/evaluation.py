#!/usr/bin/env python3
"""Full 6-stage NAVIG batch evaluation pipeline.

Stages
------
1. Reasoning   — SFT-tuned VLM writes a free-form geo-reasoning chain
2. Grounding   — GroundingDINO detects and crops road signs, houses, building signs
3. RAG         — CLIP embeds each crop; FAISS retrieves similar guidebook entries
4. Commenting  — Base VLM describes each crop in geographic terms
5. OCR / OSM   — Base VLM reads visible text; Nominatim returns candidate locations
6. Guess       — Base VLM synthesises all evidence → JSON {country, city, lat, lon}

Usage (single shard, manual)::

    python pipeline/evaluation.py \\
        --model qwen \\
        --dataset_path dataset/im2gps3k \\
        --output_path output/im2gps3k/shard_0_of_1 \\
        --results_filename results_s6_qwen.jsonl \\
        --model_path /fs/nexus-scratch/$USER/Qwen2-VL-7B-Instruct \\
        --ckpt_dir vlms/NAVIG/qwen2-vl-7b-instruct

See slurm/evaluate.sh for the recommended 4-shard SLURM array job.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone

from PIL import Image
from tqdm import tqdm

import prompts
from llm import load_model, load_sft_model
from metrics import (
    THRESHOLDS,
    THRESHOLD_NAMES,
    parse_coord,
    score_results,
    print_score_summary,
)
from utils import (
    load_data,
    dump_jsonl,
    parse_json,
    search_place_with_retry,
    PatchImages,
    retrieve_similar_images,
    _parse_osm_candidates,
    build_guess_query,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Number of 'house' crops passed to the commenting stage — houses are common
# so we cap them to keep stage-4 runtime reasonable.
_HOUSE_CROP_LIMIT = 3


class Evaluator:
    """Runs the 6-stage NAVIG pipeline on a dataset shard."""

    def __init__(
        self,
        dataset_path: str,
        model_type: str,
        output_path: str,
        results_filename: str,
        box_threshold: float,
        text_threshold: float,
        model_path: str,
        ckpt_dir: str,
        use_vllm: bool = False,
        shard_id: int = 0,
        num_shards: int = 1,
        rag_threshold: float = 100.0,
    ):
        self.dataset_path = dataset_path
        self.model_type = model_type
        self.output_path = output_path
        self.results_filename = results_filename
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model_path = model_path
        self.ckpt_dir = ckpt_dir
        self.use_vllm = use_vllm
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.rag_threshold = rag_threshold
        # Base model loaded lazily after stage 1 to avoid holding two 7B models
        # in GPU memory simultaneously.
        self.base_model = None

    # ── Stage helpers ──────────────────────────────────────────────────────────

    def _load_shard(self, jsonl_path: str) -> list[dict]:
        data = load_data(jsonl_path)
        if self.num_shards > 1:
            data = data[self.shard_id :: self.num_shards]
            logger.info(
                "Shard %d/%d: processing %d samples",
                self.shard_id,
                self.num_shards,
                len(data),
            )
        return data

    def _image_path(self, row: dict) -> str:
        return os.path.join(self.dataset_path, "images", row["ID"] + ".jpg")

    # ── Stage 1: Reasoning ────────────────────────────────────────────────────

    def get_reasoning(self):
        """Yield rows with 'image_reason' field added."""
        logger.info("Stage 1 — Reasoning")
        # Note: the SFT model always uses Swift regardless of --use_vllm because
        # vLLM's LoRA + multimodal support for LLaVA-NeXT is not stable.
        reasoning_model = load_sft_model(
            self.model_type, self.model_path, self.ckpt_dir
        )
        data = self._load_shard(os.path.join(self.dataset_path, "meta.jsonl"))
        requests = [(prompts.reasoning_prompt, self._image_path(row)) for row in data]
        responses = reasoning_model.batch_inference(requests)
        for row, response in zip(data, responses):
            row["image_reason"] = response
            yield row

    # ── Stage 2: Grounding ────────────────────────────────────────────────────

    def get_grounding(self):
        """Yield rows with 'crop' field: {category: [saved_patch_paths]}."""
        logger.info("Stage 2 — Grounding")
        data = load_data(os.path.join(self.output_path, "results_s1.jsonl"))
        ground = PatchImages(["road sign", "house", "building sign"])
        patch_dir = os.path.join(self.dataset_path, "patchesV2")
        os.makedirs(patch_dir, exist_ok=True)

        for row in tqdm(data, desc="Stage 2"):
            image_path = self._image_path(row)
            patch_result = ground(
                image_path,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            crops: dict[str, list[str]] = {}
            for category, cropped_list in patch_result.items():
                crops[category] = []
                for i, cropped_arr in enumerate(cropped_list):
                    try:
                        img = Image.fromarray(cropped_arr)
                        save_path = os.path.join(
                            patch_dir, f"{row['ID']}_{category}_{i}.jpg"
                        )
                        img.save(save_path)
                        crops[category].append(save_path)
                    except Exception as e:
                        logger.warning(
                            "Failed to save patch %s/%s/%d: %s",
                            row["ID"],
                            category,
                            i,
                            e,
                        )
            row["crop"] = crops
            yield row

    # ── Stage 3: RAG ──────────────────────────────────────────────────────────

    def get_rag(self):
        """Yield rows with 'retrieved_content' field from guidebook FAISS search."""
        logger.info("Stage 3 — RAG retrieval")
        data = load_data(os.path.join(self.output_path, "results_s2.jsonl"))
        for row in tqdm(data, desc="Stage 3"):
            retrieved: dict[str, list[dict]] = {}
            for category, images in row["crop"].items():
                if images:
                    sim_images, sim_texts, distances = retrieve_similar_images(
                        images[0], threshold=self.rag_threshold
                    )
                    retrieved[category] = [
                        {"similar_image": img, "relevant_clue": txt, "distance": d}
                        for img, txt, d in zip(sim_images, sim_texts, distances)
                    ]
                else:
                    retrieved[category] = []
            row["retrieved_content"] = retrieved
            yield row

    # ── Stage 4: Commenting ───────────────────────────────────────────────────

    def get_comment(self):
        """Yield rows with 'comment' field: {category: tab-separated VLM responses}."""
        logger.info("Stage 4 — Commenting")
        data = load_data(os.path.join(self.output_path, "results_s3.jsonl"))

        requests: list[tuple[str, str]] = []
        keys: list[tuple[int, str]] = []
        for row_idx, row in enumerate(data):
            for category, images in row["crop"].items():
                k = _HOUSE_CROP_LIMIT if category == "house" else len(images)
                query = prompts.comment_gen_template.format(item=category)
                for image in images[:k]:
                    requests.append((query, image))
                    keys.append((row_idx, category))

        responses = self.base_model.batch_inference(requests)

        grouped: dict[tuple[int, str], list[str]] = defaultdict(list)
        for (row_idx, category), response in zip(keys, responses):
            grouped[(row_idx, category)].append(response)

        for row_idx, row in enumerate(data):
            row["comment"] = {
                category: "\t".join(grouped.get((row_idx, category), []))
                for category in row["crop"]
            }
            yield row

    # ── Stage 5: OCR / OSM ────────────────────────────────────────────────────

    def get_osm(self):
        """Yield rows with 'genQuery' (VLM OCR output) and 'osm' (Nominatim results)."""
        logger.info("Stage 5 — OCR / OSM")
        data = load_data(os.path.join(self.output_path, "results_s4.jsonl"))

        requests: list[tuple[str, str]] = []
        keys: list[tuple[int, str, int]] = []
        for row_idx, row in enumerate(data):
            for category, images in row["crop"].items():
                for image_idx, image in enumerate(images):
                    requests.append((prompts.osm_gen, image))
                    keys.append((row_idx, category, image_idx))

        ocr_responses = self.base_model.batch_inference(requests)

        grouped: dict[tuple[int, str], list[tuple[int, str]]] = defaultdict(list)
        for (row_idx, category, image_idx), response in zip(keys, ocr_responses):
            grouped[(row_idx, category)].append((image_idx, response))
        for v in grouped.values():
            v.sort()

        for row_idx, row in enumerate(tqdm(data, desc="Stage 5 (OSM)")):
            row["genQuery"] = {}
            row["osm"] = None
            for category in row["crop"]:
                ocr_list = grouped.get((row_idx, category), [])
                row["genQuery"][category] = [resp for _, resp in ocr_list]
                for _, ocr_output in ocr_list:
                    candidates = _parse_osm_candidates(ocr_output)
                    if not candidates:
                        continue
                    osm_result = search_place_with_retry(ocr_output, top_k=3)
                    if osm_result is not None:
                        row["osm"] = (
                            osm_result
                            if row["osm"] is None
                            else row["osm"] + osm_result
                        )
            yield row

    # ── Stage 6: Guess ────────────────────────────────────────────────────────

    def guess_coordinates(self, only_ids: set | None = None):
        """Yield rows with 'answer' (parsed JSON prediction) and 'usage' fields."""
        logger.info("Stage 6 — Coordinate guessing")
        data = load_data(os.path.join(self.output_path, "results_s5.jsonl"))
        if only_ids is not None:
            data = [row for row in data if row["ID"] in only_ids]

        queries_usages = [
            build_guess_query(row, rag_threshold=self.rag_threshold) for row in data
        ]
        requests = [
            (query, self._image_path(row))
            for (query, _), row in zip(queries_usages, data)
        ]
        responses = self.base_model.batch_inference(requests)

        for row, (_, usage), raw in zip(data, queries_usages, responses):
            row["answer"] = parse_json(raw)
            row["usage"] = usage
            yield row

    # ── Model lifecycle ────────────────────────────────────────────────────────

    def _load_base_model(self) -> None:
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        self.base_model = load_model(
            self.model_type, self.model_path, use_vllm=self.use_vllm
        )

    # ── Top-level runners ──────────────────────────────────────────────────────

    def _write_metadata(self) -> None:
        """Write a metadata.json alongside the output directory for provenance."""
        try:
            git_hash = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:
            git_hash = "unknown"

        meta = {
            "model": self.model_type,
            "model_path": self.model_path,
            "ckpt_dir": self.ckpt_dir,
            "dataset": self.dataset_path,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "shard": f"{self.shard_id}_of_{self.num_shards}",
            "use_vllm": self.use_vllm,
            "git_commit": git_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = os.path.join(self.output_path, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Metadata written to %s", meta_path)

    def forward(self, start_stage: int = 1) -> None:
        """Run stages start_stage–5, writing intermediate results to disk."""
        os.makedirs(self.output_path, exist_ok=True)
        self._write_metadata()

        if start_stage <= 1:
            dump_jsonl(
                self.get_reasoning(), os.path.join(self.output_path, "results_s1.jsonl")
            )

        self._load_base_model()

        if start_stage <= 2:
            dump_jsonl(
                self.get_grounding(), os.path.join(self.output_path, "results_s2.jsonl")
            )
        if start_stage <= 3:
            dump_jsonl(self.get_rag(), os.path.join(self.output_path, "results_s3.jsonl"))
        if start_stage <= 4:
            dump_jsonl(
                self.get_comment(), os.path.join(self.output_path, "results_s4.jsonl")
            )
        if start_stage <= 5:
            dump_jsonl(self.get_osm(), os.path.join(self.output_path, "results_s5.jsonl"))

    def guess_forward(self) -> None:
        """Run stage 6, writing final predictions to disk."""
        out_file = os.path.join(self.output_path, self.results_filename)
        dump_jsonl(self.guess_coordinates(), out_file)

    def retry_guess_forward(self) -> None:
        """Re-run stage 6 only for rows where answer=None (JSON parse failed).

        Rows where the model returned {"latitude": "Unknown"} are valid outputs
        and are NOT retried — the model expressed genuine uncertainty.
        Patches the results file in-place.
        """
        existing_path = os.path.join(self.output_path, self.results_filename)
        if not os.path.exists(existing_path):
            logger.info(
                "No existing results at %s; running full stage 6.", existing_path
            )
            self.guess_forward()
            return
        existing = load_data(existing_path)
        failed_ids = {row["ID"] for row in existing if row.get("answer") is None}
        if not failed_ids:
            logger.info("No failed rows found in %s. Nothing to retry.", existing_path)
            return
        logger.info(
            "Retrying %d failed rows out of %d total...", len(failed_ids), len(existing)
        )
        retry_map = {
            row["ID"]: row for row in self.guess_coordinates(only_ids=failed_ids)
        }
        merged = [retry_map.get(row["ID"], row) for row in existing]
        dump_jsonl(merged, existing_path)
        still_failed = sum(1 for row in merged if row.get("answer") is None)
        logger.info(
            "Updated %d rows. Still failing after retry: %d.",
            len(retry_map),
            still_failed,
        )

    # ── Scoring ────────────────────────────────────────────────────────────────

    def calculate_score(self) -> None:
        data = load_data(os.path.join(self.output_path, self.results_filename))
        stats = score_results(data)
        print_score_summary(stats, label=self.results_filename)

    def calculate_score_cc(self) -> None:
        data = load_data(os.path.join(self.output_path, self.results_filename))
        country_correct = city_correct = 0
        total = 0
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
            print(
                f"Country match: {country_correct/total:.4f} ({country_correct}/{total})"
            )
            print(f"City match   : {city_correct/total:.4f} ({city_correct}/{total})")


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Root of dataset directory (must contain meta.jsonl and images/)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=["qwen", "llava", "cpm", "deepseek", "falcon", "llama32vision"],
    )
    p.add_argument(
        "--output_path",
        type=str,
        default=".",
        help="Output directory (created if absent)",
    )
    p.add_argument(
        "--results_filename",
        type=str,
        default="results_s6.jsonl",
        help="Filename for the stage-6 output file",
    )
    p.add_argument(
        "--box_threshold",
        type=float,
        default=0.3,
        help="GroundingDINO box confidence threshold (0–1)",
    )
    p.add_argument(
        "--text_threshold",
        type=float,
        default=0.25,
        help="GroundingDINO text confidence threshold (0–1)",
    )
    p.add_argument(
        "--model_path", type=str, required=True, help="Local path to base model weights"
    )
    p.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Path to LoRA SFT adapter for stage 1",
    )
    p.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM acceleration for stages 4–6 (LLaVA only)",
    )
    p.add_argument(
        "--num_shards", type=int, default=1, help="Total number of parallel shards"
    )
    p.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Which shard this process handles (0-indexed)",
    )
    p.add_argument(
        "--start_stage",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Resume pipeline from this stage (prior stages must already be complete).",
    )
    p.add_argument(
        "--stage6_only",
        action="store_true",
        help="Skip stages 1–5; run stage 6 on existing results_s5.jsonl",
    )
    p.add_argument(
        "--retry_failed",
        action="store_true",
        help="Re-run stage 6 only for rows with answer=None (implies --stage6_only)",
    )
    p.add_argument(
        "--rag_threshold",
        type=float,
        default=30.0,
        help="Max distance (km) for a RAG hit to be included in the stage-6 prompt",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Args: %s", args)

    evaluator = Evaluator(
        dataset_path=args.dataset_path,
        model_type=args.model,
        output_path=args.output_path,
        results_filename=args.results_filename,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        model_path=args.model_path,
        ckpt_dir=args.ckpt_dir or "",
        use_vllm=args.use_vllm,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        rag_threshold=args.rag_threshold,
    )

    if args.stage6_only or args.retry_failed:
        evaluator._load_base_model()
    else:
        evaluator.forward(start_stage=args.start_stage)
        evaluator._load_base_model()

    if args.retry_failed:
        evaluator.retry_guess_forward()
    else:
        evaluator.guess_forward()

    evaluator.calculate_score()
    evaluator.calculate_score_cc()
