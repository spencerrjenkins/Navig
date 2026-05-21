#!/usr/bin/env python3
"""Single-image demo — runs all 6 NAVIG stages on one image.

Both the SFT reasoning model and the base model are loaded simultaneously,
so VRAM usage is higher than evaluation.py (which loads them sequentially).
Use evaluation.py for batch evaluation; use this for debugging a single image.

Usage::

    python pipeline/inference.py \\
        --model qwen \\
        --image_path dataset/im2gps3k_rgb_images/images/example.jpg \\
        --model_path /fs/nexus-scratch/$USER/Qwen2-VL-7B-Instruct \\
        --ckpt_dir vlms/NAVIG/qwen2-vl-7b-instruct \\
        --box_threshold 0.3 \\
        --text_threshold 0.25

The final prediction is printed to stdout and also available as
``inference.results['answer']``.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import os

from PIL import Image
from tqdm import tqdm

import prompts
from llm import load_model, load_sft_model
from metrics import parse_coord, haversine_distance
from utils import (
    load_data, parse_json,
    search_place_with_retry, PatchImages,
    retrieve_similar_images, _parse_osm_candidates,
    build_guess_query,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_HOUSE_CROP_LIMIT = 3


class Inference:
    """Runs the full 6-stage pipeline on a single image."""

    def __init__(
        self,
        model_type: str,
        box_threshold: float,
        text_threshold: float,
        model_path: str,
        ckpt_dir: str,
    ):
        self.model_type = model_type
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.results: dict = {}

        self.base_model = load_model(model_type, model_path)
        self.reasoning_model = load_sft_model(model_type, model_path, ckpt_dir)

    def get_reasoning(self, image_path: str) -> None:
        logger.info("Stage 1 — Reasoning")
        self.results = {}
        response = self.reasoning_model.base_inference(
            prompts.reasoning_prompt, image_path
        )
        logger.info("Reasoning: %s", response)
        self.results["image_reason"] = response

    def get_grounding(self, image_path: str) -> None:
        logger.info("Stage 2 — Grounding")
        ground = PatchImages(["road sign", "house", "building sign"])
        patch_dir = "output/patches"
        os.makedirs(patch_dir, exist_ok=True)

        filename = Path(image_path).stem
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
                    save_path = os.path.join(patch_dir, f"{filename}_{category}_{i}.jpg")
                    img.save(save_path)
                    crops[category].append(save_path)
                except Exception as e:
                    logger.warning("Failed to save patch %s/%s/%d: %s",
                                   filename, category, i, e)
        self.results["crop"] = crops
        logger.info("Grounding patches saved to %s", patch_dir)

    def get_rag(self) -> None:
        logger.info("Stage 3 — RAG retrieval")
        retrieved: dict[str, list[dict]] = {}
        for category, images in self.results["crop"].items():
            if images:
                sim_images, sim_texts, distances = retrieve_similar_images(
                    images[0], threshold=40
                )
                retrieved[category] = [
                    {"similar_image": img, "relevant_clue": txt, "distance": d}
                    for img, txt, d in zip(sim_images, sim_texts, distances)
                ]
            else:
                retrieved[category] = []
        self.results["retrieved_content"] = retrieved
        logger.info("RAG results: %s", retrieved)

    def get_comment(self) -> None:
        logger.info("Stage 4 — Commenting")
        commented: dict[str, str] = {}
        for category, images in self.results["crop"].items():
            k = _HOUSE_CROP_LIMIT if category == "house" else len(images)
            query = prompts.comment_gen_template.format(item=category)
            comments = []
            for image in images[:k]:
                comments.append(self.base_model.base_inference(query, image))
            commented[category] = "\t".join(comments)
        self.results["comment"] = commented
        logger.info("Comments: %s", commented)

    def get_osm(self) -> None:
        logger.info("Stage 5 — OCR / OSM")
        self.results["genQuery"] = {}
        self.results["osm"] = None
        for category, images in self.results["crop"].items():
            self.results["genQuery"][category] = []
            for image in images:
                ocr_output = self.base_model.base_inference(prompts.osm_gen, image)
                self.results["genQuery"][category].append(ocr_output)
                candidates = _parse_osm_candidates(ocr_output)
                if not candidates:
                    continue
                osm_result = search_place_with_retry(ocr_output, top_k=3)
                if osm_result is not None:
                    self.results["osm"] = (
                        osm_result
                        if self.results["osm"] is None
                        else self.results["osm"] + osm_result
                    )
        logger.info("OSM results: %s", self.results["osm"])

    def guess_coordinates(self, image_path: str) -> None:
        logger.info("Stage 6 — Coordinate guessing")
        query, usage = build_guess_query(self.results)
        raw = self.base_model.base_inference(query, image_path)
        answer = parse_json(raw)
        self.results["usage"] = usage
        self.results["answer"] = answer
        logger.info("Prediction: %s", answer)

    def forward(self, image_path: str) -> None:
        self.get_reasoning(image_path)
        self.get_grounding(image_path)
        self.get_rag()
        self.get_comment()
        self.get_osm()
        self.guess_coordinates(image_path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image_path", type=str, required=True,
                   help="Path to the input image")
    p.add_argument("--model", type=str, default="qwen",
                   choices=["qwen", "llava", "cpm"])
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="Path to LoRA SFT adapter for stage 1")
    p.add_argument("--box_threshold", type=float, default=0.3)
    p.add_argument("--text_threshold", type=float, default=0.25)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer = Inference(
        model_type=args.model,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        model_path=args.model_path,
        ckpt_dir=args.ckpt_dir,
    )
    infer.forward(args.image_path)
    print("\nFinal prediction:", infer.results.get("answer"))
