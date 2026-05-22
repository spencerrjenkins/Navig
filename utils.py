"""Shared utilities for the NAVIG geo-localization pipeline.

Covers: JSONL I/O, JSON parsing, CLIP-based RAG retrieval, GroundingDINO
patching, Nominatim geocoding, and stage-6 query construction.
"""

import hashlib
import json
import logging
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import cast

import cv2
import faiss
import filelock
import numpy as np
import requests
import torch
from PIL import Image
from retry import retry
from torchvision.ops import box_convert

from groundingdino.util.inference import annotate, load_image, load_model, predict
from prompts import osm_gen
import prompts

logger = logging.getLogger(__name__)

# Words that appear in the OSM extraction prompt itself — filter these out so
# the model's own instructions are not accidentally sent to Nominatim.
_OSM_PROMPT_WORDS: frozenset[str] = frozenset(osm_gen.lower().split())

# Paths anchored to the project root (same directory as this file).
_ROOT = Path(__file__).parent
_GUIDEBOOK_DIR = _ROOT / "guidebook"
_GROUNDING_DINO_CONFIG = (
    _ROOT / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
)
_GROUNDING_DINO_WEIGHTS = (
    _ROOT / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth"
)
_NOMINATIM_CACHE_DIR = _ROOT / ".cache" / "nominatim"
# Nominatim policy: max 1 request/second for research/batch use.
# We enforce 1 s between requests across ALL parallel shards via a shared lock.
# (Previous value was 15 s / 4 per minute — needlessly conservative for a pipeline
# that runs infrequently; 1 s keeps us well within the published limit.)
_NOMINATIM_MIN_INTERVAL = 1.0
_NOMINATIM_LOCK_PATH = _NOMINATIM_CACHE_DIR / ".rate_limit.lock"
_NOMINATIM_TS_PATH = _NOMINATIM_CACHE_DIR / ".rate_limit.ts"
_NOMINATIM_CACHE_FILE = _NOMINATIM_CACHE_DIR / "cache.json"

# ── In-memory Nominatim cache ─────────────────────────────────────────────────
# Loaded once from a single JSON file on first access.  Individual .json shards
# written by earlier runs are merged in on startup and then ignored.
# All in-process writes go to both the in-memory dict and the single file.
_nom_cache: dict[str, list[dict]] | None = None  # keyed by sha256(candidate:top_k)


def _ensure_nom_cache() -> dict[str, list[dict]]:
    """Load the Nominatim cache into memory exactly once per process."""
    global _nom_cache
    if _nom_cache is not None:
        return _nom_cache

    _nom_cache = {}
    _NOMINATIM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load the consolidated cache file if it exists.
    if _NOMINATIM_CACHE_FILE.exists():
        try:
            _nom_cache = json.loads(_NOMINATIM_CACHE_FILE.read_text())
            logger.info("Nominatim cache loaded: %d entries", len(_nom_cache))
        except Exception as e:
            logger.warning("Nominatim cache read error: %s", e)
            _nom_cache = {}

    # Absorb any per-entry .json files written by earlier code versions.
    absorbed = 0
    for f in _NOMINATIM_CACHE_DIR.glob("*.json"):
        if f.name == "cache.json":
            continue
        stem = f.stem  # sha256 hex digest
        if stem not in _nom_cache:
            try:
                _nom_cache[stem] = json.loads(f.read_text())
                absorbed += 1
            except Exception:
                pass
    if absorbed:
        logger.info("Absorbed %d legacy per-file cache entries", absorbed)
        _flush_nom_cache()

    return _nom_cache


def _flush_nom_cache() -> None:
    """Write the in-memory cache to disk atomically."""
    tmp = _NOMINATIM_CACHE_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(_nom_cache))
        tmp.replace(_NOMINATIM_CACHE_FILE)
    except Exception as e:
        logger.warning("Nominatim cache flush error: %s", e)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

# ── CLIP module-level cache ────────────────────────────────────────────────────
# Loaded once on first call to retrieve_similar_images; reused for every
# subsequent call so the model is not re-read from disk per image.
_clip_model = None
_clip_preprocess = None
_clip_image_features: np.ndarray | None = None
_clip_index = None
_clip_text_descriptions: list[str] | None = None
_clip_image_paths: list[str] | None = None


def _load_clip_resources() -> None:
    """Initialise CLIP model and guidebook resources exactly once."""
    global _clip_model, _clip_preprocess, _clip_image_features
    global _clip_index, _clip_text_descriptions, _clip_image_paths

    if _clip_model is not None:
        return  # already loaded

    import clip  # imported here so non-RAG code paths don't require CLIP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading CLIP ViT-B/32 and guidebook index...")
    _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
    _clip_image_features = np.load(str(_GUIDEBOOK_DIR / "image_features.npy"))
    _clip_index = faiss.read_index(str(_GUIDEBOOK_DIR / "faiss_index.index"))

    with open(_GUIDEBOOK_DIR / "text_descriptions.txt", encoding="utf-8") as f:
        _clip_text_descriptions = [line.strip() for line in f]
    with open(_GUIDEBOOK_DIR / "image_paths.txt", encoding="utf-8") as f:
        _clip_image_paths = [line.strip() for line in f]

    logger.info("CLIP and guidebook loaded.")


def _cache_key(candidate: str, top_k: int) -> str:
    return hashlib.sha256(f"{candidate}:{top_k}".encode()).hexdigest()


def _cache_get(candidate: str, top_k: int) -> list[dict] | None:
    cache = _ensure_nom_cache()
    key = _cache_key(candidate, top_k)
    return cache.get(key)  # None if missing; [] if cached-empty


def _cache_set(candidate: str, top_k: int, results: list[dict]) -> None:
    cache = _ensure_nom_cache()
    cache[_cache_key(candidate, top_k)] = results
    _flush_nom_cache()


def _nominatim_get(url: str, params: dict, headers: dict) -> requests.Response:
    """Make a single Nominatim GET request respecting the cross-process rate limit.

    Nominatim policy: max 1 request/second for batch/research use.
    We enforce _NOMINATIM_MIN_INTERVAL across ALL parallel shards via a shared
    lock file + timestamp file on NFS.
    """
    _NOMINATIM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(str(_NOMINATIM_LOCK_PATH), timeout=300)
    with lock:
        now = time.time()
        try:
            last_ts = float(_NOMINATIM_TS_PATH.read_text())
        except (FileNotFoundError, ValueError):
            last_ts = 0.0
        wait = _NOMINATIM_MIN_INTERVAL - (now - last_ts)
        if wait > 0:
            logger.debug("Nominatim rate-limit: sleeping %.1f s", wait)
            time.sleep(wait)
        _NOMINATIM_TS_PATH.write_text(str(time.time()))

    return requests.get(url, params=params, headers=headers, timeout=30)


# ── JSONL I/O ─────────────────────────────────────────────────────────────────


def load_data(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dump_jsonl(objects, file_name: str) -> None:
    with open(file_name, "w", encoding="utf-8") as out_file:
        for obj in objects:
            out_file.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out_file.flush()


# ── OSM candidate parsing ──────────────────────────────────────────────────────


def _parse_osm_candidates(query) -> list[str]:
    """Extract location-string candidates from a VLM response.

    Handles: JSON arrays, markdown-fenced arrays, raw strings, and Python lists.
    Returns a deduplicated list of non-empty, non-'None' strings.
    """
    if query is None:
        return []
    if isinstance(query, list):
        candidates = query
    elif isinstance(query, str):
        raw = query.strip()
        if not raw or raw.lower() == "none":
            return []
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].startswith("```"):
                raw = "\n".join(lines[1:])
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            candidates = [raw]
        else:
            if isinstance(parsed, list):
                candidates = parsed
            elif isinstance(parsed, str):
                candidates = [parsed]
            else:
                candidates = [raw]
    else:
        return []

    normalized = []
    for item in candidates:
        if item is None:
            continue
        if not isinstance(item, str):
            item = str(item)
        item = " ".join(item.split())
        if not item or item.lower() == "none":
            continue
        # Skip strings that cannot be place names:
        #  • pure coordinate/bounding-box arrays, e.g. "[0.12, 0.01, 0.87, 0.34]"
        #  • strings with no alphabetic characters at all
        #  • very short strings (fewer than 3 alpha chars)
        #  • strings longer than 200 chars (not a place name)
        alpha_count = sum(1 for c in item if c.isalpha())
        if alpha_count < 3 or len(item) > 200:
            continue
        normalized.append(item)
    return normalized


# ── Nominatim geocoding ────────────────────────────────────────────────────────


@retry(tries=2, delay=10)
def search_place_nominatim(query: str, top_k: int = 3) -> list[dict] | None:
    """Query Nominatim for each candidate in *query* and return deduplicated results.

    The User-Agent includes a contact address per Nominatim's usage policy:
    https://operations.osmfoundation.org/policies/nominatim/
    """
    headers = {"User-Agent": "UMIACS/NAVIG_1.1.0 (contact: kinsey.long@berkeley.edu)"}
    candidates = _parse_osm_candidates(query)
    if not candidates:
        return None

    url = "https://nominatim.openstreetmap.org/search"
    all_results: list[dict] = []
    for candidate in candidates:
        if candidate.lower() in _OSM_PROMPT_WORDS:
            continue
        cached = _cache_get(candidate, top_k)
        if cached is not None:
            logger.debug("Nominatim cache hit: %r", candidate)
            all_results.extend(cached)
            continue
        params = {"q": candidate, "format": "json", "limit": top_k}
        logger.debug("Nominatim query: %r", candidate)
        response = _nominatim_get(url, params, headers)
        if response.status_code == 429:
            logger.warning("Nominatim 429 Too Many Requests — backing off 10s")
            time.sleep(10)
            raise Exception("429 Too Many Requests")
        response.raise_for_status()
        logger.debug("Nominatim response status: %s", response.status_code)
        try:
            content = response.json()
        except Exception:
            logger.warning("Failed to decode Nominatim JSON for query %r", candidate)
            continue
        if not content:
            _cache_set(candidate, top_k, [])
            continue
        candidate_results = [
            {
                "place_name": item.get("name", "N/A"),
                "location": item["display_name"],
                "lat": item["lat"],
                "lon": item["lon"],
            }
            for item in content[:top_k]
        ]
        _cache_set(candidate, top_k, candidate_results)
        all_results.extend(candidate_results)

    if not all_results:
        return None

    seen: set[tuple] = set()
    unique: list[dict] = []
    for item in all_results:
        key = (
            item.get("place_name"),
            item.get("location"),
            item.get("lat"),
            item.get("lon"),
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def search_place_with_retry(query: str, top_k: int = 3) -> list[dict] | None:
    try:
        return search_place_nominatim(query, top_k)
    except Exception as e:
        logger.warning("All Nominatim retries failed: %s", e)
        return None


# ── CLIP-based RAG retrieval ──────────────────────────────────────────────────


def retrieve_similar_images(
    input_image_path: str, k: int = 5, threshold: float = 100.0
) -> tuple[list[str], list[str], list[float]]:
    """Return (image_paths, text_clues, distances) for guidebook entries similar
    to *input_image_path*.

    The CLIP model and FAISS index are loaded once and cached for the lifetime
    of the process.  *threshold* is a FAISS L2² distance cutoff.

    Calibration: CLIP ViT-B/32 features are stored unnormalized (L2 norm ~10–12),
    so FAISS L2² values range from ~0 to ~200 between any two images.
    L2² = ||a||²+||b||² − 2⟨a,b⟩ ≈ 200·(1 − cos θ) for norm-10 vectors.
      threshold=100 → cos_sim > 0.50  (moderately similar — recommended)
      threshold= 60 → cos_sim > 0.70  (fairly similar)
      threshold= 30 → cos_sim > 0.85  (very similar — previous default, almost never met)
    """
    _load_clip_resources()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = _clip_preprocess(Image.open(input_image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = _clip_model.encode_image(img).cpu().numpy().astype("float32")

    distances, indices = _clip_index.search(features, k)

    sim_texts, sim_images, sim_dists = [], [], []
    for dist, idx in zip(distances[0], indices[0]):
        if threshold is None or dist < threshold:
            sim_texts.append(_clip_text_descriptions[idx])
            sim_images.append(_clip_image_paths[idx])
            sim_dists.append(float(dist))

    return sim_images, sim_texts, sim_dists


# ── JSON parsing ───────────────────────────────────────────────────────────────

_NAN_LITERAL_RE = re.compile(
    r"\b(NaN|Infinity|-Infinity|undefined|None|Unknown|unknown)\b"
)


def parse_json_part(guess: str) -> dict | None:
    """Attempt to parse *guess* as a JSON object, applying a cascade of fixes
    for common model output defects."""
    if guess is None:
        return None
    if guess.startswith("```json"):
        guess = guess[7:].strip()
    if guess.endswith("```"):
        guess = guess[:-3].strip()
    # Replace bare JS/Python non-JSON literals with null.
    guess = _NAN_LITERAL_RE.sub("null", guess)
    try:
        return json.loads(guess)
    except json.JSONDecodeError as e:
        logger.debug("JSON parse error: %s", e)

    # Fix 1: strip spurious trailing quote on numeric values.
    fixed = re.sub(r'(":\s*)(-?\d+\.?\d*)"', r"\1\2", guess)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Fix 2a: insert missing commas in multi-line JSON.
    fixed = re.sub(r'([0-9"])\s*\n(\s*"[^"]+"\s*:)', r"\1,\n\2", fixed)
    # Fix 2b: insert missing commas in compact single-line JSON.
    fixed = re.sub(r'([0-9"])\s*("(?:[^"\\]|\\.)*"\s*:)', r"\1,\2", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Fix 3: strip trailing commas before closing brace.
    fixed = re.sub(r",(\s*})", r"\1", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        logger.debug("JSON parse still failing after fixes: %s", e)
        return None


def parse_json(guess: str) -> dict | None:
    """Extract the last well-formed JSON object from *guess*.

    Scans brace positions right-to-left so that preamble reasoning text is
    skipped and the final JSON answer (which models emit last) is tried first.
    """
    for start in sorted([m.start() for m in re.finditer(r"\{", guess)], reverse=True):
        depth = 0
        for i, ch in enumerate(guess[start:]):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    result = parse_json_part(guess[start : start + i + 1])
                    if result is not None:
                        return result
                    break
    return None


# ── Stage-6 query construction ─────────────────────────────────────────────────


def build_guess_query(
    row: dict,
    *,
    rag_threshold: float = 100.0,
    include_reasoning: bool = True,
    include_osm: bool = True,
    include_rag: bool = True,
    include_comment: bool = True,
) -> tuple[str, dict]:
    """Construct the stage-6 prompt from a result row's accumulated evidence.

    Optional keyword arguments allow individual evidence components to be
    suppressed (used by the ablation study).

    Returns (query_string, usage_dict).
    """
    reason = row.get("image_reason", "")
    osm_results = row.get("osm", None)
    comment = row.get("comment", {})
    rag = row.get("retrieved_content", {})

    rag_formed = ""
    for rag_key, rag_items in rag.items():
        if not rag_items:
            continue
        valid = [item for item in rag_items if item["distance"] <= rag_threshold]
        if not valid:
            continue
        clues = " ".join(set(item["relevant_clue"] for item in valid))
        rag_formed += f"the relevant clues of {rag_key} in this image are: {clues}"

    comment_formed = ""
    for category, text in comment.items():
        if text:
            comment_formed += f"{category}: {text}\n"

    filtered_query = {
        k: [v for v in vals if v != "None"]
        for k, vals in row.get("genQuery", {}).items()
    }
    filtered_query = {k: v for k, v in filtered_query.items() if v}

    k_reason = 1 if include_reasoning else 0
    k_osm = (1 if osm_results else 0) if include_osm else 0
    k_rag = (1 if rag_formed else 0) if include_rag else 0
    k_comment = (1 if comment_formed else 0) if include_comment else 0

    query = prompts.base_query + prompts.intro_query
    query += prompts.reason_query_template.format(reason=reason) * k_reason
    if k_osm:
        query += prompts.osm_query_template.format(
            filtered_Query=filtered_query, osm_results=osm_results
        )
    if k_comment:
        query += prompts.comment_query_template.format(comment_formed=comment_formed)
    if k_rag:
        query += prompts.rag_query_template.format(rag_formed=rag_formed)
    if k_reason:
        query += prompts.outro_query

    usage = {
        "reasoning": k_reason,
        "osm": k_osm,
        "rag": k_rag,
        "comment": k_comment,
    }
    return query, usage


# ── GroundingDINO patch extraction ────────────────────────────────────────────


class PatchImages:
    """Detect and crop geo-relevant objects from an image using GroundingDINO."""

    # Minimum patch size: skip crops where either dimension is below this value.
    # Crops smaller than this in any axis lack enough detail for VLM reasoning.
    MIN_PATCH_PX: int = 224

    def __init__(
        self,
        geo_objects: list[str],
        grounding_dino_config_path: str | None = None,
        weights_path: str | None = None,
    ):
        self.geo_objects = geo_objects
        config = (
            Path(grounding_dino_config_path)
            if grounding_dino_config_path
            else _GROUNDING_DINO_CONFIG
        )
        weights = Path(weights_path) if weights_path else _GROUNDING_DINO_WEIGHTS
        self.model = load_model(str(config), str(weights))

    def __call__(
        self, image_path: str, box_threshold: float = 0.3, text_threshold: float = 0.25
    ) -> dict[str, list[np.ndarray]]:
        """Detect *geo_objects* in *image_path* and return cropped numpy arrays.

        Patches where either width or height is below MIN_PATCH_PX are discarded.
        """
        image_patches: dict[str, list[np.ndarray]] = {}
        for geo_object in self.geo_objects:
            image_patches[geo_object] = []
            image_source, image = load_image(image_path)
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=geo_object,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            h, w, _ = image_source.shape
            boxes_px = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes_px, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            for x1, y1, x2, y2 in xyxy:
                if abs(x2 - x1) < self.MIN_PATCH_PX or abs(y2 - y1) < self.MIN_PATCH_PX:
                    continue
                image_patches[geo_object].append(
                    image_source[int(y1) : int(y2), int(x1) : int(x2)]
                )
        return image_patches

    def save_annotation(self, image_path: str, output_path: str) -> None:
        """Detect objects and save annotated visualisations to *output_path*."""
        image_source, image = load_image(image_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        for geo_object in self.geo_objects:
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=geo_object,
                box_threshold=0.3,
                text_threshold=0.25,
            )
            annotated = annotate(
                image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
            )
            cv2.imwrite(str(output_path / f"{geo_object}.jpg"), annotated)
