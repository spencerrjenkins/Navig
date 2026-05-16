import json
import requests
import torch
import clip
from PIL import Image
import numpy as np
import re
import time
import math
from retry import retry
import faiss

import os
from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
from torchvision.ops import box_convert
from collections import defaultdict
import cv2
from prompts import osm_gen
osm_gen_list = osm_gen.lower().split()


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def dump_jsonl(objects, file_name):
    out_file = open(file_name, "w", encoding="utf-8")
    for obj in objects:
        tmp = out_file.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out_file.flush()


def _parse_osm_candidates(query):
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
        normalized.append(item)
    return normalized


@retry(tries=3, delay=10)
def search_place_nominatim(query: str, top_k=3):
    print("Querying...", query)
    headers = {"User-Agent": "UMIACS/NAVIG_1.1.0"}
    candidates = _parse_osm_candidates(query)
    if not candidates:
        return None

    url = "https://nominatim.openstreetmap.org/search"
    all_results = []
    for candidate in candidates:
        if candidate in osm_gen_list:
            continue
        params = {"q": candidate, "format": "json", "limit": top_k}
        print("Sending query...", candidate)
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 429:
            print("429 Too Many Requests — sleeping 60s before retry")
            time.sleep(60)
            raise Exception("429 Too Many Requests")
        response.raise_for_status()
        print("Response received: ", response)
        print(response.text)
        time.sleep(2)  # Respect server rate limit
        try:
            content = response.json()
        except Exception:
            print("Error decoding JSON:", response.text)
            continue
        if not content:
            continue

        content = [
            {
                "place_name": item.get("name", "N/A"),
                "location": item["display_name"],
                "lat": item["lat"],
                "lon": item["lon"],
            }
            for item in content[:top_k]
        ]
        all_results.extend(content)

    if not all_results:
        return None
    seen = set()
    unique_results = []
    for item in all_results:
        key = (
            item.get("place_name"),
            item.get("location"),
            item.get("lat"),
            item.get("lon"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_results.append(item)
    return unique_results


def search_place_with_retry(query: str, top_k=3):
    try:
        return search_place_nominatim(query, top_k)
    except Exception as e:
        print(f"All retries failed: {e}")
        return None


def retrieve_similar_images(input_image_path, k=5, threshold=30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_features_array = np.load("guidebook/image_features.npy")
    index = faiss.read_index("guidebook/faiss_index.index")

    with open("guidebook/text_descriptions.txt", "r", encoding="utf-8") as f:
        text_descriptions = [line.strip() for line in f.readlines()]

    with open("guidebook/image_paths.txt", "r", encoding="utf-8") as f:
        image_paths = [line.strip() for line in f.readlines()]
    input_image = preprocess(Image.open(input_image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        input_image_features = model.encode_image(input_image).cpu().numpy()
    input_image_features = input_image_features.astype("float32")
    distances, indices = index.search(input_image_features, k)

    filtered_similar_texts = []
    filtered_similar_images = []
    filtered_distances = []

    for i, distance in enumerate(distances[0]):
        if threshold is None or distance < threshold:
            filtered_similar_texts.append(text_descriptions[indices[0][i]])
            filtered_similar_images.append(image_paths[indices[0][i]])
            filtered_distances.append(float(distance))

    if not filtered_similar_images:
        return [], [], []
    # print(filtered_similar_images, filtered_similar_texts, filtered_distances)
    return filtered_similar_images, filtered_similar_texts, filtered_distances


def parse_guess(guess):
    pattern = r"\(([^)]+),\s*([^)]+)\)"
    match = re.search(pattern, guess)
    if match:
        return [float(match.group(1)), float(match.group(2))]
    else:
        raise ValueError("The answer is not in the correct format.")


_NAN_LITERAL_RE = re.compile(r'\b(NaN|Infinity|-Infinity|undefined|None|Unknown|unknown)\b')


def parse_json_part(guess):
    if guess is None:
        return None
    if guess.startswith("```json"):
        guess = guess[7:].strip()
    if guess.endswith("```"):
        guess = guess[:-3].strip()
    # Pre-process: replace bare JS/Python non-JSON literals with null so that
    # json.loads can proceed (models sometimes emit NaN or None as coord values).
    guess = _NAN_LITERAL_RE.sub('null', guess)
    try:
        return json.loads(guess)
    except json.JSONDecodeError as e:
        error_message = str(e)
        print(f"Error: {error_message}")
        # Fix 1: strip spurious trailing quote on numeric values
        fixed = re.sub(r'(":\s*)(-?\d+\.?\d*)"', r"\1\2", guess)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        # Fix 2a: insert missing commas in multi-line JSON (value then newline then key)
        fixed = re.sub(r'([0-9"])\s*\n(\s*"[^"]+"\s*:)', r'\1,\n\2', fixed)
        # Fix 2b: insert missing commas in compact single-line JSON (value then optional space then key)
        fixed = re.sub(r'([0-9"])\s*("(?:[^"\\]|\\.)*"\s*:)', r'\1,\2', fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        # Fix 3: strip trailing commas before closing brace (apply on top of Fix 1+2)
        fixed = re.sub(r",(\s*})", r"\1", fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            print(f"Again Error: {e}")
            return None


def parse_json(guess):
    # Scan all '{' positions right-to-left so that preamble reasoning text is
    # skipped and the final JSON answer (which models emit last) is tried first.
    # Properly tracks brace depth so nested structures are captured whole.
    for start in sorted([m.start() for m in re.finditer(r'\{', guess)], reverse=True):
        depth = 0
        for i, ch in enumerate(guess[start:]):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    result = parse_json_part(guess[start:start + i + 1])
                    if result is not None:
                        return result
                    break
    return None


def haversine_distance(guess, answer):
    """
    guess and answer are both:
    [lat, lng]
    """
    lat1, lon1 = guess
    lat2, lon2 = answer
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * (math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


class PatchImages:
    def __init__(
        self, geo_objects: list, groundingDinoConfigPath=None, WeightsPath=None
    ):
        self.geo_objects = geo_objects
        if groundingDinoConfigPath is None:
            CONFIG_PATH = os.path.join(
                os.getcwd(),
                "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            )
        else:
            CONFIG_PATH = os.path.join(os.getcwd(), groundingDinoConfigPath)

        if WeightsPath is None:
            WEIGHTS_PATH = os.path.join(
                os.getcwd(), "GroundingDINO", "weights", "groundingdino_swint_ogc.pth"
            )
        else:
            WEIGHTS_PATH = os.path.join(os.getcwd(), WeightsPath)

        self.model = load_model(CONFIG_PATH, WEIGHTS_PATH)
        self.saveImgs = defaultdict(list)

    def __call__(
        self, image_path: str, BOX_TRESHOLD: float = 0.3, TEXT_TRESHOLD: float = 0.25
    ) -> dict:
        """
        Input: Picture, list of geo-objects(string)
        Output: list of image patches
        """
        image_patches = {}
        for geo_object in self.geo_objects:
            TEXT_PROMPT = geo_object
            image_patches[geo_object] = []
            image_source, image = load_image(image_path)
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
            )
            annotated_frame = annotate(
                image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
            )
            self.saveImgs[geo_object].append(annotated_frame)
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                if abs(x2 - x1) < 224 and abs(y2 - y1) < 224:
                    continue
                image_patches[geo_object].append(
                    image_source[int(y1) : int(y2), int(x1) : int(x2)]
                )

        return image_patches

    def save_annotation(self, output_path: str):
        """
        Input: list of image patches: category->list of image patches
        Output: save the patches
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        for geo_object in self.geo_objects:
            for i, img in enumerate(self.saveImgs[geo_object]):
                cv2.imwrite(os.path.join(output_path, f"{geo_object}_{i}.jpg"), img)
