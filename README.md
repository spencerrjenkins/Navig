# NAVIG 2026

Spencer's research fork of the original NAVIG geo-localization project. The goal is to improve the pipeline's performance and methodology toward a revised, publishable system.

**Original paper:** [Awesome-Geolocalization](https://github.com/SparrowZheyuan18/Awesome-Geolocalization)

---

## Table of contents

1. [Pipeline overview](#pipeline-overview)
2. [Repository structure](#repository-structure)
3. [Environment setup](#environment-setup)
4. [Data preparation](#data-preparation)
5. [Running the full pipeline](#running-the-full-pipeline)
6. [Single-image inference](#single-image-inference)
7. [Stage-6 swap experiment](#stage-6-swap-experiment)
8. [Comparing results](#comparing-results)
9. [Evaluation metrics](#evaluation-metrics)
10. [Customization](#customization)
11. [Model reference](#model-reference)

---

## Pipeline overview

NAVIG runs a fixed 6-stage pipeline on each input image to predict `(latitude, longitude, country, city)`.

```
Image
  │
  ▼
Stage 1 — Reasoning      SFT-tuned VLM writes a free-form geo-reasoning chain
  │                       (natural features, structures, landmarks → likely country)
  ▼
Stage 2 — Grounding      GroundingDINO detects and crops patches of
  │                       road signs, houses, and building signs
  ▼
Stage 3 — RAG            CLIP embeds each crop; FAISS retrieves the top-5
  │                       most similar entries from the guidebook knowledge base
  ▼
Stage 4 — Commenting     Base VLM describes each crop in geographic terms
  │
  ▼
Stage 5 — OCR / OSM      Base VLM reads visible text; that text is searched
  │                       on OpenStreetMap (Nominatim) for candidate locations
  ▼
Stage 6 — Guess          Base VLM synthesizes all evidence and outputs JSON:
                          {"country":"…","city":"…","latitude":…,"longitude":…}
```

Stages 1–5 are deterministic given the same model and weights. Stage 6 is the most sensitive to model quality and is the primary target for the swap experiment.

---

## Repository structure

```
Navig/
├── evaluation.py          Full 6-stage batch pipeline (main entry point)
├── inference.py           Single-image demo (stages 1–6 on one file)
├── guess_only.py          Stage-6 swap experiment runner
├── compare_results.py     Side-by-side comparison of two result files
├── merge_shards.py        Merge per-shard JSONL output and compute scores
├── llm.py                 Model wrappers (LLaVA, Qwen, CPM, Llama32Vision, InternVL2)
├── prompts.py             All LLM prompt strings
├── utils.py               Data loading, FAISS retrieval, Nominatim search, JSON parsing
├── Ablation.py            Ablation study runner
├── rouge.py               Reasoning-chain quality evaluation (requires OpenAI key)
├── configuration.py       OpenAI API key (used by rouge.py only)
│
├── script.sh              SLURM array job — full pipeline, 4 shards
├── script_guess_only.sh   SLURM array job — stage-6 swap experiment, 4 shards
│
├── environment.txt        Conda environment spec (Python 3.10, CUDA 11.8)
├── install_env.sh         Environment installation helper
│
├── dataset/               Evaluation datasets (see Data preparation)
│   ├── im2gps3k_rgb_images/
│   │   ├── meta.jsonl     One record per image: {ID, LAT, LON, [country, city]}
│   │   └── images/        {ID}.jpg files
│   └── gws15k/
├── vlms/                  Local model weight directories
│   ├── llava/             LLaVA-1.6-Vicuna-7B base weights
│   ├── NAVIG/             SFT LoRA adapters (checkpoint-534)
│   ├── qwen/              Qwen2-VL-7B-Instruct weights + adapter
│   └── cpm/               MiniCPM-V-2.6 weights + adapter
├── NaviClues/             SFT training dataset
│   ├── raw_data.jsonl
│   ├── filtered_data.jsonl
│   ├── quality_data.jsonl
│   ├── filtered_clues.jsonl
│   ├── quality_clues.jsonl
│   └── images/
├── guidebook/             CLIP knowledge base for RAG
│   ├── faiss_index.index
│   ├── image_features.npy
│   ├── image_paths.txt
│   └── text_descriptions.txt
├── GroundingDINO/         Object detection submodule
└── output/                Pipeline results (gitignored)
    └── im2gps3k_rgb_images/
        ├── shard_0_of_4/
        │   ├── results_s1.jsonl … results_s5.jsonl
        │   └── results_s6_llava.jsonl
        └── merged_results.jsonl
```

---

## Environment setup

**Requirements:** Python 3.10, CUDA 11.8, conda

```bash
# Create environment from spec
conda create --name navig --file environment.txt
conda activate navig

# Load the correct CUDA module (on the Nexus cluster)
module unload cuda
module load cuda/12.1.1
export CUDA_HOME=/opt/common/cuda/cuda-12.1.1
```

Key packages (already in `environment.txt`):
- `torch==2.2.2+cu118`, `torchvision`, `torchaudio`
- `ms-swift==2.5.0.post1` — model loading and LoRA adapters
- `transformers==4.45.2`
- `faiss-gpu` — CLIP retrieval index
- `groundingdino==0.1.0` — object detection
- `openai==1.46.0` — used only by `rouge.py`

**GroundingDINO** must be cloned into the project root and built:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install -e . && cd ..
```

---

## Data preparation

### Evaluation datasets

Two datasets are supported out of the box:

| Dataset | Path | Images | Notes |
|---|---|---|---|
| IM2GPS3K | `dataset/im2gps3k_rgb_images/` | 3,000 | Primary benchmark |
| GWS15K | `dataset/gws15k/` | 15,000 | Larger benchmark |

Each dataset directory must contain:
```
dataset/<name>/
├── meta.jsonl    # {"ID":"…","LAT":"…","LON":"…","country":"…","city":"…"}
└── images/
    └── {ID}.jpg
```

Street View images cannot be redistributed due to Google's ToS. Coordinates are provided in `NaviClues/` — use the Street View Static API to download images by lat/lon.

### NaviClues (SFT training data)

Images for the NaviClues dataset are hosted at [huggingface.co/datasets/huggingCode11/NAVICLUES](https://huggingface.co/datasets/huggingCode11/NAVICLUES). Download and place under `NaviClues/images/`.

### Model weights

SFT LoRA adapters (checkpoint-534) are at [huggingface.co/huggingCode11/NAVIG](https://huggingface.co/huggingCode11/NAVIG). Download and place in the corresponding `vlms/*/` directory.

Base model weights should be downloaded from HuggingFace to `/fs/nexus-scratch/$USER/` (scratch space) and pointed to via `--model_path`:

```bash
# LLaVA-1.6-Vicuna-7B (used in paper)
huggingface-cli download llava-hf/llava-v1.6-vicuna-7b-hf \
    --local-dir /fs/nexus-scratch/$USER/llava-v1.6-vicuna-7b-hf

# Qwen2-VL-7B-Instruct
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct \
    --local-dir /fs/nexus-scratch/$USER/Qwen2-VL-7B-Instruct

# Llama-3.2-11B-Vision-Instruct (stage-6 experiment)
huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
    --local-dir /fs/nexus-scratch/$USER/llama-3.2-11b-vision-instruct

# InternVL2-8B (alternative stage-6 experiment)
huggingface-cli download OpenGVLab/InternVL2-8B \
    --local-dir /fs/nexus-scratch/$USER/InternVL2-8B
```

---

## Running the full pipeline

### Via SLURM (recommended)

`script.sh` submits a 4-shard array job. Edit the `--model_path` and `--ckpt_dir` lines, then:

```bash
sbatch script.sh
```

Each array task writes to its own directory:
```
output/im2gps3k_rgb_images/shard_{0-3}_of_4/
├── results_s1.jsonl   (reasoning)
├── results_s2.jsonl   (grounding)
├── results_s3.jsonl   (RAG)
├── results_s4.jsonl   (commenting)
├── results_s5.jsonl   (OCR/OSM)
└── results_s6_llava.jsonl   (final predictions)
```

After all 4 shards finish, merge and score:

```bash
python merge_shards.py \
    --base_dir output/im2gps3k_rgb_images \
    --num_shards 4 \
    --results_file results_s6_llava.jsonl \
    --output merged_results_llava.jsonl
```

### Manually (single shard)

```bash
python evaluation.py \
    --model llava \
    --dataset_path dataset/im2gps3k_rgb_images \
    --reasoning_path output/im2gps3k_rgb_images/shard_0_of_1 \
    --results_file_Name results_s6_llava.jsonl \
    --model_path /fs/nexus-scratch/$USER/llava-v1.6-vicuna-7b-hf \
    --ckpt_dir vlms/NAVIG/llava1_6-vicuna-7b-instruct \
    --crop_box_treshold 0.3 \
    --crop_text_treshold 0.25 \
    --use_vllm
```

**Key arguments:**

| Argument | Description | Default |
|---|---|---|
| `--model` | Model backbone: `llava`, `qwen`, `cpm` | `qwen` |
| `--model_path` | Path to base model weights | — |
| `--ckpt_dir` | Path to LoRA SFT adapter | — |
| `--dataset_path` | Root of dataset directory | — |
| `--reasoning_path` | Output directory (created if absent) | `.` |
| `--results_file_Name` | Filename for final stage-6 output | `Final_results.jsonl` |
| `--crop_box_treshold` | GroundingDINO box score threshold (0–1) | `0.65` |
| `--crop_text_treshold` | GroundingDINO text score threshold (0–1) | `0.55` |
| `--use_vllm` | Enable vLLM acceleration for stages 4–6 | off |
| `--num_shards` | Total number of parallel shards | `1` |
| `--shard_id` | Which shard this process handles (0-indexed) | `0` |

**GroundingDINO thresholds:** lower values detect more patches (higher recall, slower). Values around 0.25–0.35 tend to work well for noisy street-view images.

---

## Single-image inference

For quick testing on a single image without SLURM:

```bash
python inference.py \
    --model qwen \
    --image_path dataset/im2gps3k_rgb_images/images/example.jpg \
    --model_path /fs/nexus-scratch/$USER/Qwen2-VL-7B-Instruct \
    --ckpt_dir vlms/NAVIG/qwen2-vl-7b-instruct \
    --crop_box_treshold 0.3 \
    --crop_text_treshold 0.25
```

`inference.py` loads both the base model and SFT adapter simultaneously (higher VRAM than `evaluation.py`, which loads them sequentially). Output is printed to stdout; the final prediction is in `inference.results['answer']`.

---

## Stage-6 swap experiment

This experiment tests whether a stronger guesser model improves final accuracy without re-running the expensive stages 1–5. It validates whether **model quality** or **pipeline architecture** is the main bottleneck.

### Step 1 — Produce stage-5 output (if not already done)

Run `sbatch script.sh` and wait for completion.

### Step 2 — Merge the stage-5 shards

```bash
python merge_shards.py \
    --base_dir output/im2gps3k_rgb_images \
    --num_shards 4 \
    --results_file results_s5.jsonl \
    --output merged_s5.jsonl
```

### Step 3 — Download the experiment model (one-time, ~22 GB)

```bash
huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
    --local-dir /fs/nexus-scratch/$USER/llama-3.2-11b-vision-instruct
```

### Step 4 — Submit the guess-only job

```bash
sbatch script_guess_only.sh
```

This runs a 4-shard array job that reads from `merged_s5.jsonl` and writes per-shard output to `output/im2gps3k_rgb_images/guess_shard_{0-3}_of_4/results_s6_llama32.jsonl`.

### Step 5 — Merge guess-only shards

```bash
python merge_shards.py \
    --base_dir output/im2gps3k_rgb_images \
    --num_shards 4 \
    --results_file results_s6_llama32.jsonl \
    --shard_prefix guess_shard \
    --output merged_results_llama32.jsonl
```

### Running manually (no SLURM)

```bash
python guess_only.py \
    --s5_path output/im2gps3k_rgb_images/merged_s5.jsonl \
    --dataset_path dataset/im2gps3k_rgb_images \
    --model llama32vision \
    --model_path /fs/nexus-scratch/$USER/llama-3.2-11b-vision-instruct \
    --output output/im2gps3k_rgb_images/results_s6_llama32.jsonl
```

`guess_only.py` also supports `--score_only` to re-score an existing output file without re-running inference:

```bash
python guess_only.py \
    --s5_path ... --dataset_path ... --model llama32vision --model_path ... \
    --output output/.../results_s6_llama32.jsonl \
    --score_only
```

**Supported `--model` values for `guess_only.py`:**

| Value | Model | VRAM (fp16) | Notes |
|---|---|---|---|
| `llava` | LLaVA-1.6-Vicuna-7B | ~14 GB | Baseline |
| `qwen` | Qwen2-VL-7B-Instruct | ~14 GB | Baseline |
| `cpm` | MiniCPM-V-2.6 | ~14 GB | Baseline |
| `llama32vision` | Llama-3.2-11B-Vision-Instruct | ~22 GB | **Primary experiment** |
| `internvl2` | InternVL2-8B | ~16 GB | Alternative (strong on OCR) |

---

## Comparing results

```bash
python compare_results.py \
    output/im2gps3k_rgb_images/merged_results_llava.jsonl \
    output/im2gps3k_rgb_images/merged_results_llama32.jsonl \
    --label-a "LLaVA-7B (baseline)" \
    --label-b "Llama-3.2-11B (experiment)"
```

Output includes:
- Per-model GeoScore, average distance, and accuracy at 5 distance thresholds
- Win / loss / tie breakdown (which model was closer per image)
- Distance delta quartiles
- Top-10 images most improved and most hurt by the experiment

**Interpreting results:**
- GeoScore improves ≥ 5% and win rate > 55%: model quality is a real bottleneck; invest in fine-tuning the stronger model on NaviClues.
- Improvement < 2%: pipeline architecture is the bottleneck; shift focus to iterative agentic tool use.
- Parse failure rate drops: the stronger model produces more reliable structured JSON output, a secondary benefit for any agentic redesign.

---

## Evaluation metrics

All scoring uses the standard GeoGuessr scale:

**GeoScore** (per image):
```
GeoScore(d) = 5000 × exp(−d / 1492.7)
```
where `d` is the Haversine distance in km between the prediction and the ground truth. Maximum 5000 points; score is 0 at ~10,000 km.

**Accuracy thresholds:**

| Level | Threshold | Interpretation |
|---|---|---|
| Street | 1 km | Nearly exact |
| City | 25 km | Correct city |
| Region | 200 km | Correct region |
| Country | 750 km | Correct country |
| Continent | 2500 km | Correct continent |

`evaluation.py` also reports **country match** and **city match** accuracy (string containment).

---

## Customization

The highest-leverage entry points for research experiments:

### Prompts

All prompts live in `prompts.py`. The most impactful to modify:

| Variable | Used in | What to change |
|---|---|---|
| `reasoning_prompt` | Stage 1 | Alter the reasoning strategy (e.g. chain-of-thought structure, clue weighting) |
| `comment_gen_template` | Stage 4 | How the VLM describes cropped patches |
| `osm_gen` | Stage 5 | What text the VLM extracts for Nominatim search |
| `base_query` + `outro_query` | Stage 6 | The guesser's core instruction and output format |

### GroundingDINO object categories

In `evaluation.py` line 69:
```python
ground = PatchImages(['road sign', 'house', 'building sign'])
```
Adding `'storefront'`, `'vehicle'`, or `'license plate'` will feed more crops to stages 3–5.

### RAG distance threshold

In `evaluation.py` and `guess_only.py`, `rag_threshold = 30` controls how close (in km) a retrieved guidebook entry must be to the crop for it to be included in the stage-6 prompt. Lower = fewer but more precise clues.

### CLIP model

`utils.py` loads `ViT-B/32` for RAG embeddings. Swapping to `ViT-L/14` improves retrieval quality at the cost of higher VRAM and a required guidebook index rebuild:
```python
# utils.py — change this line
model, preprocess = clip.load("ViT-L/14", device=device)
```
Then rebuild `guidebook/faiss_index.index` and `guidebook/image_features.npy` by re-embedding all guidebook images.

### Reasoning quality evaluation

`rouge.py` scores the quality of generated reasoning chains against human references using ROUGE and an OpenAI LLM judge. Requires an OpenAI API key in `configuration.py`:
```python
class Config:
    OPENAI_API_KEY = "sk-..."
```

---

## Model reference

| Model | `--model` value | ms-Swift type | Stages used |
|---|---|---|---|
| LLaVA-1.6-Vicuna-7B (base) | `llava` | `llava1_6-vicuna-7b-instruct` | 4, 5, 6 |
| LLaVA-1.6-Vicuna-7B + SFT | `llava` (auto) | same + LoRA | 1 |
| Qwen2-VL-7B-Instruct (base) | `qwen` | `qwen2-vl-7b-instruct` | 4, 5, 6 |
| Qwen2-VL-7B-Instruct + SFT | `qwen` (auto) | same + LoRA | 1 |
| MiniCPM-V-2.6 (base) | `cpm` | `minicpm-v-v2_6-chat` | 4, 5, 6 |
| MiniCPM-V-2.6 + SFT | `cpm` (auto) | same + LoRA | 1 |
| Llama-3.2-11B-Vision-Instruct | `llama32vision` | `llama3_2-11b-vision-instruct` | 6 (experiment) |
| InternVL2-8B | `internvl2` | `internvl2-8b` | 6 (experiment) |

The SFT adapter for stage 1 is always loaded automatically from `--ckpt_dir`; `--model` selects the base architecture for both the stage-1 adapter and the base model used in stages 4–6.

For `guess_only.py`, `--model` controls only the stage-6 guesser and no SFT adapter is loaded.
