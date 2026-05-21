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
8. [Analyzing and comparing results](#analyzing-and-comparing-results)
9. [Generating figures](#generating-figures)
10. [Evaluation metrics](#evaluation-metrics)
11. [Customization](#customization)
12. [Model reference](#model-reference)

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
├── pipeline/                  Pipeline entry points
│   ├── evaluation.py          Full 6-stage batch pipeline (main entry point)
│   ├── inference.py           Single-image demo (stages 1–6 on one file)
│   ├── guess_only.py          Stage-6 swap experiment runner
│   └── ablation.py            Ablation study runner
│
├── analysis/                  Post-hoc analysis and comparison tools
│   ├── analyze_results.py     Detailed breakdown of a single result file
│   ├── compare_results.py     Side-by-side comparison of two result files
│   ├── compare_all.py         Multi-model comparison table
│   ├── merge_shards.py        Merge per-shard JSONL output and compute scores
│   ├── merge_all_shards.py    Auto-merge all shard groups in a directory
│   └── rouge.py               Reasoning-chain quality evaluation (ROUGE + GPT-4o)
│
├── figures/                   Publication figure generation
│   ├── plot_results.py        Nine PDF figures for model_justification.tex
│   └── make_dataset_figures.py  Dataset illustration figures
│
├── slurm/                     SLURM job scripts for the NEXUS cluster
│   ├── evaluate.sh            Array job — full pipeline, 4 shards
│   └── guess_only.sh          Array job — stage-6 swap experiment, 4 shards
│
├── llm.py                     Model wrappers (LLaVA, Qwen, CPM, Llama32Vision, InternVL2)
├── prompts.py                 All LLM prompt strings
├── utils.py                   Data loading, FAISS retrieval, Nominatim search, JSON parsing
├── metrics.py                 GeoScore, Haversine distance, threshold accuracy helpers
├── configuration.py           OpenAI API key loader (reads OPENAI_API_KEY env var)
│
├── environment.yml            Conda environment specification
├── install_env.sh             Environment installation helper (alternative to environment.yml)
│
├── dataset/                   Evaluation datasets (see Data preparation)
│   ├── im2gps3k_rgb_images/
│   │   ├── meta.jsonl         One record per image: {ID, LAT, LON, country, city}
│   │   └── images/            {ID}.jpg files
│   └── gws15k/
├── vlms/                      Local model weight directories
│   ├── llava/                 LLaVA-1.6-Vicuna-7B base weights
│   ├── NAVIG/                 SFT LoRA adapters (checkpoint-534)
│   ├── qwen/                  Qwen2-VL-7B-Instruct weights + adapter
│   └── cpm/                   MiniCPM-V-2.6 weights + adapter
├── NaviClues/                 SFT training dataset
│   ├── raw_data.jsonl
│   ├── filtered_data.jsonl
│   ├── quality_data.jsonl
│   └── images/
├── guidebook/                 CLIP knowledge base for RAG
│   ├── faiss_index.index
│   ├── image_features.npy
│   ├── image_paths.txt
│   └── text_descriptions.txt
├── GroundingDINO/             Object detection submodule
└── output/                    Pipeline results (gitignored)
    └── im2gps3k_rgb_images/
        ├── shard_0_of_4/
        │   ├── results_s1.jsonl … results_s5.jsonl
        │   └── results_s6_llava.jsonl
        └── merged_results.jsonl
```

---

## Environment setup

**Requirements:** Python 3.10, CUDA 12.1, conda, GCC 11

### Option A — conda environment file (recommended)

```bash
conda env create -f environment.yml
conda activate navig
```

Then build GroundingDINO from the local source (requires GCC 11 and CUDA 12.1):

```bash
module load gcc/11.2.0 cuda/12.1.1
export CUDA_HOME=/opt/common/cuda/cuda-12.1.1
export TORCH_CUDA_ARCH_LIST="8.6"   # RTX A6000 — adjust for other GPUs
pip install --no-build-isolation -e GroundingDINO/
```

### Option B — manual install script

```bash
conda create -n navig python=3.10 -y
conda activate navig
bash install_env.sh
```

`install_env.sh` installs everything in order:
- `torch==2.4.0+cu121` + matching `torchvision` and `torchaudio`
- `ms-swift==2.5.0.post1` — model loading and LoRA adapters
- `transformers==4.45.2`, `peft`, `accelerate`
- OpenAI CLIP (from GitHub source)
- `faiss-gpu` (falls back to `faiss-cpu` if the conda channel is unavailable)
- `vllm==0.5.5` — batch inference acceleration for stages 4–6
- GroundingDINO built from the local `GroundingDINO/` source

The script targets the **RTX A6000 (SM 8.6)** GPU architecture. If you are on a different GPU, edit the `TORCH_CUDA_ARCH_LIST` line in `install_env.sh` before running.

### Session setup

Load the correct modules before each session (or add to `~/.bashrc`):

```bash
module unload cuda
module load cuda/12.1.1
export CUDA_HOME=/opt/common/cuda/cuda-12.1.1
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

`slurm/evaluate.sh` submits a 4-shard array job. Edit the `--model_path` and `--ckpt_dir` lines, then:

```bash
sbatch slurm/evaluate.sh
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
python analysis/merge_shards.py \
    --base_dir output/im2gps3k_rgb_images \
    --num_shards 4 \
    --results_file results_s6_llava.jsonl \
    --output merged_results_llava.jsonl
```

### Manually (single shard)

```bash
python pipeline/evaluation.py \
    --model llava \
    --dataset_path dataset/im2gps3k_rgb_images \
    --output_path output/im2gps3k_rgb_images/shard_0_of_1 \
    --results_filename results_s6_llava.jsonl \
    --model_path /fs/nexus-scratch/$USER/llava-v1.6-vicuna-7b-hf \
    --ckpt_dir vlms/NAVIG/llava1_6-vicuna-7b-instruct \
    --box_threshold 0.3 \
    --text_threshold 0.25 \
    --use_vllm
```

**Key arguments:**

| Argument | Description | Default |
|---|---|---|
| `--model` | Model backbone: `llava`, `qwen`, `cpm` | `qwen` |
| `--model_path` | Path to base model weights | — |
| `--ckpt_dir` | Path to LoRA SFT adapter | — |
| `--dataset_path` | Root of dataset directory | — |
| `--output_path` | Output directory (created if absent) | `.` |
| `--results_filename` | Filename for final stage-6 output | `Final_results.jsonl` |
| `--box_threshold` | GroundingDINO box score threshold (0–1) | `0.65` |
| `--text_threshold` | GroundingDINO text score threshold (0–1) | `0.55` |
| `--use_vllm` | Enable vLLM acceleration for stages 4–6 | off |
| `--num_shards` | Total number of parallel shards | `1` |
| `--shard_id` | Which shard this process handles (0-indexed) | `0` |

**GroundingDINO thresholds:** lower values detect more patches (higher recall, slower). Values around 0.25–0.35 tend to work well for noisy street-view images.

---

## Single-image inference

For quick testing on a single image without SLURM:

```bash
python pipeline/inference.py \
    --model qwen \
    --image_path dataset/im2gps3k_rgb_images/images/example.jpg \
    --model_path /fs/nexus-scratch/$USER/Qwen2-VL-7B-Instruct \
    --ckpt_dir vlms/NAVIG/qwen2-vl-7b-instruct \
    --box_threshold 0.3 \
    --text_threshold 0.25
```

`pipeline/inference.py` loads both the base model and SFT adapter simultaneously (higher VRAM than `pipeline/evaluation.py`, which loads them sequentially). Output is printed to stdout; the final prediction is in `inference.results['answer']`.

---

## Stage-6 swap experiment

This experiment tests whether a stronger guesser model improves final accuracy without re-running the expensive stages 1–5. It validates whether **model quality** or **pipeline architecture** is the main bottleneck.

### Step 1 — Produce stage-5 output (if not already done)

Run `sbatch slurm/evaluate.sh` and wait for completion.

### Step 2 — Merge the stage-5 shards

```bash
python analysis/merge_shards.py \
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
sbatch slurm/guess_only.sh
```

This runs a 4-shard array job that reads from `merged_s5.jsonl` and writes per-shard output to `output/im2gps3k_rgb_images/guess_shard_{0-3}_of_4/results_s6_llama32.jsonl`.

### Step 5 — Merge guess-only shards

```bash
python analysis/merge_shards.py \
    --base_dir output/im2gps3k_rgb_images \
    --num_shards 4 \
    --results_file results_s6_llama32.jsonl \
    --shard_prefix guess_shard \
    --output merged_results_llama32.jsonl
```

### Running manually (no SLURM)

```bash
python pipeline/guess_only.py \
    --s5_path output/im2gps3k_rgb_images/merged_s5.jsonl \
    --dataset_path dataset/im2gps3k_rgb_images \
    --model llama32vision \
    --model_path /fs/nexus-scratch/$USER/llama-3.2-11b-vision-instruct \
    --output output/im2gps3k_rgb_images/results_s6_llama32.jsonl
```

`pipeline/guess_only.py` also supports `--score_only` to re-score an existing output file without re-running inference:

```bash
python pipeline/guess_only.py \
    --s5_path ... --dataset_path ... --model llama32vision --model_path ... \
    --output output/.../results_s6_llama32.jsonl \
    --score_only
```

**Supported `--model` values for `pipeline/guess_only.py`:**

| Value | Model | VRAM (fp16) | Notes |
|---|---|---|---|
| `llava` | LLaVA-1.6-Vicuna-7B | ~14 GB | Baseline |
| `qwen` | Qwen2-VL-7B-Instruct | ~14 GB | Baseline |
| `cpm` | MiniCPM-V-2.6 | ~14 GB | Baseline |
| `llama32vision` | Llama-3.2-11B-Vision-Instruct | ~22 GB | **Primary experiment** |
| `internvl2` | InternVL2-8B | ~16 GB | Alternative (strong on OCR) |

---

## Analyzing and comparing results

### Detailed analysis of one result file

```bash
python analysis/analyze_results.py output/im2gps3k_rgb_images/merged_results_llava.jsonl
```

Prints per-stage score breakdown, accuracy at all thresholds, country/city match rates, and failure-mode statistics.

### Side-by-side comparison of two result files

```bash
python analysis/compare_results.py \
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

### Multi-model comparison table

```bash
python analysis/compare_all.py \
    output/im2gps3k_rgb_images/merged_results_llava.jsonl \
    output/im2gps3k_rgb_images/merged_results_qwen.jsonl \
    output/im2gps3k_rgb_images/merged_results_llama32.jsonl \
    --labels "LLaVA-1.6" "Qwen2-VL" "Llama-3.2"
```

### Merging shards from multiple model runs at once

```bash
python analysis/merge_all_shards.py output/im2gps3k_rgb_images/
```

Automatically detects all `*_N_of_M` directories and merges each group into a `*_merged/` directory.

### Reasoning chain quality evaluation

`analysis/rouge.py` scores the quality of generated reasoning chains against human references using ROUGE and an optional GPT-4o judge:

```bash
# Set the API key before running quality eval
export OPENAI_API_KEY="sk-..."

python analysis/rouge.py \
    --model qwen_sft \
    --test_path dataset/im2gps3k_rgb_images/meta.jsonl \
    --ref_path path/to/reference_responses.jsonl \
    --output_path output/rouge_eval \
    --quality_eval
```

---

## Generating figures

Publication-quality figures are generated by scripts in `figures/`. All scripts are run from the project root directory.

### Model performance figures (for model_justification.tex)

```bash
python figures/plot_results.py [--output_dir figures/]
```

Produces nine PDF figures:

| File | Content |
|---|---|
| `fig1_geoscore.pdf` | Overall GeoScore (all vs. excluding failures) |
| `fig2_thresholds.pdf` | Accuracy at five distance thresholds |
| `fig3_distribution.pdf` | Distance-percentile violin + IQR chart |
| `fig4_evidence.pdf` | Evidence-component GeoScore deltas |
| `fig5_geographic.pdf` | Geographic GeoScore heatmap |
| `fig6_cdf.pdf` | Cumulative distance CDF (log x-axis) |
| `fig7_difficulty.pdf` | Accuracy breakdown by image difficulty tercile |
| `fig8_agreement.pdf` | Pairwise joint accuracy heatmap |
| `fig9_failure_modes.pdf` | Prediction outcome decomposition (stacked bar) |

### Dataset illustration figures

```bash
python figures/make_dataset_figures.py [--output_dir figures/] \
    [--img_dir dataset/im2gps3k_rgb_images/images] \
    [--meta_path dataset/im2gps3k_rgb_images/meta.jsonl]
```

Produces `fig_dataset_examples.pdf` (best/worst prediction grid) and `fig_dataset_geo.pdf` (geographic distribution bar chart).

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

`pipeline/evaluation.py` also reports **country match** and **city match** accuracy (string containment). All metric logic lives in `metrics.py` and is shared across pipeline and analysis scripts.

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

In `pipeline/evaluation.py`:
```python
ground = PatchImages(['road sign', 'house', 'building sign'])
```
Adding `'storefront'`, `'vehicle'`, or `'license plate'` will feed more crops to stages 3–5.

### RAG distance threshold

In `pipeline/evaluation.py` and `pipeline/guess_only.py`, `rag_threshold=30` controls how close (in km) a retrieved guidebook entry must be to the crop for it to be included in the stage-6 prompt. Lower = fewer but more precise clues. The `build_guess_query()` function in `utils.py` also accepts `include_reasoning`, `include_osm`, `include_rag`, and `include_comment` keyword arguments for ablation studies — see `pipeline/ablation.py` for usage.

### CLIP model

`utils.py` loads `ViT-B/32` for RAG embeddings. Swapping to `ViT-L/14` improves retrieval quality at the cost of higher VRAM and a required guidebook index rebuild:
```python
# utils.py — change this line in _load_clip_resources()
model, preprocess = clip.load("ViT-L/14", device=device)
```
Then rebuild `guidebook/faiss_index.index` and `guidebook/image_features.npy` by re-embedding all guidebook images.

### OpenAI API key (for rouge.py quality eval)

Set the key via environment variable before running `analysis/rouge.py`:
```bash
export OPENAI_API_KEY="sk-..."
```

`configuration.py` reads this variable at import time. Do not hard-code the key in source files.

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

For `pipeline/guess_only.py`, `--model` controls only the stage-6 guesser and no SFT adapter is loaded.
