#!/bin/bash
#SBATCH --job-name=comparison
#SBATCH --output=%j.comparison.out
#SBATCH --error=%j.comparison.err
#SBATCH --time=12:00:00
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4g
# run_comparison.sh — Queue all stage-6 model comparison jobs, then merge and analyze.
#
# WHAT THIS DOES
# --------------
# Runs the full NAVIG pipeline (all stages 1-6) independently for each
# comparison model, so each model's evidence chain is entirely its own.
#
# Models:
#   llama32vision  Llama-3.2-11B-Vision — zero-shot, no NAVIG SFT
#   deepseek       DeepSeek-VL-7B-Chat  — zero-shot, no NAVIG SFT
#   falcon         Falcon-11B-VLM       — zero-shot, no NAVIG SFT
#   cpm            MiniCPM-V-2.6 — Stage 1 uses NAVIG LoRA adapter (CPM_sft)
#   qwen           Qwen2.5-VL-7B — Stage 1 uses NAVIG LoRA adapter (Qwen_sft)
#
# SKIP LOGIC
# ----------
# A model is skipped if all its shard folders already exist:
#   output/im2gps3k_rgb_images/cmp_shard_<model>_{0..N-1}_of_N/
# Delete those folders to force a re-run.
#
# USAGE
# -----
#   sbatch run_comparison.sh [--only <model>] [--rerun_stage6 <model>]
#
#   --only <model> [<model> ...]
#                          Run and analyze only the named model(s).  Accepts multiple names.
#   --rerun_stage6 <model> Skip stages 1-5; re-run stage 6 on all rows of the named model's shards.
#   --retry_failed <model> Like --rerun_stage6 but only re-runs rows whose answer is None,
#                          patching the existing output file in-place.

set -euo pipefail

# ── Argument parsing ─────────────────────────────────────────────────────────
ONLY_MODELS=()
RERUN_STAGE6=""
RETRY_FAILED=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --only)
            shift
            while [[ $# -gt 0 ]] && [[ "${1}" != --* ]]; do
                ONLY_MODELS+=("$1"); shift
            done
            ;;
        --rerun_stage6) RERUN_STAGE6="$2"; shift 2 ;;
        --retry_failed) RETRY_FAILED="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

cd /nfshomes/srjnk01/Navig

BASE_DIR=output/im2gps3k_rgb_images
DATASET=dataset/im2gps3k_rgb_images
NUM_SHARDS=4

# ── Model paths ─────────────────────────────────────────────────────────────
LLAMA_PATH=/fs/nexus-scratch/srjnk01/llama-3.2-11b-vision-instruct
DEEPSEEK_PATH=/fs/nexus-scratch/srjnk01/deepseek-vl-7b-chat
FALCON_PATH=/fs/nexus-scratch/srjnk01/falcon-11B-vlm
CPM_PATH=/fs/nexus-scratch/srjnk01/MiniCPM-V-2_6
CPM_CKPT=vlms/NAVIG/minicpm-v-v2_6-chat

# ── Early exit: stage-6-only rerun ──────────────────────────────────────────
# Usage: sbatch run_comparison.sh --rerun_stage6 <model>
# Submits an array job that skips stages 1-5 and re-runs only stage 6 on the
# existing results_s5.jsonl for each shard of the named model.
if [[ -n "${RERUN_STAGE6}" ]]; then
    MODEL="${RERUN_STAGE6}"
    SHARD_PREFIX="cmp_shard_${MODEL}"
    RESULTS_NAME="results_s6_${MODEL}.jsonl"

    # Resolve model path and optional ckpt from the entries table
    MODEL_PATH=""
    CKPT_DIR=""
    case "${MODEL}" in
        llama32vision) MODEL_PATH="${LLAMA_PATH}" ;;
        deepseek)      MODEL_PATH="${DEEPSEEK_PATH}" ;;
        falcon)        MODEL_PATH="${FALCON_PATH}" ;;
        cpm)           MODEL_PATH="${CPM_PATH}"; CKPT_DIR="${CPM_CKPT}" ;;
        qwen)          MODEL_PATH="/fs/nexus-scratch/srjnk01/Qwen2.5-VL-7B-Instruct"
                       CKPT_DIR="vlms/NAVIG/qwen2-vl-7b-instruct" ;;
        *) echo "Unknown model for --rerun_stage6: ${MODEL}" >&2; exit 1 ;;
    esac

    CKPT_ARG=""
    [[ -n "${CKPT_DIR}" ]] && CKPT_ARG="--ckpt_dir ${CKPT_DIR}"

    echo "==> Submitting stage-6-only rerun for ${MODEL} (${NUM_SHARDS} shards)..."
    sbatch --parsable \
        --job-name=s6-${MODEL} \
        --output=%j_%a.s6_${MODEL}.out \
        --error=%j_%a.s6_${MODEL}.err \
        --time=4:00:00 \
        --account=nexus \
        --partition=tron \
        --qos=default \
        --nodes=1 \
        --ntasks=1 \
        --requeue \
        --gres=gpu:rtxa6000:1 \
        --mem=32g \
        --array=0-$((NUM_SHARDS - 1)) \
        --wrap="
source /nfshomes/srjnk01/miniconda3/etc/profile.d/conda.sh
conda activate navig
module unload cuda; module load cuda/12.1.1
export CUDA_HOME=/opt/common/cuda/cuda-12.1.1
cd /nfshomes/srjnk01/Navig

SHARD_OUT=${BASE_DIR}/${SHARD_PREFIX}_\${SLURM_ARRAY_TASK_ID}_of_${NUM_SHARDS}

python3 evaluation.py \\
    --dataset_path      ${DATASET} \\
    --model             ${MODEL} \\
    --model_path        ${MODEL_PATH} \\
    ${CKPT_ARG} \\
    --reasoning_path    \${SHARD_OUT} \\
    --results_file_Name ${RESULTS_NAME} \\
    --crop_box_treshold 0.3 \\
    --crop_text_treshold 0.25 \\
    --num_shards        ${NUM_SHARDS} \\
    --shard_id          \${SLURM_ARRAY_TASK_ID} \\
    --stage6_only
"
    echo "==> Stage-6-only job submitted for ${MODEL}."
    echo "==> Monitor: squeue -u \$USER"
    exit 0
fi

# ── Early exit: retry failed rows only ───────────────────────────────────────
# Usage: sbatch run_comparison.sh --retry_failed <model>
# Submits an array job that re-runs stage 6 only for rows where answer=None.
if [[ -n "${RETRY_FAILED}" ]]; then
    MODEL="${RETRY_FAILED}"
    SHARD_PREFIX="cmp_shard_${MODEL}"
    RESULTS_NAME="results_s6_${MODEL}.jsonl"

    MODEL_PATH=""
    CKPT_DIR=""
    case "${MODEL}" in
        llama32vision) MODEL_PATH="${LLAMA_PATH}" ;;
        deepseek)      MODEL_PATH="${DEEPSEEK_PATH}" ;;
        falcon)        MODEL_PATH="${FALCON_PATH}" ;;
        cpm)           MODEL_PATH="${CPM_PATH}"; CKPT_DIR="${CPM_CKPT}" ;;
        qwen)          MODEL_PATH="/fs/nexus-scratch/srjnk01/Qwen2.5-VL-7B-Instruct"
                       CKPT_DIR="vlms/NAVIG/qwen2-vl-7b-instruct" ;;
        *) echo "Unknown model for --retry_failed: ${MODEL}" >&2; exit 1 ;;
    esac

    CKPT_ARG=""
    [[ -n "${CKPT_DIR}" ]] && CKPT_ARG="--ckpt_dir ${CKPT_DIR}"

    echo "==> Submitting stage-6 retry-failed job for ${MODEL} (${NUM_SHARDS} shards)..."
    sbatch --parsable \
        --job-name=retry-${MODEL} \
        --output=%j_%a.retry_${MODEL}.out \
        --error=%j_%a.retry_${MODEL}.err \
        --time=2:00:00 \
        --account=nexus \
        --partition=tron \
        --qos=default \
        --nodes=1 \
        --ntasks=1 \
        --requeue \
        --gres=gpu:rtxa6000:1 \
        --mem=32g \
        --array=0-$((NUM_SHARDS - 1)) \
        --wrap="
source /nfshomes/srjnk01/miniconda3/etc/profile.d/conda.sh
conda activate navig
module unload cuda; module load cuda/12.1.1
export CUDA_HOME=/opt/common/cuda/cuda-12.1.1
cd /nfshomes/srjnk01/Navig

SHARD_OUT=${BASE_DIR}/${SHARD_PREFIX}_\${SLURM_ARRAY_TASK_ID}_of_${NUM_SHARDS}

python3 evaluation.py \\
    --dataset_path      ${DATASET} \\
    --model             ${MODEL} \\
    --model_path        ${MODEL_PATH} \\
    ${CKPT_ARG} \\
    --reasoning_path    \${SHARD_OUT} \\
    --results_file_Name ${RESULTS_NAME} \\
    --crop_box_treshold 0.3 \\
    --crop_text_treshold 0.25 \\
    --num_shards        ${NUM_SHARDS} \\
    --shard_id          \${SLURM_ARRAY_TASK_ID} \\
    --retry_failed
"
    echo "==> Retry-failed job submitted for ${MODEL}."
    echo "==> Monitor: squeue -u \$USER"
    exit 0
fi

# ── Helper: submit one full-pipeline SLURM array job ────────────────────────
# Returns the job ID.  Skips if all shard folders already exist.
submit_model() {
    local model=$1
    local model_path=$2
    local ckpt_dir=${3:-}   # optional; non-empty only for SFT models (e.g. cpm)
    local results_name=results_s6_${model}.jsonl
    local shard_prefix=cmp_shard_${model}

    # A shard is complete only when results_s5.jsonl exists inside it.
    # Checking directory existence is insufficient: preempted jobs leave the
    # directory behind with only a partial results_s1.jsonl.
    local all_shards_present=true
    for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
        if [[ ! -f "${BASE_DIR}/${shard_prefix}_${shard_id}_of_${NUM_SHARDS}/results_s5.jsonl" ]]; then
            all_shards_present=false
            break
        fi
    done
    if [[ "${all_shards_present}" == "true" ]]; then
        echo "==> SKIP ${model}: all ${NUM_SHARDS} shard folders already complete (results_s5.jsonl present)." >&2
        echo "SKIP"
        return
    fi

    local ckpt_arg=""
    if [[ -n "${ckpt_dir}" ]]; then
        ckpt_arg="--ckpt_dir ${ckpt_dir}"
    fi

    local jid
    jid=$(sbatch --parsable \
        --job-name=cmp-${model} \
        --output=%j_%a.cmp_${model}.out \
        --error=%j_%a.cmp_${model}.err \
        --time=12:00:00 \
        --account=nexus \
        --partition=tron \
        --qos=default \
        --nodes=1 \
        --ntasks=1 \
        --requeue \
        --gres=gpu:rtxa6000:1 \
        --mem=32g \
        --array=0-$((NUM_SHARDS - 1)) \
        --wrap="
source /nfshomes/srjnk01/miniconda3/etc/profile.d/conda.sh
conda activate navig
module unload cuda; module load cuda/12.1.1
export CUDA_HOME=/opt/common/cuda/cuda-12.1.1
cd /nfshomes/srjnk01/Navig

SHARD_OUT=${BASE_DIR}/${shard_prefix}_\${SLURM_ARRAY_TASK_ID}_of_${NUM_SHARDS}
mkdir -p \"\${SHARD_OUT}\"

python3 evaluation.py \\
    --dataset_path      ${DATASET} \\
    --model             ${model} \\
    --model_path        ${model_path} \\
    ${ckpt_arg} \\
    --reasoning_path    \${SHARD_OUT} \\
    --results_file_Name ${results_name} \\
    --crop_box_treshold 0.3 \\
    --crop_text_treshold 0.25 \\
    --num_shards        ${NUM_SHARDS} \\
    --shard_id          \${SLURM_ARRAY_TASK_ID}
")
    echo "==> Submitted ${model} → job ${jid} (array 0-$((NUM_SHARDS-1)))" >&2
    echo "${jid}"
}

# ── Step 2: Submit jobs for all models that need running ─────────────────────
JOB_IDS=()
ANALYZE_FILES=()
ANALYZE_LABELS=()

# Format: "model:path:label" or "model:path:label:ckpt_dir" for SFT models
for entry in \
    "llama32vision:${LLAMA_PATH}:LLaMA-3.2-11B:" \
    "deepseek:${DEEPSEEK_PATH}:DeepSeek-7B:" \
    "falcon:${FALCON_PATH}:Falcon-11B:" \
    "cpm:${CPM_PATH}:MiniCPM-V-2.6-SFT:${CPM_CKPT}" \
    "qwen:/fs/nexus-scratch/srjnk01/Qwen2.5-VL-7B-Instruct:Qwen2-VL-7B-SFT:vlms/NAVIG/qwen2-vl-7b-instruct"
do
    model="${entry%%:*}"
    rest="${entry#*:}"
    path="${rest%%:*}"
    rest="${rest#*:}"
    label="${rest%%:*}"
    ckpt="${rest#*:}"

    if [[ ${#ONLY_MODELS[@]} -gt 0 ]]; then
        match=false
        for m in "${ONLY_MODELS[@]}"; do
            [[ "${model}" == "${m}" ]] && { match=true; break; }
        done
        [[ "${match}" == "false" ]] && continue
    fi

    result=$(submit_model "${model}" "${path}" "${ckpt}")
    if [[ "${result}" != "SKIP" ]]; then
        JOB_IDS+=("${result}")
    fi
    ANALYZE_FILES+=("${BASE_DIR}/cmp_shard_${model}_merged/results_s6_${model}.jsonl")
    ANALYZE_LABELS+=("${label}")
done

# ── Step 3: Submit merge + analyze job ──────────────────────────────────────

build_merge_analyze_script() {
    cat <<'INNERSCRIPT'
source /nfshomes/srjnk01/miniconda3/etc/profile.d/conda.sh
conda activate navig
cd /nfshomes/srjnk01/Navig
INNERSCRIPT

    cat <<EOF
echo "==> Merging all shard groups → ${BASE_DIR}/*_merged/"
python3 merge_all_shards.py ${BASE_DIR}

echo "==> Running analysis..."
python3 analyze_results.py \\
    --canonical \\
    --output ${BASE_DIR}/comparison_report.txt
echo "==> Report saved to ${BASE_DIR}/comparison_report.txt"
EOF
}

if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    DEPEND="afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
    echo ""
    echo "==> Submitting merge+analyze job (depends on: ${JOB_IDS[*]})..."
    sbatch \
        --job-name=analyze \
        --output=%j.analyze.out \
        --error=%j.analyze.err \
        --time=12:00:00 \
        --account=nexus \
        --partition=tron \
        --qos=default \
        --nodes=1 \
        --ntasks=1 \
        --mem=16g \
        --dependency="${DEPEND}" \
        --wrap="$(build_merge_analyze_script)"
    echo "    Merge+analyze job submitted."
else
    # All models already done — merge and run analysis directly
    echo ""
    echo "==> All models already complete. Merging shards and running analysis..."
    python3 merge_all_shards.py ${BASE_DIR}
    python3 analyze_results.py \
        --canonical \
        --output ${BASE_DIR}/comparison_report.txt
fi

echo ""
echo "==> Done. Monitor jobs: squeue -u \$USER"
echo "    Output directory:   ${BASE_DIR}/"
echo "    Final report:       ${BASE_DIR}/comparison_report.txt"
