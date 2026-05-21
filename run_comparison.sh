#!/bin/bash
#SBATCH --job-name=comparison
#SBATCH --output=logs/im2gps200/comparison-%j.out
#SBATCH --error=logs/im2gps200/comparison-%j.err
#SBATCH --time=00:10:00
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2gb

# run_comparison.sh — Submit the full 9-model comparison experiment.
#
# WHAT THIS DOES
# --------------
# Runs the full NAVIG pipeline (all 6 stages) independently for each model,
# then merges shards, generates figures, and optionally runs ablation studies.
#
# Models (9 full runs + 1 existing swap experiment):
#   llama32vision  Llama-3.2-11B-Vision       — zero-shot, no NAVIG SFT
#   deepseek       DeepSeek-VL-7B-Chat         — zero-shot, no NAVIG SFT
#   falcon         Falcon-11B-VLM              — zero-shot, no NAVIG SFT
#   llava          LLaVA-1.6-Vicuna-7B         — base model for all 6 stages
#   llava_sft      LLaVA-1.6-Vicuna-7B + LoRA — NAVIG SFT adapter for stage 1
#   cpm            MiniCPM-V-2.6               — base model for all 6 stages
#   cpm_sft        MiniCPM-V-2.6 + LoRA       — NAVIG SFT adapter for stage 1
#   qwen           Qwen2.5-VL-7B               — base model for all 6 stages
#   qwen_sft       Qwen2.5-VL-7B + LoRA       — NAVIG SFT adapter for stage 1
#
# SKIP LOGIC
# ----------
# A model is skipped if results_s5.jsonl already exists in ALL shard dirs:
#   output/im2gps200/cmp_shard_<model>_{0..N-1}_of_N/results_s5.jsonl
# This correctly distinguishes completed from preempted/partially-run shards.
# Delete those files (or the whole shard dirs) to force a re-run.
#
# USAGE
# -----
#   sbatch run_comparison.sh
#   sbatch run_comparison.sh --only llava llava_sft
#   sbatch run_comparison.sh --rerun_stage6 qwen_sft
#   sbatch run_comparison.sh --retry_failed cpm
#   sbatch run_comparison.sh --no_ablation

set -uo pipefail

# ── Argument parsing ─────────────────────────────────────────────────────────
ONLY_MODELS=()
RERUN_STAGE6=""
RETRY_FAILED=""
NO_ABLATION=false
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
        --no_ablation)  NO_ABLATION=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

cd /nfshomes/srjnk01/Navig
mkdir -p logs

BASE_DIR=output/im2gps200
DATASET=dataset/im2gps200
NUM_SHARDS=4

# ── Model weight paths ───────────────────────────────────────────────────────
SCRATCH=/fs/nexus-scratch/srjnk01
LLAMA_PATH=${SCRATCH}/llama-3.2-11b-vision-instruct
DEEPSEEK_PATH=${SCRATCH}/deepseek-vl-7b-chat
FALCON_PATH=${SCRATCH}/falcon-11B-vlm
LLAVA_PATH=${SCRATCH}/llava-v1.6-vicuna-7b-hf
CPM_PATH=${SCRATCH}/MiniCPM-V-2_6
QWEN_PATH=${SCRATCH}/Qwen2.5-VL-7B-Instruct

# ── SFT adapter paths (relative to project root) ────────────────────────────
LLAVA_CKPT=vlms/NAVIG/llava1_6-vicuna-7b-instruct
CPM_CKPT=vlms/NAVIG/minicpm-v-v2_6-chat
QWEN_CKPT=vlms/NAVIG/qwen2-vl-7b-instruct

# ── Shared SLURM header for GPU evaluation jobs ──────────────────────────────
GPU_HEADER="
source /nfshomes/srjnk01/miniconda3/etc/profile.d/conda.sh
conda activate navig
module unload cuda; module load cuda/12.1.1
export CUDA_HOME=/opt/common/cuda/cuda-12.1.1
cd /nfshomes/srjnk01/Navig
mkdir -p logs
"

# ── Early exit: stage-6-only rerun ──────────────────────────────────────────
# Usage: sbatch run_comparison.sh --rerun_stage6 <model_key>
# Re-runs stage 6 on existing results_s5.jsonl for each shard of the named model.
if [[ -n "${RERUN_STAGE6}" ]]; then
    MODEL_KEY="${RERUN_STAGE6}"
    SHARD_PREFIX="cmp_shard_${MODEL_KEY}"
    RESULTS_NAME="results_s6_${MODEL_KEY}.jsonl"
    MODEL_TYPE="" MODEL_PATH="" CKPT_DIR=""
    case "${MODEL_KEY}" in
        llama32vision) MODEL_TYPE=llama32vision; MODEL_PATH="${LLAMA_PATH}" ;;
        deepseek)      MODEL_TYPE=deepseek;      MODEL_PATH="${DEEPSEEK_PATH}" ;;
        falcon)        MODEL_TYPE=falcon;         MODEL_PATH="${FALCON_PATH}" ;;
        llava)         MODEL_TYPE=llava;          MODEL_PATH="${LLAVA_PATH}" ;;
        llava_sft)     MODEL_TYPE=llava;          MODEL_PATH="${LLAVA_PATH}"; CKPT_DIR="${LLAVA_CKPT}" ;;
        cpm)           MODEL_TYPE=cpm;            MODEL_PATH="${CPM_PATH}" ;;
        cpm_sft)       MODEL_TYPE=cpm;            MODEL_PATH="${CPM_PATH}";  CKPT_DIR="${CPM_CKPT}" ;;
        qwen)          MODEL_TYPE=qwen;           MODEL_PATH="${QWEN_PATH}" ;;
        qwen_sft)      MODEL_TYPE=qwen;           MODEL_PATH="${QWEN_PATH}"; CKPT_DIR="${QWEN_CKPT}" ;;
        *) echo "Unknown model key for --rerun_stage6: ${MODEL_KEY}" >&2; exit 1 ;;
    esac
    CKPT_ARG=""; [[ -n "${CKPT_DIR}" ]] && CKPT_ARG="--ckpt_dir ${CKPT_DIR}"
    VLLM_FLAG=""; [[ "${MODEL_TYPE}" == "llava" ]] && VLLM_FLAG="--use_vllm"
    echo "==> Submitting stage-6-only rerun for ${MODEL_KEY} (${NUM_SHARDS} shards)..."
    sbatch --parsable \
        --job-name=s6-${MODEL_KEY} \
        --output=logs/im2gps200/s6-${MODEL_KEY}-%j_%a.out \
        --error=logs/im2gps200/s6-${MODEL_KEY}-%j_%a.err \
        --time=4:00:00 --account=nexus --partition=tron --qos=default \
        --nodes=1 --ntasks=1 --requeue \
        --gres=gpu:rtxa6000:1 --mem=32g \
        --array=0-$((NUM_SHARDS - 1)) \
        --wrap="${GPU_HEADER}
SHARD_OUT=${BASE_DIR}/${SHARD_PREFIX}_\${SLURM_ARRAY_TASK_ID}_of_${NUM_SHARDS}
python3 pipeline/evaluation.py \
    --dataset_path      ${DATASET} \
    --model             ${MODEL_TYPE} \
    --model_path        ${MODEL_PATH} \
    ${CKPT_ARG} \
    ${VLLM_FLAG} \
    --output_path       \${SHARD_OUT} \
    --results_filename  ${RESULTS_NAME} \
    --box_threshold     0.3 \
    --text_threshold    0.25 \
    --num_shards        ${NUM_SHARDS} \
    --shard_id          \${SLURM_ARRAY_TASK_ID} \
    --stage6_only"
    echo "==> Stage-6-only rerun submitted for ${MODEL_KEY}."
    exit 0
fi

# ── Early exit: retry rows where answer=None ─────────────────────────────────
# Usage: sbatch run_comparison.sh --retry_failed <model_key>
if [[ -n "${RETRY_FAILED}" ]]; then
    MODEL_KEY="${RETRY_FAILED}"
    SHARD_PREFIX="cmp_shard_${MODEL_KEY}"
    RESULTS_NAME="results_s6_${MODEL_KEY}.jsonl"
    MODEL_TYPE="" MODEL_PATH="" CKPT_DIR=""
    case "${MODEL_KEY}" in
        llama32vision) MODEL_TYPE=llama32vision; MODEL_PATH="${LLAMA_PATH}" ;;
        deepseek)      MODEL_TYPE=deepseek;      MODEL_PATH="${DEEPSEEK_PATH}" ;;
        falcon)        MODEL_TYPE=falcon;         MODEL_PATH="${FALCON_PATH}" ;;
        llava)         MODEL_TYPE=llava;          MODEL_PATH="${LLAVA_PATH}" ;;
        llava_sft)     MODEL_TYPE=llava;          MODEL_PATH="${LLAVA_PATH}"; CKPT_DIR="${LLAVA_CKPT}" ;;
        cpm)           MODEL_TYPE=cpm;            MODEL_PATH="${CPM_PATH}" ;;
        cpm_sft)       MODEL_TYPE=cpm;            MODEL_PATH="${CPM_PATH}";  CKPT_DIR="${CPM_CKPT}" ;;
        qwen)          MODEL_TYPE=qwen;           MODEL_PATH="${QWEN_PATH}" ;;
        qwen_sft)      MODEL_TYPE=qwen;           MODEL_PATH="${QWEN_PATH}"; CKPT_DIR="${QWEN_CKPT}" ;;
        *) echo "Unknown model key for --retry_failed: ${MODEL_KEY}" >&2; exit 1 ;;
    esac
    CKPT_ARG=""; [[ -n "${CKPT_DIR}" ]] && CKPT_ARG="--ckpt_dir ${CKPT_DIR}"
    VLLM_FLAG=""; [[ "${MODEL_TYPE}" == "llava" ]] && VLLM_FLAG="--use_vllm"
    echo "==> Submitting stage-6 retry-failed job for ${MODEL_KEY} (${NUM_SHARDS} shards)..."
    sbatch --parsable \
        --job-name=retry-${MODEL_KEY} \
        --output=logs/im2gps200/retry-${MODEL_KEY}-%j_%a.out \
        --error=logs/im2gps200/retry-${MODEL_KEY}-%j_%a.err \
        --time=2:00:00 --account=nexus --partition=tron --qos=default \
        --nodes=1 --ntasks=1 --requeue \
        --gres=gpu:rtxa6000:1 --mem=32g \
        --array=0-$((NUM_SHARDS - 1)) \
        --wrap="${GPU_HEADER}
SHARD_OUT=${BASE_DIR}/${SHARD_PREFIX}_\${SLURM_ARRAY_TASK_ID}_of_${NUM_SHARDS}
python3 pipeline/evaluation.py \
    --dataset_path      ${DATASET} \
    --model             ${MODEL_TYPE} \
    --model_path        ${MODEL_PATH} \
    ${CKPT_ARG} \
    ${VLLM_FLAG} \
    --output_path       \${SHARD_OUT} \
    --results_filename  ${RESULTS_NAME} \
    --box_threshold     0.3 \
    --text_threshold    0.25 \
    --num_shards        ${NUM_SHARDS} \
    --shard_id          \${SLURM_ARRAY_TASK_ID} \
    --retry_failed"
    echo "==> Retry-failed job submitted for ${MODEL_KEY}."
    exit 0
fi

# ── Helper: submit one full-pipeline SLURM array job ─────────────────────────
# Args: model_key model_type model_path label [ckpt_dir]
# Prints the submitted job ID, or "SKIP" if all shards are already complete.
submit_model() {
    local model_key=$1
    local model_type=$2
    local model_path=$3
    local label=$4
    local ckpt_dir=${5:-}
    local results_name=results_s6_${model_key}.jsonl
    local shard_prefix=cmp_shard_${model_key}

    # A shard is complete only when results_s5.jsonl exists inside it —
    # directory existence alone is insufficient (preempted jobs leave partial dirs).
    local all_complete=true
    for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
        if [[ ! -f "${BASE_DIR}/${shard_prefix}_${shard_id}_of_${NUM_SHARDS}/results_s5.jsonl" ]]; then
            all_complete=false
            break
        fi
    done
    if [[ "${all_complete}" == "true" ]]; then
        echo "==> SKIP ${model_key} (${label}): all ${NUM_SHARDS} shards already complete." >&2
        echo "SKIP"
        return
    fi

    local ckpt_arg=""
    [[ -n "${ckpt_dir}" ]] && ckpt_arg="--ckpt_dir ${ckpt_dir}"
    local vllm_flag=""
    [[ "${model_type}" == "llava" ]] && vllm_flag="--use_vllm"

    local jid
    jid=$(sbatch --parsable \
        --job-name=cmp-${model_key} \
        --output=logs/im2gps200/cmp-${model_key}-%j_%a.out \
        --error=logs/im2gps200/cmp-${model_key}-%j_%a.err \
        --time=24:00:00 --account=nexus --partition=tron --qos=default \
        --nodes=1 --ntasks=1 --requeue \
        --gres=gpu:rtxa6000:1 --mem=32g \
        --array=0-$((NUM_SHARDS - 1)) \
        --wrap="${GPU_HEADER}
SHARD_OUT=${BASE_DIR}/${shard_prefix}_\${SLURM_ARRAY_TASK_ID}_of_${NUM_SHARDS}
mkdir -p \"\${SHARD_OUT}\"
python3 pipeline/evaluation.py \\
    --dataset_path      ${DATASET} \\
    --model             ${model_type} \\
    --model_path        ${model_path} \\
    ${ckpt_arg} \\
    ${vllm_flag} \\
    --output_path       \${SHARD_OUT} \\
    --results_filename  ${results_name} \\
    --box_threshold     0.3 \\
    --text_threshold    0.25 \\
    --num_shards        ${NUM_SHARDS} \\
    --shard_id          \${SLURM_ARRAY_TASK_ID}")
    echo "==> Submitted ${model_key} (${label}) → job ${jid} (array 0-$((NUM_SHARDS-1)))" >&2
    echo "${jid}"
}

# ── Step 1: Submit all 9 evaluation jobs ─────────────────────────────────────
# Format: "model_key:model_type:path:label:ckpt_dir"
# ckpt_dir is empty for zero-shot/base models.
EVAL_JOB_IDS=()

for entry in \
    "llama32vision:llama32vision:${LLAMA_PATH}:LLaMA-3.2-11B:" \
    "deepseek:deepseek:${DEEPSEEK_PATH}:DeepSeek-7B:" \
    "falcon:falcon:${FALCON_PATH}:Falcon-11B:" \
    "llava:llava:${LLAVA_PATH}:LLaVA-1.6:" \
    "llava_sft:llava:${LLAVA_PATH}:LLaVA-1.6 (SFT):${LLAVA_CKPT}" \
    "cpm:cpm:${CPM_PATH}:MiniCPM-V-2.6:" \
    "cpm_sft:cpm:${CPM_PATH}:MiniCPM-V-2.6 (SFT):${CPM_CKPT}" \
    "qwen:qwen:${QWEN_PATH}:Qwen2.5-VL-7B:" \
    "qwen_sft:qwen:${QWEN_PATH}:Qwen2.5-VL-7B (SFT):${QWEN_CKPT}"
do
    model_key="${entry%%:*}"; rest="${entry#*:}"
    model_type="${rest%%:*}"; rest="${rest#*:}"
    model_path="${rest%%:*}"; rest="${rest#*:}"
    label="${rest%%:*}"
    ckpt_dir="${rest#*:}"

    # --only filter
    if [[ ${#ONLY_MODELS[@]} -gt 0 ]]; then
        match=false
        for m in "${ONLY_MODELS[@]}"; do
            [[ "${model_key}" == "${m}" ]] && { match=true; break; }
        done
        [[ "${match}" == "false" ]] && continue
    fi

    result=$(submit_model "${model_key}" "${model_type}" "${model_path}" "${label}" "${ckpt_dir}")
    if [[ "${result}" != "SKIP" ]]; then
        EVAL_JOB_IDS+=("${result}")
    fi
done

# ── Step 2: Submit merge + analysis job ──────────────────────────────────────
# Runs after all eval jobs finish (even partially failed ones).
# Merges all shard groups, generates comparison report, and produces figures.

ANALYSIS_SCRIPT="
source /nfshomes/srjnk01/miniconda3/etc/profile.d/conda.sh
conda activate navig
cd /nfshomes/srjnk01/Navig
mkdir -p logs

echo '==> Merging all shard groups...'
python3 analysis/merge_all_shards.py ${BASE_DIR}

echo '==> Running comprehensive comparison report...'
python3 analysis/analyze_results.py \\
    --dir    ${BASE_DIR} \\
    --output ${BASE_DIR}/comparison_report.txt
echo '    Report: ${BASE_DIR}/comparison_report.txt'

echo '==> Generating comparison figures...'
python3 figures/plot_comparison.py \\
    --base_dir   ${BASE_DIR} \\
    --output_dir figures/
echo '    Figures written to figures/'

echo '==> Analysis complete.'
"

if [[ -n "${REF_PATH:-}" ]]; then
    ANALYSIS_SCRIPT+="
echo '==> Running ROUGE evaluation (REF_PATH is set)...'
for merged_dir in ${BASE_DIR}/cmp_shard_*_merged; do
    s1_file=\"\${merged_dir}/results_s1.jsonl\"
    [[ -f \"\${s1_file}\" ]] || continue
    model_key=\$(basename \"\${merged_dir}\" | sed 's/^cmp_shard_//; s/_merged$//')
    rouge_out=\"\${merged_dir}/rouge\"
    mkdir -p \"\${rouge_out}\"
    echo \"  ROUGE for \${model_key}...\"
    python3 analysis/rouge.py \\\\
        --results_s1_path \"\${s1_file}\" \\\\
        --ref_path        \"${REF_PATH}\" \\\\
        --output_path     \"\${rouge_out}\" || true
done
echo '==> ROUGE done.'
"
fi

if [[ ${#EVAL_JOB_IDS[@]} -gt 0 ]]; then
    DEPEND="afterany:$(IFS=:; echo "${EVAL_JOB_IDS[*]}")"
    echo ""
    echo "==> Submitting merge+analysis job (depends on: ${EVAL_JOB_IDS[*]})..."
    MERGE_JID=$(sbatch --parsable \
        --job-name=analyze \
        --output=logs/im2gps200/analyze-%j.out \
        --error=logs/im2gps200/analyze-%j.err \
        --time=01:00:00 --account=nexus --partition=tron --qos=default \
        --nodes=1 --ntasks=1 --mem=32g \
        --dependency="${DEPEND}" \
        --wrap="${ANALYSIS_SCRIPT}")
    echo "    Merge+analysis job: ${MERGE_JID}"
else
    echo ""
    echo "==> All models already complete. Running merge+analysis directly..."
    eval "${ANALYSIS_SCRIPT}"
    MERGE_JID=""
fi

# ── Step 3: Submit ablation jobs ─────────────────────────────────────────────
# One GPU job per model; all 4 ablation modes run sequentially within each job.
# Depends on the merge job so that merged s1/s5 files are available.
if [[ "${NO_ABLATION}" == "false" ]]; then
    echo ""
    echo "==> Submitting ablation jobs..."

    for entry in \
        "llama32vision:llama32vision:${LLAMA_PATH}:" \
        "deepseek:deepseek:${DEEPSEEK_PATH}:" \
        "falcon:falcon:${FALCON_PATH}:" \
        "llava:llava:${LLAVA_PATH}:" \
        "llava_sft:llava:${LLAVA_PATH}:${LLAVA_CKPT}" \
        "cpm:cpm:${CPM_PATH}:" \
        "cpm_sft:cpm:${CPM_PATH}:${CPM_CKPT}" \
        "qwen:qwen:${QWEN_PATH}:" \
        "qwen_sft:qwen:${QWEN_PATH}:${QWEN_CKPT}"
    do
        model_key="${entry%%:*}"; rest="${entry#*:}"
        model_type="${rest%%:*}"; rest="${rest#*:}"
        model_path="${rest%%:*}"
        ckpt_dir="${rest#*:}"

        # --only filter
        if [[ ${#ONLY_MODELS[@]} -gt 0 ]]; then
            match=false
            for m in "${ONLY_MODELS[@]}"; do
                [[ "${model_key}" == "${m}" ]] && { match=true; break; }
            done
            [[ "${match}" == "false" ]] && continue
        fi

        CKPT_ARG=""; [[ -n "${ckpt_dir}" ]] && CKPT_ARG="--ckpt_dir ${ckpt_dir}"
        VLLM_FLAG=""; [[ "${model_type}" == "llava" || "${model_type}" == "qwen" ]] && VLLM_FLAG="--use_vllm"
        MERGED_DIR="${BASE_DIR}/cmp_shard_${model_key}_merged"

        DEPEND_ARG=""
        [[ -n "${MERGE_JID}" ]] && DEPEND_ARG="--dependency=afterany:${MERGE_JID}"

        ABL_JID=$(sbatch --parsable \
            --job-name=abl-${model_key} \
            --output=logs/im2gps200/abl-${model_key}-%j.out \
            --error=logs/im2gps200/abl-${model_key}-%j.err \
            --time=12:00:00 --account=nexus --partition=tron --qos=default \
            --nodes=1 --ntasks=1 \
            --gres=gpu:rtxa6000:1 --mem=32g \
            ${DEPEND_ARG} \
            --wrap="${GPU_HEADER}
run_ablation() {
    local mode=\$1
    echo \"=== Ablation: \${mode} for ${model_key} ===\"
    python3 pipeline/ablation.py \\
        --dataset_path     ${DATASET} \\
        --model            ${model_type} \\
        --model_path       ${model_path} \\
        ${CKPT_ARG} \\
        ${VLLM_FLAG} \\
        --output_path      ${MERGED_DIR} \\
        --results_filename ablation_\${mode}.jsonl \\
        --\${mode} || echo \"WARNING: ablation \${mode} failed for ${model_key}\"
}
run_ablation without_reasoning
run_ablation without_tools
run_ablation base_reasoning
run_ablation direct_guess
echo 'Ablation complete for ${model_key}.'")
        echo "    ablation ${model_key} → job ${ABL_JID}"
    done
fi

echo ""
echo "==> All jobs submitted."
echo "    Monitor:      squeue -u \${USER}"
echo "    Output dir:   ${BASE_DIR}/"
echo "    Final report: ${BASE_DIR}/comparison_report.txt"
echo "    Figures:      figures/fig_comparison_*.pdf"
