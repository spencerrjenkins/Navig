#!/bin/bash
#SBATCH --job-name=navig-guess
#SBATCH --output=navig-%j_%a.out
#SBATCH --error=navig-%j_%a.err
#SBATCH --time=12:00:00
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=32g
#SBATCH --array=0-3

# ---------------------------------------------------------------------------
# Stage-6 swap experiment: run coordinate guessing with a stronger model
# on top of existing results_s5 evidence from the original NAVIG pipeline.
#
# Prerequisites:
#   1. Run the original pipeline (slurm/evaluate.sh) to produce results_s5.jsonl
#      in each shard directory.
#   2. Merge the sharded s5 files into a single file:
#          python analysis/merge_shards.py \
#              --base_dir output/im2gps3k \
#              --num_shards 4 \
#              --results_file results_s5.jsonl \
#              --output merged_s5.jsonl
#   3. Download the guesser model (one-time, ~22 GB):
#          huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
#              --local-dir /fs/nexus-scratch/$USER/llama-3.2-11b-vision-instruct
#   4. Submit this job:
#          sbatch slurm/guess_only.sh
# ---------------------------------------------------------------------------

source /nfshomes/srjnk01/miniconda3/etc/profile.d/conda.sh
conda activate navig

module unload cuda
module load cuda/12.1.1
export CUDA_HOME=/opt/common/cuda/cuda-12.1.1

cd /nfshomes/srjnk01/Navig

NUM_SHARDS=4
SHARD_ID=${SLURM_ARRAY_TASK_ID}

DATASET=dataset/test
# Reads from the merged s5 file; sharding splits the rows across array tasks.
S5_FILE=output/test/merged_s5.jsonl

# ── Model selection ─────────────────────────────────────────────────────────
# Override any of these at submission time:
#   MODEL=deepseek MODEL_PATH=... RESULTS_NAME=... sbatch slurm/guess_only.sh
#
# Supported MODEL values: llava, qwen, cpm, cpm_sft, llama32vision, internvl2, deepseek, falcon
# For cpm_sft, also set: CKPT_DIR=vlms/cpm/checkpoint-534
MODEL=${MODEL:-llama32vision}
MODEL_PATH=${MODEL_PATH:-/fs/nexus-scratch/srjnk01/llama-3.2-11b-vision-instruct}
CKPT_DIR=${CKPT_DIR:-}
RESULTS_NAME=${RESULTS_NAME:-results_s6_${MODEL}.jsonl}

SHARD_OUTPUT=output/test/guess_shard_${SHARD_ID}_of_${NUM_SHARDS}
mkdir -p "${SHARD_OUTPUT}"

CKPT_ARG=""
if [[ -n "${CKPT_DIR}" ]]; then
    CKPT_ARG="--ckpt_dir ${CKPT_DIR}"
fi

python3 pipeline/guess_only.py \
    --s5_path      "${S5_FILE}" \
    --dataset_path "${DATASET}" \
    --model        "${MODEL}" \
    --model_path   "${MODEL_PATH}" \
    ${CKPT_ARG} \
    --output       "${SHARD_OUTPUT}/${RESULTS_NAME}" \
    --num_shards   ${NUM_SHARDS} \
    --shard_id     ${SHARD_ID}

# After all 4 array tasks complete, merge and score with:
#   python analysis/merge_shards.py \
#       --base_dir output/im2gps3k \
#       --num_shards 4 \
#       --results_file ${RESULTS_NAME} \
#       --shard_prefix guess_shard \
#       --output merged_${RESULTS_NAME}
