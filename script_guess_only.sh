#!/bin/bash
#SBATCH --job-name=navig-guess
#SBATCH --output=%j_%a.guess.out
#SBATCH --error=%j_%a.guess.err
#SBATCH --time=02:00:00
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
#   1.  Run the original pipeline (script.sh) to produce results_s5.jsonl
#       in each shard directory.
#   2.  Merge the sharded s5 files into a single file:
#           python merge_shards.py \
#               --base_dir output/im2gps3k_rgb_images \
#               --num_shards 4 \
#               --results_file results_s5.jsonl \
#               --output merged_s5.jsonl
#   3.  Download the guesser model (one-time, ~22 GB):
#           huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
#               --local-dir /fs/nexus-scratch/$USER/llama-3.2-11b-vision-instruct
#   4.  Submit this job:
#           sbatch script_guess_only.sh
# ---------------------------------------------------------------------------

source /nfshomes/srjnk01/miniconda3/etc/profile.d/conda.sh
conda activate navig

module unload cuda
module load cuda/12.1.1
export CUDA_HOME=/opt/common/cuda/cuda-12.1.1

cd /nfshomes/srjnk01/Navig

NUM_SHARDS=4
SHARD_ID=${SLURM_ARRAY_TASK_ID}

DATASET=dataset/im2gps3k_rgb_images
# Reads from the merged s5 file; sharding splits the rows across array tasks.
S5_FILE=output/im2gps3k_rgb_images/merged_s5.jsonl

# ── Llama 3.2 11B Vision (primary experiment) ──────────────────────────────
MODEL=llama32vision
MODEL_PATH=/fs/nexus-scratch/srjnk01/llama-3.2-11b-vision-instruct
RESULTS_NAME=results_s6_llama32.jsonl

# ── InternVL2-8B (alternative — swap this block to run it instead) ─────────
# MODEL=internvl2
# MODEL_PATH=/fs/nexus-scratch/srjnk01/InternVL2-8B
# RESULTS_NAME=results_s6_internvl2.jsonl

SHARD_OUTPUT=output/im2gps3k_rgb_images/guess_shard_${SHARD_ID}_of_${NUM_SHARDS}
mkdir -p "${SHARD_OUTPUT}"

python3 guess_only.py \
    --s5_path      "${S5_FILE}" \
    --dataset_path "${DATASET}" \
    --model        "${MODEL}" \
    --model_path   "${MODEL_PATH}" \
    --output       "${SHARD_OUTPUT}/${RESULTS_NAME}" \
    --num_shards   ${NUM_SHARDS} \
    --shard_id     ${SHARD_ID}

# After all 4 array tasks complete, merge and score with:
#   python merge_shards.py \
#       --base_dir output/im2gps3k_rgb_images \
#       --num_shards 4 \
#       --results_file ${RESULTS_NAME} \
#       --shard_prefix guess_shard \
#       --output merged_${RESULTS_NAME}
