#!/bin/bash
#SBATCH --job-name=navig-eval
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

# Activate conda environment
source /nfshomes/srjnk01/miniconda3/etc/profile.d/conda.sh
conda activate navig

# Load CUDA 12.1 (matches the torch==2.4.0+cu121 install)
module unload cuda
module load cuda/12.1.1
# Export CUDA_HOME so GroundingDINO's custom C++ ops can be found at runtime
export CUDA_HOME=/opt/common/cuda/cuda-12.1.1

# Run from the project directory
cd /nfshomes/srjnk01/Navig

NUM_SHARDS=4
SHARD_ID=${SLURM_ARRAY_TASK_ID}
SHARD_OUTPUT=output/im2gps3k/qwen_shard_${SHARD_ID}_of_${NUM_SHARDS}

python3 pipeline/evaluation.py \
    --model "qwen" \
    --dataset_path dataset/im2gps3k \
    --output_path ${SHARD_OUTPUT} \
    --results_filename "results_s6_qwen.jsonl" \
    --box_threshold 0.3 \
    --text_threshold 0.25 \
    --model_path /fs/nexus-scratch/srjnk01/Qwen2.5-VL-7B-Instruct \
    --ckpt_dir vlms/NAVIG/qwen2-vl-7b-instruct \
    --num_shards ${NUM_SHARDS} \
    --shard_id ${SHARD_ID}
