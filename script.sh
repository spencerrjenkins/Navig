#!/bin/bash

#SBATCH --job-name=navig-eval
#SBATCH --output=%j.test.out
#SBATCH --error=%j.test.err
#SBATCH --time=04:00:00
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
SHARD_OUTPUT=output/im2gps3k_rgb_images/shard_${SHARD_ID}_of_${NUM_SHARDS}

python3 evaluation.py \
    --model "llava" \
    --dataset_path dataset/im2gps3k_rgb_images \
    --reasoning_path ${SHARD_OUTPUT} \
    --results_file_Name "results_s6_llava.jsonl" \
    --crop_box_treshold 0.3 \
    --crop_text_treshold 0.25 \
    --model_path /fs/nexus-scratch/srjnk01/llava-v1.6-vicuna-7b-hf \
    --ckpt_dir vlms/NAVIG/llava1_6-vicuna-7b-instruct \
    --num_shards ${NUM_SHARDS} \
    --shard_id ${SHARD_ID} \
    --use_vllm
