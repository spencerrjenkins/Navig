#!/bin/bash
# Run this script to install all required packages into the 'navig' conda environment.
# Usage: bash install_env.sh

set -e

CONDA_ENV="navig"
PYTHON="/nfshomes/srjnk01/miniconda3/envs/${CONDA_ENV}/bin/python"
PIP="/nfshomes/srjnk01/miniconda3/envs/${CONDA_ENV}/bin/pip"

echo "==> Step 1: Install PyTorch 2.4.0 with CUDA 12.1 support"
# torch 2.4.0 + matching torchvision 0.19.0 and torchaudio 2.4.0 (must all match)
$PIP install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo "==> Step 2: Install core packages"
$PIP install \
    transformers==4.45.2 \
    openai==1.46.0 \
    Pillow==10.4.0 \
    retry==0.9.2 \
    rouge-score==0.1.2 \
    opencv-python==4.10.0.84 \
    "numpy<2.0" \
    requests==2.32.3 \
    tqdm==4.66.5 \
    sentencepiece==0.2.0 \
    accelerate==1.0.1 \
    peft==0.12.0 \
    einops==0.8.0 \
    timm==1.0.9

echo "==> Step 3: Install ms-swift (for LLaVA/Qwen inference)"
$PIP install "ms-swift==2.5.0.post1"
# PyPI pyairports 0.0.1 is a squatter package; install the real one from GitHub
$PIP install "git+https://github.com/NICTA/pyairports.git"

echo "==> Step 4: Install OpenAI CLIP"
$PIP install git+https://github.com/openai/CLIP.git

echo "==> Step 5: Install faiss (GPU version via conda, falls back to CPU)"
/nfshomes/srjnk01/miniconda3/bin/conda install -n ${CONDA_ENV} -c conda-forge faiss-gpu -y 2>/dev/null \
    || $PIP install faiss-cpu

echo "==> Step 5b: Install vLLM and ninja"
# vLLM 0.5.5 is the minimum version supporting LlavaNextForConditionalGeneration.
# ninja speeds up C++ extension builds.
$PIP install vllm==0.5.5 ninja

echo "==> Step 6: Install GroundingDINO from local source"
# GCC 8 (system default) cannot compile PyTorch 2.3 constexpr headers — load GCC 11.
# CUDA_HOME must match the PyTorch build (cu121); without it ops fall back to CPU.
module load gcc/11.2.0 2>/dev/null || true
module load cuda/12.1.1 2>/dev/null || true
export CUDA_HOME=/opt/common/cuda/cuda-12.1.1
# Target the RTX A6000 (SM 8.6) so the CUDA extension compiles for the right arch.
export TORCH_CUDA_ARCH_LIST="8.6"
cd "$(dirname "$0")/GroundingDINO"
$PIP install --no-build-isolation -e .
cd -

echo ""
echo "==> Done! Next steps:"
echo "    1. Create configuration.py in the project root (see instructions below)"
echo "    2. Activate the environment with: conda activate navig"
echo "    3. Run: python evaluation.py --model llava ..."
echo ""
echo "    To create configuration.py, run:"
echo "    cat > configuration.py << 'EOF'"
echo "    class Config:"
echo "        OPENAI_API_KEY = 'your-openai-api-key-here'"
echo "    EOF"
