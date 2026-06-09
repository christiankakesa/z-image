#!/usr/bin/env bash

set -euo pipefail

python3 -m venv ~/python-envs/z-image && \
source ~/python-envs/z-image/bin/activate && \
pip install --upgrade pip setuptools && \
pip install --upgrade invisible_watermark transformers accelerate safetensors && \
pip install --upgrade git+https://github.com/huggingface/diffusers && \
pip install packaging ninja && \
# pip install flash-attn --no-build-isolation && \
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128 && \
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 && \
pip install bitsandbytes && \
pip install compel && \
pip install peft && \
# pip install git+https://github.com/nunchaku-ai/nunchaku.git --upgrade && \
# pip install git+https://github.com/nunchaku-ai/nunchaku.git --upgrade --no-build-isolation && \
pip install sageattention && \
pip install imageio imageio-ffmpeg && \
exit 0
