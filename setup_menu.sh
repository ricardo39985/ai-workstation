#!/bin/bash

WORKSPACE="/workspace"
MODEL_DIR="$WORKSPACE/models"
SERVER_DIR="$WORKSPACE/servers"
CACHE_DIR="$WORKSPACE/hf_cache"

mkdir -p $MODEL_DIR
mkdir -p $SERVER_DIR
mkdir -p $CACHE_DIR

export HF_HOME=$CACHE_DIR

system_status() {

echo ""
echo "===== SYSTEM STATUS ====="
nvidia-smi
echo ""
python -V
echo ""
df -h
echo ""

}

install_environment() {

echo ""
echo "Installing environment..."

pip install --upgrade pip setuptools wheel

echo "Installing PyTorch..."

pip install \
torch==2.3.0 \
torchvision \
torchaudio \
--index-url https://download.pytorch.org/whl/cu121

echo "Installing xformers..."

pip install xformers==0.0.26.post1

echo "Installing base libraries..."

pip install \
transformers \
accelerate \
safetensors \
huggingface_hub \
sentencepiece \
pillow \
tqdm \
gradio

echo "Installing Diffusers from source..."

pip install git+https://github.com/huggingface/diffusers

echo "Installing vLLM dependencies..."

pip install \
cmake \
ninja \
py-cpuinfo \
openai \
ray \
tiktoken==0.6.0 \
nvidia-ml-py \
prometheus-fastapi-instrumentator \
lm-format-enforcer==0.9.8 \
outlines==0.0.34

echo "Installing vLLM..."

pip install vllm==0.4.2

echo ""
echo "Environment ready."

}

download_qwen() {

echo ""
echo "Downloading Qwen..."

python <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
repo_id="Qwen/Qwen3.5-27B-FP8",
local_dir="/workspace/models/qwen",
local_dir_use_symlinks=False
)
print("Qwen download complete")
EOF

}

download_zimage() {

echo ""
echo "Downloading Z-Image-Turbo..."

python <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
repo_id="Tongyi-MAI/Z-Image-Turbo",
local_dir="/workspace/models/zimage",
local_dir_use_symlinks=False
)
print("ZImage download complete")
EOF

}

launch_image_server() {

echo "Starting Z-Image server..."

cat << 'PY' > $SERVER_DIR/zimage_server.py
import torch
import gradio as gr
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained(
"/workspace/models/zimage",
torch_dtype=torch.bfloat16,
low_cpu_mem_usage=False
)

pipe.to("cuda")

def generate(prompt):
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=8
    ).images[0]
    return image

demo = gr.Interface(
    fn=generate,
    inputs="text",
    outputs="image",
    title="Z-Image Turbo"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
PY

python $SERVER_DIR/zimage_server.py

}

launch_qwen_server() {

echo "Starting Qwen vLLM server..."

python -m vllm.entrypoints.openai.api_server \
--model /workspace/models/qwen \
--host 0.0.0.0 \
--port 8000 \
--gpu-memory-utilization 0.90

}

python_shell() {

python

}

menu() {

while true
do

echo ""
echo "==== AI WORKSTATION ===="
echo "1 System Status"
echo "2 Install Environment"
echo "3 Download Qwen 27B"
echo "4 Download Z-Image-Turbo"
echo "5 Launch Image Server"
echo "6 Launch Qwen Server"
echo "7 Python Shell"
echo "8 Exit"

read -p "Select option: " choice

case $choice in

1) system_status ;;
2) install_environment ;;
3) download_qwen ;;
4) download_zimage ;;
5) launch_image_server ;;
6) launch_qwen_server ;;
7) python_shell ;;
8) exit 0 ;;
*) echo "Invalid option" ;;

esac

done

}

menu