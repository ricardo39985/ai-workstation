#!/bin/bash

WORKSPACE="/workspace"
MODEL_DIR="$WORKSPACE/models"
CACHE_DIR="$WORKSPACE/cache"
SERVER_DIR="$WORKSPACE/servers"

mkdir -p $MODEL_DIR/llm
mkdir -p $MODEL_DIR/image
mkdir -p $CACHE_DIR
mkdir -p $SERVER_DIR

export HF_HOME=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_DATASETS_CACHE=$CACHE_DIR

system_status() {

echo ""
echo "==== SYSTEM STATUS ===="

nvidia-smi

echo ""
df -h

echo ""
echo "Workspace contents:"
ls -al $WORKSPACE

}

install_env() {

echo ""
echo "Installing environment..."

pip install --upgrade pip

pip install \
torch torchvision \
transformers diffusers \
accelerate \
safetensors \
huggingface_hub \
sentencepiece \
pillow \
tqdm \
gradio

pip install vllm==0.4.2 --no-deps

echo ""
echo "Environment ready."

}

download_zimage() {

echo ""
echo "Downloading Z-Image-Turbo..."

python <<EOF
from diffusers import DiffusionPipeline

model="Tongyi-MAI/Z-Image-Turbo"

print("Downloading:",model)

pipe=DiffusionPipeline.from_pretrained(model)

print("Download complete")
EOF

}

launch_image_ui() {

echo ""
echo "Launching image UI..."

cat << 'EOF' > $SERVER_DIR/zimg_server.py
import torch
from diffusers import DiffusionPipeline
import gradio as gr

model_id="Tongyi-MAI/Z-Image-Turbo"

print("Loading model",model_id)

pipe=DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

pipe.to("cuda")

def generate(prompt):

    image=pipe(
        prompt,
        num_inference_steps=4,
        guidance_scale=0
    ).images[0]

    return image

demo=gr.Interface(
    fn=generate,
    inputs="text",
    outputs="image",
    title="Z-Image Turbo"
)

demo.launch(server_name="0.0.0.0",server_port=7860)
EOF

python $SERVER_DIR/zimg_server.py

}

launch_llm_server() {

MODEL="Qwen/Qwen3.5-27B-FP8"

echo ""
echo "Launching Qwen server..."

python -m vllm.entrypoints.openai.api_server \
--model $MODEL \
--host 0.0.0.0 \
--port 8000 \
--gpu-memory-utilization 0.9

}

menu() {

while true
do

echo ""
echo "==== AI WORKSTATION ===="
echo "1 System Status"
echo "2 Install Environment"
echo "3 Download Z-Image-Turbo"
echo "4 Launch Image UI"
echo "5 Launch Qwen LLM Server"
echo "6 Python Shell"
echo "7 Exit"

read -p "Select option: " choice

case $choice in

1) system_status ;;
2) install_env ;;
3) download_zimage ;;
4) launch_image_ui ;;
5) launch_llm_server ;;
6) python ;;
7) exit 0 ;;
*) echo "Invalid option" ;;

esac

done

}

menu