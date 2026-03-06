#!/bin/bash

set -e

WORKSPACE="/workspace"
MODEL_DIR="$WORKSPACE/models"
CACHE_DIR="$WORKSPACE/cache"

mkdir -p $MODEL_DIR/llm
mkdir -p $MODEL_DIR/image
mkdir -p $CACHE_DIR
mkdir -p $WORKSPACE/projects

export HF_HOME=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_DATASETS_CACHE=$CACHE_DIR

gpu_check() {
echo ""
echo "Checking GPU..."
nvidia-smi
}

system_status() {
echo ""
echo "==== SYSTEM STATUS ===="
nvidia-smi
echo ""
df -h
}

install_env() {

python - <<EOF
import importlib
packages = ["torch","transformers","diffusers","accelerate","safetensors","huggingface_hub"]
missing=[]
for p in packages:
    try:
        importlib.import_module(p)
    except:
        missing.append(p)

if missing:
    print("Missing:",missing)
    exit(1)
else:
    print("Environment already installed")
EOF

if [ $? -ne 0 ]; then

pip install --upgrade pip

pip install \
torch torchvision \
transformers diffusers \
accelerate \
safetensors \
huggingface_hub \
xformers \
sentencepiece \
pillow \
tqdm

fi
}

download_qwen() {

python <<EOF
from transformers import AutoTokenizer, AutoModelForCausalLM

model="Qwen/Qwen2.5-7B-Instruct"

print("Downloading",model)

tokenizer=AutoTokenizer.from_pretrained(model)
model=AutoModelForCausalLM.from_pretrained(model)

print("Qwen ready")
EOF

}

download_zimage() {

python <<EOF
from diffusers import DiffusionPipeline

model="Tongyi-MAI/Z-Image-Turbo"

print("Downloading",model)

pipe=DiffusionPipeline.from_pretrained(model)

print("Z-Image ready")
EOF

}

menu() {

while true
do

echo ""
echo "==== AI WORKSTATION ===="
echo "1 System Status"
echo "2 Install Environment"
echo "3 Download Qwen"
echo "4 Download Z-Image-Turbo"
echo "5 Python Shell"
echo "6 Exit"

read -p "Select option: " choice

case $choice in

1) system_status ;;
2) install_env ;;
3) download_qwen ;;
4) download_zimage ;;
5) python ;;
6) exit 0 ;;
*) echo "Invalid option" ;;

esac

done

}

gpu_check
menu