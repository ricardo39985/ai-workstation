#!/usr/bin/env bash

# ====== Paths ======
WORKSPACE="/workspace"
MODEL_DIR="$WORKSPACE/models"
CACHE_DIR="$WORKSPACE/cache"
SERVER_DIR="$WORKSPACE/servers"

mkdir -p "$MODEL_DIR/llm" "$MODEL_DIR/image" "$CACHE_DIR" "$SERVER_DIR"

export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR"

# ====== Helpers ======
banner() {
  echo ""
  echo "=============================="
  echo "  AI WORKSTATION (RunPod)"
  echo "=============================="
  echo ""
}

system_status() {
  echo ""
  echo "==== SYSTEM STATUS ===="
  echo ""
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "nvidia-smi not found"
  fi
  echo ""
  df -h
  echo ""
  echo "Workspace:"
  ls -al "$WORKSPACE"
  echo ""
  echo "Python:"
  python -V || true
  echo ""
}

install_env() {
  echo ""
  echo "Installing / verifying Python dependencies..."

  python - <<'EOF'
import importlib
packages = [
    "torch",
    "transformers",
    "diffusers",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "gradio",
    "vllm"
]
missing=[]
for p in packages:
    try:
        importlib.import_module(p)
    except:
        missing.append(p)

if missing:
    print("Missing:", missing)
    exit(1)
else:
    print("Environment already installed.")
EOF

  if [ $? -ne 0 ]; then
    echo ""
    echo "Installing packages..."

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
      gradio \
      vllm

    echo ""
    echo "Environment ready."
  fi
}

download_zimage() {
  echo ""
  echo "Downloading Z-Image-Turbo weights..."

  python - <<'EOF'
from diffusers import DiffusionPipeline
model="Tongyi-MAI/Z-Image-Turbo"
print("Downloading:", model)
pipe = DiffusionPipeline.from_pretrained(model)
print("Download complete.")
EOF
}

launch_image_ui() {
  echo ""
  echo "Launching Z-Image-Turbo UI..."

  cat <<'PY' > "$SERVER_DIR/zimg_server.py"
import torch
from diffusers import DiffusionPipeline
import gradio as gr

model_id="Tongyi-MAI/Z-Image-Turbo"

print("Loading model:", model_id)

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

pipe.to("cuda")

def generate(prompt):
    image = pipe(
        prompt,
        num_inference_steps=4,
        guidance_scale=0
    ).images[0]
    return image

demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Prompt"),
    outputs="image",
    title="Z-Image Turbo"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
PY

  echo ""
  echo "Starting image server on port 7860..."
  python "$SERVER_DIR/zimg_server.py"
}

launch_llm_server() {
  echo ""
  echo "Starting Qwen LLM server with vLLM..."

  MODEL="Qwen/Qwen3.5-27B-FP8"

  echo "Model: $MODEL"
  echo ""

  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90
}

menu() {
  while true
  do
    banner

    echo "1  System Status"
    echo "2  Install Environment"
    echo "3  Download Z-Image-Turbo"
    echo "4  Launch Image UI (port 7860)"
    echo "5  Launch Qwen LLM Server (port 8000)"
    echo "6  Python Shell"
    echo "7  Exit"
    echo ""

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