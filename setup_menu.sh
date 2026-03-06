#!/usr/bin/env bash
set -u

WORKSPACE="/workspace"
CACHE_DIR="$WORKSPACE/hf_cache"
MODEL_DIR="$WORKSPACE/models"
SERVER_DIR="$WORKSPACE/servers"

VENV_ZIMG="$WORKSPACE/venv_zimg"
VENV_LLM="$WORKSPACE/venv_llm"

mkdir -p "$CACHE_DIR" "$MODEL_DIR" "$SERVER_DIR"
mkdir -p "$MODEL_DIR/zimage" "$MODEL_DIR/qwen"

# Use only HF_HOME going forward (TRANSFORMERS_CACHE is deprecated).
export HF_HOME="$CACHE_DIR"

say() { echo -e "\n[$(date +%H:%M:%S)] $*\n"; }

system_status() {
  say "SYSTEM STATUS"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found"
  echo
  python -V || true
  echo
  df -h
  echo
  ls -al "$WORKSPACE" || true
}

ensure_venv() {
  local venv_path="$1"
  if [ ! -d "$venv_path" ]; then
    say "Creating venv: $venv_path"
    python -m venv "$venv_path"
  fi
}

pip_in_venv() {
  local venv_path="$1"
  shift
  "$venv_path/bin/python" -m pip "$@"
}

python_in_venv() {
  local venv_path="$1"
  shift
  "$venv_path/bin/python" "$@"
}

# --- Z-IMAGE ENV (guaranteed path) ---
install_zimage_env() {
  ensure_venv "$VENV_ZIMG"

  say "Installing Z-Image environment (Torch 2.4.1 + cu124, xformers matching cu124, diffusers from source)"

  # Clean inside venv only
  pip_in_venv "$VENV_ZIMG" uninstall -y torch torchvision torchaudio xformers diffusers transformers >/dev/null 2>&1 || true

  pip_in_venv "$VENV_ZIMG" install -U pip setuptools wheel

  # Torch + CUDA (pod driver shows CUDA 12.4; use cu124 wheels)
  pip_in_venv "$VENV_ZIMG" install \
    torch==2.4.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

  # xformers compatible with torch 2.4.1 + cu124 (avoid mismatched builds)
  # Use PyTorch wheel index for CUDA-matched binaries
  pip_in_venv "$VENV_ZIMG" install \
    xformers==0.0.28.post1 \
    --index-url https://download.pytorch.org/whl/cu124

  # Core deps (keep transformers <5; Z-Image uses Qwen3 internals)
  pip_in_venv "$VENV_ZIMG" install \
    "transformers>=4.45,<5" \
    accelerate safetensors huggingface_hub sentencepiece pillow tqdm gradio

  # Diffusers: model card explicitly requires source install for Z-Image support
  # and shows `from diffusers import ZImagePipeline`. :contentReference[oaicite:2]{index=2}
  pip_in_venv "$VENV_ZIMG" install --upgrade --force-reinstall \
    git+https://github.com/huggingface/diffusers

  say "Verifying ZImagePipeline import..."
  python_in_venv "$VENV_ZIMG" - <<'PY'
from diffusers import ZImagePipeline
print("OK: ZImagePipeline is importable")
PY
}

download_zimage() {
  ensure_venv "$VENV_ZIMG"
  say "Downloading Tongyi-MAI/Z-Image-Turbo into /workspace/models/zimage"
  python_in_venv "$VENV_ZIMG" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Tongyi-MAI/Z-Image-Turbo",
    local_dir="/workspace/models/zimage",
    local_dir_use_symlinks=False,
)
print("Download complete: /workspace/models/zimage")
PY
}

launch_zimage_ui() {
  ensure_venv "$VENV_ZIMG"

  say "Writing Z-Image Gradio server to /workspace/servers/zimage_server.py"

  cat > "$SERVER_DIR/zimage_server.py" <<'PY'
import torch
import gradio as gr
from diffusers import ZImagePipeline

MODEL_PATH = "/workspace/models/zimage"

# Model card quick start uses bfloat16 and recommends guidance_scale=0 for Turbo. :contentReference[oaicite:3]{index=3}
pipe = ZImagePipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

def generate(prompt: str, seed: int = 42):
    g = torch.Generator("cuda").manual_seed(int(seed))
    img = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,   # results in 8 DiT forwards per model card :contentReference[oaicite:4]{index=4}
        guidance_scale=0.0,      # Turbo models: guidance should be 0 :contentReference[oaicite:5]{index=5}
        generator=g,
    ).images[0]
    return img

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Number(value=42, precision=0, label="Seed"),
    ],
    outputs=gr.Image(type="pil", label="Output"),
    title="Z-Image-Turbo (Gradio)",
)

demo.launch(server_name="0.0.0.0", server_port=7860)
PY

  say "Launching Z-Image UI on port 7860 (RunPod will expose an HTTP link for 7860)"
  "$VENV_ZIMG/bin/python" "$SERVER_DIR/zimage_server.py"
}

# --- LLM ENV (best-effort, isolated) ---
install_llm_env() {
  ensure_venv "$VENV_LLM"

  say "Installing LLM environment (best-effort vLLM in isolated venv)"

  pip_in_venv "$VENV_LLM" install -U pip setuptools wheel

  # Use same torch/cu124 stack to match the pod (keeps GPU runtime consistent)
  pip_in_venv "$VENV_LLM" install \
    torch==2.4.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

  # vLLM is sensitive; keep it isolated from Z-Image stack.
  # If this step fails due to upstream changes, Z-Image still works.
  pip_in_venv "$VENV_LLM" install "vllm==0.5.4" || {
    echo
    echo "vLLM install failed in venv_llm. Z-Image stack is unaffected."
    echo "If you want, we can swap to an Ollama-based LLM server instead."
    echo
    return 1
  }

  say "vLLM installed in $VENV_LLM"
}

launch_qwen_vllm() {
  ensure_venv "$VENV_LLM"

  # Model choice you picked earlier: Qwen/Qwen3.5-27B-FP8
  # vLLM will download into HF_HOME (shared cache) unless you set a separate cache.
  local model="Qwen/Qwen3.5-27B-FP8"

  say "Launching vLLM OpenAI-compatible API server on :8000 for $model"
  say "Endpoint will be http://<pod-ip>:8000/v1 (connect Open WebUI to that)"

  "$VENV_LLM/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$model" \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90
}

menu() {
  while true; do
    echo ""
    echo "==== AI WORKSTATION ===="
    echo "1  System Status"
    echo "2  Install Z-Image Environment (guaranteed)"
    echo "3  Download Z-Image-Turbo"
    echo "4  Launch Z-Image UI (port 7860)"
    echo "5  Install LLM Environment (best-effort vLLM, isolated)"
    echo "6  Launch Qwen vLLM Server (port 8000)"
    echo "7  Python Shell (Z-Image venv)"
    echo "8  Exit"
    echo ""

    read -r -p "Select option: " choice

    case "$choice" in
      1) system_status ;;
      2) install_zimage_env ;;
      3) download_zimage ;;
      4) launch_zimage_ui ;;
      5) install_llm_env ;;
      6) launch_qwen_vllm ;;
      7) "$VENV_ZIMG/bin/python" ;;
      8) exit 0 ;;
      *) echo "Invalid option" ;;
    esac
  done
}

menu