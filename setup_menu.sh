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

# Prefer HF_HOME (TRANSFORMERS_CACHE is deprecated)
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
  local venv_path="$1"; shift
  "$venv_path/bin/python" -m pip "$@"
}

python_in_venv() {
  local venv_path="$1"; shift
  "$venv_path/bin/python" "$@"
}

# ---------------------------
# Z-IMAGE (guaranteed path)
# ---------------------------
install_zimage_env() {
  ensure_venv "$VENV_ZIMG"

  say "Installing Z-Image env (torch 2.4.1 cu124 + diffusers from source w/ ZImagePipeline)"

  pip_in_venv "$VENV_ZIMG" install -U pip setuptools wheel

  # Clean only inside the venv (never touches system python)
  pip_in_venv "$VENV_ZIMG" uninstall -y torch torchvision torchaudio diffusers transformers xformers >/dev/null 2>&1 || true

  # PyTorch >= 2.4 requirement you hit earlier -> use cu124 wheels
  pip_in_venv "$VENV_ZIMG" install \
    torch==2.4.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

  # Core deps
  pip_in_venv "$VENV_ZIMG" install \
    "transformers>=4.45,<5" \
    accelerate safetensors huggingface_hub sentencepiece pillow tqdm gradio

  # Official repo says install diffusers from source for Z-Image support. :contentReference[oaicite:2]{index=2}
  pip_in_venv "$VENV_ZIMG" install --upgrade --force-reinstall \
    git+https://github.com/huggingface/diffusers

  say "Verifying ZImagePipeline import..."
  python_in_venv "$VENV_ZIMG" - <<'PY'
from diffusers import ZImagePipeline
print("OK: ZImagePipeline import works")
PY
}

download_zimage() {
  ensure_venv "$VENV_ZIMG"
  say "Downloading Tongyi-MAI/Z-Image-Turbo to /workspace/models/zimage"
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

# Official repo example uses bfloat16 and recommends:
# num_inference_steps=9 (8 forwards) and guidance_scale=0.0 for Turbo. :contentReference[oaicite:3]{index=3}
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
        num_inference_steps=9,
        guidance_scale=0.0,
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

  say "Launching Z-Image UI on :7860 (RunPod will expose an HTTP link for 7860)"
  "$VENV_ZIMG/bin/python" "$SERVER_DIR/zimage_server.py"
}

# ---------------------------
# LLM / vLLM (isolated, best-effort)
# ---------------------------
install_llm_env() {
  ensure_venv "$VENV_LLM"

  say "Installing LLM env (isolated) - best effort vLLM"
  pip_in_venv "$VENV_LLM" install -U pip setuptools wheel

  # Keep CUDA stack aligned with pod
  pip_in_venv "$VENV_LLM" install \
    torch==2.4.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

  # vLLM versions move fast; isolate so failures never break Z-Image.
  # If this fails, Z-Image still works.
  pip_in_venv "$VENV_LLM" install "vllm==0.5.4" || {
    echo
    echo "vLLM install failed in venv_llm. Z-Image is unaffected."
    echo "If you want guaranteed LLM serving, switch this menu option to Ollama."
    echo
    return 1
  }

  say "vLLM installed in $VENV_LLM"
}

launch_qwen_vllm() {
  ensure_venv "$VENV_LLM"
  local model="Qwen/Qwen3.5-27B-FP8"

  say "Launching vLLM OpenAI-compatible server on :8000"
  say "Open WebUI should point to: http://<pod-ip>:8000/v1"

  "$VENV_LLM/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$model" \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.88
}

menu() {
  while true; do
    echo ""
    echo "==== AI WORKSTATION ===="
    echo "1  System Status"
    echo "2  Install Z-Image Environment (recommended)"
    echo "3  Download Z-Image-Turbo"
    echo "4  Launch Z-Image UI (port 7860)"
    echo "5  Install LLM Environment (isolated, best-effort)"
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