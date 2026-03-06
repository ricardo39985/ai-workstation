#!/usr/bin/env bash
set -u

###############################################################################
# AI WORKSTATION BOOTSTRAP SCRIPT
#
# PURPOSE
# -------
# Bootstraps an ephemeral RunPod AI workstation with:
# - ComfyUI for image generation UI
# - SDXL starter pack for immediate use
# - optional FLUX.1-dev download through Hugging Face auth
# - vLLM for Qwen in a separate virtual environment
#
# NOTES
# -----
# - ComfyUI runs on port 8888
# - Use the RunPod-exposed 8888 link in browser
# - Do NOT open http://0.0.0.0:8888 directly
# - vLLM runs on port 8000
###############################################################################

WORKSPACE="/workspace"
CACHE_DIR="$WORKSPACE/hf_cache"
MODEL_DIR="$WORKSPACE/models"
REPO_DIR="$WORKSPACE/repos"
VENV_IMG="$WORKSPACE/venv_img"
VENV_LLM="$WORKSPACE/venv_llm"

COMFY_DIR="$REPO_DIR/ComfyUI"
COMFY_PORT="8888"
LLM_PORT="8000"

mkdir -p "$CACHE_DIR" "$MODEL_DIR" "$REPO_DIR"
mkdir -p "$MODEL_DIR/checkpoints" "$MODEL_DIR/loras" "$MODEL_DIR/text_encoders" "$MODEL_DIR/vae" "$MODEL_DIR/clip_vision"

export HF_HOME="$CACHE_DIR"

say() { echo -e "\n[$(date +%H:%M:%S)] $*\n"; }

print_paths() {
  echo "Workspace:        $WORKSPACE"
  echo "HF cache:         $CACHE_DIR"
  echo "Models root:      $MODEL_DIR"
  echo "ComfyUI repo:     $COMFY_DIR"
  echo "Image venv:       $VENV_IMG"
  echo "LLM venv:         $VENV_LLM"
  echo "ComfyUI port:     $COMFY_PORT"
  echo "LLM API port:     $LLM_PORT"
}

print_model_layout() {
  say "MODEL DIRECTORY STATUS"

  echo "Expected model folders:"
  echo "  $MODEL_DIR/checkpoints"
  echo "  $MODEL_DIR/loras"
  echo "  $MODEL_DIR/text_encoders"
  echo "  $MODEL_DIR/vae"
  echo "  $MODEL_DIR/clip_vision"
  echo ""

  echo "Directory sizes:"
  du -sh "$MODEL_DIR"/* 2>/dev/null || true
  echo ""

  echo "Files found:"
  find "$MODEL_DIR" -maxdepth 3 -type f | sort || true
  echo ""

  local file_count
  file_count=$(find "$MODEL_DIR" -maxdepth 3 -type f | wc -l)

  if [ "$file_count" -eq 0 ]; then
    echo "WARNING: No model files were found under $MODEL_DIR"
    echo "ComfyUI is installed, but it has nothing to load yet."
  fi
}

system_status() {
  say "SYSTEM STATUS"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found"
  echo
  python -V || true
  echo
  df -h
  echo
  print_paths
  echo
  ls -al "$WORKSPACE" || true
  echo
  print_model_layout
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

ensure_comfyui_repo() {
  if [ ! -d "$COMFY_DIR/.git" ]; then
    say "Cloning ComfyUI"
    git clone https://github.com/Comfy-Org/ComfyUI.git "$COMFY_DIR"
  else
    say "Updating ComfyUI"
    git -C "$COMFY_DIR" pull --ff-only || true
  fi
}

install_image_env() {
  ensure_venv "$VENV_IMG"
  ensure_comfyui_repo

  say "Installing image environment for ComfyUI"

  pip_in_venv "$VENV_IMG" install -U pip setuptools wheel

  pip_in_venv "$VENV_IMG" install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

  pip_in_venv "$VENV_IMG" install -r "$COMFY_DIR/requirements.txt"
  pip_in_venv "$VENV_IMG" install huggingface_hub hf_transfer

  say "Image environment ready"
  print_paths
  print_model_layout
}

prepare_model_layout() {
  ensure_comfyui_repo

  say "Preparing ComfyUI model path configuration"

  cat > "$COMFY_DIR/extra_model_paths.yaml" <<EOF
comfyui:
  base_path: /workspace/models
  checkpoints: checkpoints
  loras: loras
  vae: vae
  text_encoders: text_encoders
  clip_vision: clip_vision
EOF

  echo "ComfyUI extra model path config written to:"
  echo "  $COMFY_DIR/extra_model_paths.yaml"
  echo ""
  print_model_layout
}

setup_hf_auth() {
  ensure_venv "$VENV_IMG"

  say "SETTING UP HUGGING FACE AUTH"

  if [ -z "${HF_TOKEN:-}" ]; then
    read -rsp "Enter Hugging Face token: " HF_TOKEN
    echo ""
    export HF_TOKEN
  else
    echo "HF_TOKEN already set in environment."
  fi

  python_in_venv "$VENV_IMG" - <<'PY'
from huggingface_hub import login, HfApi
import os

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN is missing.")

login(token=token, add_to_git_credential=False)

info = HfApi().whoami(token=token)
print("Authenticated as:", info.get("name") or info.get("fullname") or info)
PY
}

install_sdxl_starter_pack() {
  ensure_venv "$VENV_IMG"
  prepare_model_layout

  say "Installing SDXL starter pack"

  python_in_venv "$VENV_IMG" - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path

checkpoint_dir = Path("/workspace/models/checkpoints")
vae_dir = Path("/workspace/models/vae")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
vae_dir.mkdir(parents=True, exist_ok=True)

hf_hub_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    filename="sd_xl_base_1.0_0.9vae.safetensors",
    local_dir=str(checkpoint_dir),
    local_dir_use_symlinks=False,
)

hf_hub_download(
    repo_id="madebyollin/sdxl-vae-fp16-fix",
    filename="sdxl.vae.safetensors",
    local_dir=str(vae_dir),
    local_dir_use_symlinks=False,
)

print("SDXL starter pack download complete.")
PY

  print_model_layout
}

install_flux_dev() {
  ensure_venv "$VENV_IMG"
  prepare_model_layout

  local flux_file="$MODEL_DIR/checkpoints/flux1-dev.safetensors"

  if [ -f "$flux_file" ]; then
    say "FLUX.1-dev already exists"
    echo "Found:"
    echo "  $flux_file"
    echo ""
    print_model_layout
    return 0
  fi

  if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: Hugging Face auth is not set."
    echo "Run option 4 first."
    return 1
  fi

  say "Installing FLUX.1-dev checkpoint"

  python_in_venv "$VENV_IMG" - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
import os

token = os.environ.get("HF_TOKEN")
target = Path("/workspace/models/checkpoints")
target.mkdir(parents=True, exist_ok=True)

try:
    info = HfApi().whoami(token=token)
    print("Authenticated as:", info.get("name") or info.get("fullname") or info)
except Exception as e:
    print("ERROR: Hugging Face auth not available.")
    print(e)
    raise SystemExit(1)

try:
    path = hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        filename="flux1-dev.safetensors",
        local_dir=str(target),
        local_dir_use_symlinks=False,
        token=token,
    )
    print("FLUX download complete:")
    print(path)
except Exception as e:
    print("")
    print("FLUX download failed.")
    print("Most common reason: your account is not approved for FLUX.1-dev yet.")
    print("Model page:")
    print("  https://huggingface.co/black-forest-labs/FLUX.1-dev")
    print("")
    print("Raw error:")
    print(str(e))
    raise SystemExit(1)
PY

  print_model_layout
}

write_access_notes() {
  cat > "$WORKSPACE/ACCESS_NOTES.txt" <<EOF
AI WORKSTATION ACCESS NOTES

ComfyUI:
- Internal bind: http://0.0.0.0:${COMFY_PORT}
- Browser access: use the RunPod-exposed link for port ${COMFY_PORT}

LLM API:
- Internal bind: http://0.0.0.0:${LLM_PORT}
- Client base URL: http://<pod-ip>:${LLM_PORT}/v1

Models root:
- ${MODEL_DIR}
EOF
}

launch_comfyui() {
  ensure_venv "$VENV_IMG"
  ensure_comfyui_repo
  prepare_model_layout
  write_access_notes

  say "LAUNCHING COMFYUI"
  echo "Use the RunPod 8888 link in your browser."
  echo ""

  print_model_layout

  cd "$COMFY_DIR"
  "$VENV_IMG/bin/python" main.py \
    --listen 0.0.0.0 \
    --port "$COMFY_PORT" \
    --extra-model-paths-config "$COMFY_DIR/extra_model_paths.yaml"
}

install_llm_env() {
  ensure_venv "$VENV_LLM"

  say "Installing isolated LLM environment"

  pip_in_venv "$VENV_LLM" install -U pip setuptools wheel

  pip_in_venv "$VENV_LLM" install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

  pip_in_venv "$VENV_LLM" install "vllm==0.5.4" || {
    echo
    echo "vLLM install failed. Image stack is unaffected."
    echo
    return 1
  }

  say "LLM environment ready"
}

launch_qwen_vllm() {
  ensure_venv "$VENV_LLM"
  local model="Qwen/Qwen3.5-27B-FP8"

  write_access_notes

  say "LAUNCHING QWEN vLLM SERVER"
  echo "Use this in Open WebUI:"
  echo "  http://<pod-ip>:${LLM_PORT}/v1"
  echo ""

  "$VENV_LLM/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$model" \
    --host 0.0.0.0 \
    --port "$LLM_PORT" \
    --gpu-memory-utilization 0.88
}

menu() {
  while true; do
    echo ""
    echo "==== AI WORKSTATION ===="
    echo "1  System Status"
    echo "2  Install Image Environment (ComfyUI)"
    echo "3  Prepare Model Folder Layout"
    echo "4  Setup Hugging Face Auth"
    echo "5  Install SDXL Starter Pack"
    echo "6  Install FLUX.1-dev Checkpoint"
    echo "7  Show Model Directory Status"
    echo "8  Launch ComfyUI (port ${COMFY_PORT})"
    echo "9  Install LLM Environment"
    echo "10 Launch Qwen vLLM Server (port ${LLM_PORT})"
    echo "11 Show Access Notes"
    echo "12 Exit"
    echo ""

    read -r -p "Select option: " choice

    case "$choice" in
      1) system_status ;;
      2) install_image_env ;;
      3) prepare_model_layout ;;
      4) setup_hf_auth ;;
      5) install_sdxl_starter_pack ;;
      6) install_flux_dev ;;
      7) print_model_layout ;;
      8) launch_comfyui ;;
      9) install_llm_env ;;
      10) launch_qwen_vllm ;;
      11) write_access_notes; cat "$WORKSPACE/ACCESS_NOTES.txt" ;;
      12) exit 0 ;;
      *) echo "Invalid option" ;;
    esac
  done
}

menu