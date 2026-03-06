#!/usr/bin/env bash
set -u

WORKSPACE="/workspace"
CACHE_DIR="$WORKSPACE/hf_cache"
MODEL_DIR="$WORKSPACE/models"
REPO_DIR="$WORKSPACE/repos"
VENV_IMG="$WORKSPACE/venv_img"
VENV_LLM="$WORKSPACE/venv_llm"

COMFY_DIR="$REPO_DIR/ComfyUI"

mkdir -p "$CACHE_DIR" "$MODEL_DIR" "$REPO_DIR"
mkdir -p "$MODEL_DIR/checkpoints" "$MODEL_DIR/loras" "$MODEL_DIR/text_encoders" "$MODEL_DIR/vae" "$MODEL_DIR/clip_vision"

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

  say "Installing image environment for ComfyUI + FLUX"

  pip_in_venv "$VENV_IMG" install -U pip setuptools wheel

  pip_in_venv "$VENV_IMG" install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

  pip_in_venv "$VENV_IMG" install -r "$COMFY_DIR/requirements.txt"

  say "Image environment ready"
}

download_flux_base() {
  ensure_venv "$VENV_IMG"

  say "This script does not install the uncensored LoRA."
  say "It is prepared for standard FLUX workflows in ComfyUI."

  cat <<'EOF'

You will still need the normal FLUX model assets placed in the ComfyUI model folders.
Recommended folders:
- /workspace/models/checkpoints
- /workspace/models/text_encoders
- /workspace/models/vae
- /workspace/models/loras

Then point ComfyUI at /workspace/models using extra_model_paths.yaml.

EOF
}

write_comfy_model_paths() {
  ensure_comfyui_repo

  say "Writing ComfyUI extra_model_paths.yaml"

  cat > "$COMFY_DIR/extra_model_paths.yaml" <<EOF
comfyui:
  base_path: /workspace/models
  checkpoints: checkpoints
  loras: loras
  vae: vae
  text_encoders: text_encoders
  clip_vision: clip_vision
EOF

  say "Wrote $COMFY_DIR/extra_model_paths.yaml"
}

launch_comfyui() {
  ensure_venv "$VENV_IMG"
  ensure_comfyui_repo
  write_comfy_model_paths

  say "Launching ComfyUI on port 8188"
  cd "$COMFY_DIR"
  "$VENV_IMG/bin/python" main.py --listen 0.0.0.0 --port 8188 --extra-model-paths-config "$COMFY_DIR/extra_model_paths.yaml"
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

  say "Launching Qwen vLLM server on :8000"
  say "Open WebUI base URL: http://<pod-ip>:8000/v1"

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
    echo "2  Install Image Environment (ComfyUI)"
    echo "3  Prepare FLUX Model Folders"
    echo "4  Launch ComfyUI (port 8188)"
    echo "5  Install LLM Environment"
    echo "6  Launch Qwen vLLM Server (port 8000)"
    echo "7  Exit"
    echo ""

    read -r -p "Select option: " choice

    case "$choice" in
      1) system_status ;;
      2) install_image_env ;;
      3) download_flux_base ;;
      4) launch_comfyui ;;
      5) install_llm_env ;;
      6) launch_qwen_vllm ;;
      7) exit 0 ;;
      *) echo "Invalid option" ;;
    esac
  done
}

menu