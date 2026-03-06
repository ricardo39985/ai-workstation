#!/usr/bin/env bash
set -u

###############################################################################
# AI WORKSTATION BOOTSTRAP SCRIPT
#
# PURPOSE
# -------
# This script bootstraps an ephemeral RunPod-based AI workstation.
# It is designed for a disposable GPU pod where the filesystem is not expected
# to persist after termination. The script rebuilds the environment on demand.
#
# PRIMARY GOALS
# -------------
# 1. Provide a stable image-generation UI through ComfyUI.
# 2. Provide a separate LLM inference server using Qwen via vLLM.
# 3. Keep image tooling and LLM tooling isolated so dependency conflicts in one
#    stack do not break the other.
# 4. Make the setup easy for humans and future agents: clone repo, run script,
#    choose menu option, use exposed RunPod ports.
#
# CURRENT INFRA / ASSUMPTIONS
# ---------------------------
# - Runtime target: RunPod GPU pod
# - Current GPU target: RTX A6000 48GB VRAM
# - Pod is ephemeral:
#   - Local filesystem is assumed disposable
#   - Script and repo live in GitHub and are re-cloned each session
# - User wants an "AI server but not permanent":
#   - Start pod when needed
#   - Launch services/UI
#   - Stop or terminate pod when done
#
# SERVICE ARCHITECTURE
# --------------------
# IMAGE STACK
# - ComfyUI runs in its own Python virtual environment
# - Exposed on port 8188
# - Model folders live under /workspace/models
# - This script prepares the directory structure and ComfyUI model paths
#
# LLM STACK
# - vLLM runs in a separate Python virtual environment
# - Exposed on port 8000
# - Default model target: Qwen/Qwen3.5-27B-FP8
# - Intended consumer: Open WebUI or any OpenAI-compatible client
#
# WHY TWO VIRTUAL ENVS
# --------------------
# ComfyUI / image stack and vLLM / LLM stack have historically conflicting
# dependency requirements, especially around torch/xformers/transformers.
# To avoid breakage, they are isolated:
#
# - /workspace/venv_img  -> ComfyUI + image stack
# - /workspace/venv_llm  -> vLLM + Qwen stack
#
# USER / AGENT INTENT
# -------------------
# This script is not meant to be a permanent deployment orchestrator.
# It is meant to quickly reconstruct a working workstation on a fresh pod.
#
# Expected human workflow:
#   1. Start RunPod pod
#   2. SSH into pod
#   3. Clone repo
#   4. Run this script
#   5. Install image env and/or LLM env
#   6. Launch ComfyUI and/or Qwen server
#   7. Use RunPod-exposed ports in browser/UI
#   8. Stop/terminate pod when done
#
# IMPORTANT OPERATIONAL NOTES
# ---------------------------
# - ComfyUI binds to 0.0.0.0:8188 inside the pod. The browser should use the
#   RunPod HTTP/proxy link for port 8188, NOT http://0.0.0.0:8188 directly.
# - vLLM binds to 0.0.0.0:8000 inside the pod. Open WebUI or other clients
#   should use the RunPod-exposed URL or pod IP/port as appropriate.
# - This script does NOT automatically download gated or restricted model assets.
#   It prepares the folder structure and service environment.
# - This script intentionally does NOT configure uncensored or policy-violating
#   image pipelines.
#
# DIRECTORY LAYOUT
# ----------------
# /workspace/hf_cache         - Hugging Face cache
# /workspace/models           - Shared model storage
# /workspace/models/checkpoints
# /workspace/models/loras
# /workspace/models/text_encoders
# /workspace/models/vae
# /workspace/models/clip_vision
# /workspace/repos/ComfyUI    - ComfyUI git repo
# /workspace/venv_img         - image stack venv
# /workspace/venv_llm         - llm stack venv
#
# IF ANOTHER AGENT IS READING THIS
# --------------------------------
# The user wants pragmatic execution, not endless architecture discussion.
# Favor reliability, separation of concerns, and minimal manual steps.
# Preserve the two-venv approach unless there is a very strong reason not to.
###############################################################################

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

  say "Installing image environment for ComfyUI"

  pip_in_venv "$VENV_IMG" install -U pip setuptools wheel

  pip_in_venv "$VENV_IMG" install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

  pip_in_venv "$VENV_IMG" install -r "$COMFY_DIR/requirements.txt"

  say "Image environment ready"
}

prepare_flux_model_layout() {
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

  cat <<'EOF'

Model folders are ready under /workspace/models.

Expected subfolders:
- /workspace/models/checkpoints
- /workspace/models/loras
- /workspace/models/text_encoders
- /workspace/models/vae
- /workspace/models/clip_vision

ComfyUI will use:
  /workspace/repos/ComfyUI/extra_model_paths.yaml

This script prepares the layout but does not auto-download gated model assets.

EOF
}

launch_comfyui() {
  ensure_venv "$VENV_IMG"
  ensure_comfyui_repo
  prepare_flux_model_layout

  say "Launching ComfyUI on port 8188"
  say "Use the RunPod HTTP/proxy link for port 8188 in your browser"

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
  say "Open WebUI base URL should point to the RunPod-exposed endpoint for port 8000"
  say "Model: $model"

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
    echo "3  Prepare Model Folder Layout"
    echo "4  Launch ComfyUI (port 8188)"
    echo "5  Install LLM Environment"
    echo "6  Launch Qwen vLLM Server (port 8000)"
    echo "7  Exit"
    echo ""

    read -r -p "Select option: " choice

    case "$choice" in
      1) system_status ;;
      2) install_image_env ;;
      3) prepare_flux_model_layout ;;
      4) launch_comfyui ;;
      5) install_llm_env ;;
      6) launch_qwen_vllm ;;
      7) exit 0 ;;
      *) echo "Invalid option" ;;
    esac
  done
}

menu