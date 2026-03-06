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
# INFRA / INTENT
# --------------
# - Target runtime: ephemeral RunPod GPU pod
# - Current GPU target: RTX A6000 48GB VRAM
# - Repo/script lives in GitHub and is re-run on fresh pods
# - Image stack and LLM stack stay isolated to avoid dependency conflicts
#
# ACCESS
# ------
# - ComfyUI runs on port 8888
# - In browser, use the RunPod-exposed 8888 link (often the Jupyter link)
# - Do NOT try to open http://0.0.0.0:8888 directly
#
# - vLLM runs on port 8000
# - Open WebUI / other clients should use:
#   http://<pod-ip>:8000/v1
#
# MODEL LAYOUT
# ------------
# /workspace/models/checkpoints
# /workspace/models/loras
# /workspace/models/text_encoders
# /workspace/models/vae
# /workspace/models/clip_vision
#
# IMPORTANT
# ---------
# - SDXL starter pack is auto-downloadable
# - FLUX.1-dev is gated on Hugging Face and requires:
#   1) HF login already configured
#   2) access approved for the account
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

check_hf_auth() {
  ensure_venv "$VENV_IMG"

  say "CHECKING HUGGING FACE AUTH"

  python_in_venv "$VENV_IMG" - <<'PY'
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

try:
    info = HfApi().whoami()
    print("Hugging Face auth is configured.")
    print("User:", info.get("name") or info.get("fullname") or info)
except Exception as e:
    print("Hugging Face auth not available in this environment.")
    print("You need to login on the pod first.")
    print("Error:", e)
PY
}

install_sdxl_starter_pack() {
  ensure_venv "$VENV_IMG"
  prepare_model_layout

  say "Installing SDXL starter pack"
  echo "This will download:"
  echo "  - SDXL base checkpoint -> $MODEL_DIR/checkpoints"
  echo "  - SDXL fp16 VAE fix    -> $MODEL_DIR/vae"
  echo ""

  python_in_venv "$VENV_IMG" - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path

checkpoint_dir = Path("/workspace/models/checkpoints")
vae_dir = Path("/workspace/models/vae")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
vae_dir.mkdir(parents=True, exist_ok=True)

print("Downloading SDXL base checkpoint...")
hf_hub_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    filename="sd_xl_base_1.0_0.9vae.safetensors",
    local_dir=str(checkpoint_dir),
    local_dir_use_symlinks=False,
)

print("Downloading SDXL VAE fp16 fix...")
hf_hub_download(
    repo_id="madebyollin/sdxl-vae-fp16-fix",
    filename="sdxl.vae.safetensors",
    local_dir=str(vae_dir),
    local_dir_use_symlinks=False,
)

print("SDXL starter pack download complete.")
PY

  echo ""
  echo "Installed files should now exist in:"
  echo "  $MODEL_DIR/checkpoints"
  echo "  $MODEL_DIR/vae"
  echo ""

  print_model_layout
}

install_flux_dev() {
  ensure_venv "$VENV_IMG"
  prepare_model_layout

  say "Installing FLUX.1-dev checkpoint"

  python_in_venv "$VENV_IMG" - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, LocalTokenNotFoundError
from huggingface_hub.utils import HfHubHTTPError

target = Path("/workspace/models/checkpoints")
target.mkdir(parents=True, exist_ok=True)

print("Target directory:", target)

try:
    info = HfApi().whoami()
    print("Authenticated as:", info.get("name") or info.get("fullname") or info)
except Exception as e:
    print("ERROR: Hugging Face auth is not configured for this environment.")
    print("Login on the pod first, then retry.")
    print("Details:", e)
    raise SystemExit(1)

try:
    print("Downloading FLUX.1-dev...")
    path = hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        filename="flux1-dev.safetensors",
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    print("FLUX download complete:")
    print(path)

except Exception as e:
    text = str(e)
    print("")
    print("FLUX download failed.")
    print("Most common reason: your Hugging Face account is not approved for FLUX.1-dev yet.")
    print("Open the model page, request/confirm access, then retry.")
    print("")
    print("Model page:")
    print("  https://huggingface.co/black-forest-labs/FLUX.1-dev")
    print("")
    print("Raw error:")
    print(text)
    raise SystemExit(1)
PY

  echo ""
  print_model_layout
}

write_access_notes() {
  cat > "$WORKSPACE/ACCESS_NOTES.txt" <<EOF
AI WORKSTATION ACCESS NOTES

ComfyUI:
- Internal bind: http://0.0.0.0:${COMFY_PORT}
- Browser access: use the RunPod-exposed link for port ${COMFY_PORT}
- In your current setup this is usually the Jupyter/HTTP link for ${COMFY_PORT}

LLM API:
- Internal bind: http://0.0.0.0:${LLM_PORT}
- Client base URL: http://<pod-ip>:${LLM_PORT}/v1
- Open WebUI should point to the RunPod-exposed endpoint for port ${LLM_PORT}

Models root:
- ${MODEL_DIR}

Expected image model folders:
- ${MODEL_DIR}/checkpoints
- ${MODEL_DIR}/loras
- ${MODEL_DIR}/text_encoders
- ${MODEL_DIR}/vae
- ${MODEL_DIR}/clip_vision
EOF
}

launch_comfyui() {
  ensure_venv "$VENV_IMG"
  ensure_comfyui_repo
  prepare_model_layout
  write_access_notes

  say "LAUNCHING COMFYUI"

  echo "ComfyUI is starting on internal address:"
  echo "  http://0.0.0.0:${COMFY_PORT}"
  echo ""
  echo "DO NOT open 0.0.0.0 in your browser."
  echo "Use the RunPod HTTP/Jupyter link that maps to port ${COMFY_PORT}."
  echo ""
  echo "Access notes also written to:"
  echo "  $WORKSPACE/ACCESS_NOTES.txt"
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
  print_paths
}

launch_qwen_vllm() {
  ensure_venv "$VENV_LLM"
  local model="Qwen/Qwen3.5-27B-FP8"

  write_access_notes

  say "LAUNCHING QWEN vLLM SERVER"

  echo "Model:"
  echo "  $model"
  echo ""
  echo "Internal API bind:"
  echo "  http://0.0.0.0:${LLM_PORT}/v1"
  echo ""
  echo "Use this in Open WebUI or another OpenAI-compatible client:"
  echo "  http://<pod-ip>:${LLM_PORT}/v1"
  echo ""
  echo "Access notes also written to:"
  echo "  $WORKSPACE/ACCESS_NOTES.txt"
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
    echo "4  Check Hugging Face Auth"
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
      4) check_hf_auth ;;
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