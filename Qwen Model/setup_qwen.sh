#!/bin/bash

# Stop on error
set -e

echo "--- Setting up ComfyUI Environment for Qwen Image Edit ---"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

echo "Script directory: $SCRIPT_DIR"
echo "Workspace directory: $WORKSPACE_DIR"

# 1. Clone/Enter Directory
cd "$SCRIPT_DIR"
if [ ! -d "ComfyUI" ]; then
    echo "Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
else
    cd ComfyUI
fi

# 2. Create Virtual Env
echo "Creating Virtual Environment..."
rm -rf qwen
python3 -m venv qwen
source qwen/bin/activate

# 3. Install Build Tools
echo "Installing build tools..."
pip install --upgrade pip setuptools wheel ninja packaging psutil --default-timeout=1000

# 4. Install PyTorch STABLE (2.5.1 works with FP8)
echo "Installing PyTorch 2.5.1 (stable with FP8 support)..."
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --default-timeout=1000

# 5. Remove torch from requirements to avoid overwriting
echo "Preparing requirements (preserving PyTorch version)..."
grep -vE "^(torch|torchvision|torchaudio)$" requirements.txt > requirements_notorch.txt || true

# 6. Install ComfyUI Dependencies
echo "Installing ComfyUI dependencies..."
pip install -r requirements_notorch.txt --default-timeout=1000
rm -f requirements_notorch.txt

# 7. Install Evaluation Metrics
echo "Installing Evaluation Libraries (LPIPS, FID, SSIM)..."
pip install lpips scikit-image pytorch-fid --default-timeout=1000

# 8. Install HuggingFace Hub for model downloads
echo "Installing huggingface_hub..."
pip install huggingface_hub --default-timeout=1000

# 9. Create model directories
echo "Creating model directories..."
mkdir -p models/text_encoders
mkdir -p models/diffusion_models
mkdir -p models/vae
mkdir -p models/loras
mkdir -p input
mkdir -p output

# 10. Download Qwen Image Edit Models
echo "Downloading Qwen Image Edit models from HuggingFace..."

python << 'PYEOF'
from huggingface_hub import hf_hub_download
import os
import shutil

MODEL_DIR = "models"

# Text encoder (FP8)
print("Downloading text encoder (8.8GB)...")
hf_hub_download(
    repo_id="Comfy-Org/Qwen-Image_ComfyUI",
    filename="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    local_dir=MODEL_DIR,
)
os.rename(
    f"{MODEL_DIR}/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    f"{MODEL_DIR}/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
)

# VAE
print("Downloading VAE (243MB)...")
hf_hub_download(
    repo_id="Comfy-Org/Qwen-Image_ComfyUI",
    filename="split_files/vae/qwen_image_vae.safetensors",
    local_dir=MODEL_DIR,
)
os.rename(
    f"{MODEL_DIR}/split_files/vae/qwen_image_vae.safetensors",
    f"{MODEL_DIR}/vae/qwen_image_vae.safetensors"
)

# Diffusion model (Edit version, FP8)
print("Downloading diffusion model - Edit (20GB)...")
hf_hub_download(
    repo_id="Comfy-Org/Qwen-Image-Edit_ComfyUI",
    filename="split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors",
    local_dir=MODEL_DIR,
)
os.rename(
    f"{MODEL_DIR}/split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors",
    f"{MODEL_DIR}/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors"
)

# Cleanup split_files directory
shutil.rmtree(f"{MODEL_DIR}/split_files", ignore_errors=True)

print("All models downloaded successfully!")
PYEOF

# 11. Copy LoRA from parent Qwen_Model folder if it exists
if [ -f "$SCRIPT_DIR/qwen_5.safetensors" ]; then
    echo "Copying LoRA from $SCRIPT_DIR/qwen_5.safetensors..."
    cp "$SCRIPT_DIR/qwen_5.safetensors" models/loras/
else
    echo "WARNING: qwen_5.safetensors not found in $SCRIPT_DIR"
    echo "Please copy your LoRA to: $SCRIPT_DIR/ComfyUI/models/loras/"
fi

# 12. Copy run_batch.py from parent Qwen_Model folder
if [ -f "$SCRIPT_DIR/run_batch.py" ]; then
    echo "Copying run_batch.py from $SCRIPT_DIR..."
    cp "$SCRIPT_DIR/run_batch.py" .
    chmod +x run_batch.py
else
    echo "ERROR: run_batch.py not found in $SCRIPT_DIR"
    echo "Please ensure run_batch.py exists in the Qwen_Model folder"
    exit 1
fi

# 13. Create images structure if it doesn't exist
mkdir -p "$WORKSPACE_DIR/images/CLRerNet_Viz"
mkdir -p "$WORKSPACE_DIR/images/Zero-Shot"
mkdir -p "$WORKSPACE_DIR/images/Fine-Tuned"

# 14. Verify downloads
echo ""
echo "=== Verifying downloaded models ==="
ls -lh models/text_encoders/
ls -lh models/diffusion_models/
ls -lh models/vae/
ls -lh models/loras/

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "To activate and run:"
echo "  source $SCRIPT_DIR/ComfyUI/qwen/bin/activate"
echo "  cd $SCRIPT_DIR/ComfyUI"
echo "  python run_batch.py"
echo ""
