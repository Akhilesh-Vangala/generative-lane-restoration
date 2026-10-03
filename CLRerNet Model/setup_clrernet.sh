#!/bin/bash
set -e

# Configuration
REPO_URL="https://github.com/hirotomusiker/CLRerNet.git"
REPO_DIR="CLRerNet"

echo "======================================================="
echo "   CLRerNet Installation Pipeline"
echo "======================================================="

# --- STEP 1: System Cleanup & Dependencies ---
echo "[1/9] Preparing System (Python 3.11 & Build Tools)..."
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
sudo apt-get install -y libopencv-dev libffi-dev liblapack-dev libsqlite3-dev \
    build-essential libssl-dev libbz2-dev libreadline-dev \
    git zip curl wget vim python3-opencv nvidia-cuda-toolkit

# --- STEP 2: Start ---
if [ -d "$REPO_DIR" ]; then
    echo " Removing old '$REPO_DIR' for a clean slate..."
    rm -rf "$REPO_DIR"
fi
echo "[2/9] Cloning Repository..."
git clone $REPO_URL
cd $REPO_DIR

# --- STEP 3: Environment Creation ---
echo "[3/9] Creating Python 3.11 Virtual Environment..."
python3.11 -m venv clrer
source clrer/bin/activate
pip install --upgrade pip

# --- STEP 4: Apply Constraints ---
echo "numpy<2.0" > constraints.txt
echo "pillow<10.0.0" >> constraints.txt
echo "[4/9] Constraints applied: NumPy < 2.0, Pillow < 10.0"

# --- STEP 5: PyTorch Installation ---
echo "[5/9] Installing PyTorch 2.1.0..."
pip install -c constraints.txt torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# --- STEP 6: OpenMMLab & Requirements ---
echo "[6/9] Installing Libraries..."
pip install -c constraints.txt -U openmim
# Manual MMCV install to bypass 'mim' dependency bugs
pip install -c constraints.txt mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
pip install -c constraints.txt mmengine==0.10.5
pip install -c constraints.txt mmdet==3.3.0

# Install remaining deps (ignoring their version conflicts)
pip install -c constraints.txt albumentations==0.4.6 --no-deps
pip install -c constraints.txt imgaug --no-deps
pip install -c constraints.txt p_tqdm==1.4.2 pytest pytest-cov tensorboard yapf==0.40.1
pip install scikit-image

# --- STEP 7: Patching PyTorch ---
echo "[7/9] Patching PyTorch to allow CUDA 13.0..."
TARGET_FILE="clrer/lib/python3.11/site-packages/torch/utils/cpp_extension.py"

if [ -f "$TARGET_FILE" ]; then
    # This command replaces the error-raising line with 'pass'
    sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE.*/pass/' "$TARGET_FILE"
    echo "PyTorch patched: CUDA version check disabled."
else
    echo "Error: Could not find PyTorch file to patch!"
    exit 1
fi

# --- STEP 8: Compile NMS ---
echo "[8/9] Compiling NMS Kernel..."
cd libs/models/layers/nms

# Clean artifacts
rm -rf build dist src/*.egg-info

# Set GPU Architectures
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"

# Run Install
python setup.py install

cd ../../../..

# --- STEP 9: Adding Dataset Path ---
echo "[9/9] Creating Dummy Dataset Files..."
mkdir -p dataset/culane/list
touch dataset/culane/list/test.txt
touch dataset/culane/list/val.txt
touch dataset/culane/list/train_gt.txt
echo "Dummy dataset files created inside $REPO_DIR."

echo "==================================================="
echo "SUCCESS! CLRerNet is fully installed."
echo "   Environment Location: $REPO_DIR/clrer"
echo "To start using it: source $REPO_DIR/clrer/bin/activate"
echo "==================================================="