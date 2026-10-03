# Road Lane Overlay: Generative Restoration of Lane Markings in Dashcam Footage

A deep learning pipeline for detecting and restoring road lane markings in dashcam footage using CLRerNet for lane detection and a fine-tuned Qwen Image Edit model for realistic lane restoration.

## Overview

This project implements a **detect → repair** pipeline for lane-marking recovery:

1. **CLRerNet** detects lane polylines and generates visualization overlays with green lane markers
2. **Qwen Image Edit** (fine-tuned) replaces the green detection lines with realistic road lane markings

## Pipeline Architecture

```
Input Images
     │
     ▼
┌─────────────────────────────────┐
│  CLRerNet Lane Detection        │
│  - Detects lane polylines       │
│  - Generates green line overlay │
│  - Outputs: visualization +     │
│    lane coordinates             │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Qwen Image Edit                │
│  - Replaces green lines with    │
│    realistic lane markings      │
│  - Zero-shot & Fine-tuned modes │
└─────────────────────────────────┘
     │
     ▼
Restored Images
```

## Demo Results

The `images/` folder contains demo images showing each stage of the pipeline:

| Stage | Folder | Description |
|-------|--------|-------------|
| Input | `Targets/` | Original ground truth image |
| Detection | `CLRerNet_Viz/` | CLRerNet output with green lane overlays |
| Zero-Shot | `Zero-Shot/` | Base Qwen model output (no fine-tuning) |
| Fine-Tuned | `Fine-Tuned/` | LoRA fine-tuned model output |

## Project Structure

```
Road-Lane-Overlay/
├── CLRerNet Model/
│   ├── generate_lanes.py          # Lane detection inference script
│   ├── setup_clrernet.sh          # Environment setup
│   └── run_clrernet.sh            # Batch processing wrapper
│
├── Qwen Model/
│   ├── run_batch.py               # Batch inference (zero-shot & fine-tuned)
│   ├── setup_qwen.sh              # ComfyUI environment setup
│   └── config.yaml                # LoRA training configuration
│
├── Lama/                          # LaMa inpainting (exploratory, not used in final pipeline)
│   ├── notebooks/                 # Jupyter notebooks for testing
│   ├── saicinpainting/            # LaMa model package
│   └── models/big-lama/           # Pre-trained LaMa model
│
├── visibility_boost.py            # Dark Channel Prior dehazing (optional)
├── evaluate.py                    # Metrics evaluation (LPIPS, SSIM, FID)
└── images/                        # Input/output image directories (demo included)
    ├── Targets/                   # Ground truth images
    ├── CLRerNet_Viz/              # Lane detection outputs
    ├── Zero-Shot/                 # Base model outputs
    └── Fine-Tuned/                # LoRA fine-tuned outputs
```

## Hardware Requirements

- **GPU:** NVIDIA GPU with CUDA support (tested on CUDA 12.1/12.4)
- **VRAM:** ~24GB recommended for Qwen inference
- **Storage:** ~35GB for all models

## Installation

### 1. CLRerNet Setup (Ubuntu/Linux with CUDA)

```bash
cd "CLRerNet Model"
bash setup_clrernet.sh
```

This installs:
- Python 3.11 virtual environment
- PyTorch 2.1.0 with CUDA 12.1
- OpenMMLab ecosystem (mmcv, mmengine, mmdet)
- CLRerNet repository and NMS kernel

### 2. Qwen Model Setup

```bash
cd "Qwen Model"
bash setup_qwen.sh
```

This installs:
- ComfyUI framework
- PyTorch 2.5.1 with CUDA 12.4
- Qwen Image Edit models (~29GB total):
  - Text encoder (8.8GB)
  - Diffusion model (20GB)
  - VAE (243MB)
- Evaluation libraries (LPIPS, pytorch-fid, scikit-image)

### 3. Download Model Weights

The model checkpoints are hosted on Google Drive (not included in the repository due to size).

#### CLRerNet Model (63.5MB)
**[Download clrernet_culane_dla34_ema.pth](https://drive.google.com/file/d/1ikZjjnGq5aNpEPMJ9-Mm8cRKpABA4_8j/view?usp=sharing)**

Place the file in:
```
CLRerNet Model/clrernet_culane_dla34_ema.pth
```

#### Qwen LoRA Weights (281MB)
**[Download qwen_5.safetensors](https://drive.google.com/file/d/1Mo16H1ASopDS6AQx2qN_urGmz8AyrIPS/view?usp=sharing)**

Place the file in one of these locations:

**Option A** (before running setup):
```
Qwen Model/qwen_5.safetensors
```
The setup script will automatically copy it to the correct location.

**Option B** (after running setup):
```
Qwen Model/ComfyUI/models/loras/qwen_5.safetensors
```

## Usage

> **Note:** All commands below assume you are starting from the main project directory (`Road-Lane-Overlay/`).

### Step 1: Lane Detection with CLRerNet

```bash
cd "CLRerNet Model"
source CLRerNet/clrer/bin/activate

python generate_lanes.py \
    --input ../images/Targets \
    --output_dir ../images/CLRerNet_Viz \
    --device cuda
```

Or use the wrapper script:
```bash
bash run_clrernet.sh
```

**Outputs:**
- `images/CLRerNet_Viz/` - Images with green lane overlays
- `images/CLRerNet_Viz/coords/` - Lane coordinates as `.lines.txt` files

### Step 2: Lane Restoration with Qwen

```bash
cd "Qwen Model/ComfyUI"
source qwen/bin/activate

python run_batch.py
```

**Configuration** (in `run_batch.py`):
```python
# Paths are auto-calculated relative to script location
INPUT_DIR = WORKSPACE_DIR / "images" / "CLRerNet_Viz"   # CLRerNet outputs
OUTPUT_DIR = WORKSPACE_DIR / "images"                   # Output location
SEED = 42                                               # Fixed seed for reproducibility
RUN_ZERO_SHOT = True                                    # Run base model
RUN_FINE_TUNED = True                                   # Run with LoRA
```

**Outputs:**
- `images/Zero-Shot/` - Base model results
- `images/Fine-Tuned/` - LoRA fine-tuned results

### Step 3: Evaluation

```bash
python evaluate.py \
    --target_dir ./images/Targets \
    --zero_shot_dir ./images/Zero-Shot \
    --finetuned_dir ./images/Fine-Tuned
```

**Metrics:**
| Metric | Description | Better |
|--------|-------------|--------|
| LPIPS  | Perceptual similarity | Lower |
| FID    | Distribution distance | Lower |
| F1     | Lane detection accuracy | Higher |
| Recall | Lane detection coverage | Higher |

> **Note:** FID requires multiple images to calculate distribution statistics. It will return an error if only 1 image is provided.

## Results

| Category | Metric | Zero-Shot | Fine-Tuned | Improvement |
|----------|--------|-----------|------------|-------------|
| Image Quality | FID ↓ | 34.54 | 30.28 | +12.3% |
| Image Quality | LPIPS ↓ | 0.1213 | 0.1207 | +0.4% |
| Lane Preservation | F1 ↑ | 0.6733 | 0.7214 | +7.1% |
| Lane Preservation | Recall ↑ | 0.6208 | 0.6850 | +10.3% |

*Lane preservation metrics evaluated at IoU threshold = 0.5*

Fine-tuning significantly improves lane preservation (F1 +7.1%, Recall +10.3%) while also improving image quality (FID +12.3%).

## Model Details

### CLRerNet
- **Architecture:** DLA-34 backbone with FPN
- **Training:** CULane dataset, 800x320 resolution
- **Performance:** F1@0.5 = 81.43% on CULane
- **Reference:** [CLRerNet: Improving Confidence of Lane Detection with LaneIoU](https://github.com/hirotomusiker/CLRerNet)

### Qwen Image Edit
- **Base Model:** Qwen/Qwen-Image-Edit-2509
- **Fine-tuning:** LoRA (rank 16, 3000 steps)
- **Training Resolution:** 1024x1024
- **Inference Resolution:** 1664x928
- **Prompt:** "Replace the green lane detection lines with realistic road lane markings. The final image must not contain green lane lines. Do not remove vehicles, signs, or objects."

### LoRA Training with ai-toolkit

The LoRA weights (`qwen_5.safetensors`) were trained using [ai-toolkit](https://github.com/ostris/ai-toolkit).

- **Pre-trained weights:** [Download from Google Drive](https://drive.google.com/file/d/1Mo16H1ASopDS6AQx2qN_urGmz8AyrIPS/view?usp=sharing)
- **Training dataset:** 900 image pairs (CLRerNet visualizations + ground truth)
- **Training config:** `Qwen Model/config.yaml`

**To train your own LoRA:**

1. Install ai-toolkit:
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Prepare your dataset:
   - `datasets/target/` - Target images (ground truth with real lane markings)
   - `datasets/control/` - Control images (CLRerNet visualizations with green lines)
   - Each image needs a corresponding `.txt` caption file

3. Run training:
```bash
python run.py "path/to/config.yaml"
```

**Training Configuration (`config.yaml`):**
| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen-Image-Edit-2509 |
| LoRA Rank | 16 |
| Optimizer | AdamW8bit |
| Learning Rate | 0.0001 |
| Batch Size | 1 |
| Steps | 3000 |
| Resolution | 1024x1024 |
| Quantization | uint3 (model), qfloat8 (text encoder) |
| Sampler | Flowmatch, 25 steps |

**Training Prompt:**
```
Replace the green lane detection lines with realistic road lane markings.
The final image must not contain green lane lines. Do not remove vehicles, signs, or objects.
```

## Optional: Visibility Boost

The `visibility_boost.py` implements Dark Channel Prior dehazing for adverse weather conditions. Not needed for clean datasets like CULane.

```python
from visibility_boost import dehaze_image_bgr
import cv2

img = cv2.imread("hazy_image.jpg")
dehazed, transmission, A, dark = dehaze_image_bgr(img)
cv2.imwrite("dehazed.jpg", dehazed)
```

## Dataset

The project was evaluated on the [CULane dataset](https://xingangpan.github.io/projects/CULane.html):
- 62,532 training images
- 9 test categories: normal, crowded, dazzle, no-line, shadow, arrow, cross, curve, night

## Team

- **Arunesh Mishra** - CLRerNet pipeline, Visibility Boost, Qwen Research
- **Mrunmay Deshmukh** - Qwen fine-tuning, ComfyUI integration
- **Shreya Muppidi** - LaMa exploration, fine-tuning research
- **Prachi Dudhe** - Documentation, LaMa setup

## References

1. Honda, H., & Uchida, Y. (2024). [CLRerNet: Improving Confidence of Lane Detection with LaneIoU](https://github.com/hirotomusiker/CLRerNet)
2. He, K., Sun, J., & Tang, X. (2009). [Single Image Haze Removal Using Dark Channel Prior](https://ieeexplore.ieee.org/document/5567108)
3. Suvorov, R., et al. (2022). [Resolution-Robust Large Mask Inpainting with Fourier Convolutions (LaMa)](https://github.com/advimman/lama)
4. Qwen Team. [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
5. ostris. [ai-toolkit - LoRA Training Framework](https://github.com/ostris/ai-toolkit)
6. comfyanonymous. [ComfyUI - Diffusion Model GUI/Backend](https://github.com/comfyanonymous/ComfyUI)
7. CULane Dataset. [CULane: A Large Scale Dataset for Lane Detection](https://xingangpan.github.io/projects/CULane.html)
