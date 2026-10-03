# LAMA Inpainting for Lane Detection

This folder contains experiments with the LAMA (Large Mask Inpainting) model for lane detection inpainting. Although the final project used the Qwen model, these experiments demonstrate the exploration of LAMA as an alternative approach.

## 📁 Folder Structure

```
Lama/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── notebooks/
│   ├── lama-test.ipynb               # Inpainting with pre-generated lane masks
│   └── lama-test-manual-mask.ipynb   # Inpainting with interactive manual masks
├── saicinpainting/                    # Core LAMA package
├── models/
│   └── big-lama/                      # Pre-trained LAMA model weights (~381MB)
│       ├── config.yaml
│       └── models/best.ckpt
└── data/
    ├── sample_images/                 # Sample road images for testing
    └── sample_masks/                  # Corresponding lane masks
```

## 🚀 Quick Start

### 1. Environment Setup

Create a conda environment with Python 3.9:

```bash
conda create -n lama python=3.9
conda activate lama
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with PyTorch, install it separately first:

```bash
# For CPU
pip install torch torchvision torchaudio

# For CUDA (adjust version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Download Perceptual Loss Weights (Optional)

The notebooks use perceptual loss weights. Download them if needed:

```bash
mkdir -p ~/.cache/torch/ade20k/ade20k-resnet50dilated-ppm_deepsup
wget -P ~/.cache/torch/ade20k/ade20k-resnet50dilated-ppm_deepsup/ \
  http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
```

### 4. Run the Notebooks

Navigate to the notebooks directory and launch Jupyter:

```bash
cd Lama
cd notebooks
jupyter notebook
```

## 📓 Notebooks

### `lama-test.ipynb`
- **Purpose**: Demonstrates LAMA inpainting with pre-generated lane masks
- **Input**: Images from `data/sample_images/` and masks from `data/sample_masks/`
- **Process**: 
  - Loads images and corresponding lane masks
  - Dilates masks for better coverage
  - Runs LAMA inpainting to remove lane markings
  - Saves results with visualization
- **Output**: Inpainted images in `data/output/`

### `lama-test-manual-mask.ipynb`
- **Purpose**: Interactive mask drawing for custom inpainting
- **Input**: Images from `data/sample_images/`
- **Process**:
  - Opens each image in an interactive window
  - Draw mask with mouse (left-click and drag)
  - Press 's' to save and proceed, 'r' to reset, 'q' to quit
  - Runs LAMA inpainting on drawn mask
- **Output**: Inpainted images in `data/output_manual/`

## 🔧 Adding Your Own Data

To test with your own images:

1. **For pre-generated masks** (lama-test.ipynb):
   - Place images in `data/sample_images/`
   - Place corresponding masks in `data/sample_masks/`
   - Mask naming convention: `{image_name}_full.png`
   - Example: `image001.jpg` → `image001_full.png`

2. **For manual masks** (lama-test-manual-mask.ipynb):
   - Simply place images in `data/sample_images/`
   - Draw masks interactively when running the notebook

## 📊 Model Information

- **Model**: LAMA (Large Mask Inpainting with Fourier Convolutions)
- **Weights**: big-lama (trained on Places365-Challenge dataset)
- **Size**: ~381MB
- **Resolution**: Trained on 256x256, generalizes to higher resolutions

## 🔗 References

- **Original LAMA Repository**: [advimman/lama](https://github.com/advimman/lama)
- **Paper**: [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161)
- **Project Page**: [https://advimman.github.io/lama-project/](https://advimman.github.io/lama-project/)

## 📝 Notes

- The notebooks use relative paths, so they should work out of the box after environment setup
- GPU is recommended but not required (will use CPU if CUDA is not available)
- Sample images are from the CULane dataset used in lane detection experiments
- The model can handle various mask sizes and shapes

## ⚠️ Troubleshooting

### Import Errors
If you get import errors for `saicinpainting`, make sure you're running the notebook from the `notebooks/` directory, as the notebooks add the parent directory to the Python path.

### CUDA Out of Memory
If you run out of GPU memory, the model will automatically fall back to CPU. You can also reduce the number of images processed at once.

### Missing Perceptual Loss Weights
If the perceptual loss weights are not found, the model will still work but may show warnings. Download them using the command in step 3 above.

