import argparse
import os
import glob
import sys
import cv2
import torch
import numpy as np
import tempfile
from tqdm import tqdm
from PIL import Image

# =========================================================
# 1. SETUP PATHS
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description='CLRerNet Inference')
    parser.add_argument('--input', default='../images', help='Path to image OR folder')
    parser.add_argument('--output_dir', default='../images/CLRerNet_Viz', help='Where to save results')
    parser.add_argument('--repo_dir', default='CLRerNet', help='Path to CLRerNet repository')
    parser.add_argument('--config', default='configs/clrernet/culane/clrernet_culane_dla34_ema.py')
    parser.add_argument('--checkpoint', default='clrernet_culane_dla34_ema.pth')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--target_width', type=int, default=1640, help='Resize all images to this width')
    parser.add_argument('--target_height', type=int, default=590, help='Resize all images to this height')
    return parser.parse_args()

args = parse_args()
abs_input = os.path.abspath(args.input)
abs_output = os.path.abspath(args.output_dir)
abs_repo = os.path.abspath(args.repo_dir)
abs_checkpoint = os.path.abspath(args.checkpoint)

print(f"--- CLRerNet Processing ---")
print(f"Repo:       {abs_repo}")
print(f"Device:     {args.device}")

if not os.path.exists(abs_repo):
    sys.exit(f"Error: Repo dir '{abs_repo}' not found.")

sys.path.insert(0, abs_repo)
print(f"Switching context to: {abs_repo}")
os.chdir(abs_repo)

# =========================================================
# 2. NATIVE IMPORTS
# =========================================================
from mmdet.apis import init_detector
from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes

# =========================================================
# 3. PROCESSING
# =========================================================

def resize_to_target(img_path, target_width, target_height):
    """Always resize to target size."""
    with Image.open(img_path) as img:
        img_resized = img.resize((target_width, target_height), Image.LANCZOS)
        
        suffix = os.path.splitext(img_path)[1]
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        img_resized.save(temp_file.name)
        temp_file.close()
        
        return temp_file.name

def save_lane_txt(preds, txt_path):
    with open(txt_path, 'w') as f:
        for lane in preds:
            coords = []
            for (x, y) in lane:
                coords.extend([f"{x:.5f}", f"{y:.5f}"])
            if coords:
                f.write(" ".join(coords) + "\n")

def process_file(model, img_path, viz_dir, txt_dir, target_width, target_height):
    img_name = os.path.basename(img_path)
    img_stem = os.path.splitext(img_name)[0]
    
    temp_path = resize_to_target(img_path, target_width, target_height)
    
    try:
        src, preds = inference_one_image(model, temp_path)
        
        save_viz_path = os.path.join(viz_dir, img_name)
        visualize_lanes(src, preds, save_path=save_viz_path)
        
        save_txt_path = os.path.join(txt_dir, img_stem + ".lines.txt")
        save_lane_txt(preds, save_txt_path)
        
        return True
    except Exception as e:
        print(f"Error on {img_name}: {e}")
        return False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def main():
    print(f"Loading Model on {args.device}...")
    
    model = init_detector(args.config, abs_checkpoint, device=args.device)
    
    viz_dir = os.path.join(abs_output)
    txt_dir = os.path.join(abs_output, "coords")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    
    if os.path.isfile(abs_input):
        img_paths = [abs_input]
    else:
        img_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            img_paths.extend(glob.glob(os.path.join(abs_input, ext)))
            img_paths.extend(glob.glob(os.path.join(abs_input, ext.upper())))
    
    print(f"Processing {len(img_paths)} images...")
    
    for img_path in tqdm(img_paths):
        process_file(model, img_path, viz_dir, txt_dir, args.target_width, args.target_height)

if __name__ == '__main__':
    main()