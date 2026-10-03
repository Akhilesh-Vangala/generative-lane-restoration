import torch
import lpips
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
import os
import argparse
import glob
from pytorch_fid import fid_score

# ============================================
# HELPER FUNCTIONS
# ============================================

def load_image_as_tensor(image_path, target_size=None):
    img = Image.open(image_path).convert('RGB')
    if target_size is not None:
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    return transform(img).unsqueeze(0)

def load_image_as_array(image_path, target_size=None):
    img = Image.open(image_path).convert('RGB')
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
    return np.array(img)

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

def get_common_images(target_dir, zero_dir, fine_dir):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    def get_basenames(directory):
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(directory, ext)))
            files.extend(glob.glob(os.path.join(directory, ext.upper())))
        return set(os.path.basename(f) for f in files)

    targets = get_basenames(target_dir)
    zeros = get_basenames(zero_dir)
    fines = get_basenames(fine_dir)

    common = sorted(list(targets & zeros & fines))
    
    if len(common) == 0:
        print("WARNING: No matching image filenames found across all three folders!")
        print(f"  Target: {target_dir}")
        print(f"  Zero-Shot: {zero_dir}")
        print(f"  Fine-Tuned: {fine_dir}")
    return common

# ============================================
# METRIC CALCULATIONS (LPIPS, SSIM, FID)
# ============================================

def calculate_lpips(model_dir, target_dir, image_names):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_scores = []
    
    print(f"  Processing {len(image_names)} images for LPIPS...")
    for img_name in image_names:
        model_path = os.path.join(model_dir, img_name)
        target_path = os.path.join(target_dir, img_name)
        target_size = get_image_size(target_path)
        
        model_img = load_image_as_tensor(model_path, target_size=(target_size[1], target_size[0])).to(device)
        target_img = load_image_as_tensor(target_path, target_size=None).to(device)
        
        with torch.no_grad():
            distance = lpips_model(model_img, target_img)
        lpips_scores.append(distance.item())
    return np.mean(lpips_scores), lpips_scores

def calculate_ssim(model_dir, target_dir, image_names):
    ssim_scores = []
    print(f"  Processing {len(image_names)} images for SSIM...")
    for img_name in image_names:
        model_path = os.path.join(model_dir, img_name)
        target_path = os.path.join(target_dir, img_name)
        target_size = get_image_size(target_path)
        
        model_img = load_image_as_array(model_path, target_size=target_size)
        target_img = load_image_as_array(target_path, target_size=None)
        
        if model_img.shape != target_img.shape:
            model_img = np.array(Image.fromarray(model_img).resize(
                (target_img.shape[1], target_img.shape[0]), Image.LANCZOS
            ))
        
        min_dim = min(model_img.shape[0], model_img.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        
        score = ssim(model_img, target_img, channel_axis=2, data_range=255, win_size=win_size)
        ssim_scores.append(score)
    return np.mean(ssim_scores), ssim_scores

def calculate_fid_simple(dir1, dir2):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fid_value = fid_score.calculate_fid_given_paths([dir1, dir2], batch_size=4, device=device, dims=2048, num_workers=0)
        return fid_value
    except Exception as e:
        print(f"    ERROR calculating FID: {e}")
        return None

# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Lane Restoration Metrics')
    
    # Defaults: Look for folders in the same directory as this script (root workspace)
    parser.add_argument('--target_dir', default='./images/Targets', help='Ground Truth Images')
    parser.add_argument('--zero_shot_dir', default='./images/Zero-Shot', help='Zero Shot Results')
    parser.add_argument('--finetuned_dir', default='./images/Fine-Tuned', help='Fine Tuned Results')
    
    args = parser.parse_args()
    target_dir = os.path.abspath(args.target_dir)
    zero_dir = os.path.abspath(args.zero_shot_dir)
    fine_dir = os.path.abspath(args.finetuned_dir)

    print(f"--- Configuration ---")
    print(f"Targets:    {target_dir}")
    print(f"Zero-Shot:  {zero_dir}")
    print(f"Fine-Tuned: {fine_dir}")

    for d in [target_dir, zero_dir, fine_dir]:
        if not os.path.exists(d):
            print(f"\nError: Directory not found: {d}")
            print("Ensure you have run inference and created 'images_folder' in workspace.")
            return

    test_images = get_common_images(target_dir, zero_dir, fine_dir)
    if not test_images: return

    print(f"\nFound {len(test_images)} common images.")

    # ZERO-SHOT
    print("\n" + "="*40 + "\n EVALUATING ZERO-SHOT\n" + "="*40)
    lpips_zero, _ = calculate_lpips(zero_dir, target_dir, test_images)
    ssim_zero, _ = calculate_ssim(zero_dir, target_dir, test_images)
    # Dont use FID for a single image
    fid_zero = calculate_fid_simple(zero_dir, target_dir)

    # FINE-TUNED
    print("\n" + "="*40 + "\n EVALUATING FINE-TUNED\n" + "="*40)
    lpips_ft, _ = calculate_lpips(fine_dir, target_dir, test_images)
    ssim_ft, _ = calculate_ssim(fine_dir, target_dir, test_images)
    # Dont use FID for a single image
    fid_ft = calculate_fid_simple(fine_dir, target_dir)

    # RESULTS
    imp_lpips = ((lpips_zero - lpips_ft) / lpips_zero) * 100
    imp_ssim = ((ssim_ft - ssim_zero) / ssim_zero) * 100
    
    fid_z_str = f"{fid_zero:.2f}" if fid_zero else "N/A"
    fid_f_str = f"{fid_ft:.2f}" if fid_ft else "N/A"
    imp_fid_str = f"{((fid_zero - fid_ft) / fid_zero) * 100:+.1f}%" if (fid_zero and fid_ft) else "N/A"

    print("\n" + "="*60 + "\n FINAL RESULTS TABLE\n" + "="*60)
    print(f"""
| Metric    | Zero-Shot | Fine-Tuned | Improvement |
|-----------|-----------|------------|-------------|
| LPIPS ↓   | {lpips_zero:.4f}    | {lpips_ft:.4f}     | {imp_lpips:+.1f}%      |
| FID ↓     | {fid_z_str}      | {fid_f_str}      | {imp_fid_str}      |
| SSIM ↑    | {ssim_zero:.4f}    | {ssim_ft:.4f}     | {imp_ssim:+.1f}%      |
    """)

if __name__ == "__main__":
    main()