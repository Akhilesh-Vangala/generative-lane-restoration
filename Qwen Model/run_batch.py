#!/usr/bin/env python3
"""
Batch inference for Qwen Image Edit - Zero-Shot and Fine-Tuned
"""

import os
import sys
from typing import Sequence, Mapping, Any, Union
from pathlib import Path

# ============== CONFIGURATION ==============
SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE_DIR = SCRIPT_DIR.parent.parent

INPUT_DIR = str(WORKSPACE_DIR / "images" / "CLRerNet_Viz")
OUTPUT_DIR = str(WORKSPACE_DIR / "images")
#Fixed Seed
SEED = 42
PROMPT = "Replace the green lane detection lines with realistic road lane markings. The final image must not contain green lane lines. Do not remove vehicles, signs, or objects."
LORA_NAME = "qwen_5.safetensors"
RUN_ZERO_SHOT = True
RUN_FINE_TUNED = True
# ===========================================

import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    if path is None:
        path = os.getcwd()
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name
    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    try:
        from main import load_extra_path_config
    except ImportError:
        from utils.extra_config import load_extra_path_config
    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    import asyncio
    import execution
    from nodes import init_extra_nodes
    sys.path.insert(0, find_path("ComfyUI"))
    import server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    asyncio.run(init_extra_nodes())


from nodes import NODE_CLASS_MAPPINGS


def process_single_image(image_name, output_path, use_lora, vae, clip, unet):
    """Process a single image - either zero-shot or fine-tuned."""
    
    # Load image
    loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
    loadimage_6 = loadimage.load_image(image=image_name)
    
    # Scale image
    imagescaletototalpixels = NODE_CLASS_MAPPINGS["ImageScaleToTotalPixels"]()
    imagescaletototalpixels_7 = imagescaletototalpixels.EXECUTE_NORMALIZED(
        upscale_method="nearest-exact",
        megapixels=1,
        image=get_value_at_index(loadimage_6, 0),
    )
    
    # Get model (with or without LoRA)
    if use_lora:
        print(f"    Applying LoRA: {LORA_NAME}")
        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        loraloadermodelonly_10 = loraloadermodelonly.load_lora_model_only(
            lora_name=LORA_NAME,
            strength_model=1,
            model=get_value_at_index(unet, 0),
        )
        model_for_sampling = get_value_at_index(loraloadermodelonly_10, 0)
    else:
        model_for_sampling = get_value_at_index(unet, 0)
    
    # VAE Encode
    vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
    vaeencode_13 = vaeencode.encode(
        pixels=get_value_at_index(imagescaletototalpixels_7, 0),
        vae=get_value_at_index(vae, 0),
    )
    
    # Text encode positive
    print("    Encoding prompts...")
    textencodeqwenimageedit = NODE_CLASS_MAPPINGS["TextEncodeQwenImageEdit"]()
    textencodeqwenimageedit_8 = textencodeqwenimageedit.EXECUTE_NORMALIZED(
        prompt=PROMPT,
        clip=get_value_at_index(clip, 0),
        vae=get_value_at_index(vae, 0),
        image=get_value_at_index(imagescaletototalpixels_7, 0),
    )
    
    # Text encode negative (empty)
    textencodeqwenimageedit_11 = textencodeqwenimageedit.EXECUTE_NORMALIZED(
        prompt="",
        clip=get_value_at_index(clip, 0),
        vae=get_value_at_index(vae, 0),
        image=get_value_at_index(imagescaletototalpixels_7, 0),
    )
    
    # Model sampling
    print("    Applying ModelSamplingAuraFlow...")
    modelsamplingauraflow = NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()
    modelsamplingauraflow_12 = modelsamplingauraflow.patch_aura(
        shift=3,
        model=model_for_sampling,
    )
    
    # CFG Norm
    print("    Applying CFGNorm...")
    cfgnorm = NODE_CLASS_MAPPINGS["CFGNorm"]()
    cfgnorm_14 = cfgnorm.EXECUTE_NORMALIZED(
        strength=1,
        model=get_value_at_index(modelsamplingauraflow_12, 0),
    )
    
    # Sample
    print(f"    Sampling (seed={SEED})...")
    ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
    ksampler_15 = ksampler.sample(
        seed=SEED,
        steps=25,
        cfg=2.5,
        sampler_name="euler",
        scheduler="simple",
        denoise=1,
        model=get_value_at_index(cfgnorm_14, 0),
        positive=get_value_at_index(textencodeqwenimageedit_8, 0),
        negative=get_value_at_index(textencodeqwenimageedit_11, 0),
        latent_image=get_value_at_index(vaeencode_13, 0),
    )
    
    # Decode
    print("    Decoding...")
    vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
    vaedecode_16 = vaedecode.decode(
        samples=get_value_at_index(ksampler_15, 0),
        vae=get_value_at_index(vae, 0),
    )
    
    # Save using PIL (to custom path)
    print(f"    Saving to: {output_path}")
    from PIL import Image
    import numpy as np
    img_tensor = get_value_at_index(vaedecode_16, 0)
    img_array = (img_tensor.cpu().numpy()[0] * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, quality=95)


def main():
    import_custom_nodes()
    
    # Setup output directories
    zero_shot_dir = Path(OUTPUT_DIR) / "Zero-Shot"
    fine_tuned_dir = Path(OUTPUT_DIR) / "Fine-Tuned"
    zero_shot_dir.mkdir(parents=True, exist_ok=True)
    fine_tuned_dir.mkdir(parents=True, exist_ok=True)
    
    # Get images
    input_path = Path(INPUT_DIR)
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = sorted([f for f in input_path.iterdir() if f.suffix.lower() in extensions])
    
    print(f"Found {len(images)} images")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Seed: {SEED}")
    print(f"LoRA: {LORA_NAME}")
    print("=" * 60)
    
    with torch.inference_mode():
        # Load models once
        print("\nLoading models...")
        
        print("  Loading VAE...")
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vae = vaeloader.load_vae(vae_name="qwen_image_vae.safetensors")
        
        print("  Loading CLIP...")
        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        clip = cliploader.load_clip(
            clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors",
            type="qwen_image",
            device="default",
        )
        
        print("  Loading UNET...")
        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unet = unetloader.load_unet(
            unet_name="qwen_image_edit_fp8_e4m3fn.safetensors",
            weight_dtype="fp8_e4m3fn",
        )
        
        print("Models loaded!\n")
        
        # Process each image
        for i, img_path in enumerate(images, 1):
            print(f"\n[{i}/{len(images)}] {img_path.name}")
            
            # Copy image to ComfyUI input folder
            import shutil
            comfyui_input = Path(find_path("ComfyUI")) / "input"
            comfyui_input.mkdir(exist_ok=True)
            shutil.copy(str(img_path), str(comfyui_input / img_path.name))
            
            # Zero-Shot (no LoRA)
            if RUN_ZERO_SHOT:
                print("\n  --- Zero-Shot ---")
                process_single_image(
                    image_name=img_path.name,
                    output_path=str(zero_shot_dir / img_path.name),
                    use_lora=False,
                    vae=vae,
                    clip=clip,
                    unet=unet,
                )
            
            # Fine-Tuned (with LoRA)
            if RUN_FINE_TUNED:
                print("\n  --- Fine-Tuned ---")
                process_single_image(
                    image_name=img_path.name,
                    output_path=str(fine_tuned_dir / img_path.name),
                    use_lora=True,
                    vae=vae,
                    clip=clip,
                    unet=unet,
                )
    
    print("\n" + "=" * 60)
    print("Done!")
    print(f"Zero-Shot outputs: {zero_shot_dir}")
    print(f"Fine-Tuned outputs: {fine_tuned_dir}")


if __name__ == "__main__":
    main()
