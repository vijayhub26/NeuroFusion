"""
Gradio Web Interface for Brain Tumor Analysis.
Supports Unified Model (Segmentation + Synthesis).
"""

import gradio as gr
import torch
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import io
from PIL import Image

from config import Config
from models.unified_detector import UnifiedTumorDetector
from data import get_brats_dataloaders_2d

# Global variables to hold data
current_sample = None
model = None

def load_model():
    global model
    if model is not None:
        return model, "Model already loaded."
    
    print("Loading model...")
    config = Config()
    model = UnifiedTumorDetector(config)
    
    # Prioritize Unified Segmentation logs
    unified_dir = "logs/unified_segmentation/version_0/checkpoints"
    simple_dir = "logs/simple_tumor_detection/version_2/checkpoints"
    
    checkpoints = []
    
    # 1. Look for Unified Checkpoints first
    if os.path.exists(unified_dir):
        for root, dirs, files in os.walk(unified_dir):
            for file in files:
                if file.endswith(".ckpt"):
                    checkpoints.append(os.path.join(root, file))
    
    # If no unified checkpoints, fall back to simple
    if not checkpoints and os.path.exists(simple_dir):
        for root, dirs, files in os.walk(simple_dir):
            for file in files:
                 if file.endswith(".ckpt"):
                    checkpoints.append(os.path.join(root, file))

    if not checkpoints:
        return None, "Error: No checkpoints found!"
        
    # Sort to find latest/best (assume last one is best/latest)
    latest_ckpt = checkpoints[-1]
    print(f"Loading checkpoint: {latest_ckpt}")
    
    # Load state dict
    try:
        checkpoint = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
    except Exception as e:
         return None, f"Error loading checkpoint: {str(e)}"
    
    # Handle PL state dict keys
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
           
    try: 
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Strict load failed: {e}. Trying non-strict.")
        model.load_state_dict(new_state_dict, strict=False)
        
    model.eval()
    model.eval()
    return f"Model loaded successfully from {os.path.basename(latest_ckpt)}"

def get_random_sample():
    global current_sample
    
    # Initialize dataloader on demand
    config = Config()
    _, val_loader = get_brats_dataloaders_2d(
        data_root=config.data.data_root,
        batch_size=1,
        num_workers=0,
        num_slices_per_scan=5,
        missing_prob=0.0
    )
    
    # fetch a random batch
    iterator = iter(val_loader)
    
    # Initialize batch to ensure it's defined even if loop doesn't run or errors
    try:
        batch = next(iterator)
    except StopIteration:
        # Should not happen on fresh iterator but safe fallback
        return None, None, None, None, None, "Error: Validation set empty?"

    steps = random.randint(0, 5) 
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(val_loader)
            batch = next(iterator)
            break # Stop skipping if we hit end
            
    # Modalities: (1, num_slices, 4, H, W)
    modalities = batch["modalities"][0] 
    seg = batch["seg"][0] 
    
    # Find slice with most tumor
    tumor_pixels = seg.view(seg.shape[0], -1).sum(dim=1)
    if tumor_pixels.max() > 0:
        best_slice_idx = torch.argmax(tumor_pixels).item()
    else:
        best_slice_idx = random.randint(0, modalities.shape[0]-1)
    
    slice_data = modalities[best_slice_idx] # (4, H, W)
    slice_seg = seg[best_slice_idx] # (H, W)
    
    current_sample = {
        "data": slice_data,
        "seg": slice_seg,
        "has_tumor": tumor_pixels[best_slice_idx] > 0
    }
    
    # Return images (normalize)
    def to_img(tensor):
        img = tensor.numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return (img * 255).astype(np.uint8)
    
    t1 = to_img(slice_data[0])
    t1ce = to_img(slice_data[1])
    t2 = to_img(slice_data[2])
    flair = to_img(slice_data[3])
    
    # Create mask overlay on FLAIR
    # Mask is 0 (bg), 1, 2, 3 (tumor parts)
    # We want binary mask > 0
    mask = slice_seg.numpy()
    
    # Create overlay image
    # We'll return just the raw mask for now, let Gradio handle it?
    # Or creating a colored overlay manually might be nicer
    
    status_text = "Sample Loaded. " + ("Tumor Present!" if current_sample["has_tumor"] else "No Tumor.")
    
    return t1, t1ce, t2, flair, mask * 80, status_text # Scale mask for visibility

def process_sample(keep_t1, keep_t1ce, keep_t2, keep_flair):
    global current_sample, model
    
    if current_sample is None:
        return [None]*5 + ["Please load a sample first!", 0.0]
        
    msg = load_model()
    if model is None:
        return [None]*5 + [msg, 0.0]
        
    # Construct mask (1=Keep, 0=Drop)
    mask = torch.tensor([
        1.0 if keep_t1 else 0.0,
        1.0 if keep_t1ce else 0.0,
        1.0 if keep_t2 else 0.0,
        1.0 if keep_flair else 0.0
    ])
    
    input_tensor = current_sample["data"].clone()
    
    # Apply mask
    for i in range(4):
        if mask[i] == 0:
            input_tensor[i] = 0
            
    input_batch = input_tensor.unsqueeze(0)
    mask_batch = mask.unsqueeze(0)
    
    # Run Inference
    with torch.no_grad():
        results = model(input_batch, mask_batch)
        
    # Segmentation
    seg_logits = results["seg_logits"] # (1, 1, H, W)
    pred_mask = (torch.sigmoid(seg_logits) > 0.5).float().squeeze().cpu().numpy()
    
    # Synthesis
    synthesized = results["synthesized_modalities"][0].cpu()
    
    def to_img(tensor):
        img = tensor.numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return (img * 255).astype(np.uint8)
    
    out_t1 = to_img(synthesized[0])
    out_t1ce = to_img(synthesized[1])
    out_t2 = to_img(synthesized[2])
    out_flair = to_img(synthesized[3])
    
    # Cast Mask to uint8
    pred_mask_viz = (pred_mask * 255).astype(np.uint8)

    tumor_size = pred_mask.sum()
    result_text = f"Tumor Detected: {tumor_size} pixels." if tumor_size > 0 else "No Tumor Detected."
    
    return out_t1, out_t1ce, out_t2, out_flair, pred_mask_viz, result_text


# CSS for better layout
css = """
.gradio-container { background-color: #f0f2f6; }
"""

with gr.Blocks(title="NeuroFusion", css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 NeuroFusion: Brain Tumor Analysis")
    gr.Markdown("**Capabilities**: 🔎 Segmentation + ✨ Missing Modality Synthesis")
    
    with gr.Row():
        # Left Column: Input
        with gr.Column(scale=1):
            load_btn = gr.Button("🎲 Load New Patient", variant="primary")
            status = gr.Textbox(label="System Status", value="Ready.")
            
            gr.Markdown("### Input Modalities (Ground Truth)")
            with gr.Row():
                img_t1 = gr.Image(label="T1", type="numpy", height=150)
                img_t1ce = gr.Image(label="T1ce", type="numpy", height=150)
            with gr.Row():
                img_t2 = gr.Image(label="T2", type="numpy", height=150)
                img_flair = gr.Image(label="FLAIR", type="numpy", height=150)
            
            img_gt_mask = gr.Image(label="Ground Truth Tumor", type="numpy", height=150)

        # Middle Column: Controls
        with gr.Column(scale=0.5):
            gr.Markdown("### ⚙️ Simulate Missing Data")
            gr.Markdown("Uncheck to hide modality from AI:")
            chk_t1 = gr.Checkbox(label="T1 Data", value=True)
            chk_t1ce = gr.Checkbox(label="T1ce Data", value=True)
            chk_t2 = gr.Checkbox(label="T2 Data", value=True)
            chk_flair = gr.Checkbox(label="FLAIR Data", value=True)
            
            run_btn = gr.Button("🚀 RUN AI DIAGNOSIS", variant="stop", size="lg")
            
            result_msg = gr.Textbox(label="AI Diagnosis Results")

        # Right Column: Output
        with gr.Column(scale=1):
            gr.Markdown("### AI Reconstruction (Synthesized)")
            with gr.Row():
                syn_t1 = gr.Image(label="Recon T1", type="numpy", height=150)
                syn_t1ce = gr.Image(label="Recon T1ce", type="numpy", height=150)
            with gr.Row():
                syn_t2 = gr.Image(label="Recon T2", type="numpy", height=150)
                syn_flair = gr.Image(label="Recon FLAIR", type="numpy", height=150)
            
            gr.Markdown("### 🎯 Predicted Segmentation")
            pred_mask = gr.Image(label="AI Tumor Mask", type="numpy", height=300)

    # Wiring
    load_btn.click(
        fn=get_random_sample,
        outputs=[img_t1, img_t1ce, img_t2, img_flair, img_gt_mask, status]
    )
    
    run_btn.click(
        fn=process_sample,
        inputs=[chk_t1, chk_t1ce, chk_t2, chk_flair],
        outputs=[syn_t1, syn_t1ce, syn_t2, syn_flair, pred_mask, result_msg]
    )
    
    # Init
    demo.load(fn=load_model, outputs=[status])

if __name__ == "__main__":
    demo.launch(share=True)
