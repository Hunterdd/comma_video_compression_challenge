import os
import sys
import math
import zipfile
import argparse
from pathlib import Path
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Add root folder to path to import utilities
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(HERE))

import av
from frame_utils import yuv420_to_rgb
from lrconv_nerv import HNeRVModel, save_quantized_weights


# Quantization Aware Training (QAT) with Straight-Through Estimator (STE)
def fake_quantize(tensor, n_levels=255):
    v_min = tensor.min()
    v_max = tensor.max()
    if v_max == v_min:
        return tensor
    scale = n_levels / (v_max - v_min)
    q = ((tensor - v_min) * scale).round().clamp(0, n_levels)
    dq = v_min + q / scale
    return (dq - tensor).detach() + tensor

def apply_qat(model):
    originals = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.is_floating_point():
            originals[name] = p.data.clone()
            p.data.copy_(fake_quantize(p.data))
    return originals

def restore_qat(model, originals):
    for name, p in model.named_parameters():
        if name in originals:
            p.data.copy_(originals[name])


# Loader logic
def load_video_frames(video_path, target_size=(384, 512)):
    print(f"Loading video {video_path}...")
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = []
    
    for frame in container.decode(stream):
        arr = yuv420_to_rgb(frame)  # (H, W, C) uint8
        
        # Resize to target size for training
        x = arr.permute(2, 0, 1).unsqueeze(0).float()
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        arr = x.squeeze(0).permute(1, 2, 0).to(torch.uint8)
        frames.append(arr)
        
    container.close()
    frames_tensor = torch.stack(frames) # (N, H, W, C)
    print(f"Loaded {len(frames_tensor)} frames. Shape: {frames_tensor.shape}")
    return frames_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, default=str(ROOT / "videos"))
    parser.add_argument("--archive-dir", type=str, default=str(HERE / "archive"))
    parser.add_argument("--video-names-file", type=str, default=str(ROOT / "public_test_video_names.txt"))
    parser.add_argument("--epochs", type=int, default=150, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size for overfitting")
    parser.add_argument("--embed-dim", type=int, default=16, help="HNeRV frame embedding dimension")
    parser.add_argument("--fc-dim", type=int, default=128, help="MLP projected starting channels")
    parser.add_argument("--bottleneck-ratio", type=float, default=0.25, help="LRConv bottleneck ratio")
    parser.add_argument("--ft-epochs", type=int, default=30, help="number of epochs for QAT + sqrt loss fine-tuning")
    parser.add_argument("--seed", type=int, default=1234, help="random seed for reproducibility")
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Training on device: {device}")

    # No distortion network needed (training with Pixel MSE only)

    # Read target videos
    with open(args.video_names_file, "r") as f:
        video_names = [line.strip() for line in f.readlines() if line.strip()]

    # Ensure archive directory exists
    os.makedirs(args.archive_dir, exist_ok=True)

    # Spatial configuration (Strides [2, 2, 2, 2, 2, 2] maps 7x9 to 448x576)
    fc_hw = (7, 9)
    dec_strides = [2, 2, 2, 2, 2, 2]
    dec_channels = [128, 96, 64, 48, 32, 24]
    ks_dec = 3

    for video_name in video_names:
        video_path = Path(args.in_dir) / video_name
        if not video_path.exists():
            print(f"Skipping {video_path} (not found)")
            continue

        # Load video frames (resizing to 384x512)
        frames = load_video_frames(video_path, target_size=(448,576))
        num_frames = len(frames)

        # Initialize HNeRV model with LRConv-NeRV decoder
        model = HNeRVModel(
            num_frames=num_frames,
            embed_dim=args.embed_dim,
            fc_hw=fc_hw,
            dec_strides=dec_strides,
            fc_dim=args.fc_dim,
            dec_channels=dec_channels,
            ks_dec=ks_dec,
            conv_type='lrconv',
            bottleneck_ratio=args.bottleneck_ratio
        ).to(device)

        # Calculate number of parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model initialized. Total trainable parameters: {num_params:,} (~{num_params * 4 / 1024:.1f} KB in float32, ~{num_params / 1024:.1f} KB quantized)")

        # Optimizer and Scheduler (Standard AdamW + Cosine Annealing)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

        base_epochs = max(0, args.epochs - args.ft_epochs)
        model.train()
        print(f"Starting training on {video_name} for {args.epochs} epochs ({base_epochs} Base epochs + {args.epochs - base_epochs} FT epochs)...")
        
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            is_ft = (epoch > base_epochs)
            
            # Shuffle indices
            indices = torch.randperm(num_frames)
            
            for i in range(0, num_frames, args.batch_size):
                batch_indices_cpu = indices[i:i + args.batch_size]
                batch_indices_device = batch_indices_cpu.to(device)
                
                # Fetch targets on CPU, then send to device
                targets = frames[batch_indices_cpu].to(device).float().permute(0, 3, 1, 2) / 255.0
                
                if optimizer:
                    optimizer.zero_grad()
                
                # Apply QAT fake-quantization to weights before forward pass only in FT stage
                if is_ft:
                    originals = apply_qat(model)
                
                outputs = model(batch_indices_device)
                
                # Compute standard MSE
                mse = F.mse_loss(outputs, targets)
                
                # Use PoseNet-aligned square-root loss in FT stage, standard MSE in base stage
                if is_ft:
                    # Add 1e-12 to prevent NaN errors when taking the derivative of a square root near zero
                    loss = torch.sqrt(10.0 * mse + 1e-12)
                else:
                    loss = mse
                
                loss.backward()
                
                # Restore original full-precision weights before optimizer step only in FT stage
                if is_ft:
                    restore_qat(model, originals)
                
                # Gradient norm clipping to prevent NaNs/exploding gradients under the square-root loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if optimizer:
                    optimizer.step()
                
                epoch_loss += loss.item() * len(batch_indices_cpu)
                
            if scheduler:
                scheduler.step()
            
            epoch_loss /= num_frames
            
            if epoch % 10 == 0 or epoch == args.epochs:
                stage_str = "FT (QAT+Sqrt)" if is_ft else "Base (MSE)"
                lr_curr = optimizer.param_groups[0]['lr']
                # Calculate gradient norm for debugging / sanity checking
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                print(f"Epoch [{epoch}/{args.epochs}] ({stage_str}) | Loss: {epoch_loss:.6f} | LR: {lr_curr:.6f} | Grad Norm: {grad_norm:.6f}")

        # Quantize and save model state dict along with metadata
        model.eval()
        base_name = Path(video_name).stem
        weight_file = Path(args.archive_dir) / f"{base_name}.pth"
        print(f"Quantizing and saving weights to {weight_file}...")
        metadata = {
            'num_frames': num_frames,
            'embed_dim': args.embed_dim,
            'fc_hw': fc_hw,
            'dec_strides': dec_strides,
            'fc_dim': args.fc_dim,
            'dec_channels': dec_channels,
            'ks_dec': ks_dec,
            'conv_type': 'lrconv',
            'bottleneck_ratio': args.bottleneck_ratio
        }
        save_quantized_weights(model.state_dict(), metadata, weight_file)

    # Package into archive.zip
    archive_zip = HERE / "archive.zip"
    print(f"Packaging archive.zip at {archive_zip}...")
    with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in Path(args.archive_dir).glob("*.pth"):
            zipf.write(file, arcname=file.name)
    print("Compression complete!")

if __name__ == "__main__":
    main()
