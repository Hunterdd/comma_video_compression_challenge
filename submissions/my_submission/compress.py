import os
import sys
import math
import zipfile
import argparse
from pathlib import Path

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
from lrconv_nerv import HNeRVModel, save_quantized_weights, Muon


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
    args = parser.parse_args()

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

        # Split model parameters for Muon + AdamW
        muon_params = []
        adamw_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Muon for 2D/4D weights (excluding embeddings)
            if p.ndim >= 2 and 'embeddings' not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        optimizer_muon = Muon(muon_params, lr=args.lr, weight_decay=1e-4) if muon_params else None
        optimizer_adamw = optim.AdamW(adamw_params, lr=args.lr * 0.1, weight_decay=1e-4) if adamw_params else None

        scheduler_muon = optim.lr_scheduler.CosineAnnealingLR(optimizer_muon, T_max=args.epochs, eta_min=args.lr * 0.01) if optimizer_muon else None
        scheduler_adamw = optim.lr_scheduler.CosineAnnealingLR(optimizer_adamw, T_max=args.epochs, eta_min=args.lr * 0.1 * 0.01) if optimizer_adamw else None
        
        # Loss functions - We use pixel-level MSE and frozen networks (PoseNet, SegNet)
        # We don't need l1_loss_fn or ssim_loss_fn

        model.train()
        print(f"Starting training on {video_name} for {args.epochs} epochs...")
        
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            
            # Shuffle indices
            indices = torch.randperm(num_frames)
            
            for i in range(0, num_frames, args.batch_size):
                batch_indices_cpu = indices[i:i + args.batch_size]
                batch_indices_device = batch_indices_cpu.to(device)
                
                # Fetch targets on CPU, then send to device
                targets = frames[batch_indices_cpu].to(device).float().permute(0, 3, 1, 2) / 255.0
                
                if optimizer_muon:
                    optimizer_muon.zero_grad()
                if optimizer_adamw:
                    optimizer_adamw.zero_grad()
                
                outputs = model(batch_indices_device)
                
                # Pixel MSE Loss
                loss = F.mse_loss(outputs, targets)
                
                loss.backward()
                
                if optimizer_muon:
                    optimizer_muon.step()
                if optimizer_adamw:
                    optimizer_adamw.step()
                
                epoch_loss += loss.item() * len(batch_indices_cpu)
                
            if scheduler_muon:
                scheduler_muon.step()
            if scheduler_adamw:
                scheduler_adamw.step()
            
            epoch_loss /= num_frames
            
            if epoch % 10 == 0 or epoch == args.epochs:
                print(f"Epoch [{epoch}/{args.epochs}] | Loss: {epoch_loss:.6f}")

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
