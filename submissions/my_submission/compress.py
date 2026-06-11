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
from lrconv_nerv import HNeRVModel, save_quantized_weights

# SSIM Loss Implementation in PyTorch
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        
    def forward(self, img1, img2):
        return 1 - ssim(img1, img2, self.window_size, self.size_average)


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

    # Read target videos
    with open(args.video_names_file, "r") as f:
        video_names = [line.strip() for line in f.readlines() if line.strip()]

    # Ensure archive directory exists
    os.makedirs(args.archive_dir, exist_ok=True)

    # Spatial configuration (Strides [2, 2, 2, 2, 2, 2] maps 6x8 to 384x512)
    fc_hw = (6, 8)
    dec_strides = [2, 2, 2, 2, 2, 2]
    dec_channels = [96, 64, 48, 32, 24, 16]
    ks_dec = 3

    for video_name in video_names:
        video_path = Path(args.in_dir) / video_name
        if not video_path.exists():
            print(f"Skipping {video_path} (not found)")
            continue

        # Load video frames (resizing to 384x512)
        frames = load_video_frames(video_path, target_size=(384, 512))
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

        # Optimizer and schedulers
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        
        # Loss functions
        l1_loss_fn = nn.L1Loss()
        ssim_loss_fn = SSIMLoss()

        model.train()
        print(f"Starting training on {video_name} for {args.epochs} epochs...")
        
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            epoch_l1 = 0.0
            epoch_ssim = 0.0
            
            # Shuffle indices
            indices = torch.randperm(num_frames)
            
            for i in range(0, num_frames, args.batch_size):
                batch_indices = indices[i:i + args.batch_size].to(device)
                
                # Fetch targets and permute to NCHW, normalized to [0, 1]
                targets = frames[batch_indices].to(device).float().permute(0, 3, 1, 2) / 255.0
                
                optimizer.zero_grad()
                outputs = model(batch_indices)
                
                # Hybrid L1 + SSIM Loss
                loss_l1 = l1_loss_fn(outputs, targets)
                loss_ssim = ssim_loss_fn(outputs, targets)
                loss = 0.15 * loss_l1 + 0.85 * loss_ssim
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_indices)
                epoch_l1 += loss_l1.item() * len(batch_indices)
                epoch_ssim += (1 - loss_ssim.item()) * len(batch_indices)
                
            scheduler.step()
            
            epoch_loss /= num_frames
            epoch_l1 /= num_frames
            epoch_ssim /= num_frames
            
            if epoch % 10 == 0 or epoch == args.epochs:
                print(f"Epoch [{epoch}/{args.epochs}] | Loss: {epoch_loss:.6f} | L1: {epoch_l1:.6f} | SSIM: {epoch_ssim:.4f}")

        # Quantize and save model state dict along with metadata
        model.eval()
        base_name = Path(video_name).stem
        weight_file = Path(args.archive_dir) / f"{base_name}.pth"
        print(f"Quantizing and saving weights to {weight_file}...")
        metadata = {'num_frames': num_frames}
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
