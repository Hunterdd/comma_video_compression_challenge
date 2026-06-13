import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

# Add root folder to path to import utilities
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(HERE))

from frame_utils import camera_size
from lrconv_nerv import HNeRVModel, load_quantized_weights

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <src_weight_file> <dst_raw>")
        sys.exit(1)
        
    src, dst = sys.argv[1], sys.argv[2]
    
    if not os.path.exists(src):
        print(f"ERROR: Weights file {src} not found!")
        sys.exit(1)
        
    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Decoding on device: {device}")
    
    # Load raw quantized dict to read metadata first (N = num_frames)
    print(f"Reading metadata from {src}...")
    quantized_dict = torch.load(src, map_location='cpu')
    metadata = quantized_dict.get('metadata', {})
    N = metadata.get('num_frames', 1200)
    
    target_w, target_h = camera_size
    print(f"Video frames to reconstruct: {N} | Target Resolution: {target_w}x{target_h}")
    
    # Read model configuration from metadata or use defaults
    embed_dim = metadata.get('embed_dim', 16)
    fc_hw = metadata.get('fc_hw', (6, 8))
    dec_strides = metadata.get('dec_strides', [2, 2, 2, 2, 2, 2])
    fc_dim = metadata.get('fc_dim', 128)
    dec_channels = metadata.get('dec_channels', [96, 64, 48, 32, 24, 16])
    ks_dec = metadata.get('ks_dec', 3)
    conv_type = metadata.get('conv_type', 'lrconv')
    bottleneck_ratio = metadata.get('bottleneck_ratio', 0.25)
    
    # Initialize HNeRV model
    model = HNeRVModel(
        num_frames=N,
        embed_dim=embed_dim,
        fc_hw=fc_hw,
        dec_strides=dec_strides,
        fc_dim=fc_dim,
        dec_channels=dec_channels,
        ks_dec=ks_dec,
        conv_type=conv_type,
        bottleneck_ratio=bottleneck_ratio
    ).to(device)
    
    # Load quantized weights
    print(f"Loading weights into model...")
    load_quantized_weights(model, src, device)
    model.eval()
    
    batch_size = 16
    n_written = 0
    
    # Reconstruct frames in batches and write to binary raw file
    with open(dst, 'wb') as f:
        with torch.no_grad():
            for i in range(0, N, batch_size):
                end_idx = min(i + batch_size, N)
                batch_indices = torch.arange(i, end_idx, dtype=torch.long, device=device)
                
                # Reconstruct downscaled frames: (B, 3, 384, 512)
                outputs = model(batch_indices)
                
                # Upscale to original resolution (H, W) = (874, 1164)
                outputs_resized = F.interpolate(
                    outputs,
                    size=(target_h, target_w),
                    mode='bicubic',
                    align_corners=False
                )
                
                # Nudge to red channel
                # outputs_resized[:, 0, :, :].add_(1.0 / 255.0)
                
                # Convert back to uint8 RGB: (B, H, W, 3)
                frames = outputs_resized.clamp(0.0, 1.0) * 255.0
                frames = frames.round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                
                # Dump flat bytes
                f.write(frames.tobytes())
                n_written += len(frames)
                
                if n_written % 160 == 0 or n_written == N:
                    print(f"Decoded {n_written}/{N} frames...")
                    
    print(f"Successfully saved {n_written} raw frames to {dst}")

if __name__ == "__main__":
    main()
