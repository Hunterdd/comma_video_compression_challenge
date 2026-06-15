"""HNeRV-style decoder: 229K params, single-video memorization.

Per-frame-pair latent (28-d) -> 6 upsample stages -> 384x512 RGB pair.

Each stage: Conv(in, out*4, 3x3) + PixelShuffle(2) + bilinear-skip + sin().
Final: dilated-conv refine residual + sigmoid RGB heads (separate frame 0 and 1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_divisible(v, divisor=4, min_value=None):
    """Ensures that all channel counts are cleanly divisible by the divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SineAct(nn.Module):
    """Custom Sine Activation layer for use inside nn.Sequential"""
    def forward(self, x):
        return torch.sin(x)

class HNeRVDecoder(nn.Module):
    def __init__(self, latent_dim=28, base_channels=36, eval_size=(384, 512), reduction=2):
        super().__init__()
        self.eval_size = eval_size
        self.base_h, self.base_w = 6, 8
        C = base_channels

        # 7 stages from 6x8 to 384x512; channel taper matches HNeRV paper
        self.channels = [C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)]

        # In your __init__:
        stem_channels = 12 # Project to 1/3rd of the base channels initially
        self.stem_linear = nn.Linear(latent_dim, stem_channels * self.base_h * self.base_w)
        self.stem_expand = nn.Conv2d(stem_channels, self.channels[0], kernel_size=1)
        
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        
        for i in range(6):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            
            # --- PROGRESSIVE APPLICATION ---
            # Keep early stages (i < 3) dense to preserve structural information.
            # Apply LRConv only to the later, computationally expensive stages.
            if i < 2:
                block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1)
                )
            else:
                # --- LRConv SPATIAL BOTTLENECK ---
                mid_ch = max(in_ch // reduction, 8) 
                
                block = nn.Sequential(
                    # 1. Compress spatially and channel-wise (3x1 vertical filter)
                    nn.Conv2d(in_ch, mid_ch, kernel_size=(3, 1), padding=(1, 0)),
                    SineAct(),
                    # 2. Expand spatially and channel-wise (1x3 horizontal filter)
                    # Inside your LRConv bottleneck (for i >= 2):
                    nn.Conv2d(mid_ch, out_ch * 4, kernel_size=(1, 3), padding=(0, 1), groups=2)
                )
            
            self.blocks.append(block)
            self.skips.append(nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())
            
        self.ps = nn.PixelShuffle(2)

        final_ch = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, 3, padding=2, dilation=2),
            nn.Conv2d(final_ch // 2, final_ch, 3, padding=1),
        )
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    def forward(self, z):
        B = z.shape[0]
        x = self.stem_linear(z).view(B, 12, self.base_h, self.base_w)
        x = self.stem_expand(x) # Expands from 12 back to 36 channels
        x = torch.sin(x)
        
        for block, skip in zip(self.blocks, self.skips):
            identity = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            identity = skip(identity)
            
            x = self.ps(block(x)) 
            x = torch.sin(x + identity)
            
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        
        return torch.stack([f0, f1], dim=1)

        
class HNeRVDecoder_grouped(nn.Module):
    def __init__(self, latent_dim=28, base_channels=36, eval_size=(384, 512), reduction=2):
        super().__init__()
        self.eval_size = eval_size
        self.base_h, self.base_w = 6, 8
        C = base_channels

        # Enforce divisibility by 4 on all channel counts to prevent grouping runtime errors
        self.channels = [
            C, 
            C, 
            C, 
            make_divisible(C * 0.75),   # 27 becomes 28
            make_divisible(C * 0.58),   # 20 stays 20
            make_divisible(C * 0.5),    # 18 becomes 20
            make_divisible(C * 0.5)     # 18 becomes 20
        ]

        # STRATEGY 1: Factorized Stem (Saves ~32k parameters instantly)
        stem_channels = 12 
        self.stem_linear = nn.Linear(latent_dim, stem_channels * self.base_h * self.base_w)
        self.stem_expand = nn.Conv2d(stem_channels, self.channels[0], kernel_size=1)

        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        
        for i in range(6):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            
            # Shifting progressive boundary to 2 (as per your successful test)
            if i < 2:
                block = nn.Sequential(
                    # nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1)
                    nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1, groups=4)
                )
            else:
                mid_ch = max(in_ch // reduction, 8)
                # Ensure mid_ch is also divisible by our groups parameter (2)
                mid_ch = make_divisible(mid_ch, divisor=2) 
                
                # STRATEGY 2: Spatial Bottleneck with Grouped Expansion
                block = nn.Sequential(
                    # Vertical spatial filter
                    nn.Conv2d(in_ch, mid_ch, kernel_size=(3, 1), padding=(1, 0)),
                    SineAct(),
                    # Horizontal spatial filter + Grouped channel expansion (Saves ~30% per block)
                    nn.Conv2d(mid_ch, out_ch * 4, kernel_size=(1, 3), padding=(0, 1), groups=2)
                )
            
            self.blocks.append(block)
            self.skips.append(nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())
            
        self.ps = nn.PixelShuffle(2)

        final_ch = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, 3, padding=2, dilation=2),
            nn.Conv2d(final_ch // 2, final_ch, 3, padding=1),
        )
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    def forward(self, z):
        B = z.shape[0]
        # Factorized stem forward pass
        x = self.stem_linear(z).view(B, 12, self.base_h, self.base_w)
        x = self.stem_expand(x)
        x = torch.sin(x)
        
        for block, skip in zip(self.blocks, self.skips):
            identity = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            identity = skip(identity)
            
            x = self.ps(block(x)) 
            x = torch.sin(x + identity)
            
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        
        return torch.stack([f0, f1], dim=1)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SineAct(nn.Module):
#     """Custom Sine Activation layer for use inside nn.Sequential"""
#     def forward(self, x):
#         return torch.sin(x)

