"""HNeRV-style decoder: 229K params, single-video memorization.

Per-frame-pair latent (28-d) -> 6 upsample stages -> 384x512 RGB pair.

Each stage: Conv(in, out*4, 3x3) + PixelShuffle(2) + bilinear-skip + sin().
Final: dilated-conv refine residual + sigmoid RGB heads (separate frame 0 and 1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.stem = nn.Linear(latent_dim, self.channels[0] * self.base_h * self.base_w)

        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        
        for i in range(6):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            
            # Bottleneck architecture: 1x1 (compress) -> 3x3 (process) -> 1x1 (expand)
            # A minimum channel limit (e.g., 8) prevents the bottleneck from getting too narrow.
            mid_ch = max(in_ch // reduction, 8) 
            
            bottleneck = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=1),
                SineAct(),
                nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
                SineAct(),
                nn.Conv2d(mid_ch, out_ch * 4, kernel_size=1) # *4 for PixelShuffle
            )
            
            self.blocks.append(bottleneck)
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
        x = self.stem(z).view(B, self.channels[0], self.base_h, self.base_w)
        x = torch.sin(x)
        
        for block, skip in zip(self.blocks, self.skips):
            identity = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            identity = skip(identity)
            
            # block(x) now passes through the 1x1->3x3->1x1 bottleneck
            x = self.ps(block(x)) 
            x = torch.sin(x + identity)
            
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        
        return torch.stack([f0, f1], dim=1)