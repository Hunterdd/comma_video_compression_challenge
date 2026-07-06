"""CompactTINCHNeRV decoder with FiLM conditioning and chunk-based architecture.

Single network mapping entire 600 frame-pair video using 4 chunk branches/leaves.
Base channels=27 (~186K params) with FiLM conditioning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TINCStage(nn.Module):
    """TINC stage: Conv3x3 + PixelShuffle + Conv1x1/Identity skip."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        identity = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        identity = self.skip(identity)
        x = self.ps(self.conv(x))
        return torch.sin(x + identity)


class CompactTINCHNeRV(nn.Module):
    def __init__(self, latent_dim=28, base_channels=27, eval_size=(384, 512)):
        super().__init__()
        self.eval_size = eval_size
        self.base_h, self.base_w = 6, 8
        C = base_channels

        # Stem
        self.stem = nn.Linear(latent_dim, C * self.base_h * self.base_w)

        # 2 shared root stages
        self.root_stages = nn.ModuleList([
            TINCStage(C, C),
            TINCStage(C, C),
        ])

        # Level 1: 2 branches of 2 stages each
        self.lvl1_stages = nn.ModuleList([
            nn.ModuleList([TINCStage(C, C), TINCStage(C, C)]) for _ in range(2)
        ])

        # Level 2: 4 chunks of 2 stages each
        self.lvl2_stages_5 = nn.ModuleList([
            TINCStage(C, C) for _ in range(4)
        ])
        self.lvl2_stages_6 = nn.ModuleList([
            TINCStage(C, int(C * 0.75)) for _ in range(4)
        ])

        # Refinement layers: 4 chunks of Conv2d(20->10) + Conv2d(10->10)
        final_ch = int(C * 0.75)  # This is 20 for base_channels=27
        self.refines = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(final_ch, 10, 3, padding=1),  # 20 -> 10
                nn.Conv2d(10, 10, 3, padding=1)         # 10 -> 10
            ) for _ in range(4)
        ])

        # RGB heads: 4 chunks each
        self.rgb_0_heads = nn.ModuleList([
            nn.Conv2d(10, 3, 3, padding=1) for _ in range(4)
        ])
        self.rgb_1_heads = nn.ModuleList([
            nn.Conv2d(10, 3, 3, padding=1) for _ in range(4)
        ])

        # FiLM conditioning embeddings
        self.film_gamma0 = nn.Embedding(4, C)
        self.film_beta0 = nn.Embedding(4, C)
        self.film_gamma1 = nn.Embedding(4, C)
        self.film_beta1 = nn.Embedding(4, C)

    def forward(self, z, chunk_id):
        """
        Args:
            z: (B, latent_dim) latent codes
            chunk_id: (1,) int tensor indicating which chunk [0,1,2,3]
        """
        B = z.shape[0]
        chunk_idx = chunk_id.item()

        # Stem
        x = self.stem(z).view(B, -1, self.base_h, self.base_w)
        x = torch.sin(x)

        # Apply FiLM conditioning after stem
        gamma0 = self.film_gamma0(chunk_id).view(1, -1, 1, 1)
        beta0 = self.film_beta0(chunk_id).view(1, -1, 1, 1)
        x = x * gamma0 + beta0

        # Root stages (shared)
        for stage in self.root_stages:
            x = stage(x)

        # Apply second FiLM conditioning
        gamma1 = self.film_gamma1(chunk_id).view(1, -1, 1, 1)
        beta1 = self.film_beta1(chunk_id).view(1, -1, 1, 1)
        x = x * gamma1 + beta1

        # Level 1 branches (choose based on chunk_id)
        branch_idx = chunk_idx // 2  # chunks 0,1 -> branch 0; chunks 2,3 -> branch 1
        for stage in self.lvl1_stages[branch_idx]:
            x = stage(x)

        # Level 2 stages (chunk-specific)
        x = self.lvl2_stages_5[chunk_idx](x)
        x = self.lvl2_stages_6[chunk_idx](x)

        # Refinement (chunk-specific)
        x = x + 0.1 * torch.sin(self.refines[chunk_idx](x))

        # RGB heads (chunk-specific)
        f0 = torch.sigmoid(self.rgb_0_heads[chunk_idx](x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1_heads[chunk_idx](x)) * 255.0

        return torch.stack([f0, f1], dim=1)


class HNeRVDecoder(nn.Module):
    """Legacy HNeRV decoder for backward compatibility."""
    def __init__(self, latent_dim=28, base_channels=36, eval_size=(384, 512), stem_dim=14):
        super().__init__()
        self.eval_size = eval_size
        # Minimal implementation for compatibility
        # This is just a placeholder - you may need the full implementation
        # if you have legacy archives to decode
        pass
    
    def forward(self, z):
        # Placeholder implementation
        raise NotImplementedError("HNeRVDecoder not fully implemented - use CompactTINCHNeRV")
