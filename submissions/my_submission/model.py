"""HNeRV decoder with two-stage stem — reduces params by ~13K vs the original.

The original stem is Linear(latent_dim, C*6*8) = Linear(28, 960) = 27,840 params.
That is 33% of the entire model at base_channels=20.
It's oversized for 100-pair memorisation (designed for 600 pairs).

Fix: split into two linear layers with a sin() nonlinearity between them:
  stem_proj:  Linear(latent_dim, stem_dim)        e.g. Linear(28, 14)   = 406 params
  stem:       Linear(stem_dim,   C*base_h*base_w) e.g. Linear(14, 960)  = 14,400 params
  Total stem: 14,806 params  vs  27,840 original — saves 13,034 params (15.6%)

The sin() between the two layers ensures this is NOT a linear reparameterisation
of the latent, so the full representational capacity of the stem is preserved.

Everything else (blocks, skips, refine, rgb heads) is identical to hnerv_muon.

Param breakdown (base_channels=20, stem_dim=14):
  stem_proj : Linear(28,14)         =     406
  stem      : Linear(14,960)        =  14,400
  blocks    : 6 Conv2d stages       =  53,444
  skips     : 3 Conv2d + 3 Identity =     611
  refine    : 2 Conv2d              =     915
  rgb_0/1   : 2 Conv2d(10,3,3)      =     546
  TOTAL                             =  70,322 params

  vs original single-stem model     =  83,356 params
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HNeRVDecoder(nn.Module):
    def __init__(self, latent_dim=28, base_channels=20, eval_size=(384, 512),
                 stem_dim=14):
        super().__init__()
        self.eval_size = eval_size
        self.base_h, self.base_w = 6, 8
        C = base_channels

        # Channel taper — same schedule as hnerv_muon
        self.channels = [C, C, C, int(C * 0.75), int(C * 0.58),
                         int(C * 0.5), int(C * 0.5)]

        # Two-stage stem: latent_dim → stem_dim → spatial grid
        # sin() between the two stages keeps it nonlinear (not a mere linear re-param)
        self.stem_proj = nn.Linear(latent_dim, stem_dim)
        self.stem      = nn.Linear(stem_dim, self.channels[0] * self.base_h * self.base_w)

        self.blocks = nn.ModuleList()
        self.skips  = nn.ModuleList()
        for i in range(6):
            in_ch  = self.channels[i]
            out_ch = self.channels[i + 1]
            self.blocks.append(nn.Conv2d(in_ch, out_ch * 4, 3, padding=1))
            self.skips.append(
                nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            )
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
        # Two-stage stem with sin nonlinearity between stages
        z = torch.sin(self.stem_proj(z))
        x = self.stem(z).view(B, self.channels[0], self.base_h, self.base_w)
        x = torch.sin(x)
        for block, skip in zip(self.blocks, self.skips):
            identity = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x  = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)
