import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

class LRConv2d(nn.Module):
    """
    Low Rank Convolution (LRConv2d)
    Decomposes a standard k x k convolution with C_in input channels and C_out output channels
    into a vertical k x 1 convolution followed by a horizontal 1 x k convolution with an
    intermediate bottleneck rank r.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bottleneck_ratio=0.25):
        super().__init__()
        
        # Parse kernel size, padding, stride
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        ph, pw = (padding, padding) if isinstance(padding, int) else padding
        sh, sw = (stride, stride) if isinstance(stride, int) else stride
            
        # Compute bottleneck rank r
        # Enforce a minimum rank (e.g., minimum of 8 or 12)
        self.r = max(8, ceil(bottleneck_ratio * min(in_channels, out_channels)))
        
        # First Stage: Vertical spatial convolution (kh x 1) compressing channels C_in -> r
        self.conv_v = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.r,
            kernel_size=(kh, 1),
            stride=(sh, 1),
            padding=(ph, 0),
            bias=False
        )
        
        # Second Stage: Horizontal spatial convolution (1 x kw) expanding channels r -> C_out
        self.conv_h = nn.Conv2d(
            in_channels=self.r,
            out_channels=out_channels,
            kernel_size=(1, kw),
            stride=(1, sw),
            padding=(0, pw),
            bias=bias
        )

    def forward(self, x):
        return self.conv_h(self.conv_v(x))


class UpConvLR(nn.Module):
    """
    UpConvLR: A custom decoder upsampling convolution block.
    Supports either dense Conv2d ('conv') or Low Rank Conv2d ('lrconv').
    """
    def __init__(self, ngf, new_ngf, strd, ks, conv_type='lrconv', bias=True, bottleneck_ratio=0.25):
        super().__init__()
        self.conv_type = conv_type
        
        if conv_type == 'conv':
            self.conv = nn.Conv2d(ngf, new_ngf * strd * strd, ks, 1, ks // 2, bias=bias)
            self.up = nn.PixelShuffle(strd)
        elif conv_type == 'lrconv':
            self.conv = LRConv2d(
                in_channels=ngf,
                out_channels=new_ngf * strd * strd,
                kernel_size=ks,
                stride=1,
                padding=ks // 2,
                bias=bias,
                bottleneck_ratio=bottleneck_ratio
            )
            self.up = nn.PixelShuffle(strd)
        elif conv_type == 'deconv':
            self.conv = nn.ConvTranspose2d(ngf, new_ngf, strd, stride=strd)
            self.up = nn.Identity()
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

    def forward(self, x):
        return self.up(self.conv(x))


class DownConvLR(nn.Module):
    """
    DownConvLR: A custom encoder downsampling convolution block.
    Supports either dense Conv2d ('conv') or Low Rank Conv2d ('lrconv').
    """
    def __init__(self, ngf, new_ngf, strd, ks, conv_type='lrconv', bias=True, bottleneck_ratio=0.25):
        super().__init__()
        self.conv_type = conv_type
        
        if conv_type == 'conv':
            self.conv = nn.Conv2d(ngf, new_ngf, ks, stride=strd, padding=ks // 2, bias=bias)
        elif conv_type == 'lrconv':
            self.conv = LRConv2d(
                in_channels=ngf,
                out_channels=new_ngf,
                kernel_size=ks,
                stride=strd,
                padding=ks // 2,
                bias=bias,
                bottleneck_ratio=bottleneck_ratio
            )
        else:
            raise ValueError(f"Unsupported down conv_type: {conv_type}")

    def forward(self, x):
        return self.conv(x)


class NormLayer(nn.Module):
    def __init__(self, norm_type, channels):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == 'layer':
            self.norm = nn.GroupNorm(1, channels)
        elif norm_type == 'group':
            self.norm = nn.GroupNorm(8, channels)
        else:
            self.norm = nn.Identity()
            
    def forward(self, x):
        return self.norm(x)


class ActivationLayer(nn.Module):
    def __init__(self, act_type):
        super().__init__()
        self.act_type = act_type
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'gelu':
            self.act = nn.GELU()
        elif act_type == 'silu' or act_type == 'swish':
            self.act = nn.SiLU(inplace=True)
        elif act_type == 'leaky':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Identity()
            
    def forward(self, x):
        return self.act(x)


class NeRVBlockLR(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        dec_block = kargs.get('dec_block', True)
        conv_fn = UpConvLR if dec_block else DownConvLR
        self.stride = kargs.get('strd', 1)
        
        self.conv = conv_fn(
            ngf=kargs['ngf'],
            new_ngf=kargs['new_ngf'],
            strd=kargs['strd'],
            ks=kargs['ks'],
            conv_type=kargs.get('conv_type', 'lrconv'),
            bias=kargs.get('bias', True),
            bottleneck_ratio=kargs.get('bottleneck_ratio', 0.25)
        )
        
        self.norm = NormLayer(kargs.get('norm', 'none'), kargs['new_ngf'])
        self.act = ActivationLayer(kargs.get('act', 'gelu'))

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class HNeRVGenerator(nn.Module):
    def __init__(self, embed_dim, fc_hw, dec_strides, fc_dim, dec_channels, ks_dec, conv_type='lrconv', bottleneck_ratio=0.25):
        super().__init__()
        self.fc_h, self.fc_w = fc_hw
        self.fc_dim = fc_dim
        
        # Linear layer mapping embedding -> starting feature map
        self.mlp = nn.Linear(embed_dim, fc_dim * self.fc_h * self.fc_w)
        
        # Decoder stages
        self.layers = nn.ModuleList()
        self.skips = nn.ModuleList()
        in_ch = fc_dim
        for stride, out_ch in zip(dec_strides, dec_channels):
            self.layers.append(
                NeRVBlockLR(
                    dec_block=True,
                    conv_type=conv_type,
                    ngf=in_ch,
                    new_ngf=out_ch,
                    strd=stride,
                    ks=ks_dec,
                    bias=True,
                    norm='layer',
                    act='gelu',
                    bottleneck_ratio=bottleneck_ratio
                )
            )
            # Bilinear 1x1 conv skip projection
            self.skips.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()
            )
            in_ch = out_ch
            
        # Final color projection
        self.final_conv = nn.Conv2d(in_ch, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.mlp(x)
        x = x.view(-1, self.fc_dim, self.fc_h, self.fc_w)
        for layer, skip in zip(self.layers, self.skips):
            identity = F.interpolate(x, scale_factor=layer.stride, mode='bilinear', align_corners=False)
            identity = skip(identity)
            x = layer(x) + identity
        x = self.final_conv(x)
        return torch.sigmoid(x)


class HNeRVModel(nn.Module):
    def __init__(self, num_frames, embed_dim, fc_hw, dec_strides, fc_dim, dec_channels, ks_dec, conv_type='lrconv', bottleneck_ratio=0.25):
        super().__init__()
        self.num_frames = num_frames
        self.embeddings = nn.Embedding(num_frames, embed_dim)
        
        self.generator = HNeRVGenerator(
            embed_dim=embed_dim,
            fc_hw=fc_hw,
            dec_strides=dec_strides,
            fc_dim=fc_dim,
            dec_channels=dec_channels,
            ks_dec=ks_dec,
            conv_type=conv_type,
            bottleneck_ratio=bottleneck_ratio
        )
        
    def forward(self, frame_indices):
        embeds = self.embeddings(frame_indices)
        return self.generator(embeds)


# Custom 8-bit weight quantization utility functions to reduce file size by 75%
def save_quantized_weights(state_dict, metadata, file_path):
    quantized_dict = {'metadata': metadata}
    for k, v in state_dict.items():
        if v.is_floating_point():
            v_min = v.min().item()
            v_max = v.max().item()
            if v_max == v_min:
                q_v = torch.zeros_like(v, dtype=torch.uint8)
            else:
                scale = 255.0 / (v_max - v_min)
                q_v = ((v - v_min) * scale).round().clamp(0, 255).to(torch.uint8)
            quantized_dict[k] = {
                'q_weight': q_v.cpu(),
                'min': v_min,
                'max': v_max
            }
        else:
            quantized_dict[k] = v.cpu()
    torch.save(quantized_dict, file_path)


def load_quantized_weights(model, file_path, device):
    quantized_dict = torch.load(file_path, map_location=device)
    state_dict = {}
    metadata = quantized_dict.get('metadata', {})
    for k, v in quantized_dict.items():
        if k == 'metadata':
            continue
        if isinstance(v, dict) and 'q_weight' in v:
            q_v = v['q_weight'].to(device).float()
            v_min = v['min']
            v_max = v['max']
            state_dict[k] = v_min + (v_max - v_min) * (q_v / 255.0)
        else:
            state_dict[k] = v.to(device)
    model.load_state_dict(state_dict)
    return metadata

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    if G.size(-2) > G.size(-1):
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01, ns_steps=5, nesterov=True):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            nesterov=nesterov
        )
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                
                # Apply decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                
                # Update momentum buffer
                buf.lerp_(g, 1 - momentum)
                
                # Nesterov momentum
                if nesterov:
                    update = g.lerp(buf, momentum)
                else:
                    update = buf
                
                # If parameter has >= 2 dimensions, apply Newton-Schulz orthogonalization
                if update.ndim >= 2:
                    orig_shape = update.shape
                    if update.ndim > 2:
                        update = update.flatten(1)
                    
                    # Compute orthogonal update
                    orth_update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    
                    # Aspect ratio scaling
                    scale = (max(update.size(0), update.size(1)) / min(update.size(0), update.size(1))) ** 0.5
                    orth_update = orth_update * scale
                    
                    # Reshape back to original parameter shape
                    orth_update = orth_update.view(orig_shape)
                    
                    p.add_(orth_update, alpha=-lr)
                else:
                    # Fallback standard SGD update for 1D parameters (biases, gains, etc.)
                    p.add_(update, alpha=-lr)
                    
        return loss