"""
SNR-aware conditional UAP generator for SemCom-UAP (Section IV-C).

Architecture:
  blocks 1..N-1 : ConvTranspose2d -> GroupNorm -> ReLU -> SpatialTransformer(cond)
  block  N      : ConvTranspose2d -> GroupNorm -> ReLU
  output layer  : ConvTranspose2d -> Tanh

The condition vector is cond = [ref_feats ; snr_emb], i.e. a cross-modal
reference feature concatenated with the shared SNR embedding
(context_dim = 256 + 64 = 320 for BERT-family, 512 + 64 = 576 for CLIP).
It is injected through cross attention only.
"""

import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange


# ==========================================
# Cross-attention blocks
# ==========================================

def _exists(val):
    return val is not None

def _default(val, d):
    return val if _exists(val) else (d() if callable(d) else d)

def _zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def _normalize(in_channels):
    return nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)


class _FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = _default(dim_out, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class _CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim    = dim_head * heads
        context_dim  = _default(context_dim, query_dim)
        self.scale   = dim_head ** -0.5
        self.heads   = heads
        self.to_q    = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k    = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v    = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out  = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        context = _default(context, x)
        k, v = self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim  = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out  = einsum('b i j, b j d -> b i d', attn, v)
        out  = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class _BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None):
        super().__init__()
        self.attn1 = _CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.attn2 = _CrossAttention(query_dim=dim, context_dim=context_dim,
                                     heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff    = _FeedForward(dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class _SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm     = _normalize(in_channels)
        self.proj_in  = nn.Conv2d(in_channels, inner_dim, kernel_size=1)
        self.transformer_blocks = nn.ModuleList([
            _BasicTransformerBlock(inner_dim, n_heads, d_head,
                                   dropout=dropout, context_dim=context_dim)
            for _ in range(depth)
        ])
        self.proj_out = _zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1))

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x    = self.proj_in(self.norm(x))
        x    = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return self.proj_out(x) + x_in


# ==========================================
# SNR Encoder
# ==========================================

class SNREncoder(nn.Module):
    """Shared SNR encoder (Eq. (2)).

    Maps the scalar channel SNR (dB) to a d_snr-dimensional embedding through
    a two-layer MLP (1 -> 64 -> 64) with SiLU activation. The embedding is
    concatenated to the cross-modal reference feature to form the generator
    condition, and is shared by the image and text generators.
    """

    def __init__(self, snr_emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, snr_emb_dim),
        )

    def forward(self, snr_db, device):
        """
        snr_db: float or (B,) tensor
        returns: (B, snr_emb_dim)
        """
        if isinstance(snr_db, float):
            snr_t = torch.tensor([[snr_db]], dtype=torch.float32, device=device)
        else:
            snr_t = snr_db.float().view(-1, 1).to(device)
        return self.net(snr_t)  # (1 or B, snr_emb_dim)


# ==========================================
# Conditional UAP generator
# ==========================================

class Generator(nn.Module):
    """Conditional UAP generator.

    Emits one perturbation per row of the input latent code. The image
    attacker clips every row to the l_inf budget and adds row i to image i,
    and the evaluation deploys row 0 of the exported image tensor. The text
    attacker norm-clamps every row and substitutes the first `text_budget`
    rows, and the evaluation maps the mean of the exported rows to one word.
    """

    def __init__(self, input_dim, num_filters, output_dim, batch_size,
                 first_kernel_size, num_heads=1, context_dim=320):
        super().__init__()

        self.length = len(num_filters)

        self.deconv_blocks = nn.ModuleList()
        self.norm_blocks   = nn.ModuleList()
        self.act = nn.ReLU()

        for i, filters in enumerate(num_filters):
            deconvs = nn.ModuleList()
            norms   = nn.ModuleList()

            for j, out_ch in enumerate(filters):
                if i == 0 and j == 0:
                    in_ch  = input_dim
                    deconv = nn.ConvTranspose2d(in_ch, out_ch,
                                               kernel_size=first_kernel_size,
                                               stride=1, padding=0)
                elif j == 0:
                    in_ch  = num_filters[i - 1][-1]
                    deconv = nn.ConvTranspose2d(in_ch, out_ch,
                                               kernel_size=4, stride=2, padding=1)
                else:
                    in_ch  = filters[j - 1]
                    deconv = nn.ConvTranspose2d(in_ch, out_ch,
                                               kernel_size=4, stride=2, padding=1)

                nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
                nn.init.constant_(deconv.bias, 0.0)

                deconvs.append(deconv)
                norms.append(nn.GroupNorm(min(8, out_ch), out_ch))

            self.deconv_blocks.append(deconvs)
            self.norm_blocks.append(norms)

        # cross attention on the first N-1 blocks
        self.cross_attns = nn.ModuleList()
        for i in range(len(num_filters) - 1):
            last_ch = num_filters[i][-1]
            self.cross_attns.append(_SpatialTransformer(
                last_ch, num_heads,
                last_ch // num_heads,
                depth=1, context_dim=context_dim))

        # Output layer
        last_ch  = num_filters[-1][-1]
        out_conv = nn.ConvTranspose2d(last_ch, output_dim,
                                      kernel_size=4, stride=2, padding=1)
        nn.init.normal_(out_conv.weight, mean=0.0, std=0.02)
        nn.init.constant_(out_conv.bias, 0.0)
        self.output_layer = nn.Sequential(out_conv, nn.Tanh())

    def forward(self, x, cond):
        """
        x:       (N, z_dim, h, w)   latent code
        cond:    (N, context_dim)   [ref_feats ; snr_emb]
        returns: (N, output_dim, H, W)
        """
        cond_seq = cond.unsqueeze(1)  # (N, 1, context_dim)

        for i in range(self.length):
            for j in range(len(self.deconv_blocks[i])):
                x = self.deconv_blocks[i][j](x)
                x = self.norm_blocks[i][j](x)
                x = self.act(x)

            if i < self.length - 1:
                x = self.cross_attns[i](x, cond_seq)

        return self.output_layer(x)
