"""RGB-D Multi-Head Depth-guided Cross-Attention (DCA) variant.

Multi-head version of `rgbd_dca`. Each `_attend` call splits Q/K/V across
H heads (default H=4), runs scaled dot-product per head, concatenates, and
applies a learnable output projection W_O. W_O is zero-initialized so the
fused contribution starts at 0 and the model begins identical to RGB-only
(same warm-start property as the single-head variant).

Like `rgbd_dca`, parameterized by `num_iters` (K). Initial step is `D->I`;
each iter k=1..K adds `I->D` then `D->I`. Only the RGB stream feeds into
the Swin Transformer stages.

Architecture:
  Input (B, 4, H, W)
    |- RGB (B, 3, H, W)   -> patch_embed_rgb   -> (B, N, C) rgb_tokens
    `- Depth (B, 1, H, W) -> patch_embed_depth -> (B, N, C) depth_tokens
                                  |
                          DCAMultiHeadFusion (H heads, 1 + 2K attn steps)
                                  |
                          rgb_K (B, N, C)
                                  |
                          Swin Transformer stages 1-4
"""

import logging
import types

import torch
import torch.nn as nn

from variants.rgbd_early import RGBDMapper, _compute_depth_stats

logger = logging.getLogger(__name__)


# ===================================================================
# Fusion
# ===================================================================

class _IterBlock(nn.Module):
    """One bidirectional refinement cycle: I->D, then D->I (multi-head)."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # I guides D
        self.W_Iq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_Ik = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_Dv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O_iD = nn.Linear(embed_dim, embed_dim, bias=False)
        # D guides I
        self.W_Dq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_Dk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_Iv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O_dI = nn.Linear(embed_dim, embed_dim, bias=False)

        for L in (self.W_Iq, self.W_Ik, self.W_Dv,
                  self.W_Dq, self.W_Dk, self.W_Iv):
            nn.init.xavier_uniform_(L.weight)
        # Zero-init output projections so each iter starts as a no-op.
        nn.init.zeros_(self.W_O_iD.weight)
        nn.init.zeros_(self.W_O_dI.weight)


class DCAMultiHeadFusion(nn.Module):

    def __init__(self, embed_dim, num_heads=4, num_iters=0):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim={embed_dim} not divisible by num_heads={num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_iters = num_iters

        # Initial D guides RGB
        self.W_D1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_D2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_I1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O_init = nn.Linear(embed_dim, embed_dim, bias=False)

        for L in (self.W_D1, self.W_D2, self.W_I1):
            nn.init.xavier_uniform_(L.weight)
        # Zero-init W_O so the initial fusion output is 0; combined with the
        # residual in _attend(), the model starts identical to RGB-only and
        # learns the depth contribution gradually.
        nn.init.zeros_(self.W_O_init.weight)

        self.iter_blocks = nn.ModuleList([
            _IterBlock(embed_dim, num_heads) for _ in range(num_iters)
        ])

    def _attend(self, Q_src, K_src, V_src, Wq, Wk, Wv, Wo, residual):
        B, N, C = Q_src.shape
        H, d = self.num_heads, self.head_dim

        Q = Wq(Q_src).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        K = Wk(K_src).view(B, N, H, d).transpose(1, 2)
        V = Wv(V_src).view(B, N, H, d).transpose(1, 2)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = torch.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn, V)                                       # (B, H, N, d)

        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return residual + Wo(out)

    def forward(self, rgb_tokens, depth_tokens):
        """
        Args:
            rgb_tokens:   (B, N, C)
            depth_tokens: (B, N, C)
        Returns:
            rgb:   (B, N, C) -- (1 + K)-times-refined rgb
            depth: (B, N, C) -- K-times-refined depth (depth_tokens if K=0)
        """
        rgb = self._attend(
            depth_tokens, depth_tokens, rgb_tokens,
            self.W_D1, self.W_D2, self.W_I1, self.W_O_init,
            residual=rgb_tokens,
        )
        depth = depth_tokens

        for block in self.iter_blocks:
            depth = self._attend(
                rgb, rgb, depth,
                block.W_Iq, block.W_Ik, block.W_Dv, block.W_O_iD,
                residual=depth,
            )
            rgb = self._attend(
                depth, depth, rgb,
                block.W_Dq, block.W_Dk, block.W_Iv, block.W_O_dI,
                residual=rgb,
            )

        return rgb, depth


# ===================================================================
# Custom forward for the Swin backbone
# ===================================================================

def _dca_forward(self, x):
    """SwinTransformer.forward() with dual PatchEmbed + DCA fusion."""
    rgb = x[:, :3, :, :]
    depth = x[:, 3:, :, :]

    rgb_embed = self.patch_embed_rgb(rgb)
    depth_embed = self.patch_embed_depth(depth)

    Wh, Ww = rgb_embed.size(2), rgb_embed.size(3)

    rgb_tokens = rgb_embed.flatten(2).transpose(1, 2)
    depth_tokens = depth_embed.flatten(2).transpose(1, 2)

    x, _ = self.fusion(rgb_tokens, depth_tokens)

    x = self.pos_drop(x)

    outs = {}
    for i in range(self.num_layers):
        layer = self.layers[i]
        x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
        if i in self.out_indices:
            norm_layer = getattr(self, f"norm{i}")
            x_out = norm_layer(x_out)
            out = (
                x_out.view(-1, H, W, self.num_features[i])
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            outs[f"res{i + 2}"] = out

    return outs


# ===================================================================
# Variant interface
# ===================================================================

def update_config(cfg):
    """Extend PIXEL_MEAN and PIXEL_STD to 4 channels (RGB + depth)."""
    depth_mean, depth_std = _compute_depth_stats()
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530, depth_mean]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375, depth_std]


def update_model(model, cfg):
    """Replace the single PatchEmbed with dual streams + multi-head DCA fusion."""
    PatchEmbed = type(model.backbone.patch_embed)

    backbone = model.backbone
    old_pe = backbone.patch_embed
    embed_dim = backbone.embed_dim
    device = old_pe.proj.weight.device
    dtype = old_pe.proj.weight.dtype

    # RGB PatchEmbed: copy pretrained weights exactly
    patch_embed_rgb = PatchEmbed(
        patch_size=old_pe.patch_size,
        in_chans=3,
        embed_dim=embed_dim,
        norm_layer=nn.LayerNorm,
    ).to(device=device, dtype=dtype)
    patch_embed_rgb.load_state_dict(old_pe.state_dict())

    # Depth PatchEmbed: mean-init from RGB filters
    patch_embed_depth = PatchEmbed(
        patch_size=old_pe.patch_size,
        in_chans=1,
        embed_dim=embed_dim,
        norm_layer=nn.LayerNorm,
    ).to(device=device, dtype=dtype)
    patch_embed_depth.proj.weight.data = (
        old_pe.proj.weight.data.mean(dim=1, keepdim=True)
    )
    patch_embed_depth.proj.bias.data = old_pe.proj.bias.data.clone()
    if hasattr(old_pe, "norm") and old_pe.norm is not None:
        patch_embed_depth.norm.load_state_dict(old_pe.norm.state_dict())

    num_iters = getattr(cfg.MODEL, "DCA_ITERS", 0)
    num_heads = getattr(cfg.MODEL, "DCA_HEADS", 4)
    fusion = DCAMultiHeadFusion(
        embed_dim=embed_dim, num_heads=num_heads, num_iters=num_iters,
    ).to(device=device, dtype=dtype)

    backbone.patch_embed_rgb = patch_embed_rgb
    backbone.patch_embed_depth = patch_embed_depth
    backbone.fusion = fusion
    del backbone.patch_embed

    backbone.forward = types.MethodType(_dca_forward, backbone)

    logger.info(
        "DCA-MHA variant (H=%d heads, K=%d iters): dual PatchEmbed "
        "(rgb=3-ch, depth=1-ch) + %d-step multi-head depth-guided "
        "cross-attention at embed_dim=%d (head_dim=%d)",
        num_heads, num_iters, 1 + 2 * num_iters, embed_dim,
        embed_dim // num_heads,
    )
    return model


def get_mapper(cfg, is_train):
    """Return a custom dataset mapper for 4-channel RGBD input."""
    return RGBDMapper(cfg, is_train)
