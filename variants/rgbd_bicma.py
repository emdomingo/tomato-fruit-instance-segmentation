"""RGB-D Bidirectional Cross-Modal Attention (BiCMA) variant.

Two separate PatchEmbed streams for RGB and depth. A parameter-free
bidirectional cross-modal attention module iteratively uses each
modality's self-similarity matrix to re-weight the other modality's
tokens. Only the final RGB tokens feed into the Swin Transformer stages.

Fusion formula (repeated for num_iters iterations):
    X_rgb = softmax(X_D @ X_D^T / sqrt(C)) @ X_rgb      (depth-guided)
    X_D   = softmax(X_rgb @ X_rgb^T / sqrt(C)) @ X_D    (RGB-guided)
"""

import logging
import types

import torch
import torch.nn as nn

from variants.rgbd_early import RGBDMapper, _compute_depth_stats

logger = logging.getLogger(__name__)


# ===================================================================
# BiCMA Fusion Module
# ===================================================================

class BiCMAFusion(nn.Module):
    """Parameter-free bidirectional cross-modal attention fusion.

    Uses each modality's self-similarity matrix (softmax-normalized)
    to re-weight the other modality's token sequence.
    """

    def __init__(self, embed_dim, num_iters=2):
        super().__init__()
        self.num_iters = num_iters
        self.scale = embed_dim ** -0.5

    def forward(self, rgb_tokens, depth_tokens):
        """
        Args:
            rgb_tokens:   (B, N, C) from patch_embed_rgb
            depth_tokens: (B, N, C) from patch_embed_depth
        Returns:
            (B, N, C) depth-informed RGB tokens
        """
        for _ in range(self.num_iters):
            # Depth-guided RGB update: S(X_D · X_D^T) · X_rgb
            depth_sim = torch.bmm(
                depth_tokens, depth_tokens.transpose(1, 2)
            ) * self.scale
            rgb_tokens = torch.bmm(
                torch.softmax(depth_sim, dim=-1), rgb_tokens
            )

            # RGB-guided Depth update: S(X_rgb · X_rgb^T) · X_D
            rgb_sim = torch.bmm(
                rgb_tokens, rgb_tokens.transpose(1, 2)
            ) * self.scale
            depth_tokens = torch.bmm(
                torch.softmax(rgb_sim, dim=-1), depth_tokens
            )

        return rgb_tokens


# ===================================================================
# Custom forward for the Swin backbone
# ===================================================================

def _bicma_forward(self, x):
    """Modified SwinTransformer forward with dual PatchEmbed + BiCMA fusion.

    Expects x to be (B, 4, H, W) where channels 0-2 are RGB and
    channel 3 is depth (concatenated by RGBDMapper).
    """
    rgb = x[:, :3, :, :]
    depth = x[:, 3:, :, :]

    # Dual-stream patch embedding → (B, C, Wh, Ww) each
    rgb_embed = self.patch_embed_rgb(rgb)
    depth_embed = self.patch_embed_depth(depth)

    Wh, Ww = rgb_embed.size(2), rgb_embed.size(3)

    # Flatten to token sequences → (B, N, C)
    rgb_tokens = rgb_embed.flatten(2).transpose(1, 2)
    depth_tokens = depth_embed.flatten(2).transpose(1, 2)

    # BiCMA fusion → (B, N, C) depth-informed RGB tokens
    x = self.fusion(rgb_tokens, depth_tokens)

    x = self.pos_drop(x)

    # Standard Swin stages
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
    """Replace single PatchEmbed with dual streams + BiCMA fusion."""
    PatchEmbed = type(model.backbone.patch_embed)

    backbone = model.backbone
    old_pe = backbone.patch_embed
    embed_dim = backbone.embed_dim
    device = old_pe.proj.weight.device
    dtype = old_pe.proj.weight.dtype

    # --- RGB PatchEmbed: copy pretrained weights exactly ---
    patch_embed_rgb = PatchEmbed(
        patch_size=old_pe.patch_size,
        in_chans=3,
        embed_dim=embed_dim,
        norm_layer=nn.LayerNorm,
    ).to(device=device, dtype=dtype)
    patch_embed_rgb.load_state_dict(old_pe.state_dict())

    # --- Depth PatchEmbed: mean-init from RGB filters ---
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

    # --- BiCMA fusion module (parameter-free) ---
    fusion = BiCMAFusion(embed_dim=embed_dim, num_iters=2).to(
        device=device, dtype=dtype
    )

    # --- Attach to backbone and monkey-patch forward ---
    backbone.patch_embed_rgb = patch_embed_rgb
    backbone.patch_embed_depth = patch_embed_depth
    backbone.fusion = fusion
    del backbone.patch_embed

    backbone.forward = types.MethodType(_bicma_forward, backbone)

    logger.info(
        "BiCMA variant: dual PatchEmbed (rgb=%d-ch, depth=%d-ch) "
        "+ %d-iter fusion at embed_dim=%d",
        3, 1, fusion.num_iters, embed_dim,
    )
    return model


def get_mapper(cfg, is_train):
    """Return a mapper that loads RGB + depth as a 4-channel input."""
    return RGBDMapper(cfg, is_train)
