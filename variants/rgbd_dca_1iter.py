"""RGB-D Depth-guided Cross-Attention (DCA), 3-step iterative bidirectional.

Three-step iterative bidirectional cross-modal fusion with no weight sharing
between steps:

    Step 1: depth   guides rgb     -> rgb_1
    Step 2: rgb_1   guides depth   -> depth_1
    Step 3: depth_1 guides rgb_1   -> rgb_2

Only the RGB stream (rgb_2) feeds into the Swin Transformer stages; the
refined depth (depth_1) is discarded after fusion, matching the single-pass
DCA variant.

Architecture:
  Input (B, 4, H, W)
    |- RGB (B, 3, H, W)   -> patch_embed_rgb   -> (B, N, C) rgb_tokens
    `- Depth (B, 1, H, W) -> patch_embed_depth -> (B, N, C) depth_tokens
                                  |
                          DCAFusion (3-step iterative bidirectional)
                                  |
                          rgb_2 (B, N, C)
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

class DCAFusion(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.scale = embed_dim ** -0.5

        # 1 Depth (D) Guides RGB (I)
        self.W_D1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_D2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_I1 = nn.Linear(embed_dim, embed_dim, bias=False)

        # 2 RGB Guides Depth
        self.W_I2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_I3 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_D3 = nn.Linear(embed_dim, embed_dim, bias=False)

        # 3 Depth Guides RGB
        self.W_D4 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_D5 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_I4 = nn.Linear(embed_dim, embed_dim, bias=False)

        # Initializing Weights - Xavier for 
        nn.init.xavier_uniform_(self.W_D1.weight)  # step 1 Q
        nn.init.xavier_uniform_(self.W_D2.weight)  # step 1 K
        nn.init.xavier_uniform_(self.W_I2.weight)  # step 2 Q
        nn.init.xavier_uniform_(self.W_I3.weight)  # step 2 K
        nn.init.xavier_uniform_(self.W_D4.weight)  # step 3 Q
        nn.init.xavier_uniform_(self.W_D5.weight)  # step 3 K

        # Zero Init for Values
        nn.init.zeros_(self.W_I1.weight)  # step 1 V
        nn.init.zeros_(self.W_D3.weight)  # step 2 V
        nn.init.zeros_(self.W_I4.weight)  # step 3 V

    def forward(self, rgb_tokens, depth_tokens):
        """
        Args:
            rgb_tokens:   (B, N, C)
            depth_tokens: (B, N, C)
        Returns:
            rgb_2:   (B, N, C) -- twice-refined rgb
            depth_1: (B, N, C) -- once-refined depth
        """

        # D guides RGB
        Q = self.W_D1(depth_tokens)                                    # (B, N, C)
        K = self.W_D2(depth_tokens)                                    # (B, N, C)
        V = self.W_I1(rgb_tokens)                                      # (B, N, C)
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) * self.scale     # (B, N, N)
        attn = torch.softmax(attn_logits, dim=-1) # apply row-wise
        rgb_1 = rgb_tokens + torch.bmm(attn, V)                        # (B, N, C)

        # RGB guides Depth
        Q = self.W_I2(rgb_1)                                           # (B, N, C)
        K = self.W_I3(rgb_1)                                           # (B, N, C)
        V = self.W_D3(depth_tokens)                                    # (B, N, C)
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) * self.scale     # (B, N, N)
        attn = torch.softmax(attn_logits, dim=-1)
        depth_1 = depth_tokens + torch.bmm(attn, V)                    # (B, N, C)

        # D guides RGB
        Q = self.W_D4(depth_1)                                         # (B, N, C)
        K = self.W_D5(depth_1)                                         # (B, N, C)
        V = self.W_I4(rgb_1)                                           # (B, N, C)
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) * self.scale     # (B, N, N)
        attn = torch.softmax(attn_logits, dim=-1)
        rgb_2 = rgb_1 + torch.bmm(attn, V)                             # (B, N, C)

        return rgb_2, depth_1


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
    """Replace the single PatchEmbed with dual streams + DCA fusion."""
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

    fusion = DCAFusion(embed_dim=embed_dim).to(device=device, dtype=dtype)

    backbone.patch_embed_rgb = patch_embed_rgb
    backbone.patch_embed_depth = patch_embed_depth
    backbone.fusion = fusion
    del backbone.patch_embed

    backbone.forward = types.MethodType(_dca_forward, backbone)

    logger.info(
        "DCA 1iter variant: dual PatchEmbed (rgb=3-ch, depth=1-ch) "
        "+ 3-step iterative bidirectional cross-attention at embed_dim=%d",
        embed_dim,
    )
    return model


def get_mapper(cfg, is_train):
    """Return a custom dataset mapper for 4-channel RGBD input."""
    return RGBDMapper(cfg, is_train)
