"""RGB-D Depth-guided Cross-Attention with invalid-depth attention masking.

Same architecture as rgbd_dca, but the cross-attention is given a
key-padding mask derived from per-patch depth validity. Patches whose
underlying depth was mostly the sensor's invalid-zero sentinel are
blocked from being attended to (as keys) by the RGB stream, instead of
being silently neutralized via mean-fill alone.

How invalid depth flows through the pipeline:
  1. RGBDMapper (Part 1) fills invalid depth pixels with the training
     mean, so the depth channel itself carries no -2σ outlier.
  2. RGBDMapper(include_validity=True) appends a 5th channel — a binary
     {0, 255} validity mask of the original depth. PIXEL_STD[4] = 255
     normalizes it to {0.0, 1.0}.
  3. _dca_mask_forward splits the input into (rgb, depth, validity),
     average-pools validity at the Swin patch size to get a per-token
     valid fraction, and thresholds at 0.5 to produce a boolean
     key-padding mask.
  4. DCAFusionMasked passes the mask to F.scaled_dot_product_attention
     as an additive (-inf) attn_mask, but only on the steps where depth
     tokens act as keys (initial fusion + the D→I step inside each iter
     block). RGB-as-key steps are unmasked since RGB has no validity
     concept.

K-iter semantics are identical to rgbd_dca:
    Initial:                  D guides RGB        -> rgb_0     [masked]
    Iter k (1..K):  rgb_{k-1} guides depth_{k-1}  -> depth_k   [unmasked]
                    depth_k   guides rgb_{k-1}    -> rgb_k     [masked]
The mask is computed once from the input depth and reused across all
masked steps (static-input-validity).
"""

import logging
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from variants.rgbd_early import RGBDMapper, _compute_depth_stats

logger = logging.getLogger(__name__)


# ===================================================================
# Fusion
# ===================================================================

class _IterBlock(nn.Module):
    """One bidirectional refinement cycle: I->D, then D->I."""

    def __init__(self, embed_dim):
        super().__init__()
        # I guides D (RGB-as-key — unmasked at attention time)
        self.W_Iq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_Ik = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_Dv = nn.Linear(embed_dim, embed_dim, bias=False)
        # D guides I (depth-as-key — masked at attention time)
        self.W_Dq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_Dk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_Iv = nn.Linear(embed_dim, embed_dim, bias=False)

        nn.init.xavier_uniform_(self.W_Iq.weight)
        nn.init.xavier_uniform_(self.W_Ik.weight)
        nn.init.xavier_uniform_(self.W_Dq.weight)
        nn.init.xavier_uniform_(self.W_Dk.weight)
        nn.init.zeros_(self.W_Dv.weight)
        nn.init.zeros_(self.W_Iv.weight)


class DCAFusionMasked(nn.Module):
    """DCAFusion with an optional key_padding_mask on depth-as-key steps."""

    def __init__(self, embed_dim, num_iters=0):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.num_iters = num_iters

        # Initial D guides RGB (depth-as-key)
        self.W_D1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_D2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_I1 = nn.Linear(embed_dim, embed_dim, bias=False)

        nn.init.xavier_uniform_(self.W_D1.weight)
        nn.init.xavier_uniform_(self.W_D2.weight)
        # Zero-init W_I1 so initial fusion is a no-op; model starts
        # identical to RGB-only and learns depth contribution gradually.
        nn.init.zeros_(self.W_I1.weight)

        self.iter_blocks = nn.ModuleList([
            _IterBlock(embed_dim) for _ in range(num_iters)
        ])

    def _attend(self, Q_src, K_src, V_src, Wq, Wk, Wv, residual, attn_mask=None):
        Q = Wq(Q_src).unsqueeze(1)   # (B, 1, N, C) — head dim for SDPA
        K = Wk(K_src).unsqueeze(1)
        V = Wv(V_src).unsqueeze(1)
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        return residual + out.squeeze(1)

    def forward(self, rgb_tokens, depth_tokens, key_padding_mask=None):
        """
        Args:
            rgb_tokens:        (B, N, C)
            depth_tokens:      (B, N, C)
            key_padding_mask:  (B, N) bool, True where depth token is
                               valid (attend) and False where invalid
                               (block). When None, behavior matches
                               rgbd_dca.DCAFusion.
        Returns:
            rgb:   (B, N, C) -- (1 + K)-times-refined rgb
            depth: (B, N, C) -- K-times-refined depth (depth_tokens if K=0)
        """
        # Build additive attn_mask for depth-as-key steps.
        # Shape (B, 1, 1, N) broadcasts over heads (1) and queries (N).
        if key_padding_mask is not None:
            depth_attn_mask = torch.zeros_like(
                key_padding_mask, dtype=rgb_tokens.dtype,
            ).masked_fill(~key_padding_mask, float("-inf"))
            depth_attn_mask = depth_attn_mask[:, None, None, :]
        else:
            depth_attn_mask = None

        rgb = self._attend(
            depth_tokens, depth_tokens, rgb_tokens,
            self.W_D1, self.W_D2, self.W_I1,
            residual=rgb_tokens,
            attn_mask=depth_attn_mask,
        )
        depth = depth_tokens

        for block in self.iter_blocks:
            # I-guides-D: RGB is K/V, no mask (RGB has no validity concept).
            depth = self._attend(
                rgb, rgb, depth,
                block.W_Iq, block.W_Ik, block.W_Dv,
                residual=depth,
                attn_mask=None,
            )
            # D-guides-I: depth is K/V, mask invalid-depth tokens.
            rgb = self._attend(
                depth, depth, rgb,
                block.W_Dq, block.W_Dk, block.W_Iv,
                residual=rgb,
                attn_mask=depth_attn_mask,
            )

        return rgb, depth


# ===================================================================
# Custom forward for the Swin backbone
# ===================================================================

def _dca_mask_forward(self, x):
    """SwinTransformer.forward() with dual PatchEmbed + masked DCA fusion.

    Input x: (B, 5, H, W) = [RGB(3), depth(1), validity(1)]
    """
    rgb = x[:, :3, :, :]
    depth = x[:, 3:4, :, :]
    validity = x[:, 4:5, :, :]  # in [0, 1] after PIXEL_STD=255 normalization

    # Per-patch valid fraction at Swin patch resolution (typically 4x4).
    patch_h, patch_w = self.patch_embed_rgb.patch_size
    valid_frac = F.avg_pool2d(
        validity, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w),
    )  # (B, 1, H/p, W/p)
    # Token-aligned boolean mask: True = attend, False = block.
    key_padding_mask = (valid_frac > 0.5).flatten(2).squeeze(1)  # (B, N)

    rgb_embed = self.patch_embed_rgb(rgb)
    depth_embed = self.patch_embed_depth(depth)

    Wh, Ww = rgb_embed.size(2), rgb_embed.size(3)

    rgb_tokens = rgb_embed.flatten(2).transpose(1, 2)
    depth_tokens = depth_embed.flatten(2).transpose(1, 2)

    # Sanity: validity-pool resolution must match the patch_embed output;
    # if Swin's internal F.pad-to-patch-multiple changed the token count,
    # crop the mask accordingly. Validity was pooled from the same input
    # H,W though, so this should align by construction.
    if key_padding_mask.shape[1] != rgb_tokens.shape[1]:
        raise RuntimeError(
            f"valid-mask tokens ({key_padding_mask.shape[1]}) != "
            f"rgb tokens ({rgb_tokens.shape[1]})"
        )

    x, _ = self.fusion(
        rgb_tokens, depth_tokens, key_padding_mask=key_padding_mask,
    )

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
    """Extend PIXEL_MEAN/STD to 5 channels (RGB + depth + validity)."""
    depth_mean, depth_std = _compute_depth_stats(cfg)
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530, depth_mean, 0.0]
    # PIXEL_STD=255 for the validity channel so 255→1.0, 0→0.0 after
    # Detectron2's `(x - mean) / std` normalization.
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375, depth_std, 255.0]


def update_model(model, cfg):
    """Replace PatchEmbed with dual streams + masked DCA fusion."""
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
    fusion = DCAFusionMasked(embed_dim=embed_dim, num_iters=num_iters).to(
        device=device, dtype=dtype
    )

    backbone.patch_embed_rgb = patch_embed_rgb
    backbone.patch_embed_depth = patch_embed_depth
    backbone.fusion = fusion
    del backbone.patch_embed

    backbone.forward = types.MethodType(_dca_mask_forward, backbone)

    logger.info(
        "DCA-mask variant (K=%d iters): dual PatchEmbed (rgb=3-ch, depth=1-ch) "
        "+ %d-step depth-guided cross-attention at embed_dim=%d, "
        "with per-token validity masking on depth-as-key steps",
        num_iters, 1 + 2 * num_iters, embed_dim,
    )
    return model


def get_mapper(cfg, is_train):
    """RGBDMapper with validity channel enabled (5-channel input)."""
    return RGBDMapper(cfg, is_train, include_validity=True)
