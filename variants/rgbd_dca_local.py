"""RGB-D Local Depth-guided Cross-Attention (local DCA) variant.

Identical in spirit to `rgbd_dca` — dual PatchEmbed stems, depth guides RGB,
residual warm-start so the model begins as RGB-only — but the cross-attention
is restricted to non-overlapping w×w spatial windows instead of attending
globally across the whole feature map.

Why local?
----------
In `rgbd_dca`, the fused RGB token at position i is a depth-weighted sum over
*every* RGB token in the image:

    out_i = Σ_j  softmax(depth_i · depth_j) · rgb_j     (all j)

That lets a token pull RGB from anywhere, which dilutes depth's main value for
instance segmentation: separating *touching* fruits that sit at different
depths. Restricting attention to a local window keeps the depth-guided
"selection" mechanism (a token can still attend to nearby RGB on the correct
side of a depth edge) while preventing far-away, depth-similar regions from
leaking in:

    out_i = Σ_{j ∈ window(i)}  softmax(depth_i · depth_j) · rgb_j

window_size = 1   collapses to a purely pointwise (FiLM-like) fusion.
window_size = ∞   recovers global `rgbd_dca`.

Parameterized by `num_iters` (K), same as `rgbd_dca`: one initial depth-guided
RGB update, then K bidirectional refinement cycles. All attention — initial and
iterative — happens within windows.

Architecture:
  Input (B, 4, H, W)
    |- RGB (B, 3, H, W)   -> patch_embed_rgb   -> (B, N, C) rgb_tokens
    `- Depth (B, 1, H, W) -> patch_embed_depth -> (B, N, C) depth_tokens
                                  |
                          DCALocalFusion (windowed, 1 + 2K attn steps)
                                  |
                          rgb_K (B, N, C)
                                  |
                          Swin Transformer stages 1-4
"""

import logging
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from variants.rgbd_early import RGBDMapper, _compute_depth_stats

logger = logging.getLogger(__name__)


# ===================================================================
# Window partition / reverse helpers
# ===================================================================

def _window_partition(x, ws):
    """(B, H, W, C) -> (num_windows*B, ws*ws, C).

    H and W must already be multiples of `ws` (the caller pads first).
    Windows are flattened batch-major: index = b*(nH*nW) + wh*nW + ww.
    """
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    # -> (B, nH, nW, ws, ws, C) -> (B*nH*nW, ws*ws, C)
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, ws * ws, C)
    )


def _window_reverse(windows, ws, H, W):
    """(num_windows*B, ws*ws, C) -> (B, H, W, C). Inverse of _window_partition."""
    C = windows.shape[-1]
    nH, nW = H // ws, W // ws
    B = windows.shape[0] // (nH * nW)
    x = windows.view(B, nH, nW, ws, ws, C)
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B, H, W, C)
    )


# ===================================================================
# Fusion
# ===================================================================

class _IterBlock(nn.Module):
    """One bidirectional refinement cycle: I->D, then D->I."""

    def __init__(self, embed_dim):
        super().__init__()
        # I guides D
        self.W_Iq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_Ik = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_Dv = nn.Linear(embed_dim, embed_dim, bias=False)
        # D guides I
        self.W_Dq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_Dk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_Iv = nn.Linear(embed_dim, embed_dim, bias=False)

        nn.init.xavier_uniform_(self.W_Iq.weight)
        nn.init.xavier_uniform_(self.W_Ik.weight)
        nn.init.xavier_uniform_(self.W_Dq.weight)
        nn.init.xavier_uniform_(self.W_Dk.weight)
        # Zero-init values so each iter starts as a no-op.
        nn.init.zeros_(self.W_Dv.weight)
        nn.init.zeros_(self.W_Iv.weight)


class DCALocalFusion(nn.Module):
    """Windowed depth-guided cross-attention.

    Same attention algebra as `rgbd_dca.DCAFusion`, but tokens are partitioned
    into non-overlapping `window_size` × `window_size` windows first, so each
    attention step only mixes tokens within a window. The window dimension is
    folded into the batch dim, so `F.scaled_dot_product_attention` does the
    local attention without any explicit neighbourhood gather.
    """

    def __init__(self, embed_dim, window_size=7, num_iters=0):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.window_size = window_size
        self.num_iters = num_iters

        # Initial D guides RGB
        self.W_D1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_D2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_I1 = nn.Linear(embed_dim, embed_dim, bias=False)

        nn.init.xavier_uniform_(self.W_D1.weight)
        nn.init.xavier_uniform_(self.W_D2.weight)
        # Zero-init W_I1 so the initial fusion output is 0; combined with the
        # residual in _attend(), the model starts identical to RGB-only and
        # learns the depth contribution gradually.
        nn.init.zeros_(self.W_I1.weight)

        self.iter_blocks = nn.ModuleList([
            _IterBlock(embed_dim) for _ in range(num_iters)
        ])

    def _attend(self, Q_src, K_src, V_src, Wq, Wk, Wv, residual, attn_mask):
        # Q_src/K_src/V_src are windowed: (num_windows*B, ws*ws, C).
        Q = Wq(Q_src).unsqueeze(1)   # (nW*B, 1, ws*ws, C) -- SDPA needs a head dim
        K = Wk(K_src).unsqueeze(1)
        V = Wv(V_src).unsqueeze(1)
        # attn_mask (when not None) is additive, broadcastable to
        # (nW*B, 1, ws*ws, ws*ws); it masks out padded key positions.
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        return residual + out.squeeze(1)

    def _build_key_mask(self, valid, ws, dtype):
        """Additive attention mask that hides padded key tokens.

        Args:
            valid: (B, Hp, Wp, 1) float, 1.0 for real tokens and 0.0 for the
                bottom/right padding added to make Hp, Wp multiples of `ws`.
        Returns:
            (nW*B, 1, 1, ws*ws) tensor of {0, large_negative}, broadcast over
            query rows. No window is ever fully padded (padding is < ws on
            each edge), so every softmax row keeps at least one real key and
            cannot produce NaNs.
        """
        vw = _window_partition(valid, ws).squeeze(-1)  # (nW*B, ws*ws)
        neg = torch.finfo(dtype).min
        mask = torch.where(
            vw > 0,
            torch.zeros((), dtype=dtype, device=vw.device),
            torch.full((), neg, dtype=dtype, device=vw.device),
        )
        return mask[:, None, None, :]  # (nW*B, 1, 1, ws*ws)

    def forward(self, rgb_tokens, depth_tokens, Wh, Ww):
        """
        Args:
            rgb_tokens:   (B, N, C)  with N == Wh*Ww
            depth_tokens: (B, N, C)
            Wh, Ww: token-grid height/width (from the patch embed output)
        Returns:
            rgb:   (B, N, C) -- (1 + K)-times-refined rgb
            depth: (B, N, C) -- K-times-refined depth (depth_tokens if K=0)
        """
        B, N, C = rgb_tokens.shape
        ws = self.window_size

        rgb = rgb_tokens.view(B, Wh, Ww, C)
        depth = depth_tokens.view(B, Wh, Ww, C)

        # Pad bottom/right so the grid tiles into whole windows.
        pad_b = (ws - Wh % ws) % ws
        pad_r = (ws - Ww % ws) % ws
        if pad_b or pad_r:
            # F.pad on (B, H, W, C) pads the last dims as (C_l, C_r, W_l, W_r, H_l, H_r).
            rgb = F.pad(rgb, (0, 0, 0, pad_r, 0, pad_b))
            depth = F.pad(depth, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = Wh + pad_b, Ww + pad_r

        # Build the padded-key mask once (shared across all attention steps).
        attn_mask = None
        if pad_b or pad_r:
            valid = rgb.new_ones((B, Hp, Wp, 1))
            valid[:, Wh:, :, :] = 0
            valid[:, :, Ww:, :] = 0
            attn_mask = self._build_key_mask(valid, ws, rgb.dtype)

        rgb_w = _window_partition(rgb, ws)      # (nW*B, ws*ws, C)
        depth_w = _window_partition(depth, ws)

        # --- Initial: depth guides RGB (windowed) ---
        rgb_w = self._attend(
            depth_w, depth_w, rgb_w,
            self.W_D1, self.W_D2, self.W_I1,
            residual=rgb_w, attn_mask=attn_mask,
        )

        # --- K bidirectional refinement cycles (windowed) ---
        for block in self.iter_blocks:
            depth_w = self._attend(
                rgb_w, rgb_w, depth_w,
                block.W_Iq, block.W_Ik, block.W_Dv,
                residual=depth_w, attn_mask=attn_mask,
            )
            rgb_w = self._attend(
                depth_w, depth_w, rgb_w,
                block.W_Dq, block.W_Dk, block.W_Iv,
                residual=rgb_w, attn_mask=attn_mask,
            )

        # --- Back to token sequence ---
        rgb = _window_reverse(rgb_w, ws, Hp, Wp)      # (B, Hp, Wp, C)
        depth = _window_reverse(depth_w, ws, Hp, Wp)
        if pad_b or pad_r:
            rgb = rgb[:, :Wh, :Ww, :].contiguous()
            depth = depth[:, :Wh, :Ww, :].contiguous()
        rgb = rgb.view(B, N, C)
        depth = depth.view(B, N, C)

        return rgb, depth


# ===================================================================
# Custom forward for the Swin backbone
# ===================================================================

def _dca_local_forward(self, x):
    """SwinTransformer.forward() with dual PatchEmbed + windowed DCA fusion."""
    rgb = x[:, :3, :, :]
    depth = x[:, 3:, :, :]

    rgb_embed = self.patch_embed_rgb(rgb)
    depth_embed = self.patch_embed_depth(depth)

    Wh, Ww = rgb_embed.size(2), rgb_embed.size(3)

    rgb_tokens = rgb_embed.flatten(2).transpose(1, 2)
    depth_tokens = depth_embed.flatten(2).transpose(1, 2)

    # Pass the token grid shape so the fusion can window the tokens.
    x, _ = self.fusion(rgb_tokens, depth_tokens, Wh, Ww)

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
    depth_mean, depth_std = _compute_depth_stats(cfg)
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530, depth_mean]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375, depth_std]


def update_model(model, cfg):
    """Replace the single PatchEmbed with dual streams + windowed DCA fusion."""
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
    window_size = getattr(cfg.MODEL, "DCA_WINDOW", 7)
    fusion = DCALocalFusion(
        embed_dim=embed_dim, window_size=window_size, num_iters=num_iters,
    ).to(device=device, dtype=dtype)

    backbone.patch_embed_rgb = patch_embed_rgb
    backbone.patch_embed_depth = patch_embed_depth
    backbone.fusion = fusion
    del backbone.patch_embed

    backbone.forward = types.MethodType(_dca_local_forward, backbone)

    logger.info(
        "Local-DCA variant (w=%d window, K=%d iters): dual PatchEmbed "
        "(rgb=3-ch, depth=1-ch) + %d-step windowed depth-guided "
        "cross-attention at embed_dim=%d",
        window_size, num_iters, 1 + 2 * num_iters, embed_dim,
    )
    return model


def get_mapper(cfg, is_train):
    """Return a custom dataset mapper for 4-channel RGBD input."""
    return RGBDMapper(cfg, is_train)
