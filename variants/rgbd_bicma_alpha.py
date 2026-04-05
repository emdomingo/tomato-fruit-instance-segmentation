"""RGB-D Bidirectional Cross-Modal Attention (BiCMA) variant.

Two separate PatchEmbed streams for RGB and depth. A parameter-free
bidirectional cross-modal attention module iteratively uses each
modality's self-similarity matrix to re-weight the other modality's
tokens. Only the final RGB tokens feed into the Swin Transformer stages.

Fusion formula (repeated for num_iters iterations):
    X_rgb = softmax(X_D @ X_D^T / sqrt(C)) @ X_rgb      (depth-guided)
    X_D   = softmax(X_rgb @ X_rgb^T / sqrt(C)) @ X_D    (RGB-guided)

Overview of BiCMA Fusion
------------------------
Unlike early fusion (which simply concatenates depth as a 4th channel),
BiCMA keeps RGB and depth as separate "streams" through their own
PatchEmbed layers, then fuses them via cross-modal attention BEFORE
feeding into the Swin Transformer stages.

The key idea: each modality's self-similarity matrix (how similar each
patch is to every other patch within that modality) is used to re-weight
the OTHER modality's tokens. Intuitively:
  - If two depth patches are very similar, those same spatial positions
    in the RGB stream should attend to each other too.
  - And vice versa: RGB similarity guides depth token mixing.

This is "parameter-free" because it uses no learned weights — only
matrix multiplications and softmax. The cross-modal attention matrices
are derived entirely from the token embeddings themselves.

After fusion, only the depth-informed RGB tokens proceed into the
standard Swin Transformer stages. The depth stream is discarded —
its information has been "injected" into the RGB tokens via attention.

Architecture:
  Input (B, 4, H, W)
    ├─ RGB (B, 3, H, W)   → patch_embed_rgb   → (B, N, C) rgb_tokens
    └─ Depth (B, 1, H, W) → patch_embed_depth  → (B, N, C) depth_tokens
                                    ↓
                        BiCMA Fusion (iterative)
                                    ↓
                        fused_rgb_tokens (B, N, C)
                                    ↓
                        Swin Transformer stages 1-4
                                    ↓
                        Multi-scale feature maps
"""

import logging
import types

import torch
import torch.nn as nn

# Reuse the RGBD data mapper and depth stats from the early fusion variant,
# since both variants need the same data loading pipeline (4-ch RGBD input)
# and the same depth normalization statistics.
from variants.rgbd_early import RGBDMapper, _compute_depth_stats

logger = logging.getLogger(__name__)


# ===================================================================
# BiCMA Fusion Module
# ===================================================================

class BiCMAFusion(nn.Module):
    """Bidirectional cross-modal attention fusion with learnable blend rates.

    Uses each modality's self-similarity matrix (softmax-normalized)
    to re-weight the other modality's token sequence. Each stream has
    its own learnable scalar alpha (sigmoid-gated) that controls how
    much of the cross-modal update to mix in:

        X_rgb  = (1 - alpha_rgb)   * X_rgb  + alpha_rgb   * update_rgb
        X_D    = (1 - alpha_depth) * X_D    + alpha_depth * update_D

    Separate alphas allow the model to learn that RGB (which carries
    pretrained ImageNet features) needs more conservative fusion than
    depth (which starts from averaged RGB filters with no task-relevant
    pretraining). Both are initialized so sigmoid gives ~0.1.

    The fusion is "bidirectional" because information flows both ways:
      1. Depth → RGB: depth similarity guides which RGB tokens to mix
      2. RGB → Depth: RGB similarity guides which depth tokens to mix

    This can be iterated multiple times (num_iters) for deeper
    cross-modal interaction, though the default is 1 iteration.
    """

    def __init__(self, embed_dim, num_iters=2):
        super().__init__()
        self.num_iters = num_iters
        # Scaling factor 1/sqrt(C) to prevent dot products from becoming
        # too large before softmax (same trick as in standard attention).
        # Without this, softmax would produce near-one-hot distributions,
        # causing gradient vanishing.
        self.scale = embed_dim ** -0.5
        # Learnable blend rates, one per stream. Initialized to -2.2 so
        # sigmoid(-2.2) ≈ 0.1 — conservative start that protects pretrained
        # RGB features while still allowing the model to open up fusion.
        self.alpha_rgb   = nn.Parameter(torch.tensor(-2.2))
        self.alpha_depth = nn.Parameter(torch.tensor(-2.2))

    def forward(self, rgb_tokens, depth_tokens):
        """
        Args:
            rgb_tokens:   (B, N, C) token embeddings from patch_embed_rgb
                          B = batch size, N = number of patches, C = embed_dim
            depth_tokens: (B, N, C) token embeddings from patch_embed_depth
        Returns:
            (B, N, C) depth-informed RGB tokens (ready for Swin stages)
        """
        alpha_rgb   = torch.sigmoid(self.alpha_rgb)
        alpha_depth = torch.sigmoid(self.alpha_depth)

        for _ in range(self.num_iters):
            # === Step 1: Depth-guided RGB update ===
            # Compute depth self-similarity: how similar is each depth
            # patch to every other depth patch?
            # depth_sim shape: (B, N, N)
            depth_sim = torch.bmm(
                depth_tokens, depth_tokens.transpose(1, 2)
            ) * self.scale  # scale by 1/sqrt(C) for numerical stability

            # Use the depth similarity as attention weights over RGB tokens.
            # Softmax normalizes each row to sum to 1, creating a weighted
            # average. Effect: RGB patches that correspond to similar depth
            # regions get mixed together.
            attn_depth = torch.softmax(depth_sim, dim=-1)

            # compute update (DO NOT overwrite yet)
            rgb_update = torch.bmm(attn_depth, rgb_tokens)

            # residual blend — alpha_rgb learned separately from depth
            rgb_tokens = (1 - alpha_rgb) * rgb_tokens + alpha_rgb * rgb_update

            # === Step 2: RGB-guided Depth update ===
            # Same logic in reverse: use RGB similarity to re-weight
            # depth tokens. Note that we use the UPDATED rgb_tokens from
            # Step 1, so the depth update benefits from the already-fused
            # RGB information.
            # rgb_sim shape: (B, N, N)
            rgb_sim = torch.bmm(
                rgb_tokens, rgb_tokens.transpose(1, 2)
            ) * self.scale

            attn_rgb = torch.softmax(rgb_sim, dim=-1)

            # compute update
            depth_update = torch.bmm(attn_rgb, depth_tokens)

            # residual blend — alpha_depth learned separately from rgb
            depth_tokens = (1 - alpha_depth) * depth_tokens + alpha_depth * depth_update

        # Only return the RGB tokens — the depth stream has served its
        # purpose by informing the RGB features. The Swin Transformer
        # stages downstream only expect one token sequence.
        return rgb_tokens


# ===================================================================
# Custom forward for the Swin backbone
# ===================================================================

def _bicma_forward(self, x):
    """Modified SwinTransformer.forward() with dual PatchEmbed + BiCMA fusion.

    This function REPLACES the original Swin Transformer's forward method
    via monkey-patching (see update_model below). It adds the dual-stream
    patch embedding and BiCMA fusion step before the standard Swin stages.

    The original Swin forward does:
        x → patch_embed → pos_drop → stages → output features

    Our modified forward does:
        x → split RGB/depth
          → patch_embed_rgb(RGB) + patch_embed_depth(depth)
          → BiCMA fusion
          → pos_drop → stages → output features

    Args:
        self: The D2SwinTransformer backbone instance (bound via MethodType)
        x: (B, 4, H, W) tensor — channels 0-2 are RGB, channel 3 is depth
           (concatenated by RGBDMapper during data loading)

    Returns:
        dict of multi-scale feature maps: {"res2": ..., "res3": ..., ...}
        These feed into Mask2Former's pixel decoder (MSDeformAttn).
    """
    # Split the 4-channel input into RGB and depth
    rgb = x[:, :3, :, :]    # (B, 3, H, W) — RGB channels
    depth = x[:, 3:, :, :]  # (B, 1, H, W) — depth channel

    # Dual-stream patch embedding: each stream independently projects
    # its input into patch tokens.
    # PatchEmbed uses a Conv2d(in_ch, 96, kernel=4, stride=4) to convert
    # non-overlapping 4x4 pixel patches into 96-dim embeddings.
    # Output shape: (B, C, Wh, Ww) where Wh, Ww = H/4, W/4
    rgb_embed = self.patch_embed_rgb(rgb)      # (B, 96, H/4, W/4)
    depth_embed = self.patch_embed_depth(depth) # (B, 96, H/4, W/4)

    # Spatial dimensions of the patch grid (needed by Swin stages later)
    Wh, Ww = rgb_embed.size(2), rgb_embed.size(3)

    # Flatten spatial dimensions and transpose to get token sequences.
    # (B, C, Wh, Ww) → flatten(2) → (B, C, N) → transpose → (B, N, C)
    # where N = Wh * Ww is the number of patches (tokens).
    rgb_tokens = rgb_embed.flatten(2).transpose(1, 2)    # (B, N, 96)
    depth_tokens = depth_embed.flatten(2).transpose(1, 2) # (B, N, 96)

    # BiCMA fusion: cross-modal attention between RGB and depth tokens.
    # Returns depth-informed RGB tokens (B, N, C). The depth stream is
    # discarded after this point.
    x = self.fusion(rgb_tokens, depth_tokens)

    # Positional dropout (standard Swin regularization)
    x = self.pos_drop(x)

    # === Standard Swin Transformer stages ===
    # Swin-Tiny has 4 stages with progressively:
    #   - halved spatial resolution (via patch merging)
    #   - doubled channel dimension
    # Stage features at out_indices are collected for multi-scale output.
    outs = {}
    for i in range(self.num_layers):
        layer = self.layers[i]
        # Each Swin stage returns:
        #   x_out: output tokens for this stage (before downsampling)
        #   H, W: spatial dims of x_out
        #   x: tokens after downsampling (input to next stage)
        #   Wh, Ww: spatial dims of x (after downsampling)
        x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

        # Collect features from stages specified in out_indices
        # (typically all 4 stages for FPN-style multi-scale features)
        if i in self.out_indices:
            norm_layer = getattr(self, f"norm{i}")
            x_out = norm_layer(x_out)
            # Reshape from (B, H*W, C) → (B, C, H, W) for conv-based
            # downstream modules (pixel decoder, FPN, etc.)
            out = (
                x_out.view(-1, H, W, self.num_features[i])
                .permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
                .contiguous()
            )
            # Detectron2 convention: features named "res2" through "res5"
            # (matching ResNet naming for compatibility with FPN)
            outs[f"res{i + 2}"] = out

    return outs


# ===================================================================
# Variant interface
# ===================================================================

def update_config(cfg):
    """Extend PIXEL_MEAN and PIXEL_STD to 4 channels (RGB + depth).

    Identical to the early fusion variant's update_config — both variants
    use the same 4-channel input format and need the same normalization.
    The first 3 values are ImageNet RGB stats; the 4th is computed from
    the Rob2Pheno depth images.
    """
    depth_mean, depth_std = _compute_depth_stats()
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530, depth_mean]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375, depth_std]


def update_model(model, cfg):
    """Replace the single PatchEmbed with dual streams + BiCMA fusion.

    This is more invasive than early fusion's simple conv swap. We:
      1. Create TWO PatchEmbed modules (one for RGB, one for depth)
      2. Create a BiCMA fusion module
      3. Attach all three to the backbone
      4. Delete the original single PatchEmbed
      5. Monkey-patch the backbone's forward() method to use the new
         dual-stream + fusion pipeline

    Weight initialization strategy:
      - RGB PatchEmbed: exact copy of pretrained COCO weights (so the
        RGB stream starts at the same quality as the pretrained model)
      - Depth PatchEmbed: weights initialized by averaging the 3 RGB
        channels of the pretrained conv filters into 1 channel. This is
        better than random init because the pretrained filters already
        encode useful low-level features (edges, textures). Averaging
        across RGB gives a reasonable grayscale-like filter.
      - BiCMA fusion: alpha_rgb and alpha_depth initialized to sigmoid(-2.2) ≈ 0.1
    """
    # Get the PatchEmbed class from the existing instance, so we create
    # new instances of the exact same class (handles any subclassing).
    PatchEmbed = type(model.backbone.patch_embed)

    backbone = model.backbone
    old_pe = backbone.patch_embed     # The original 3-channel PatchEmbed
    embed_dim = backbone.embed_dim    # 96 for Swin-Tiny
    device = old_pe.proj.weight.device
    dtype = old_pe.proj.weight.dtype

    # --- RGB PatchEmbed: copy pretrained weights exactly ---
    # This stream handles the 3-channel RGB input identically to the
    # original pretrained model.
    patch_embed_rgb = PatchEmbed(
        patch_size=old_pe.patch_size,  # (4, 4) for Swin-Tiny
        in_chans=3,                    # 3 RGB channels
        embed_dim=embed_dim,           # 96
        norm_layer=nn.LayerNorm,       # LayerNorm after patch projection
    ).to(device=device, dtype=dtype)
    # Load the pretrained weights directly — an exact copy
    patch_embed_rgb.load_state_dict(old_pe.state_dict())

    # --- Depth PatchEmbed: mean-init from RGB filters ---
    # This stream handles the 1-channel depth input.
    patch_embed_depth = PatchEmbed(
        patch_size=old_pe.patch_size,  # same patch size as RGB
        in_chans=1,                    # single depth channel
        embed_dim=embed_dim,           # same embedding dimension
        norm_layer=nn.LayerNorm,
    ).to(device=device, dtype=dtype)
    # Initialize the depth conv filter by averaging the RGB filters:
    # old weight shape: (96, 3, 4, 4) → mean over dim=1 → (96, 1, 4, 4)
    # This gives each output filter a grayscale-like version of the
    # pretrained RGB filter — a much better starting point than random.
    patch_embed_depth.proj.weight.data = (
        old_pe.proj.weight.data.mean(dim=1, keepdim=True)
    )
    # Copy the bias directly (same output channels, same bias)
    patch_embed_depth.proj.bias.data = old_pe.proj.bias.data.clone()
    # Copy the LayerNorm parameters (same embed_dim, same normalization)
    if hasattr(old_pe, "norm") and old_pe.norm is not None:
        patch_embed_depth.norm.load_state_dict(old_pe.norm.state_dict())

    # --- BiCMA fusion module ---
    # num_iters=1 means one round of bidirectional cross-modal attention.
    # More iterations would allow deeper cross-modal interaction but
    # increase computation (quadratic in the number of tokens).
    fusion = BiCMAFusion(embed_dim=embed_dim, num_iters=2).to(
        device=device, dtype=dtype
    )

    # --- Attach new modules to backbone and monkey-patch forward ---
    # Register the new modules as attributes so PyTorch tracks their
    # parameters (for optimizer, device placement, state_dict, etc.)
    backbone.patch_embed_rgb = patch_embed_rgb
    backbone.patch_embed_depth = patch_embed_depth
    backbone.fusion = fusion
    # Remove the old single-stream PatchEmbed to avoid confusion
    # and prevent its parameters from being included in the optimizer
    del backbone.patch_embed

    # Replace the backbone's forward() method with our custom version
    # that handles dual-stream embedding + fusion.
    # types.MethodType binds _bicma_forward to the backbone instance,
    # so 'self' inside _bicma_forward refers to the backbone.
    backbone.forward = types.MethodType(_bicma_forward, backbone)

    logger.info(
        "BiCMA variant: dual PatchEmbed (rgb=%d-ch, depth=%d-ch) "
        "+ %d-iter fusion at embed_dim=%d",
        3, 1, fusion.num_iters, embed_dim,
    )
    return model


def get_mapper(cfg, is_train):
    """Return a custom dataset mapper for 4-channel RGBD input.

    Reuses RGBDMapper from rgbd_early.py — both variants need the same
    data loading pipeline (load RGB + depth TIFF, concatenate to 4-ch,
    apply geometric augmentations). The difference is entirely in how
    the model processes the 4 channels, not in how they're loaded.
    """
    return RGBDMapper(cfg, is_train)
