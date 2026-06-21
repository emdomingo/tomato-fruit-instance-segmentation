"""RGB-D DCA variant warm-started from a fine-tuned RGB checkpoint.

Identical fusion to `rgbd_dca` (same DCAFusion / K-iter semantics), but the RGB
patch embed is initialized from a fine-tuned RGB-variant checkpoint instead of
the COCO-pretrained / randomly-initialized backbone, and trains at a very small
LR so those weights stay close to their fine-tuned values.

Config (set by train.py from CLI):
    cfg.MODEL.DCA_RGB_INIT_WEIGHTS : path to the RGB checkpoint (.pth) whose
        backbone.patch_embed.* tensors warm-start patch_embed_rgb.
    cfg.MODEL.DCA_RGB_INIT_LR_MULT : LR multiplier (x BASE_LR) applied to
        patch_embed_rgb in Rob2PhenoTrainer.build_optimizer.

patch_embed_depth is mean-initialized from the fine-tuned RGB filters (not the
COCO ones), giving the depth stream a tomato-adapted starting point too.
"""

import logging
import types

import torch
import torch.nn as nn

from variants.rgbd_early import RGBDMapper, _compute_depth_stats
from variants.rgbd_dca import DCAFusion, _dca_forward

logger = logging.getLogger(__name__)


def update_config(cfg):
    """Extend PIXEL_MEAN and PIXEL_STD to 4 channels (RGB + depth)."""
    depth_mean, depth_std = _compute_depth_stats(cfg)
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530, depth_mean]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375, depth_std]


def _extract_patch_embed_state(weights_path):
    """Pull backbone.patch_embed.* tensors out of a detectron2 .pth checkpoint."""
    ckpt = torch.load(weights_path, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    prefix = "backbone.patch_embed."
    pe_sd = {
        k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)
    }
    if not pe_sd:
        raise KeyError(
            f"No '{prefix}*' tensors found in {weights_path}. "
            f"Expected a fine-tuned RGB-variant checkpoint."
        )
    return pe_sd


def update_model(model, cfg):
    """Dual PatchEmbed + DCA fusion, with patch_embed_rgb warm-started."""
    weights_path = getattr(cfg.MODEL, "DCA_RGB_INIT_WEIGHTS", "")
    if not weights_path:
        raise ValueError(
            "rgbd_dca_rgbinit requires --dca-rgb-init-weights pointing at a "
            "fine-tuned RGB checkpoint (.pth)."
        )

    PatchEmbed = type(model.backbone.patch_embed)
    backbone = model.backbone
    old_pe = backbone.patch_embed
    embed_dim = backbone.embed_dim
    device = old_pe.proj.weight.device
    dtype = old_pe.proj.weight.dtype

    rgb_pe_state = _extract_patch_embed_state(weights_path)

    # RGB PatchEmbed: load the fine-tuned RGB-checkpoint weights.
    patch_embed_rgb = PatchEmbed(
        patch_size=old_pe.patch_size,
        in_chans=3,
        embed_dim=embed_dim,
        norm_layer=nn.LayerNorm,
    ).to(device=device, dtype=dtype)
    patch_embed_rgb.load_state_dict(rgb_pe_state)

    # Depth PatchEmbed: mean-init from the fine-tuned RGB filters.
    patch_embed_depth = PatchEmbed(
        patch_size=old_pe.patch_size,
        in_chans=1,
        embed_dim=embed_dim,
        norm_layer=nn.LayerNorm,
    ).to(device=device, dtype=dtype)
    patch_embed_depth.proj.weight.data = (
        patch_embed_rgb.proj.weight.data.mean(dim=1, keepdim=True)
    )
    patch_embed_depth.proj.bias.data = patch_embed_rgb.proj.bias.data.clone()
    if hasattr(old_pe, "norm") and old_pe.norm is not None:
        patch_embed_depth.norm.load_state_dict(patch_embed_rgb.norm.state_dict())

    num_iters = getattr(cfg.MODEL, "DCA_ITERS", 0)
    fusion = DCAFusion(embed_dim=embed_dim, num_iters=num_iters).to(
        device=device, dtype=dtype
    )

    backbone.patch_embed_rgb = patch_embed_rgb
    backbone.patch_embed_depth = patch_embed_depth
    backbone.fusion = fusion
    del backbone.patch_embed

    backbone.forward = types.MethodType(_dca_forward, backbone)

    logger.info(
        "DCA-rgbinit (K=%d iters): patch_embed_rgb warm-started from %s "
        "(lr_mult=%g), patch_embed_depth mean-init from those filters",
        num_iters, weights_path,
        getattr(cfg.MODEL, "DCA_RGB_INIT_LR_MULT", 1.0),
    )
    return model


def get_mapper(cfg, is_train):
    """Return a custom dataset mapper for 4-channel RGBD input."""
    return RGBDMapper(cfg, is_train)
