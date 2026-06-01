"""Variant registry. Maps variant name strings to their modules."""

from variants import (
    rgb,
    rgbd_early,
    rgbd_bicma_alpha,
    rgbd_dca,
    rgbd_dca_local,
    rgbd_dca_mask,
    rgbd_dca_multihead,
    rgbd_dca_rgbinit,
)

VARIANTS = {
    "rgb": rgb,
    "rgbd_early": rgbd_early,
    "rgbd_bicma_alpha": rgbd_bicma_alpha,
    "rgbd_dca": rgbd_dca,
    "rgbd_dca_local": rgbd_dca_local,
    "rgbd_dca_mask": rgbd_dca_mask,
    "rgbd_dca_multihead": rgbd_dca_multihead,
    "rgbd_dca_rgbinit": rgbd_dca_rgbinit,
}
