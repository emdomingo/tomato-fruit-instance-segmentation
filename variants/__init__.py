"""Variant registry. Maps variant name strings to their modules."""

from variants import rgb, rgbd_early, rgbd_bicma, rgbd_bicma_alpha

VARIANTS = {
    "rgb": rgb,
    "rgbd_early": rgbd_early,
    "rgbd_bicma": rgbd_bicma,
    "rgbd_bicma_alpha": rgbd_bicma_alpha
}
