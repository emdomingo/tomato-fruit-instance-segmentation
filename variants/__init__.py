"""Variant registry. Maps variant name strings to their modules."""

from variants import rgb, rgbd_early

VARIANTS = {
    "rgb": rgb,
    "rgbd_early": rgbd_early,
}
