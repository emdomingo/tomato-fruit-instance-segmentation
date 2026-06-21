"""Variant registry. Maps variant name strings to their modules.

Only the three variants reported in the final results are registered here.
Experimental/ablation variants (bicma_alpha, dca_local, dca_mask,
dca_multihead, dca_rgbinit) are kept under ``variant_archive/`` for reference
and are intentionally not registered.
"""

from variants import (
    rgb,
    rgbd_early,
    rgbd_dca,
)

VARIANTS = {
    "rgb": rgb,
    "rgbd_early": rgbd_early,
    "rgbd_dca": rgbd_dca,
}
