"""RGB-D Early Fusion variant: concatenate depth as a 4th input channel.

Expands PatchEmbed.proj from Conv2d(3, 96) to Conv2d(4, 96), copying
pretrained RGB weights and zero-initializing the depth channel.

Depth TIFFs in Rob2Pheno are 8-bit RGB images. This variant converts
them to single-channel grayscale before concatenating with RGB.
"""

import copy
import logging
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
from torch.nn import functional as F

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Depth statistics — lazily computed from Rob2Pheno Depth TIFFs
# ---------------------------------------------------------------------------
_depth_stats_cache = None


def _compute_depth_stats():
    """Compute mean/std of grayscale depth across all Rob2Pheno depth TIFFs.

    Values are in [0, 255] range (8-bit grayscale), matching the scale of
    the RGB PIXEL_MEAN / PIXEL_STD values.
    """
    global _depth_stats_cache
    if _depth_stats_cache is not None:
        return _depth_stats_cache

    from PIL import Image

    depth_dir = (
        Path(__file__).resolve().parent.parent
        / "data" / "Rob2Pheno" / "Depth"
    )
    paths = sorted(depth_dir.glob("*_DEPTH.tiff"))
    if not paths:
        raise FileNotFoundError(
            f"No *_DEPTH.tiff files in {depth_dir}"
        )

    running_sum = 0.0
    running_sq_sum = 0.0
    pixel_count = 0

    for p in paths:
        img = Image.open(p).convert("L")  # → 8-bit grayscale
        depth = np.asarray(img, dtype=np.float64)
        running_sum += depth.sum()
        running_sq_sum += (depth ** 2).sum()
        pixel_count += depth.size

    mean = running_sum / pixel_count
    std = np.sqrt(running_sq_sum / pixel_count - mean ** 2)

    _depth_stats_cache = (float(mean), float(std))
    logger.info(
        "Depth stats (grayscale): mean=%.3f, std=%.3f (%d files)",
        mean, std, len(paths),
    )
    return _depth_stats_cache


# ===================================================================
# Variant interface
# ===================================================================

def update_config(cfg):
    """Extend PIXEL_MEAN and PIXEL_STD to 4 channels (RGB + depth)."""
    depth_mean, depth_std = _compute_depth_stats()
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530, depth_mean]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375, depth_std]


def update_model(model, cfg):
    """Replace 3→4 channel PatchEmbed.proj, copying RGB weights."""
    old_conv = model.backbone.patch_embed.proj  # Conv2d(3, 96, 4, 4)
    new_conv = nn.Conv2d(
        4, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
    )

    # Copy pretrained RGB weights; zero-init depth channel
    new_conv.weight.data[:, :3, :, :] = old_conv.weight.data
    new_conv.bias.data = old_conv.bias.data
    new_conv.weight.data[:, 3:, :, :] = 0

    model.backbone.patch_embed.proj = new_conv
    return model


def get_mapper(cfg, is_train):
    """Return a mapper that loads RGB + depth as a 4-channel input."""
    return RGBDMapper(cfg, is_train)


# ===================================================================
# Dataset mapper
# ===================================================================

def _depth_path_from_rgb(rgb_path):
    """Derive depth TIFF path from the corresponding RGB path.

    data/Rob2Pheno/RGB/..._RGB.tiff  →  data/Rob2Pheno/Depth/..._DEPTH.tiff
    """
    p = Path(rgb_path)
    depth_name = p.name.replace("_RGB.tiff", "_DEPTH.tiff")
    depth_dir = p.parent.parent / "Depth"
    return str(depth_dir / depth_name)


class RGBDMapper:
    """Dataset mapper that loads RGB + depth and returns a 4-ch tensor.

    Mirrors ``MaskFormerInstanceDatasetMapper`` but adds a grayscale
    depth channel concatenated after RGB before augmentations.
    """

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.img_format = cfg.INPUT.FORMAT
        self.size_divisibility = cfg.INPUT.SIZE_DIVISIBILITY

        # Build augmentation list (same as MaskFormerInstanceDatasetMapper,
        # but without ColorAugSSDTransform which is RGB-specific).
        if is_train:
            self.tfm_gens = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                ),
            ]
            if cfg.INPUT.CROP.ENABLED:
                self.tfm_gens.append(
                    T.RandomCrop(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                    )
                )
            self.tfm_gens.append(T.RandomFlip())
        else:
            self.tfm_gens = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TEST,
                    cfg.INPUT.MAX_SIZE_TEST,
                ),
            ]

        mode = "training" if is_train else "inference"
        logger.info(
            "[RGBDMapper] Augmentations used in %s: %s",
            mode, self.tfm_gens,
        )

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        # --- Load RGB (H, W, 3) uint8 --------------------------------
        rgb = utils.read_image(
            dataset_dict["file_name"], format=self.img_format
        )
        utils.check_image_size(dataset_dict, rgb)

        # --- Load depth → grayscale (H, W) uint8 ---------------------
        from PIL import Image

        depth_path = _depth_path_from_rgb(dataset_dict["file_name"])
        depth = np.asarray(
            Image.open(depth_path).convert("L"), dtype=np.uint8
        )

        # --- Concatenate → (H, W, 4) uint8 ---------------------------
        image = np.concatenate(
            [rgb, depth[..., None]], axis=-1
        )

        # --- Geometric augmentations (channel-agnostic) ---------------
        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(
            self.tfm_gens, aug_input
        )
        image = aug_input.image

        # --- Process annotations (masks) ------------------------------
        assert "annotations" in dataset_dict
        for anno in dataset_dict["annotations"]:
            anno.pop("keypoints", None)

        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image.shape[:2]
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        if len(annos):
            assert "segmentation" in annos[0]

        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            if isinstance(segm, list):
                masks.append(
                    polygons_to_bitmask(segm, *image.shape[:2])
                )
            elif isinstance(segm, dict):
                masks.append(mask_util.decode(segm))
            elif isinstance(segm, np.ndarray):
                assert segm.ndim == 2
                masks.append(segm)
            else:
                raise ValueError(
                    f"Cannot convert segmentation of type "
                    f"'{type(segm)}' to BitMasks!"
                )

        # --- To tensors -----------------------------------------------
        image = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        masks = [
            torch.from_numpy(np.ascontiguousarray(x)) for x in masks
        ]

        classes = torch.tensor(
            [int(obj["category_id"]) for obj in annos],
            dtype=torch.int64,
        )

        # --- Pad to size_divisibility ---------------------------------
        if self.size_divisibility > 0:
            h, w = image.shape[-2], image.shape[-1]
            padding_size = [
                0, self.size_divisibility - w,
                0, self.size_divisibility - h,
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            masks = [
                F.pad(x, padding_size, value=0).contiguous()
                for x in masks
            ]

        image_shape = (image.shape[-2], image.shape[-1])
        dataset_dict["image"] = image

        # --- Instance annotations -------------------------------------
        instances = Instances(image_shape)
        instances.gt_classes = classes
        if len(masks) == 0:
            instances.gt_masks = torch.zeros(
                (0, image.shape[-2], image.shape[-1])
            )
        else:
            masks = BitMasks(torch.stack(masks))
            instances.gt_masks = masks.tensor

        dataset_dict["instances"] = instances
        return dataset_dict
