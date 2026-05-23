"""RGB-D Early Fusion variant: concatenate depth as a 4th input channel.

Expands PatchEmbed.proj from Conv2d(3, 96) to Conv2d(4, 96), copying
pretrained RGB weights and zero-initializing the depth channel.

Depth TIFFs in Rob2Pheno are 8-bit RGB images. This variant converts
them to single-channel grayscale before concatenating with RGB.

Overview of Early Fusion
------------------------
"Early fusion" means we merge RGB and depth at the very first layer of the
network — before any learned feature extraction happens. Concretely:

  1. The original Swin-Tiny backbone expects a 3-channel (RGB) input.
  2. We expand the first convolution (PatchEmbed.proj) to accept 4 channels
     (RGB + depth grayscale) instead of 3.
  3. The pretrained COCO weights for the RGB channels are preserved; the new
     depth channel's convolution weights are zero-initialized so the model
     starts from the same effective state as the RGB baseline and gradually
     learns to incorporate depth information during fine-tuning.

This is the simplest possible RGB-D fusion strategy and serves as a baseline
for comparison against more sophisticated approaches (e.g., BiCMA).
"""

import copy
import logging
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util  # COCO RLE mask encode/decode utilities
import torch
import torch.nn as nn
from torch.nn import functional as F

# Detectron2 utilities for image loading, augmentations, and annotation structures
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Depth statistics — lazily computed from training-split depth TIFFs
# ---------------------------------------------------------------------------

# Cache keyed by depth-dir path so we only scan once per process per dataset.
# Both rgbd_early and rgbd_bicma call _compute_depth_stats(); the cache
# ensures the (potentially slow) I/O only happens on the first call.
_depth_stats_cache: dict = {}


def _resolve_depth_dir(cfg=None):
    """Return the Depth/ directory for the active training dataset.

    If `cfg` is passed and `cfg.DATASETS.TRAIN[0]` is registered with a
    loader (the usual case in train.py), peek at the first dataset record
    and derive `<rgb_dir>/../Depth`. Falls back to Rob2Pheno when no cfg
    is supplied — useful for ad-hoc calls / notebooks.
    """
    if cfg is not None:
        try:
            from detectron2.data import DatasetCatalog
            ds_name = cfg.DATASETS.TRAIN[0]
            records = DatasetCatalog.get(ds_name)
            if records:
                first = Path(records[0]["file_name"])
                return first.parent.parent / "Depth"
        except Exception:
            pass
    return (
        Path(__file__).resolve().parent.parent
        / "data" / "Rob2Pheno" / "Depth"
    )


def _resolve_training_depth_paths(cfg, depth_dir):
    """Return depth TIFF paths restricted to the training split.

    Reads the active training dataset's records (each has an RGB
    "file_name") and maps each RGB filename to its corresponding depth
    TIFF via `_depth_path_from_rgb`. Returns only paths that exist on
    disk; raises if none of the training records have a matching depth
    TIFF. If cfg is None or the dataset can't be resolved, returns None
    so the caller can fall back to scanning the directory.
    """
    if cfg is None:
        return None
    try:
        from detectron2.data import DatasetCatalog
        ds_name = cfg.DATASETS.TRAIN[0]
        records = DatasetCatalog.get(ds_name)
    except Exception:
        return None
    if not records:
        return None

    paths = []
    for r in records:
        depth_path = Path(_depth_path_from_rgb(r["file_name"]))
        if depth_path.exists():
            paths.append(depth_path)
    if not paths:
        raise FileNotFoundError(
            f"No depth TIFFs found for training records of '{ds_name}' "
            f"(looked under {depth_dir})"
        )
    return sorted(paths)


def _compute_depth_stats(cfg=None):
    """Compute mean/std of grayscale depth across the training depth TIFFs.

    Values are in [0, 255] range (8-bit grayscale), matching the scale of
    the RGB PIXEL_MEAN / PIXEL_STD values used by Detectron2's normalizer.

    Detectron2's ImageList normalizes every input channel with:
        pixel = (pixel - PIXEL_MEAN[ch]) / PIXEL_STD[ch]
    So we need to provide the depth channel's mean and std in the same
    [0, 255] pixel-value scale that the RGB stats use.

    Restricts the scan to depth TIFFs corresponding to the training
    split (`cfg.DATASETS.TRAIN[0]`'s records). Falls back to globbing
    the whole Depth/ directory only when no cfg is supplied (e.g.
    notebook / ad-hoc use).

    Returns:
        (mean, std) tuple of floats in [0, 255] range.
    """
    from PIL import Image

    depth_dir = _resolve_depth_dir(cfg)
    # Cache key includes whether we're scoped to a training split so that
    # an ad-hoc no-cfg call doesn't poison the cache for a later training run.
    cache_key = (str(depth_dir), cfg.DATASETS.TRAIN[0] if cfg is not None else None)
    cached = _depth_stats_cache.get(cache_key)
    if cached is not None:
        return cached

    paths = _resolve_training_depth_paths(cfg, depth_dir)
    if paths is None:
        paths = sorted(depth_dir.glob("*_DEPTH.tiff"))
        if not paths:
            raise FileNotFoundError(
                f"No *_DEPTH.tiff files in {depth_dir}"
            )

    # Use running sums (Welford-like, without the online variance part)
    # to avoid loading all images into memory at once.
    # Zero pixels are the dataset's invalid-sensor sentinel (~27% of pixels
    # in Rob2Pheno), so exclude them from the statistics — otherwise mean
    # is biased low and std inflated by a value that doesn't represent a
    # real measurement.
    running_sum = 0.0
    running_sq_sum = 0.0
    valid_count = 0
    total_count = 0

    for p in paths:
        # Convert to "L" (luminance) = 8-bit single-channel grayscale.
        # Rob2Pheno depth TIFFs are stored as 3-channel RGB images where
        # all channels have the same value, so .convert("L") collapses them.
        img = Image.open(p).convert("L")
        depth = np.asarray(img, dtype=np.float64)
        valid = depth > 0
        valid_depth = depth[valid]
        running_sum += valid_depth.sum()
        running_sq_sum += (valid_depth ** 2).sum()
        valid_count += int(valid.sum())
        total_count += depth.size

    # Population mean and std over valid pixels only.
    # std = sqrt(E[X^2] - (E[X])^2)
    mean = running_sum / valid_count
    std = np.sqrt(running_sq_sum / valid_count - mean ** 2)

    result = (float(mean), float(std))
    _depth_stats_cache[cache_key] = result
    logger.info(
        "Depth stats (valid pixels only): mean=%.3f, std=%.3f "
        "(%d files in %s; valid fraction=%.3f)",
        mean, std, len(paths), depth_dir, valid_count / total_count,
    )
    return result


# ===================================================================
# Variant interface
# ===================================================================

def update_config(cfg):
    """Extend PIXEL_MEAN and PIXEL_STD to 4 channels (RGB + depth).

    The first three values are ImageNet RGB statistics (the same ones used
    by the pretrained Mask2Former/Swin-Tiny checkpoint). The 4th value is
    computed from the Rob2Pheno depth images so the depth channel is
    normalized to roughly zero-mean, unit-variance — just like the RGB
    channels.

    This is called by train.py BEFORE the config is frozen.
    """
    depth_mean, depth_std = _compute_depth_stats(cfg)
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530, depth_mean]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375, depth_std]


def update_model(model, cfg):
    """Replace the backbone's 3-channel PatchEmbed conv with a 4-channel one.

    The PatchEmbed layer is the very first operation in the Swin Transformer.
    It projects non-overlapping image patches (4x4 pixels) into the
    embedding dimension (96 for Swin-Tiny) via a Conv2d with kernel_size=4
    and stride=4.

    Strategy:
      - Create a new Conv2d(in_channels=4, out_channels=96, kernel=4, stride=4)
      - Copy the pretrained 3-channel weights into channels 0-2
      - Zero-initialize channel 3 (depth) so the model initially ignores
        depth and gradually learns to use it during fine-tuning
      - Swap it into the model in-place

    This is called by train.py AFTER the model is constructed but BEFORE
    training begins.
    """
    # The original PatchEmbed conv: Conv2d(3, 96, kernel_size=4, stride=4)
    old_conv = model.backbone.patch_embed.proj
    new_conv = nn.Conv2d(
        4, old_conv.out_channels,              # 4 input channels instead of 3
        kernel_size=old_conv.kernel_size,       # 4×4 patch size
        stride=old_conv.stride,                # stride=4 (non-overlapping patches)
    )

    # Move new conv to same device/dtype as old conv before copying weights.
    # Important when loading pretrained weights on GPU or in mixed precision.
    new_conv = new_conv.to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)

    # Copy pretrained RGB weights into the first 3 channels.
    # old_conv.weight shape: (96, 3, 4, 4) — [out_ch, in_ch, kH, kW]
    # new_conv.weight shape: (96, 4, 4, 4)
    new_conv.weight.data[:, :3, :, :] = old_conv.weight.data  # RGB channels
    new_conv.bias.data = old_conv.bias.data                    # copy bias as-is
    new_conv.weight.data[:, 3:, :, :] = 0  # depth channel starts at zero

    # Swap the conv layer in-place within the model's PatchEmbed module
    model.backbone.patch_embed.proj = new_conv
    return model


def get_mapper(cfg, is_train):
    """Return a custom dataset mapper that loads RGB + depth as 4-ch input.

    Detectron2 uses "mappers" to transform raw dataset dicts (file paths,
    annotations) into the tensor format the model expects. The default
    mapper only loads 3-channel RGB images; our RGBDMapper additionally
    loads the corresponding depth TIFF and concatenates it as a 4th channel.
    """
    return RGBDMapper(cfg, is_train)


# ===================================================================
# Dataset mapper
# ===================================================================

def _depth_path_from_rgb(rgb_path):
    """Derive the depth TIFF path from the corresponding RGB image path.

    The Rob2Pheno dataset has a parallel directory structure:
        data/Rob2Pheno/RGB/PLANT001_RGB.tiff
        data/Rob2Pheno/Depth/PLANT001_DEPTH.tiff

    So we just swap the directory from "RGB" to "Depth" and the filename
    suffix from "_RGB.tiff" to "_DEPTH.tiff".
    """
    p = Path(rgb_path)
    depth_name = p.name.replace("_RGB.tiff", "_DEPTH.tiff")
    depth_dir = p.parent.parent / "Depth"  # Go up from RGB/, then into Depth/
    return str(depth_dir / depth_name)


class RGBDMapper:
    """Dataset mapper that loads RGB + depth and returns a 4-channel tensor.

    This is a custom replacement for Mask2Former's built-in
    ``MaskFormerInstanceDatasetMapper``. It does the same thing — loads an
    image, applies augmentations, and converts annotations to instance
    masks — but additionally loads the corresponding depth TIFF and
    concatenates it as a 4th channel.

    Why a custom mapper?
      Detectron2's default data pipeline only supports 3-channel images.
      To feed 4-channel (RGBD) tensors to the model, we need to control
      the data loading ourselves. The key difference from the default:
        1. We load the depth image alongside RGB.
        2. We concatenate them into a (H, W, 4) array BEFORE augmentation
           so that geometric transforms (resize, crop, flip) are applied
           identically to both modalities.
        3. We omit ColorAugSSDTransform (color jitter) because it would
           corrupt the depth channel — depth values are physical
           measurements, not visual colors.

    Data flow:
      dataset_dict (from Detectron2 DatasetCatalog)
        → load RGB image (H, W, 3) uint8
        → load depth TIFF → grayscale (H, W) uint8
        → concatenate → (H, W, 4) uint8
        → apply geometric augmentations (resize, crop, flip)
        → convert annotations to instance masks
        → to tensors (C, H, W) float
        → pad to size_divisibility
        → return dict with "image" and "instances"
    """

    def __init__(self, cfg, is_train=True, include_validity=False):
        self.is_train = is_train
        self.img_format = cfg.INPUT.FORMAT      # Usually "BGR" for Detectron2
        self.size_divisibility = cfg.INPUT.SIZE_DIVISIBILITY  # Usually 0 or 32

        # Invalid-depth fill value: replace sensor-zero pixels with the
        # training-set mean so that after Detectron2's per-channel
        # normalization (subtract mean, divide by std) they become ~0 —
        # a neutral, no-signal value rather than a -2σ outlier.
        depth_mean, _ = _compute_depth_stats(cfg)
        self._depth_fill = np.uint8(np.clip(round(depth_mean), 0, 255))

        # When True, append a 5th uint8 channel encoding the pre-fill
        # depth validity (255 where original depth > 0, else 0). Variants
        # like rgbd_dca_mask consume this to mask invalid tokens in
        # cross-attention. Padded regions of the validity channel are
        # filled with 0 (invalid).
        self._include_validity = include_validity

        # Build augmentation list.
        # We use the same geometric augmentations as MaskFormerInstanceDatasetMapper,
        # but OMIT ColorAugSSDTransform because color jitter would corrupt depth values.
        # Geometric transforms (resize, crop, flip) are channel-agnostic and safe.
        if is_train:
            self.tfm_gens = [
                # Resize so the shortest edge is within the configured range.
                # This maintains aspect ratio.
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,        # list of possible sizes
                    cfg.INPUT.MAX_SIZE_TRAIN,        # cap on longest edge
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,  # "choice" or "range"
                ),
            ]
            # Optional random crop (not typically enabled for instance segmentation)
            if cfg.INPUT.CROP.ENABLED:
                self.tfm_gens.append(
                    T.RandomCrop(
                        cfg.INPUT.CROP.TYPE,  # "relative" or "absolute"
                        cfg.INPUT.CROP.SIZE,  # crop dimensions
                    )
                )
            # Random horizontal flip with default 50% probability
            self.tfm_gens.append(T.RandomFlip())
        else:
            # During inference/validation, only resize (no random augmentations)
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
        """Transform a single dataset sample into the format the model expects.

        Args:
            dataset_dict: A dict from Detectron2's DatasetCatalog containing
                at minimum:
                  - "file_name": path to the RGB TIFF
                  - "annotations": list of instance annotation dicts, each
                    with "segmentation" (polygon or RLE) and "category_id"
                  - "height", "width": original image dimensions

        Returns:
            dict with:
              - "image": (4, H, W) float tensor (RGBD, unnormalized —
                normalization happens later in the model's ImageList)
              - "instances": Instances object with gt_classes and gt_masks
        """
        # Deep copy to avoid mutating the original dataset record, since
        # Detectron2 may reuse dataset_dicts across epochs.
        dataset_dict = copy.deepcopy(dataset_dict)

        # --- Load RGB (H, W, 3) uint8 --------------------------------
        # utils.read_image handles TIFF/PNG/JPG and applies format
        # conversion (e.g., BGR ordering if cfg.INPUT.FORMAT == "BGR").
        rgb = utils.read_image(
            dataset_dict["file_name"], format=self.img_format
        )
        # Verify the loaded image dimensions match the dataset metadata
        utils.check_image_size(dataset_dict, rgb)

        # --- Load depth → grayscale (H, W) uint8 ---------------------
        from PIL import Image

        # Find the matching depth TIFF for this RGB image
        depth_path = _depth_path_from_rgb(dataset_dict["file_name"])
        # Convert to single-channel grayscale ("L" mode = luminance).
        # Rob2Pheno depth TIFFs are stored as 3-channel images where R=G=B,
        # so .convert("L") collapses them to a single channel.
        raw_depth = np.asarray(
            Image.open(depth_path).convert("L"), dtype=np.uint8
        )
        # 0 is the dataset's invalid-pixel sentinel. Replace with the
        # training mean so post-normalization these pixels are ~0 (neutral)
        # instead of a -2σ outlier the model has to learn to ignore.
        depth = np.where(raw_depth > 0, raw_depth, self._depth_fill).astype(np.uint8)

        # --- Concatenate → (H, W, 4 or 5) uint8 ----------------------
        # Add a channel dimension to depth (H, W) → (H, W, 1), then
        # concatenate with RGB (H, W, 3) → RGBD (H, W, 4).
        # This must happen BEFORE augmentations so geometric transforms
        # are applied identically to both modalities.
        # If include_validity, append a 5th channel = pre-fill mask
        # (255 valid, 0 invalid). Stored in 0/255 range so it survives
        # the same geometric augs as the other channels.
        channels = [rgb, depth[..., None]]
        if self._include_validity:
            validity = ((raw_depth > 0).astype(np.uint8) * 255)[..., None]
            channels.append(validity)
        image = np.concatenate(channels, axis=-1)

        # --- Geometric augmentations (channel-agnostic) ---------------
        # AugInput wraps the image so Detectron2's transform pipeline can
        # modify it. apply_transform_gens returns the transformed image
        # AND the list of transforms applied, which we need to apply the
        # same transforms to the annotation masks.
        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(
            self.tfm_gens, aug_input
        )
        image = aug_input.image  # (H', W', 4) after resize/crop/flip

        # --- Process annotations (masks) ------------------------------
        assert "annotations" in dataset_dict
        # Remove keypoint annotations (not used in instance segmentation)
        for anno in dataset_dict["annotations"]:
            anno.pop("keypoints", None)

        # Apply the SAME geometric transforms to each annotation's
        # bounding box and segmentation mask. Skip "iscrowd" annotations
        # (COCO convention for crowd regions, not used in Rob2Pheno but
        # handled for safety).
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image.shape[:2]  # pass new (H', W')
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        if len(annos):
            assert "segmentation" in annos[0]

        # Convert segmentations to binary masks.
        # COCO annotations can be in three formats:
        #   1. list of polygons (most common) → polygons_to_bitmask
        #   2. dict (RLE-encoded mask) → mask_util.decode
        #   3. numpy array (already a bitmap)
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            if isinstance(segm, list):
                # Polygon format: list of [x1,y1,x2,y2,...] vertex lists
                masks.append(
                    polygons_to_bitmask(segm, *image.shape[:2])
                )
            elif isinstance(segm, dict):
                # COCO RLE (run-length encoding) format
                masks.append(mask_util.decode(segm))
            elif isinstance(segm, np.ndarray):
                # Already a binary mask array
                assert segm.ndim == 2
                masks.append(segm)
            else:
                raise ValueError(
                    f"Cannot convert segmentation of type "
                    f"'{type(segm)}' to BitMasks!"
                )

        # --- To tensors -----------------------------------------------
        # Convert image from (H, W, 4) numpy → (4, H, W) torch tensor.
        # .transpose(2, 0, 1) reorders from HWC → CHW (PyTorch convention).
        # np.ascontiguousarray ensures the memory layout is contiguous
        # (required for efficient tensor creation after the transpose).
        image = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        # Convert each (H, W) binary mask numpy array to a torch tensor
        masks = [
            torch.from_numpy(np.ascontiguousarray(x)) for x in masks
        ]

        # Extract class IDs (0 = ripe/red, 1 = unripe/green in Rob2Pheno)
        classes = torch.tensor(
            [int(obj["category_id"]) for obj in annos],
            dtype=torch.int64,
        )

        # --- Pad to size_divisibility ---------------------------------
        # Some architectures require spatial dimensions to be divisible by
        # a certain number (e.g., 32 for feature pyramid networks).
        # If size_divisibility > 0, pad the image and masks to meet this.
        if self.size_divisibility > 0:
            h, w = image.shape[-2], image.shape[-1]
            # F.pad format: (left, right, top, bottom)
            padding_size = [
                0, self.size_divisibility - w,    # left=0, right padding
                0, self.size_divisibility - h,    # top=0, bottom padding
            ]
            # Pad image with 128 (mid-gray), masks with 0 (background).
            # If validity channel is present, pad it with 0 (invalid)
            # instead of 128 — padded regions have no real measurement.
            if self._include_validity:
                img_pad = F.pad(image[:4], padding_size, value=128)
                mask_pad = F.pad(image[4:5], padding_size, value=0)
                image = torch.cat([img_pad, mask_pad], dim=0).contiguous()
            else:
                image = F.pad(image, padding_size, value=128).contiguous()
            masks = [
                F.pad(x, padding_size, value=0).contiguous()
                for x in masks
            ]

        image_shape = (image.shape[-2], image.shape[-1])  # final (H, W)
        dataset_dict["image"] = image  # (4, H, W) tensor

        # --- Instance annotations -------------------------------------
        # Pack ground-truth classes and masks into a Detectron2 Instances
        # object, which is the standard format the model's loss functions
        # expect.
        instances = Instances(image_shape)
        instances.gt_classes = classes
        if len(masks) == 0:
            # No instances in this image — create an empty mask tensor
            # with the correct spatial dimensions
            instances.gt_masks = torch.zeros(
                (0, image.shape[-2], image.shape[-1])
            )
        else:
            # Stack individual mask tensors into a BitMasks object:
            # (N, H, W) where N = number of instances
            masks = BitMasks(torch.stack(masks))
            instances.gt_masks = masks.tensor

        dataset_dict["instances"] = instances
        return dataset_dict
