"""
Train Mask2Former + Swin-Tiny on Rob2Pheno tomato dataset.

Usage:
    python train.py --variant rgb --num-workers 0          # local Windows
    python train.py --variant rgb --batch-size 4            # HPC
    python train.py --variant rgb --eval-only               # evaluate only
    python train.py --variant rgb --resume                  # resume training

The --variant flag selects an input layer variant from variants/.
See modifications.md for how to create new variants (RGB-D, K-iter fusion).
"""

import argparse
import collections
import copy
import itertools
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import torch

# ---------------------------------------------------------------------------
# Path setup: add Mask2Former to sys.path so its modules are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
M2F_DIR = PROJECT_ROOT / "Mask2Former"
if str(M2F_DIR) not in sys.path:
    sys.path.insert(0, str(M2F_DIR))

# ---------------------------------------------------------------------------
# Detectron2 imports
# ---------------------------------------------------------------------------
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger

# ---------------------------------------------------------------------------
# Mask2Former imports
# ---------------------------------------------------------------------------
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    MaskFormerInstanceDatasetMapper,
    add_maskformer2_config,
)

# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------
from variants import VARIANTS


# ===========================================================================
# Dataset loading & registration
# ===========================================================================

def load_rob2pheno(json_path: str, image_root: str) -> list:
    """
    Load a Rob2Pheno COCO annotation file and return a list of Detectron2
    dataset dicts (one dict per image).

    Args:
        json_path  : Path to the COCO JSON annotation file.
        image_root : Directory containing all the image files.

    Returns:
        List of dicts in Detectron2 format.
    """
    json_path = Path(json_path)
    image_root = Path(image_root)

    with open(json_path) as f:
        coco = json.load(f)

    # COCO 1-indexed → Detectron2 0-indexed
    cat_id_to_d2_id = {
        cat["id"]: idx
        for idx, cat in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))
    }

    img_to_anns = collections.defaultdict(list)
    for ann in coco["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    dataset_dicts = []
    for img_entry in coco["images"]:
        record = {
            "file_name": str(image_root / img_entry["file_name"]),
            "image_id": img_entry["id"],
            "height": img_entry["height"],
            "width": img_entry["width"],
        }

        d2_annotations = []
        for ann in img_to_anns[img_entry["id"]]:
            seg = ann["segmentation"]
            # Fix flat polygon list → nested: [x,y,...] → [[x,y,...]]
            if seg and isinstance(seg[0], (int, float)):
                seg = [seg]

            d2_annotations.append({
                "bbox": ann["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": cat_id_to_d2_id[ann["category_id"]],
                "segmentation": seg,
                "iscrowd": ann.get("iscrowd", 0),
            })

        record["annotations"] = d2_annotations
        dataset_dicts.append(record)

    return dataset_dicts


def register_datasets():
    """Register Rob2Pheno train/val datasets with Detectron2."""
    data_root = PROJECT_ROOT / "data" / "Rob2Pheno"
    rgb_dir = data_root / "RGB"
    train_json = data_root / "train_2class.JSON"
    val_json = data_root / "val_2class.JSON"

    thing_classes = ["redfruit", "greenfruit"]

    # Safe re-registration (for interactive use)
    for name in ["rob2pheno_train", "rob2pheno_val"]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)

    DatasetCatalog.register(
        "rob2pheno_train",
        lambda j=str(train_json), r=str(rgb_dir): load_rob2pheno(j, r),
    )
    MetadataCatalog.get("rob2pheno_train").set(
        thing_classes=thing_classes, evaluator_type="coco",
    )

    DatasetCatalog.register(
        "rob2pheno_val",
        lambda j=str(val_json), r=str(rgb_dir): load_rob2pheno(j, r),
    )
    MetadataCatalog.get("rob2pheno_val").set(
        thing_classes=thing_classes, evaluator_type="coco",
    )


# ===========================================================================
# Config construction
# ===========================================================================

def build_cfg(args):
    """Build the Detectron2 config from base YAML + CLI overrides."""
    cfg = get_cfg()
    add_deeplab_config(cfg)       # registers LR_SCHEDULER_NAME key
    add_maskformer2_config(cfg)   # registers Mask2Former + Swin config keys

    # Required by base YAML chain (Swin config inherits from R50 base)
    cfg.MODEL.RESNETS.STEM_TYPE = "basic"
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 1, 1]

    # Load Swin-Tiny instance segmentation config
    base_config = str(
        M2F_DIR / "configs" / "coco" / "instance-segmentation"
        / "swin" / "maskformer2_swin_tiny_bs16_50ep.yaml"
    )
    cfg.merge_from_file(base_config)

    # -- Model --
    cfg.MODEL.WEIGHTS = str(
        PROJECT_ROOT / "pretrained" / "mask2former_swin_tiny_coco_instance.pkl"
    )
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2  # redfruit, greenfruit

    # -- Datasets --
    cfg.DATASETS.TRAIN = ("rob2pheno_train",)
    cfg.DATASETS.TEST = ("rob2pheno_val",)

    # -- Solver --
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = (3500, 4500)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.TEST.EVAL_PERIOD = 1000

    # -- Input --
    cfg.INPUT.MAX_SIZE_TRAIN = 640
    cfg.INPUT.MAX_SIZE_TEST = 640
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640)
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    # -- Variant-specific config --
    cfg.MODEL.DCA_ITERS = args.dca_iters
    cfg.MODEL.DCA_HEADS = args.dca_heads

    # -- Output --
    if args.variant == "rgbd_dca_multihead":
        suffix = f"_h{args.dca_heads}_iter{args.dca_iters}"
    elif args.variant.startswith("rgbd_dca"):
        suffix = f"_iter{args.dca_iters}"
    else:
        suffix = ""
    if args.green_weight != 1.0:
        suffix += f"_gw{args.green_weight}"
    suffix += f"_mi{args.max_iter}"
    cfg.OUTPUT_DIR = str(
        PROJECT_ROOT / "output" / f"{args.variant}{suffix}_swin_tiny"
    )

    return cfg


# ===========================================================================
# Trainer
# ===========================================================================

class Rob2PhenoTrainer(DefaultTrainer):
    """
    DefaultTrainer subclass with:
      1. Correct Mask2Former dataset mapper (or custom variant mapper)
      2. COCO evaluation
      3. AdamW optimizer with backbone LR multiplier + Swin weight decay rules
         (from Mask2Former/train_net.py)
    """

    # Set by main() before training starts — allows variant to inject custom mappers
    _custom_mapper = None
    _custom_test_mapper = None
    _variant = None

    @classmethod
    def build_model(cls, cfg):
        # Apply the variant's update_model BEFORE the optimizer is built
        # (DefaultTrainer.__init__ calls build_model then build_optimizer),
        # so any new parameters the variant adds are picked up by the
        # optimizer's named_modules() walk in build_optimizer.
        model = super().build_model(cfg)
        if cls._variant is not None:
            model = cls._variant.update_model(model, cfg)
        return model

    @classmethod
    def build_train_loader(cls, cfg):
        if cls._custom_mapper is not None:
            return build_detection_train_loader(cfg, mapper=cls._custom_mapper)
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
        else:
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cls._custom_test_mapper is not None:
            return build_detection_test_loader(cfg, dataset_name, mapper=cls._custom_test_mapper)
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                # New variant modules (freshly initialized, not pretrained) live
                # under backbone.* but should train at full base LR, not the
                # 0.1x backbone multiplier intended for pretrained Swin weights.
                is_new_variant_module = any(
                    s in module_name
                    for s in ("fusion", "patch_embed_depth", "patch_embed_rgb")
                )
                if "backbone" in module_name and not is_new_variant_module:
                    hyperparams["lr"] = (
                        hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Mask2Former + Swin-Tiny on Rob2Pheno",
    )
    parser.add_argument(
        "--variant", type=str, default="rgb",
        choices=list(VARIANTS.keys()),
        help="Input layer variant (default: rgb)",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--dca-iters", type=int, default=0,
        help="K bidirectional refinement cycles after the initial D->I step "
             "(rgbd_dca variant only; ignored otherwise).",
    )
    parser.add_argument(
        "--dca-heads", type=int, default=4,
        help="Number of attention heads for rgbd_dca_multihead "
             "(must divide embed_dim; ignored otherwise).",
    )
    parser.add_argument(
        "--green-weight", type=float, default=1.0,
        help="Per-class CE weight for greenfruit (class id 1). 1.0 = no change. "
             "Values >1 prioritize greenfruit; auto-appends '_gw{val}' to OUTPUT_DIR.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Run evaluation only (no training)",
    )
    return parser.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def _apply_class_weights(trainer, green_weight, no_object_weight):
    """Overwrite SetCriterion.empty_weight to up-weight greenfruit (class id 1)."""
    if green_weight == 1.0:
        return
    model = trainer.model
    inner = model.module if hasattr(model, "module") else model
    criterion = inner.criterion
    new_weights = torch.tensor(
        [1.0, green_weight, no_object_weight],
        device=criterion.empty_weight.device,
        dtype=criterion.empty_weight.dtype,
    )
    assert new_weights.shape == criterion.empty_weight.shape, (
        f"empty_weight shape mismatch: got {criterion.empty_weight.shape}, "
        f"built {new_weights.shape}"
    )
    criterion.empty_weight.copy_(new_weights)
    logging.getLogger("mask2former").info(
        f"[class-weight] empty_weight set to {new_weights.tolist()} "
        f"(red, green, no-obj)"
    )


def main():
    args = parse_args()
    variant = VARIANTS[args.variant]

    # 1. Register datasets
    register_datasets()

    # 2. Build config
    cfg = build_cfg(args)

    # 3. Let variant modify config before freeze
    variant.update_config(cfg)
    cfg.freeze()

    # 4. Setup
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    setup_logger(output=cfg.OUTPUT_DIR, name="mask2former")
    logger = logging.getLogger("mask2former")
    logger.info(f"Variant: {args.variant}")
    logger.info(f"Output:  {cfg.OUTPUT_DIR}")

    # 5. Set custom mapper if variant provides one
    custom_mapper = variant.get_mapper(cfg, is_train=True)
    custom_test_mapper = variant.get_mapper(cfg, is_train=False)
    Rob2PhenoTrainer._custom_mapper = custom_mapper
    Rob2PhenoTrainer._custom_test_mapper = custom_test_mapper
    Rob2PhenoTrainer._variant = variant

    # 6. Eval-only mode
    if args.eval_only:
        model = build_model(cfg)
        model = variant.update_model(model, cfg)
        model.eval()
        DetectionCheckpointer(model).load(
            os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        )
        if custom_test_mapper is not None:
            val_loader = build_detection_test_loader(cfg, "rob2pheno_val", mapper=custom_test_mapper)
        else:
            val_loader = build_detection_test_loader(cfg, "rob2pheno_val")
        evaluator = COCOEvaluator(
            "rob2pheno_val",
            output_dir=os.path.join(cfg.OUTPUT_DIR, "eval_final"),
        )
        results = inference_on_dataset(model, val_loader, evaluator)
        for task, metrics in results.items():
            logger.info(f"Task: {task}")
            for name, value in metrics.items():
                logger.info(f"  {name}: {value:.1f}")
        return

    # 7. Build trainer (constructs model internally; build_model applies the variant)
    trainer = Rob2PhenoTrainer(cfg)

    # 8. Apply per-class CE weights (must happen before resume_or_load so the
    #    overridden empty_weight buffer gets saved into new checkpoints)
    _apply_class_weights(
        trainer, args.green_weight, cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
    )

    # 9. Load weights
    trainer.resume_or_load(resume=args.resume)

    # 10. Train
    trainer.train()


if __name__ == "__main__":
    main()
