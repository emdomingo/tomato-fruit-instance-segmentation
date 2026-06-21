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
from cv_splits import make_kfold_splits

# Modules that stay trainable under --freeze-non-variant. Substring-matched
# against parameter names: "patch_embed" covers backbone.patch_embed (rgb,
# rgbd_early) as well as patch_embed_rgb/patch_embed_depth (rgbd_dca*);
# "fusion" covers the DCA fusion blocks; "class_embed" keeps the randomly
# initialized 2-class head trainable (the COCO 80-class head cannot load
# into it, so freezing it would leave a random classifier).
FREEZE_TRAINABLE_KEYWORDS = ("patch_embed", "fusion", "class_embed")


# ===========================================================================
# Dataset loading & registration
# ===========================================================================

def load_rob2pheno(json_path: str, image_root: str, keep_ids=None) -> list:
    """
    Load a Rob2Pheno COCO annotation file and return a list of Detectron2
    dataset dicts (one dict per image).

    Args:
        json_path  : Path to the COCO JSON annotation file.
        image_root : Directory containing all the image files.
        keep_ids   : Optional iterable of image_ids to retain. When None,
                     all images in the JSON are loaded. Used for k-fold CV.

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

    keep = set(keep_ids) if keep_ids is not None else None
    dataset_dicts = []
    for img_entry in coco["images"]:
        if keep is not None and img_entry["id"] not in keep:
            continue
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


def register_papple():
    """Register PApple train/val/test datasets with Detectron2.

    PApple is converted to a 1-class ("fruit") COCO file by
    data/PApple/scripts/convert_annotations.py and shares the same COCO
    schema as Rob2Pheno, so we can reuse `load_rob2pheno` directly.
    """
    data_root = PROJECT_ROOT / "data" / "PApple"
    rgb_dir = data_root / "RGB"
    splits = {
        "papple_train": data_root / "train_1class.JSON",
        "papple_val": data_root / "val_1class.JSON",
        "papple_test": data_root / "test_1class.JSON",
    }
    thing_classes = ["fruit"]
    for name, jpath in splits.items():
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        DatasetCatalog.register(
            name,
            lambda j=str(jpath), r=str(rgb_dir): load_rob2pheno(j, r),
        )
        MetadataCatalog.get(name).set(
            thing_classes=thing_classes, evaluator_type="coco",
        )


def _best_checkpoint_in_dir(run_dir, metric=None):
    """Return the checkpoint in run_dir with the highest <metric> in metrics.json.

    Used to select the best-by-AP checkpoint for evaluation/visualization rather
    than the final iteration, which tends to overfit (runs often peak well before
    MAX_ITER). Detectron2 names periodic checkpoints model_{iter:07d}.pth; the
    end-of-training state is saved as model_final.pth and logged at iteration ==
    max_iter, so the best-iter == final case maps to model_final.pth.

    metric: the metrics.json key to maximize. When None, prefer the in-fold
    validation set (rob2pheno_fold_val/segm/AP, the right model-selection signal
    for CV runs) and otherwise fall back to whatever single */segm/AP eval the run
    logged (e.g. rob2pheno_val for a non-CV run).
    """
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(
            f"Cannot auto-select best checkpoint: {metrics_path} not found."
        )
    # best[key] = (iter, val); track the global max iteration for the final-eval case.
    best = {}
    max_iter_seen = -1
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("iteration") is None:
                continue
            max_iter_seen = max(max_iter_seen, row["iteration"])
            for key, val in row.items():
                # Detectron2 logs the primary mask AP as "<dataset>/segm/AP" when
                # multiple TEST sets exist (CV runs), or bare "segm/AP" for a single
                # TEST set (non-CV runs). Match both, but not AP50/AP75/AP-<class>.
                if key != "segm/AP" and not key.endswith("/segm/AP"):
                    continue
                if key not in best or val > best[key][1]:
                    best[key] = (row["iteration"], val)
    if metric is None:
        if "rob2pheno_fold_val/segm/AP" in best:
            metric = "rob2pheno_fold_val/segm/AP"
        elif len(best) == 1:
            metric = next(iter(best))
        elif not best:
            raise ValueError(
                f"No '*/segm/AP' eval rows in {metrics_path}; cannot pick a checkpoint."
            )
        else:
            raise ValueError(
                f"Multiple eval datasets in {metrics_path} ({sorted(best)}); "
                f"pass an explicit metric to _best_checkpoint_in_dir."
            )
    if metric not in best:
        raise ValueError(
            f"No '{metric}' eval rows in {metrics_path}; cannot pick a checkpoint."
        )
    best_iter, best_val = best[metric]
    # The final eval is logged at max_iter but the file is model_final.pth.
    if best_iter == max_iter_seen:
        ckpt = os.path.join(run_dir, "model_final.pth")
    else:
        ckpt = os.path.join(run_dir, f"model_{best_iter:07d}.pth")
    logging.getLogger("mask2former").info(
        "[best-ckpt] best checkpoint in %s: iter %d (%s=%.3f) -> %s",
        run_dir, best_iter, metric, best_val, os.path.basename(ckpt),
    )
    return ckpt


def register_cv_fold(fold_idx: int, k: int, seed: int, stratify: bool):
    """Register rob2pheno_fold_train / rob2pheno_fold_val by splitting train_2class.JSON.

    The original rob2pheno_val (40-image held-out test set) is registered separately
    by register_datasets() and remains identical across all folds.
    """
    if not (0 <= fold_idx < k):
        raise ValueError(f"fold_idx={fold_idx} out of range for k={k}")

    data_root = PROJECT_ROOT / "data" / "Rob2Pheno"
    rgb_dir = data_root / "RGB"
    train_json = data_root / "train_2class.JSON"
    thing_classes = ["redfruit", "greenfruit"]

    folds = make_kfold_splits(train_json, k=k, seed=seed, stratify=stratify)
    train_ids, val_ids = folds[fold_idx]

    for name, ids in [("rob2pheno_fold_train", train_ids), ("rob2pheno_fold_val", val_ids)]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        DatasetCatalog.register(
            name,
            lambda j=str(train_json), r=str(rgb_dir), keep=tuple(ids):
                load_rob2pheno(j, r, keep_ids=keep),
        )
        MetadataCatalog.get(name).set(
            thing_classes=thing_classes, evaluator_type="coco",
        )

    print(
        f"[cv] fold {fold_idx + 1}/{k} (seed={seed}, stratify={stratify}): "
        f"{len(train_ids)} train / {len(val_ids)} fold-val images",
        flush=True,
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

    # -- Datasets / classes (driven by --dataset) --
    if args.dataset == "rob2pheno":
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2  # redfruit, greenfruit
        if args.folds > 0:
            # K-fold CV: train on the fold's slice; evaluate BOTH the fold-val
            # (in-fold validation) and rob2pheno_val (held-out test) each EVAL_PERIOD.
            cfg.DATASETS.TRAIN = ("rob2pheno_fold_train",)
            cfg.DATASETS.TEST = ("rob2pheno_fold_val", "rob2pheno_val")
        else:
            cfg.DATASETS.TRAIN = ("rob2pheno_train",)
            cfg.DATASETS.TEST = ("rob2pheno_val",)
    elif args.dataset == "papple":
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1  # fruit
        cfg.DATASETS.TRAIN = ("papple_train",)
        cfg.DATASETS.TEST = ("papple_val",)
    else:
        raise ValueError(f"Unknown --dataset {args.dataset!r}")

    # -- Solver --
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    # iters_per_epoch drives both --epochs (total length) and --eval-epochs
    # (eval/checkpoint cadence). It's derived from the registered train set size
    # so the same "epoch" means the same number of passes over the data
    # regardless of fold size or batch size. The dataset is already registered
    # (main() registers before build_cfg), so the length query is safe here.
    # ceil division: iters_per_epoch = ceil(N / batch_size).
    num_train = len(DatasetCatalog.get(cfg.DATASETS.TRAIN[0]))
    iters_per_epoch = (num_train + args.batch_size - 1) // args.batch_size

    # --epochs (optional) overrides --max-iter.
    if args.epochs > 0:
        max_iter = args.epochs * iters_per_epoch
    else:
        max_iter = args.max_iter
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.LR_SCHEDULER_NAME = args.lr_scheduler
    cfg.SOLVER.STEPS = (int(0.7 * max_iter), int(0.9 * max_iter))
    cfg.SOLVER.GAMMA = 0.1
    # STEPS/GAMMA are only consumed by WarmupMultiStepLR; WarmupCosineLR and
    # WarmupPolyLR anneal smoothly to ~0 by MAX_ITER and ignore them.
    cfg.SOLVER.WARMUP_ITERS = args.warmup_iters
    cfg.SOLVER.WARMUP_FACTOR = args.warmup_factor

    # Eval/checkpoint cadence. --eval-epochs (optional) expresses it in epochs so
    # checkpoints land on epoch boundaries (best-by-AP checkpoint == an epoch);
    # otherwise fall back to --eval-period in iterations. Checkpoint and eval are
    # kept aligned so the best-by-AP checkpoint always exists on disk.
    if args.eval_epochs > 0:
        eval_period = max(1, round(args.eval_epochs * iters_per_epoch))
    else:
        eval_period = args.eval_period
    cfg.SOLVER.CHECKPOINT_PERIOD = eval_period
    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"

    logging.getLogger("mask2former").info(
        "[epochs] %d train images / batch %d = %d iters/epoch -> MAX_ITER=%d "
        "(%.1f epochs), eval/checkpoint every %d iters (%.2f epochs)",
        num_train, args.batch_size, iters_per_epoch, max_iter,
        max_iter / iters_per_epoch, eval_period, eval_period / iters_per_epoch,
    )

    # -- Input --
    max_size = args.input_max_size
    min_lo = int(round(0.75 * max_size / 32)) * 32
    raw_step = max(1, (max_size - min_lo) // 5)
    step = max(32, ((raw_step + 31) // 32) * 32)
    min_sizes = tuple(range(min_lo, max_size + 1, step))
    cfg.INPUT.MAX_SIZE_TRAIN = max_size
    cfg.INPUT.MAX_SIZE_TEST = max_size
    cfg.INPUT.MIN_SIZE_TRAIN = min_sizes
    cfg.INPUT.MIN_SIZE_TEST = int(round(0.8 * max_size / 32)) * 32
    # Force ResizeShortestEdge mapper for all variants so rgb trains at the same
    # ~640px scale as the RGB-D variants (the base YAML's "coco_instance_lsj"
    # default ignored MAX_SIZE_TRAIN and trained rgb at 1024x1024).
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_instance"
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    # -- Variant-specific config --
    cfg.MODEL.DCA_ITERS = args.dca_iters
    cfg.MODEL.DCA_LR_MULT = args.dca_lr_mult
    cfg.MODEL.FREEZE_NON_VARIANT = args.freeze_non_variant

    # -- Output --
    if args.variant.startswith("rgbd_dca"):
        suffix = f"_iter{args.dca_iters}"
    else:
        suffix = ""
    if args.green_weight != 1.0:
        suffix += f"_gw{str(args.green_weight).replace('.', '_')}"
    if args.epochs > 0:
        suffix += f"_ep{args.epochs}"
    suffix += f"_mi{max_iter}"
    if args.eval_epochs > 0:
        suffix += f"_ee{str(args.eval_epochs).replace('.', '_')}"
    elif args.eval_period != 1000:
        suffix += f"_ev{args.eval_period}"
    _sched_tag = {"WarmupCosineLR": "cos", "WarmupPolyLR": "poly"}.get(
        args.lr_scheduler, ""
    )
    if _sched_tag:
        suffix += f"_{_sched_tag}"
    if args.dca_lr_mult != 1.0:
        suffix += f"_dlr{str(args.dca_lr_mult).replace('.', '_')}"
    if args.freeze_non_variant:
        suffix += "_frz"
    if args.input_max_size != 640:
        suffix += f"_sz{args.input_max_size}"
    if args.folds > 0:
        suffix += f"_fold{args.fold_index}of{args.folds}_seed{args.cv_seed}"
    if args.run_tag:
        suffix += f"_{args.run_tag}"
    dataset_tag = "" if args.dataset == "rob2pheno" else f"_{args.dataset}"
    cfg.OUTPUT_DIR = str(
        PROJECT_ROOT / "output" / f"{args.variant}{dataset_tag}{suffix}_swin_tiny"
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
        if cfg.MODEL.FREEZE_NON_VARIANT:
            # Freeze everything except the variant input modules and the
            # 2-class head. build_optimizer skips params with
            # requires_grad=False, so this is the whole mechanism.
            logger = logging.getLogger("mask2former")
            trainable, total = 0, 0
            for name, p in model.named_parameters():
                keep = any(k in name for k in FREEZE_TRAINABLE_KEYWORDS)
                p.requires_grad_(keep)
                total += p.numel()
                if keep:
                    trainable += p.numel()
                    logger.info("[freeze] trainable: %s (%d)", name, p.numel())
            logger.info(
                "[freeze] %d / %d params trainable (%.3f%%)",
                trainable, total, 100.0 * trainable / total,
            )
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

        # The DCA fusion + depth patch embed start from zero/mean-init and must
        # grow before the model overfits (~iter 2000), while the RGB tower is
        # already near-optimal. DCA_LR_MULT gives just those groups a faster LR
        # so depth can contribute inside that window. 1.0 = unchanged.
        dca_lr_mult = getattr(cfg.MODEL, "DCA_LR_MULT", 1.0)

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
                # Under --freeze-non-variant, everything still trainable is a
                # variant module or the class head, so skip the 0.1x backbone
                # multiplier (otherwise rgb/rgbd_early's backbone.patch_embed
                # would train 10x slower than the DCA variants' modules).
                if (
                    "backbone" in module_name
                    and not is_new_variant_module
                    and not cfg.MODEL.FREEZE_NON_VARIANT
                ):
                    hyperparams["lr"] = (
                        hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                if dca_lr_mult != 1.0 and any(
                    s in module_name for s in ("fusion", "patch_embed_depth")
                ):
                    hyperparams["lr"] = hyperparams["lr"] * dca_lr_mult
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
    parser.add_argument(
        "--dataset", type=str, default="rob2pheno",
        choices=["rob2pheno", "papple"],
        help="Which dataset to train on. Also drives NUM_CLASSES "
             "(2 for rob2pheno, 1 for papple).",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument(
        "--epochs", type=int, default=0,
        help="If > 0, derive MAX_ITER (and the STEPS decay points) from the "
             "registered train set size, overriding --max-iter: "
             "iters_per_epoch = ceil(num_train_images / batch_size), "
             "MAX_ITER = epochs * iters_per_epoch. Keeps runs comparable across "
             "folds/batch sizes. Appends '_ep{val}' to OUTPUT_DIR. "
             "0 (default) = use --max-iter directly.",
    )
    parser.add_argument(
        "--eval-period", type=int, default=1000,
        help="Iterations between validation evals AND periodic checkpoints "
             "(both are set to this so the best-by-AP checkpoint always exists "
             "on disk). Lower it (e.g. 500) to pinpoint the early overfit peak. "
             "Appends '_ev{val}' to OUTPUT_DIR when != 1000. Ignored when "
             "--eval-epochs > 0.",
    )
    parser.add_argument(
        "--eval-epochs", type=float, default=0,
        help="If > 0, eval+checkpoint every N epochs instead of every "
             "--eval-period iterations, so checkpoints land on epoch boundaries "
             "and the best-by-AP checkpoint corresponds to an epoch: "
             "period = round(N * ceil(num_train / batch_size)). Fractional "
             "values allowed (e.g. 0.5). Appends '_ee{val}' to OUTPUT_DIR.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--lr-scheduler", type=str, default="WarmupMultiStepLR",
        choices=["WarmupMultiStepLR", "WarmupCosineLR", "WarmupPolyLR"],
        help="LR schedule. Default WarmupMultiStepLR: 10x step decay at "
             "0.7/0.9*max_iter. WarmupCosineLR/WarmupPolyLR anneal smoothly to "
             "~0 by max_iter (no cliff; STEPS/GAMMA ignored). Appends a short "
             "tag (e.g. '_cos') to OUTPUT_DIR when not the default.",
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=200,
        help="Linear LR warmup length. Pair with --warmup-factor < 1 to "
             "actually ramp (the base config's WARMUP_FACTOR=1.0 = no ramp).",
    )
    parser.add_argument(
        "--warmup-factor", type=float, default=1.0,
        help="Starting LR as a fraction of BASE_LR (e.g. 0.01). Default 1.0 "
             "preserves the prior no-warmup behavior.",
    )
    parser.add_argument(
        "--dca-lr-mult", type=float, default=1.0,
        help="LR multiplier (x BASE_LR) applied ONLY to the DCA fusion and "
             "patch_embed_depth groups, so the zero/mean-init depth path can "
             "grow before the model overfits. 1.0 = unchanged. Auto-appends "
             "'_dlr{val}' to OUTPUT_DIR (rgbd_dca* variants only).",
    )
    parser.add_argument(
        "--input-max-size", type=int, default=640,
        help="Max image size for train+test. MIN_SIZE_TRAIN sweeps from "
             "0.75*max..max in 5 steps; MIN_SIZE_TEST = 0.8*max. "
             "Appends '_sz{val}' to OUTPUT_DIR when != 640.",
    )
    parser.add_argument(
        "--dca-iters", type=int, default=0,
        help="K bidirectional refinement cycles after the initial D->I step "
             "(rgbd_dca variant only; ignored otherwise).",
    )
    parser.add_argument(
        "--freeze-non-variant", action="store_true",
        help="Freeze all weights except the variant input modules "
             "(patch_embed*, fusion) and the randomly-initialized 2-class "
             "class_embed head. Appends '_frz' to OUTPUT_DIR.",
    )
    parser.add_argument(
        "--run-tag", type=str, default="",
        help="Optional tag appended to OUTPUT_DIR to keep parallel runs separate "
             "(e.g. --run-tag v2 -> '..._mi5000_v2_swin_tiny'). Empty = unchanged.",
    )
    parser.add_argument(
        "--green-weight", type=float, default=1.0,
        help="Per-class CE weight for greenfruit (class id 1). 1.0 = no change. "
             "Values >1 prioritize greenfruit; auto-appends '_gw{val}' to OUTPUT_DIR "
             "(dots replaced with underscores, e.g. 2.0 -> '_gw2_0').",
    )
    parser.add_argument(
        "--folds", type=int, default=0,
        help="K for k-fold CV on the train set. 0 (default) = no CV "
             "(original single-split behavior). When >0, requires --fold-index; "
             "only the rob2pheno dataset is supported.",
    )
    parser.add_argument(
        "--fold-index", type=int, default=-1,
        help="0-based fold index to train on. Required when --folds > 0.",
    )
    parser.add_argument(
        "--cv-seed", type=int, default=42,
        help="RNG seed for the k-fold split. Splits are deterministic for "
             "a given (--folds, --cv-seed, --cv-stratify) tuple.",
    )
    parser.add_argument(
        "--cv-stratify", dest="cv_stratify", action="store_true", default=True,
        help="Stratify k-fold by minority-class image presence (default).",
    )
    parser.add_argument(
        "--no-cv-stratify", dest="cv_stratify", action="store_false",
        help="Disable stratification; use plain random k-fold.",
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
    # green_weight is a Rob2Pheno-only concept (assumes class id 1 = greenfruit).
    # Skip with a clear message when we're training on something else.
    cfg = trainer.cfg
    if cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES != 2:
        logging.getLogger("mask2former").warning(
            "[class-weight] --green-weight ignored: requires 2-class setup "
            "(got NUM_CLASSES=%d)", cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        )
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

    if args.folds > 0:
        if args.dataset != "rob2pheno":
            raise NotImplementedError(
                "--folds is currently only supported for --dataset rob2pheno"
            )
        if not (0 <= args.fold_index < args.folds):
            raise ValueError(
                f"--fold-index must be in [0, {args.folds}); got {args.fold_index}"
            )

    # 1. Register datasets (both registries; --dataset picks which is active)
    register_datasets()
    register_papple()
    if args.folds > 0:
        register_cv_fold(args.fold_index, args.folds, args.cv_seed, args.cv_stratify)

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
        test_name = cfg.DATASETS.TEST[0]
        if custom_test_mapper is not None:
            val_loader = build_detection_test_loader(cfg, test_name, mapper=custom_test_mapper)
        else:
            val_loader = build_detection_test_loader(cfg, test_name)
        evaluator = COCOEvaluator(
            test_name,
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
