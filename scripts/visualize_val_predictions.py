"""Render side-by-side prediction contact sheets on the fixed rob2pheno_val set.

For each of the 40 images in rob2pheno_val, produces ONE PNG laid out as:

    [ RGB | Ground Truth | rgb | rgbd_early | dca K=0 | dca K=1 | dca K=2 ]

REQUIRES a GPU node (MSDeformAttn CUDA op). Run on the HPC, e.g.:

Each MODELS entry points at a *run directory*; the best-by-val-AP checkpoint in
it is auto-resolved via train._best_checkpoint_in_dir (the same logic warm-start
uses), so it picks model_{best_iter}.pth, NOT the overfit model_final.pth. If
that checkpoint file isn't present in the dir, it errors telling you which file
to download from the HPC run.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402  (reuse register/build_cfg/variant machinery)
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader  # noqa: E402
from detectron2.data import detection_utils as utils  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.utils.visualizer import Visualizer, ColorMode  # noqa: E402

# (column label, variant name, dca_iters, run directory) -- the median-fold
MODELS = [
    ("RGB",          "rgb",         0, "output/rgb_ep200_mi6600_ee10_0_cos_sz1280_fold2of5_seed42_swin_tiny"),
    ("RGBD Early",   "rgbd_early",  0, "output/rgbd_early_ep200_mi6600_ee10_0_cos_sz1280_fold1of5_seed42_swin_tiny"),
    ("RGBD DCA K=0", "rgbd_dca",    0, "output/rgbd_dca_iter0_ep200_mi6600_ee10_0_cos_dlr5_0_sz1280_fold3of5_seed42_swin_tiny"),
    ("RGBD DCA K=1", "rgbd_dca",    1, "output/rgbd_dca_iter1_ep200_mi6600_ee10_0_cos_dlr5_0_sz1280_fold0of5_seed42_swin_tiny"),
    ("RGBD DCA K=2", "rgbd_dca",    2, "output/rgbd_dca_iter2_ep200_mi6600_ee10_0_cos_dlr5_0_sz1280_fold1of5_seed42_swin_tiny"),
]

DATASET = "rob2pheno_val"
# Fixed, consistent class colors across every panel (0-255 RGB).
THING_COLORS = [(220, 30, 30), (30, 190, 60)]  # redfruit, greenfruit


def build_cfg_for(variant_name, dca_iters, ckpt, max_size):
    """Reconstruct the training-time cfg for one variant and point it at ckpt."""
    argv = [
        "viz",
        "--variant", variant_name,
        "--dca-iters", str(dca_iters),
        "--input-max-size", str(max_size),
        "--dataset", "rob2pheno",
        "--num-workers", "0",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        args = train.parse_args()
    finally:
        sys.argv = old_argv
    cfg = train.build_cfg(args)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = ckpt
    cfg.DATASETS.TEST = (DATASET,)
    train.VARIANTS[variant_name].update_config(cfg)
    cfg.freeze()
    return cfg, train.VARIANTS[variant_name]


def load_model(cfg, variant_mod, ckpt):
    model = build_model(cfg)
    model = variant_mod.update_model(model, cfg)
    model.eval()
    DetectionCheckpointer(model).load(ckpt)
    return model


def rg_counts(class_ids):
    """(red, green) counts from an iterable/tensor of 0/1 class ids.

    Class indices follow THING_CLASSES = [redfruit, greenfruit] -> 0=red, 1=green.
    """
    a = np.asarray(class_ids)
    return int((a == 0).sum()), int((a == 1).sum())


@torch.no_grad()
def run_model(label, variant_name, dca_iters, run_dir, max_size, score_thr):
    """Return {image_id: Instances(cpu)} of predictions over the val set."""
    run_dir = str(PROJECT_ROOT / run_dir) if not os.path.isabs(run_dir) else run_dir
    # Select the checkpoint by the IN-FOLD validation AP (rob2pheno_fold_val), not
    # the test set we're visualizing -- selecting on the test set would bias the
    # figure, and the fold-val-best checkpoint is what cleanup_checkpoints.py kept
    # on disk. metric=None makes _best_checkpoint_in_dir prefer fold_val AP.
    ckpt = train._best_checkpoint_in_dir(run_dir, metric="rob2pheno_fold_val/segm/AP")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"[{label}] best (fold-val) checkpoint resolved to {ckpt} but it's not "
            f"on disk. List the run dir's *.pth and download that file from the HPC."
        )
    print(f"[{label}] loading {os.path.basename(ckpt)}", flush=True)
    cfg, variant_mod = build_cfg_for(variant_name, dca_iters, ckpt, max_size)
    model = load_model(cfg, variant_mod, ckpt)
    mapper = variant_mod.get_mapper(cfg, is_train=False)  # None -> default mapper
    loader = (build_detection_test_loader(cfg, DATASET, mapper=mapper)
              if mapper is not None else build_detection_test_loader(cfg, DATASET))

    preds = {}
    for batch in loader:
        out = model(batch)
        for inp, o in zip(batch, out):
            inst = o["instances"].to("cpu")
            keep = inst.scores >= score_thr
            preds[inp["image_id"]] = inst[keep]
    del model
    torch.cuda.empty_cache()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="viz_val", help="output dir for PNGs")
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--input-max-size", type=int, default=1280)
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args()

    train.register_datasets()
    records = DatasetCatalog.get(DATASET)
    meta = MetadataCatalog.get(DATASET)
    meta.thing_colors = list(THING_COLORS)

    # Run every model over the full val set, collect predictions.
    all_preds = {}  # label -> {image_id: Instances}
    for label, variant_name, dca_iters, run_dir in MODELS:
        all_preds[label] = run_model(
            label, variant_name, dca_iters, run_dir,
            args.input_max_size, args.score_thr,
        )

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for rec in records:
        img_id = rec["image_id"]
        rgb = utils.read_image(rec["file_name"], format="RGB")
        stem = Path(rec["file_name"]).stem

        # 3 rows x 2 cols: GT + 5 models. ravel() order is
        # [GT | RGB], [RGBD Early | DCA K=0], [DCA K=1 | DCA K=2].
        # Size each cell to the image's aspect ratio so imshow fills it,
        # otherwise wide images leave large vertical gaps between rows.
        h, w = rgb.shape[:2]
        cell_w = 4.0
        cell_h = cell_w * h / w
        fig, axes = plt.subplots(
            3, 2, figsize=(2 * cell_w, 3 * cell_h),
            constrained_layout=True,
        )
        axes = axes.ravel()

        # Panel 0: ground truth + GT counts
        gt_vis = Visualizer(rgb, metadata=meta, instance_mode=ColorMode.SEGMENTATION)
        axes[0].imshow(gt_vis.draw_dataset_dict(rec).get_image())
        gr, gg = rg_counts([a["category_id"] for a in rec.get("annotations", [])])
        axes[0].set_title(f"Ground Truth  (R:{gr} G:{gg})", fontsize=10)

        # Panels 1..5: each model's prediction + its counts
        for j, (label, *_rest) in enumerate(MODELS):
            inst = all_preds[label].get(img_id)
            v = Visualizer(rgb, metadata=meta, instance_mode=ColorMode.SEGMENTATION)
            if inst is not None and len(inst):
                img = v.draw_instance_predictions(inst).get_image()
                r, g = rg_counts(inst.pred_classes.numpy())
            else:
                img, r, g = rgb, 0, 0
            axes[1 + j].imshow(img)
            axes[1 + j].set_title(f"{label}  (R:{r} G:{g})", fontsize=10)

        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{stem}  (score>={args.score_thr})", fontsize=10)
        out_path = out_dir / f"val_{img_id:03d}_{stem}.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path.name}", flush=True)

    print(f"\nDone. {len(records)} contact sheets in {out_dir}")


if __name__ == "__main__":
    main()
