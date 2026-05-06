"""Run inference with a trained Mask2Former variant and visualize predictions.

Usage:
    # Single image (RGB variant)
    python inference.py --variant rgb --image data/Rob2Pheno/RGB/some_image.tiff

    # All validation images
    python inference.py --variant rgb --val-set

    # Compare all 3 variants on validation set
    python inference.py --variant rgb rgbd_early rgbd_bicma --val-set

    # Custom checkpoint
    python inference.py --variant rgb --checkpoint output/rgb_swin_tiny/model_0001999.pth --image img.tiff

    # Adjust confidence threshold
    python inference.py --variant rgb --val-set --threshold 0.5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
M2F_DIR = PROJECT_ROOT / "Mask2Former"
if str(M2F_DIR) not in sys.path:
    sys.path.insert(0, str(M2F_DIR))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import ColorMode, Visualizer

from mask2former import add_maskformer2_config
from variants import VARIANTS


THING_CLASSES = ["redfruit", "greenfruit"]
# Colors: red for ripe, green for unripe
CLASS_COLORS = [(0.9, 0.2, 0.2), (0.2, 0.8, 0.2)]


def build_cfg_for_inference(variant_name, checkpoint_path=None, output_dir=None):
    """Build a Detectron2 config suitable for inference."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.MODEL.RESNETS.STEM_TYPE = "basic"
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 1, 1]

    base_config = str(
        M2F_DIR / "configs" / "coco" / "instance-segmentation"
        / "swin" / "maskformer2_swin_tiny_bs16_50ep.yaml"
    )
    cfg.merge_from_file(base_config)

    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
    cfg.INPUT.MAX_SIZE_TEST = 640
    cfg.INPUT.MIN_SIZE_TEST = 512

    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "output" / f"{variant_name}_swin_tiny")
    cfg.OUTPUT_DIR = output_dir

    if checkpoint_path is None:
        checkpoint_path = os.path.join(output_dir, "model_final.pth")
    cfg.MODEL.WEIGHTS = checkpoint_path

    # Let variant modify config (e.g., add depth channel stats)
    variant = VARIANTS[variant_name]
    variant.update_config(cfg)
    cfg.freeze()

    return cfg


def load_model(variant_name, checkpoint_path=None, output_dir=None):
    """Build and load a trained model for a given variant."""
    cfg = build_cfg_for_inference(variant_name, checkpoint_path, output_dir)
    variant = VARIANTS[variant_name]

    model = build_model(cfg)
    model = variant.update_model(model, cfg)
    model.eval()

    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    print(f"Loaded {variant_name} from {cfg.MODEL.WEIGHTS}")

    return model, cfg


def prepare_input(image_path, variant_name, cfg):
    """Prepare a single image as model input.

    For RGB: loads 3-channel image.
    For RGBD variants: loads RGB + depth concatenated as 4 channels.
    """
    from PIL import Image

    rgb = np.array(Image.open(image_path).convert("RGB"))
    # BGR for detectron2
    rgb_bgr = rgb[:, :, ::-1]

    if variant_name in ("rgbd_early", "rgbd_bicma", "rgbd_bicma_alpha", "rgbd_dca", "rgbd_dca_1iter"):
        # Derive depth path
        p = Path(image_path)
        depth_name = p.name.replace("_RGB.tiff", "_DEPTH.tiff")
        depth_dir = p.parent.parent / "Depth"
        depth_path = str(depth_dir / depth_name)

        depth = np.array(Image.open(depth_path).convert("L"))
        image = np.concatenate([rgb_bgr, depth[..., None]], axis=-1)
    else:
        image = rgb_bgr

    # To tensor (C, H, W) float32
    image_tensor = torch.as_tensor(
        image.transpose(2, 0, 1).astype("float32")
    )

    height, width = image.shape[:2]
    return {"image": image_tensor, "height": height, "width": width}, rgb


def run_inference(model, inputs):
    """Run model inference on prepared inputs."""
    with torch.no_grad():
        outputs = model([inputs])[0]
    return outputs


def visualize_predictions(rgb_image, outputs, threshold=0.3, variant_name=""):
    """Draw instance masks and labels on the RGB image."""
    # Set up metadata for visualizer
    meta_name = f"_inference_{variant_name}"
    if meta_name not in MetadataCatalog.list():
        MetadataCatalog.get(meta_name).set(
            thing_classes=THING_CLASSES,
            thing_colors=[(int(r * 255), int(g * 255), int(b * 255))
                          for r, g, b in CLASS_COLORS],
        )
    metadata = MetadataCatalog.get(meta_name)

    instances = outputs["instances"].to("cpu")
    # Filter by confidence
    keep = instances.scores >= threshold
    instances = instances[keep]

    vis = Visualizer(
        rgb_image,
        metadata=metadata,
        scale=1.0,
        instance_mode=ColorMode.IMAGE,
    )
    vis_output = vis.draw_instance_predictions(instances)
    return vis_output.get_image(), instances


def get_val_images():
    """Get list of validation image paths from the val annotation file."""
    val_json = PROJECT_ROOT / "data" / "Rob2Pheno" / "val_2class.JSON"
    rgb_dir = PROJECT_ROOT / "data" / "Rob2Pheno" / "RGB"

    with open(val_json) as f:
        coco = json.load(f)

    return [str(rgb_dir / img["file_name"]) for img in coco["images"]]


def print_metrics_summary(all_results):
    """Print a comparison table of per-variant detection counts and scores."""
    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY")
    print("=" * 70)

    for variant_name, results in all_results.items():
        total_instances = sum(r["num_instances"] for r in results)
        total_red = sum(r["num_red"] for r in results)
        total_green = sum(r["num_green"] for r in results)
        avg_score = np.mean([
            s for r in results for s in r["scores"]
        ]) if total_instances > 0 else 0

        print(f"\n  {variant_name}:")
        print(f"    Images processed:  {len(results)}")
        print(f"    Total detections:  {total_instances}")
        print(f"      - redfruit:      {total_red}")
        print(f"      - greenfruit:    {total_green}")
        print(f"    Avg confidence:    {avg_score:.3f}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained Mask2Former variants",
    )
    parser.add_argument(
        "--variant", nargs="+", default=["rgb"],
        choices=list(VARIANTS.keys()),
        help="Variant(s) to run (default: rgb). Pass multiple to compare.",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single image for inference",
    )
    parser.add_argument(
        "--val-set", action="store_true",
        help="Run on all validation images",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint (default: model_final.pth)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Confidence threshold for detections (default: 0.3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save visualizations (default: output/<variant>/vis)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Don't show images (just save them)",
    )
    args = parser.parse_args()

    if not args.image and not args.val_set:
        parser.error("Provide --image or --val-set")

    # Gather image paths
    if args.val_set:
        image_paths = get_val_images()
        print(f"Running on {len(image_paths)} validation images")
    else:
        image_paths = [args.image]

    all_results = {}

    for variant_name in args.variant:
        print(f"\n{'='*50}")
        print(f"Variant: {variant_name}")
        print(f"{'='*50}")

        model, cfg = load_model(variant_name, args.checkpoint, args.output_dir)
        variant_results = []

        # Output directory for visualizations
        out_dir = args.output_dir or str(
            PROJECT_ROOT / "output" / f"{variant_name}_swin_tiny" / "vis"
        )
        os.makedirs(out_dir, exist_ok=True)

        for img_path in image_paths:
            img_name = Path(img_path).stem
            print(f"  Processing: {img_name}")

            inputs, rgb = prepare_input(img_path, variant_name, cfg)
            outputs = run_inference(model, inputs)

            vis_img, instances = visualize_predictions(
                rgb, outputs, threshold=args.threshold,
                variant_name=variant_name,
            )

            # Collect stats
            classes = instances.pred_classes.numpy()
            scores = instances.scores.numpy()
            result = {
                "image": img_name,
                "num_instances": len(instances),
                "num_red": int((classes == 0).sum()),
                "num_green": int((classes == 1).sum()),
                "scores": scores.tolist(),
            }
            variant_results.append(result)

            print(f"    Detections: {result['num_instances']} "
                  f"(red={result['num_red']}, green={result['num_green']})")

            # Save visualization
            save_path = os.path.join(out_dir, f"{img_name}_{variant_name}.png")
            cv2.imwrite(save_path, vis_img[:, :, ::-1])  # RGB→BGR for cv2

            # Display if requested
            if not args.no_display and len(image_paths) == 1:
                cv2.imshow(f"{variant_name}: {img_name}", vis_img[:, :, ::-1])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        all_results[variant_name] = variant_results
        print(f"\n  Visualizations saved to: {out_dir}")

    # Print comparison
    if len(args.variant) > 1 or args.val_set:
        print_metrics_summary(all_results)


if __name__ == "__main__":
    main()
