"""Visualize inference predictions from multiple models against ground truth.

Creates a grid of images for each validation sample:
  [Ground Truth | RGB | RGBD Early | BiCMA-a 1iter | BiCMA-a 2iter | BiCMA-a 3iter]

Usage:
    conda run -n mask2former python visualize_comparison.py
    conda run -n mask2former python visualize_comparison.py --threshold 0.5
    conda run -n mask2former python visualize_comparison.py --images 0 1 2
    conda run -n mask2former python visualize_comparison.py --save-dir my_vis
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
import pycocotools.mask as mask_util  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy import ndimage  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent

# Models to compare: (display_name, output_folder_name)
MODELS = [
    ("RGB", "rgb_swin_tiny"),
    ("RGBD Early", "rgbd_early_swin_tiny"),
    ("RGBD Cross-Attention (1 Iter)", "rgbd_bicma_alpha_swin_tiny_1iter"),
    ("RGBD Cross-Attention (2 Iter)", "rgbd_bicma_alpha_swin_tiny_2iter"),
    ("RGBD Cross-Attention (3 Iter)", "rgbd_bicma_alpha_swin_tiny_3iter"),
]

# RGBA overlay colors per class
COLORS = {
    "red": (255, 60, 60, 100),
    "green": (60, 200, 60, 100),
}
EDGE_COLORS = {
    "red": (255, 30, 30, 255),
    "green": (30, 180, 30, 255),
}
CLASS_LABELS = {
    "red": "red",
    "green": "green",
}
BORDER_WIDTH = 3


def mask_border(mask, width=BORDER_WIDTH):
    """Extract the border of a binary mask via erosion."""
    eroded = ndimage.binary_erosion(
        mask, iterations=width).astype(np.uint8)
    return mask.astype(np.uint8) - eroded


def polygon_to_mask(poly, h, w):
    """Rasterize a polygon to a binary mask using PIL."""
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).polygon(poly, fill=1)
    return np.array(img)


def draw_label(draw, cx, cy, label, color_key):
    """Draw a floating text label with background at (cx, cy)."""
    text = CLASS_LABELS[color_key]
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((cx, cy), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    # Center the label on the centroid
    tx = int(cx - tw / 2)
    ty = int(cy - th / 2)
    pad = 2
    bg_color = (0, 0, 0, 180)
    text_color = EDGE_COLORS[color_key][:3] + (255,)
    draw.rectangle(
        [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
        fill=bg_color)
    draw.text((tx, ty), text, fill=text_color, font=font)


def draw_gt_masks(image, annotations):
    """Draw ground truth polygon masks with thick borders and labels."""
    w, h = image.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for ann in annotations:
        # GT categories: 1=redfruit, 2=greenfruit
        color_key = "red" if ann["category_id"] == 1 else "green"
        fill = COLORS[color_key]
        edge = EDGE_COLORS[color_key]

        segs = ann["segmentation"]
        if isinstance(segs[0], list):
            polys = segs
        else:
            polys = [segs]

        # Collect all polygon pixels for centroid
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for seg in polys:
            poly = [(seg[i], seg[i + 1])
                    for i in range(0, len(seg), 2)]
            if len(poly) < 3:
                continue
            # Fill
            draw.polygon(poly, fill=fill)
            # Rasterize for border extraction
            seg_mask = polygon_to_mask(poly, h, w)
            combined_mask = np.maximum(combined_mask, seg_mask)

        # Draw thick border
        border = mask_border(combined_mask)
        border_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        border_rgba[border == 1] = edge
        border_img = Image.fromarray(border_rgba, "RGBA")
        overlay = Image.alpha_composite(overlay, border_img)
        draw = ImageDraw.Draw(overlay)

        # Label at centroid
        ys, xs = np.where(combined_mask > 0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            draw_label(draw, cx, cy, color_key, color_key)

    return Image.alpha_composite(image.convert("RGBA"), overlay)


def draw_pred_masks(image, predictions, threshold=0.3):
    """Draw predicted RLE masks with thick borders and labels."""
    w, h = image.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    # Sort by score so highest-confidence masks are drawn on top
    preds_sorted = sorted(predictions, key=lambda p: p["score"])

    for pred in preds_sorted:
        if pred["score"] < threshold:
            continue

        # Predictions: 0=redfruit, 1=greenfruit
        color_key = "red" if pred["category_id"] == 0 else "green"
        fill = COLORS[color_key]
        edge = EDGE_COLORS[color_key]

        mask = mask_util.decode(pred["segmentation"])

        # Fill overlay
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[mask == 1] = fill
        mask_img = Image.fromarray(mask_rgba, "RGBA")
        overlay = Image.alpha_composite(overlay, mask_img)

        # Thick border
        border = mask_border(mask)
        border_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        border_rgba[border == 1] = edge
        border_img = Image.fromarray(border_rgba, "RGBA")
        overlay = Image.alpha_composite(overlay, border_img)

    # Draw labels on top of everything (second pass)
    draw = ImageDraw.Draw(overlay)
    for pred in preds_sorted:
        if pred["score"] < threshold:
            continue
        color_key = "red" if pred["category_id"] == 0 else "green"
        mask = mask_util.decode(pred["segmentation"])
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            draw_label(draw, cx, cy, color_key, color_key)

    return Image.alpha_composite(image.convert("RGBA"), overlay)


def load_gt(val_json_path):
    """Load ground truth annotations grouped by image_id."""
    with open(val_json_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    return images, anns_by_image


def load_predictions(results_json_path):
    """Load predictions grouped by image_id."""
    with open(results_json_path) as f:
        preds = json.load(f)

    preds_by_image = {}
    for p in preds:
        preds_by_image.setdefault(p["image_id"], []).append(p)

    return preds_by_image


def count_detections(predictions, threshold):
    """Count detections above threshold by class."""
    filtered = [p for p in predictions if p["score"] >= threshold]
    n_red = sum(1 for p in filtered if p["category_id"] == 0)
    n_green = sum(1 for p in filtered if p["category_id"] == 1)
    return n_red, n_green


def main():
    parser = argparse.ArgumentParser(
        description="Compare model predictions against ground truth")
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Confidence threshold (default: 0.3)")
    parser.add_argument(
        "--images", nargs="*", type=int, default=None,
        help="Image indices to visualize (default: all)")
    parser.add_argument(
        "--save-dir", type=str, default="visualizations",
        help="Output directory (default: visualizations)")
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output DPI (default: 150)")
    args = parser.parse_args()

    val_json = PROJECT_ROOT / "data" / "Rob2Pheno" / "val_2class.JSON"
    rgb_dir = PROJECT_ROOT / "data" / "Rob2Pheno" / "RGB"
    gt_images, gt_anns = load_gt(val_json)

    # Load predictions for each model
    model_preds = {}
    for display_name, folder in MODELS:
        results_path = (PROJECT_ROOT / "output" / folder
                        / "inference" / "coco_instances_results.json")
        if results_path.exists():
            model_preds[display_name] = load_predictions(results_path)
            n = len(model_preds[display_name])
            print(f"Loaded: {display_name} ({n} images)")
        else:
            print(f"WARNING: not found: {results_path}")

    image_ids = sorted(gt_images.keys())
    if args.images is not None:
        image_ids = [image_ids[i]
                     for i in args.images if i < len(image_ids)]

    save_dir = PROJECT_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Layout: 2 rows x 3 cols
    # Row 1: Ground Truth, RGB, RGBD Early
    # Row 2: Cross-Attention (1 Iter), (2 Iter), (3 Iter)
    n_rows, n_cols = 2, 3

    # Build ordered list of panels: (display_name_or_GT, is_gt)
    available_models = [name for name, _ in MODELS
                        if name in model_preds]

    print(f"\nGenerating {len(image_ids)} comparison images "
          f"({n_rows}x{n_cols} grid)...")

    for idx, img_id in enumerate(image_ids):
        img_info = gt_images[img_id]
        img_path = rgb_dir / img_info["file_name"]
        if not img_path.exists():
            print(f"  Skipping {img_info['file_name']} (not found)")
            continue

        rgb = Image.open(img_path).convert("RGB")
        img_name = img_path.stem

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

        # Ground truth panel [0, 0]
        gt_annotations = gt_anns.get(img_id, [])
        gt_vis = draw_gt_masks(rgb, gt_annotations)
        n_gt_red = sum(
            1 for a in gt_annotations if a["category_id"] == 1)
        n_gt_green = sum(
            1 for a in gt_annotations if a["category_id"] == 2)

        axes[0, 0].imshow(gt_vis)
        axes[0, 0].set_title(
            f"Ground Truth\n{n_gt_red}R / {n_gt_green}G",
            fontsize=10, fontweight="bold")
        axes[0, 0].axis("off")

        # Panel positions: row 0 cols 1-2 then row 1 cols 0-2
        panel_positions = [
            (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
        ]

        for i, (display_name, _) in enumerate(MODELS):
            if display_name not in model_preds:
                continue
            r, c = panel_positions[i]
            ax = axes[r, c]
            preds = model_preds[display_name].get(img_id, [])
            pred_vis = draw_pred_masks(
                rgb, preds, threshold=args.threshold)
            n_red, n_green = count_detections(preds, args.threshold)

            ax.imshow(pred_vis)
            ax.set_title(
                f"{display_name}\n{n_red}R / {n_green}G",
                fontsize=10)
            ax.axis("off")

        legend_handles = [
            mpatches.Patch(
                color=(1, 0.23, 0.23), label="redfruit"),
            mpatches.Patch(
                color=(0.23, 0.78, 0.23), label="greenfruit"),
        ]
        fig.legend(
            handles=legend_handles, loc="lower center",
            ncol=2, fontsize=9)
        plt.suptitle(img_name, fontsize=11, y=0.99)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])

        out_path = save_dir / f"{img_name}_comparison.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"  [{idx + 1}/{len(image_ids)}] {img_name}")

    print(f"\nDone! Saved to: {save_dir}")


if __name__ == "__main__":
    main()
