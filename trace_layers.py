"""Trace every layer in each Mask2Former variant, logging tensor shapes.

Registers forward hooks on all submodules, runs a dummy forward pass in
eval/no_grad mode, and writes a hierarchical markdown report per variant.

Usage:
    python trace_layers.py                          # trace all variants
    python trace_layers.py --variant rgb            # trace one variant
    python trace_layers.py --output-dir ./my_traces # custom output dir
    python trace_layers.py --device cpu             # force CPU

Requirements:
    - Compiled MSDeformAttn CUDA ops (run on HPC, not Windows)
    - Rob2Pheno depth TIFFs in data/Rob2Pheno/Depth/ (for RGBD variants)
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
M2F_DIR = PROJECT_ROOT / "Mask2Former"
if str(M2F_DIR) not in sys.path:
    sys.path.insert(0, str(M2F_DIR))

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from variants import VARIANTS


# ===========================================================================
# Shape description helpers
# ===========================================================================

def _describe_shape(obj, depth=0):
    """Recursively describe the shape/structure of a tensor-like object."""
    if isinstance(obj, torch.Tensor):
        return "(" + ", ".join(str(d) for d in obj.shape) + ")"
    if isinstance(obj, dict):
        items = [f"{k}: {_describe_shape(v, depth + 1)}" for k, v in obj.items()]
        return "{" + ", ".join(items) + "}"
    if isinstance(obj, (tuple, list)):
        if len(obj) == 0:
            return "[]" if isinstance(obj, list) else "()"
        parts = [_describe_shape(item, depth + 1) for item in obj]
        if isinstance(obj, tuple) and len(parts) == 1:
            return parts[0]
        sep = ", "
        bracket = ("(", ")") if isinstance(obj, tuple) else ("[", "]")
        return bracket[0] + sep.join(parts) + bracket[1]
    # NestedTensor (deformable attention)
    if hasattr(obj, "tensors") and hasattr(obj, "mask"):
        return (
            f"NestedTensor(tensors={_describe_shape(obj.tensors)}, "
            f"mask={_describe_shape(obj.mask)})"
        )
    if obj is None:
        return "None"
    return type(obj).__name__


# ===========================================================================
# Hook-based layer tracer
# ===========================================================================

class LayerTracer:
    """Register pre/post forward hooks on every submodule to capture shapes.

    Uses pre-hooks for ordering (parent before children) and post-hooks
    for output shapes.  Deduplicates modules called multiple times
    (e.g. class_embed called once per decoder layer).
    """

    def __init__(self):
        self.call_order = []      # module names in pre-order
        self.input_shapes = {}    # name -> str
        self.output_shapes = {}   # name -> str
        self.class_names = {}     # name -> str
        self._hooks = []

    # ------------------------------------------------------------------

    def _make_pre_hook(self, name):
        def hook_fn(module, inputs):
            if name in self.input_shapes:
                return  # already recorded (module called more than once)
            self.call_order.append(name)
            self.class_names[name] = module.__class__.__name__
            if isinstance(inputs, tuple) and len(inputs) == 1:
                self.input_shapes[name] = _describe_shape(inputs[0])
            else:
                self.input_shapes[name] = _describe_shape(inputs)
        return hook_fn

    def _make_post_hook(self, name):
        def hook_fn(module, inputs, output):
            if name in self.output_shapes:
                return
            self.output_shapes[name] = _describe_shape(output)
        return hook_fn

    # ------------------------------------------------------------------

    def register_hooks(self, model):
        """Register pre+post hooks on every submodule (excluding root)."""
        for name, module in model.named_modules():
            if name == "":
                continue
            self._hooks.append(
                module.register_forward_pre_hook(self._make_pre_hook(name))
            )
            self._hooks.append(
                module.register_forward_hook(self._make_post_hook(name))
            )

    def wrap_forward_features(self, model):
        """Wrap pixel_decoder.forward_features so it appears in the trace.

        MaskFormerHead calls pixel_decoder.forward_features() directly
        instead of pixel_decoder(), so the normal forward hook never fires
        for the pixel_decoder module itself.  This wrapper injects a record.
        """
        pixel_decoder = model.sem_seg_head.pixel_decoder
        orig_ff = pixel_decoder.forward_features
        tracer = self
        pd_name = "sem_seg_head.pixel_decoder"

        def _wrapped(features):
            if pd_name not in tracer.input_shapes:
                tracer.call_order.append(pd_name)
                tracer.class_names[pd_name] = pixel_decoder.__class__.__name__
                tracer.input_shapes[pd_name] = _describe_shape(features)
            result = orig_ff(features)
            if pd_name not in tracer.output_shapes:
                tracer.output_shapes[pd_name] = _describe_shape(result)
            return result

        pixel_decoder.forward_features = _wrapped

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------

    def get_records(self):
        """Return an OrderedDict of records in forward-pass (pre-)order."""
        records = OrderedDict()
        for name in self.call_order:
            if name in self.output_shapes:
                records[name] = {
                    "class_name": self.class_names[name],
                    "input": self.input_shapes[name],
                    "output": self.output_shapes[name],
                }
        return records


# ===========================================================================
# Section classification
# ===========================================================================

def _classify_section(name, variant_name):
    """Map a dotted module name to a markdown section header."""
    # BiCMA-specific sections
    if variant_name == "rgbd_bicma":
        if name.startswith("backbone.patch_embed_rgb"):
            return "## RGB Patch Embedding"
        if name.startswith("backbone.patch_embed_depth"):
            return "## Depth Encoder"
        if name.startswith("backbone.fusion"):
            return "## Cross-Modal Fusion (BiCMA)"

    if name.startswith("backbone"):
        return "## Backbone (Swin Transformer)"

    if "pixel_decoder" in name:
        return "## Pixel Decoder (MSDeformAttn)"

    # Separate prediction heads from the rest of the transformer decoder
    if name.startswith("sem_seg_head.predictor.class_embed") or \
       name.startswith("sem_seg_head.predictor.mask_embed"):
        return "## Prediction Heads"

    if name.startswith("sem_seg_head.predictor"):
        return "## Transformer Decoder (Masked Attention)"

    if name.startswith("sem_seg_head"):
        return "## Semantic Segmentation Head"

    if name.startswith("criterion"):
        return "## Criterion (loss only, not called during inference)"

    return "## Other"


# ===========================================================================
# Config & model construction
# ===========================================================================

def _register_datasets_minimal():
    """Register minimal dataset metadata (no data loading needed)."""
    thing_classes = ["redfruit", "greenfruit"]
    for name in ["rob2pheno_train", "rob2pheno_val"]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        DatasetCatalog.register(name, lambda: [])
        MetadataCatalog.get(name).set(
            thing_classes=thing_classes,
            evaluator_type="coco",
        )


def _build_cfg(variant_name):
    """Build a Detectron2 config for the given variant (mirrors train.py)."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.MODEL.RESNETS.STEM_TYPE = "basic"
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 1, 1]

    base_yaml = str(
        M2F_DIR / "configs" / "coco" / "instance-segmentation"
        / "swin" / "maskformer2_swin_tiny_bs16_50ep.yaml"
    )
    cfg.merge_from_file(base_yaml)

    cfg.MODEL.WEIGHTS = ""                    # no checkpoint needed
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2    # redfruit, greenfruit
    cfg.DATASETS.TRAIN = ("rob2pheno_train",)
    cfg.DATASETS.TEST = ("rob2pheno_val",)

    cfg.INPUT.MAX_SIZE_TRAIN = 640
    cfg.INPUT.MAX_SIZE_TEST = 640
    cfg.INPUT.MIN_SIZE_TRAIN = (640,)
    cfg.INPUT.MIN_SIZE_TEST = 640

    # Let variant modify config (e.g. add depth channel to PIXEL_MEAN/STD)
    variant_module = VARIANTS[variant_name]
    variant_module.update_config(cfg)
    cfg.freeze()
    return cfg


def _build_model(variant_name, device):
    """Build the model for a variant, apply variant modifications, move to device."""
    cfg = _build_cfg(variant_name)
    model = build_model(cfg)

    variant_module = VARIANTS[variant_name]
    model = variant_module.update_model(model, cfg)

    model = model.to(device)
    model.eval()
    return model, cfg


# ===========================================================================
# Markdown formatting
# ===========================================================================

def _format_markdown(variant_name, input_shape_str, records, model):
    """Build the full markdown report for one variant."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines = [
        f"# Layer Trace: `{variant_name}`\n",
        f"- **Input shape**: `{input_shape_str}`",
        f"- **Total parameters**: {total_params:,}",
        f"- **Trainable parameters**: {trainable_params:,}",
        f"- **Total layers traced**: {len(records)}",
        "",
    ]

    current_section = None
    section_block = []  # lines for the current code block

    def _flush_section():
        """Write accumulated section_block into lines."""
        if section_block:
            lines.append("```")
            lines.extend(section_block)
            lines.append("```")
            lines.append("")
            section_block.clear()

    for name, info in records.items():
        section = _classify_section(name, variant_name)
        if section != current_section:
            _flush_section()
            lines.append(section)
            lines.append("")
            current_section = section

        depth = name.count(".")
        indent = "  " * depth
        section_block.append(f"{indent}{name} ({info['class_name']})")
        section_block.append(f"{indent}  Input:  {info['input']}")
        section_block.append(f"{indent}  Output: {info['output']}")
        section_block.append("")

    _flush_section()

    # Summary table
    lines.extend([
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total layers traced | {len(records)} |",
        f"| Total parameters | {total_params:,} |",
        f"| Trainable parameters | {trainable_params:,} |",
        f"| Non-trainable parameters | {total_params - trainable_params:,} |",
        "",
    ])
    return "\n".join(lines)


# ===========================================================================
# Trace one variant
# ===========================================================================

def trace_variant(variant_name, device):
    """Trace a single variant and return (markdown_content, success)."""
    print(f"\n{'=' * 60}")
    print(f"  Tracing variant: {variant_name}")
    print(f"{'=' * 60}")

    model, cfg = _build_model(variant_name, device)

    num_channels = len(cfg.MODEL.PIXEL_MEAN)  # 3 for RGB, 4 for RGBD
    H, W = 640, 640
    input_shape_str = f"(1, {num_channels}, {H}, {W})"
    print(f"  Input shape : {input_shape_str}")
    print(f"  Device      : {device}")

    # --- hooks ---
    tracer = LayerTracer()
    tracer.register_hooks(model)
    tracer.wrap_forward_features(model)
    hook_count = len(tracer._hooks)
    print(f"  Hooks registered on {hook_count // 2} submodules")

    # --- dummy input (matches MaskFormer.forward expectations) ---
    dummy_image = torch.randn(num_channels, H, W, device=device)
    batched_inputs = [{"image": dummy_image, "height": H, "width": W}]

    # --- forward pass ---
    print("  Running forward pass ...")
    with torch.no_grad():
        _ = model(batched_inputs)

    records = tracer.get_records()
    print(f"  Done. {len(records)} layers traced.")

    md = _format_markdown(variant_name, input_shape_str, records, model)

    # cleanup
    tracer.remove_hooks()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return md


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Trace layer shapes through Mask2Former variants",
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        choices=list(VARIANTS.keys()),
        help="Variant to trace (default: all)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "traces"),
        help="Directory for output markdown files (default: traces/)",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    _register_datasets_minimal()

    variants_to_trace = (
        [args.variant] if args.variant else list(VARIANTS.keys())
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for vname in variants_to_trace:
        md = trace_variant(vname, device)
        out_path = os.path.join(args.output_dir, f"trace_{vname}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"  Written to: {out_path}")

    print(f"\nAll done. Traces in {args.output_dir}/")


if __name__ == "__main__":
    main()
