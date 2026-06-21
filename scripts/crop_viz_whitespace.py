#!/usr/bin/env python
"""Crop excess whitespace from already-rendered viz_val contact sheets.

The sheets produced by ``visualize_val_predictions.py`` are saved with
``bbox_inches="tight"`` so the outer border is small, but they still carry
large *internal* whitespace: a tall blank band under the suptitle and wide
gaps between the three image rows. A plain border trim cannot remove those
because there is content above and below them. This script instead collapses
every run of blank (near-white) rows/columns down to a small uniform margin,
leaving all six panels and their titles intact.

Pure image processing -- no detectron2 / model loading required, so it does
not need the checkpoints or the dataset.

    # overwrite in place (default)
    python scripts/crop_viz_whitespace.py viz_val

    # write to a separate folder, keeping originals
    python scripts/crop_viz_whitespace.py viz_val --out viz_val_cropped
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def keep_indices(is_blank: np.ndarray, margin: int, pad: int) -> np.ndarray:
    """Indices to keep along one axis.

    Non-blank lines are always kept. A run of consecutive blank lines is
    shrunk to at most ``pad`` lines when it touches the start/end (outer
    border) or ``margin`` lines when it sits between content (internal gap).
    """
    n = len(is_blank)
    keep = []
    i = 0
    while i < n:
        if not is_blank[i]:
            keep.append(i)
            i += 1
            continue
        j = i
        while j < n and is_blank[j]:
            j += 1
        allow = pad if (i == 0 or j == n) else margin
        keep.extend(range(i, i + min(j - i, allow)))
        i = j
    return np.asarray(keep, dtype=int)


def crop_image(arr: np.ndarray, margin: int, pad: int, white_thr: int) -> np.ndarray:
    rows_blank = (arr >= white_thr).all(axis=(1, 2))
    cols_blank = (arr >= white_thr).all(axis=(0, 2))
    keep_rows = keep_indices(rows_blank, margin, pad)
    keep_cols = keep_indices(cols_blank, margin, pad)
    return arr[keep_rows][:, keep_cols]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default="viz_val",
                    help="folder of PNG contact sheets (default: viz_val)")
    ap.add_argument("--out", default=None,
                    help="output folder; default overwrites in place")
    ap.add_argument("--margin", type=int, default=16,
                    help="px of whitespace kept at internal gaps (default 16)")
    ap.add_argument("--pad", type=int, default=8,
                    help="px of whitespace kept at the outer border (default 8)")
    ap.add_argument("--white-thr", type=int, default=250,
                    help="min channel value for a pixel to count as white (default 250)")
    args = ap.parse_args()

    src = Path(args.dir)
    out_dir = Path(args.out) if args.out else src
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(src.glob("*.png"))
    if not pngs:
        print(f"No PNGs found in {src}")
        return

    for p in pngs:
        arr = np.asarray(Image.open(p).convert("RGB"))
        cropped = crop_image(arr, args.margin, args.pad, args.white_thr)
        out_path = out_dir / p.name
        Image.fromarray(cropped).save(out_path)
        print(f"{p.name}: {arr.shape[1]}x{arr.shape[0]} -> "
              f"{cropped.shape[1]}x{cropped.shape[0]}", flush=True)

    print(f"\nDone. {len(pngs)} sheets written to {out_dir}")


if __name__ == "__main__":
    main()
