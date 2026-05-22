"""K-fold cross-validation split utility for COCO-format annotation files.

The split is deterministic for a given (json_path, k, seed, stratify) tuple.
When stratify=True, images are bucketed by whether they contain at least one
instance of the minority class (category_id == 2 in the source 1-indexed COCO
file, i.e. greenfruit for Rob2Pheno), so every fold sees a roughly equal share
of minority-class images.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple


# Source COCO category_id for the minority class to stratify on.
# Rob2Pheno: 1=redfruit (majority), 2=greenfruit (minority).
_STRATIFY_CATEGORY_ID = 2


def _kfold_indices(n_items: int, k: int) -> List[Tuple[List[int], List[int]]]:
    """Return k (train_idx, val_idx) tuples splitting range(n_items) into k folds.

    Fold sizes differ by at most 1 (first n_items % k folds get one extra item).
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if n_items < k:
        raise ValueError(f"n_items ({n_items}) must be >= k ({k})")
    base, rem = divmod(n_items, k)
    bounds = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        bounds.append((start, start + size))
        start += size
    folds = []
    all_idx = list(range(n_items))
    for lo, hi in bounds:
        val_idx = all_idx[lo:hi]
        train_idx = all_idx[:lo] + all_idx[hi:]
        folds.append((train_idx, val_idx))
    return folds


def make_kfold_splits(
    json_path: str | Path,
    k: int,
    seed: int,
    stratify: bool = True,
) -> List[Tuple[List[int], List[int]]]:
    """Return k (train_image_ids, val_image_ids) tuples covering every image.

    Args:
        json_path: COCO annotation file to split.
        k:         Number of folds.
        seed:      RNG seed for reproducibility.
        stratify: When True, balance minority-class image counts across folds.
                   When False, plain random k-fold over all image ids.

    Returns:
        List of length k; each element is (train_ids, val_ids). Every image_id
        from the source JSON appears in exactly one val_ids list.
    """
    with open(json_path) as f:
        coco = json.load(f)

    all_ids = [img["id"] for img in coco["images"]]

    if not stratify:
        rng = random.Random(seed)
        shuffled = all_ids[:]
        rng.shuffle(shuffled)
        return [
            ([shuffled[i] for i in tr], [shuffled[i] for i in va])
            for tr, va in _kfold_indices(len(shuffled), k)
        ]

    # Stratify by presence of the minority class in each image.
    minority_ids = {
        ann["image_id"]
        for ann in coco["annotations"]
        if ann["category_id"] == _STRATIFY_CATEGORY_ID
    }
    pos = [i for i in all_ids if i in minority_ids]
    neg = [i for i in all_ids if i not in minority_ids]

    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    pos_folds = _kfold_indices(len(pos), k) if len(pos) >= k else None
    neg_folds = _kfold_indices(len(neg), k) if len(neg) >= k else None

    folds = []
    for i in range(k):
        val_ids = []
        train_ids = []
        if pos_folds is not None:
            tr, va = pos_folds[i]
            val_ids += [pos[j] for j in va]
            train_ids += [pos[j] for j in tr]
        else:
            train_ids += pos  # too few positives to stratify; reuse in all folds' train
        if neg_folds is not None:
            tr, va = neg_folds[i]
            val_ids += [neg[j] for j in va]
            train_ids += [neg[j] for j in tr]
        else:
            train_ids += neg
        folds.append((train_ids, val_ids))
    return folds


def _main():
    import argparse
    p = argparse.ArgumentParser(description="Inspect k-fold splits of a COCO JSON.")
    p.add_argument("json_path")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-stratify", action="store_true")
    args = p.parse_args()

    folds = make_kfold_splits(args.json_path, args.k, args.seed, not args.no_stratify)
    with open(args.json_path) as f:
        coco = json.load(f)
    minority_ids = {
        ann["image_id"]
        for ann in coco["annotations"]
        if ann["category_id"] == _STRATIFY_CATEGORY_ID
    }
    all_val = []
    for i, (tr, va) in enumerate(folds):
        pos_count = sum(1 for x in va if x in minority_ids)
        print(f"fold {i}: train={len(tr):3d}  val={len(va):3d}  val_minority={pos_count}")
        all_val += va
    n_total = len(coco["images"])
    print(f"coverage: {len(set(all_val))}/{n_total} unique val ids (expected {n_total})")
    if len(set(all_val)) != n_total or len(all_val) != n_total:
        print("WARNING: val ids not a clean partition!")


if __name__ == "__main__":
    _main()
