"""Aggregate k-fold CV metrics across per-fold output directories.

Usage:
    python aggregate_cv.py --variant rgb --folds 5
    python aggregate_cv.py --variant rgbd_dca --folds 5 --max-iter 5000 --cv-seed 42

Output:
    1. Console summary: mean +/- std across folds, using the LAST recorded
       value per fold (per metric, per evaluated dataset).
    2. CSV at {output_root}/cv_<variant>[_tag]_K{folds}_seed{seed}_periter.csv
       with one row per (fold, iter, dataset, metric, value). Use this to
       reproduce per-iter AP tables like the one in CLAUDE.md.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

METRICS = ["segm/AP", "segm/AP50", "segm/AP75", "segm/AP-redfruit", "segm/AP-greenfruit"]
DATASETS = ["rob2pheno_fold_val", "rob2pheno_val"]


def _iter_rows(metrics_path: Path):
    """Yield parsed JSON dicts, one per non-empty line of metrics.json."""
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _last_value(metrics_path: Path, key: str):
    last = None
    for row in _iter_rows(metrics_path):
        if key in row:
            last = row[key]
    return last


def _per_iter_evals(metrics_path: Path):
    """Yield (iteration, dataset, metric, value) tuples for every eval row.

    A metrics.json line is treated as an eval row when it contains at least one
    key of the form `{dataset}/segm/...` for a dataset we care about.
    """
    interesting_prefixes = tuple(f"{d}/" for d in DATASETS)
    for row in _iter_rows(metrics_path):
        if not any(k.startswith(interesting_prefixes) for k in row):
            continue
        # Detectron2 writes "iteration" as the step counter
        iteration = row.get("iteration")
        for key, val in row.items():
            for ds in DATASETS:
                prefix = f"{ds}/"
                if key.startswith(prefix):
                    metric = key[len(prefix):]
                    yield iteration, ds, metric, val
                    break


def _mean_std(vals):
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--folds", type=int, required=True)
    p.add_argument("--cv-seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=None,
                   help="If set, restrict glob to runs with _mi{max_iter}_.")
    p.add_argument("--dataset", default="rob2pheno",
                   help="Dataset tag in the output dir (default rob2pheno -> empty tag).")
    p.add_argument("--run-tag", default=None,
                   help="Optional run-tag suffix to match (e.g. for parallel runs).")
    p.add_argument("--output-root", default=str(PROJECT_ROOT / "output"))
    p.add_argument("--csv-path", default=None,
                   help="Where to write the per-iter CSV. Default: <output_root>/cv_<variant>..._periter.csv")
    args = p.parse_args()

    dataset_tag = "" if args.dataset == "rob2pheno" else f"_{args.dataset}"
    mi_part = f"*_mi{args.max_iter}_" if args.max_iter is not None else "*_mi*_"
    tag_part = f"{args.run_tag}_" if args.run_tag else ""
    pattern = (
        f"{args.variant}{dataset_tag}{mi_part}"
        f"fold*of{args.folds}_seed{args.cv_seed}_{tag_part}swin_tiny"
    )
    full_glob = os.path.join(args.output_root, pattern)

    matches = sorted(glob.glob(full_glob))
    if not matches:
        print(f"No output dirs matched: {full_glob}")
        return

    per_fold = {}
    for d in matches:
        name = os.path.basename(d)
        marker = "_fold"
        i = name.find(marker)
        if i < 0:
            continue
        rest = name[i + len(marker):]
        try:
            fold_idx = int(rest.split("of", 1)[0])
        except ValueError:
            continue
        per_fold[fold_idx] = d

    print(f"Found {len(per_fold)} fold dirs matching {pattern}:")
    for k in sorted(per_fold):
        print(f"  fold {k}: {os.path.basename(per_fold[k])}")
    print()

    if len(per_fold) < args.folds:
        missing = sorted(set(range(args.folds)) - set(per_fold))
        print(f"WARNING: missing folds: {missing}")
        print()

    # --- Per-iter CSV ---
    csv_name = args.csv_path or os.path.join(
        args.output_root,
        f"cv_{args.variant}{dataset_tag}"
        + (f"_{args.run_tag}" if args.run_tag else "")
        + f"_K{args.folds}_seed{args.cv_seed}_periter.csv",
    )
    n_rows = 0
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "iteration", "dataset", "metric", "value"])
        for k in sorted(per_fold):
            mp = Path(per_fold[k]) / "metrics.json"
            if not mp.exists():
                continue
            for iteration, ds, metric, val in _per_iter_evals(mp):
                writer.writerow([k, iteration, ds, metric, val])
                n_rows += 1
    print(f"Per-iter CSV: {csv_name}  ({n_rows} rows)")
    print()

    # --- Last-iter summary across folds ---
    for dataset in DATASETS:
        print(f"=== eval dataset: {dataset} (last-iter mean +/- std across {args.folds} folds) ===")
        for metric in METRICS:
            key = f"{dataset}/{metric}"
            vals = []
            for k in sorted(per_fold):
                mp = Path(per_fold[k]) / "metrics.json"
                if not mp.exists():
                    continue
                v = _last_value(mp, key)
                if v is not None:
                    vals.append(v)
            mean, std = _mean_std(vals)
            n = len(vals)
            if n == 0:
                print(f"  {metric:24s}  (no data)")
            else:
                print(f"  {metric:24s}  {mean:6.2f} +/- {std:5.2f}  (n={n})")
        print()


if __name__ == "__main__":
    main()
