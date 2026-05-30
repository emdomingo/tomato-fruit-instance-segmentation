#!/usr/bin/env python3
"""Prune Detectron2 output dirs, keeping only the best + final checkpoint.

The "best" checkpoint is the eval iteration with the highest segm/AP in
metrics.json. "Final" is model_final.pth (the last checkpoint). All small
artifacts (metrics.json, log.txt, events.*, last_checkpoint, inference/) are
always kept; only redundant model_*.pth files are removed.

Usage (run on the HPC, from the parent dir containing the output folders):

    # Dry run -- shows what WOULD be deleted, touches nothing:
    python cleanup_checkpoints.py DIR [DIR ...]

    # Actually delete:
    python cleanup_checkpoints.py --apply DIR [DIR ...]

    # All matching dirs at once (dry run):
    python cleanup_checkpoints.py *swin_tiny
"""
import argparse
import json
import os
import re
import sys

CKPT_RE = re.compile(r"^model_(\d{7})\.pth$")


def read_best_iter(metrics_path, metric="segm/AP"):
    """Return (best_iter, best_value) from the metrics.json eval lines."""
    best_iter, best_val = None, float("-inf")
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if metric in rec and "iteration" in rec:
                val = rec[metric]
                if val is not None and val > best_val:
                    best_val, best_iter = val, rec["iteration"]
    return best_iter, best_val


def process(folder, metric, apply):
    metrics_path = os.path.join(folder, "metrics.json")
    if not os.path.isfile(metrics_path):
        print(f"[SKIP] {folder}: no metrics.json")
        return 0

    best_iter, best_val = read_best_iter(metrics_path, metric)
    if best_iter is None:
        print(f"[SKIP] {folder}: no '{metric}' entries in metrics.json")
        return 0

    best_name = f"model_{best_iter:07d}.pth"
    keep = {best_name, "model_final.pth"}

    # Verify the best checkpoint file actually exists; warn if not.
    if not os.path.isfile(os.path.join(folder, best_name)):
        print(f"[WARN] {folder}: best is iter {best_iter} but {best_name} "
              f"is missing (it may be model_final.pth)")

    ckpts = sorted(n for n in os.listdir(folder) if CKPT_RE.match(n))
    to_delete = [n for n in ckpts if n not in keep]

    freed = sum(os.path.getsize(os.path.join(folder, n)) for n in to_delete)
    print(f"\n=== {os.path.basename(folder.rstrip('/'))} ===")
    print(f"  best:   {best_name}  ({metric}={best_val:.3f} @ iter {best_iter})")
    print(f"  final:  model_final.pth")
    print(f"  keep    {len(keep & set(ckpts))} ckpt(s), "
          f"delete {len(to_delete)} ckpt(s)  (~{freed/1e9:.1f} GB)")

    for n in to_delete:
        path = os.path.join(folder, n)
        if apply:
            os.remove(path)
            print(f"    deleted {n}")
        else:
            print(f"    would delete {n}")
    return freed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="output folders to prune")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default: dry run)")
    ap.add_argument("--metric", default="segm/AP",
                    help="metric to maximize (default: segm/AP)")
    args = ap.parse_args()

    total = 0
    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"[SKIP] {d}: not a directory")
            continue
        total += process(d, args.metric, args.apply)

    verb = "Freed" if args.apply else "Would free"
    print(f"\n{verb} ~{total/1e9:.1f} GB total.")
    if not args.apply:
        print("Dry run only -- re-run with --apply to delete.")


if __name__ == "__main__":
    sys.exit(main())
