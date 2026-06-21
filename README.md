# Tomato Fruit Instance Segmentation

Instance segmentation of tomato fruits — **ripe (red)** vs. **unripe (green)** — on the
[Rob2Pheno](#dataset) RGB-D dataset, using [Mask2Former](https://github.com/facebookresearch/Mask2Former)
with a Swin-Tiny backbone (COCO-pretrained).

The project compares three input variants to study whether depth helps segmentation:

| Variant      | Input             | Fusion                                                                                   |
|--------------|-------------------|------------------------------------------------------------------------------------------|
| `rgb`        | RGB (3-ch)        | Baseline — standard Mask2Former pass-through.                                             |
| `rgbd_early` | RGB + Depth (4-ch)| Early fusion — depth concatenated as a 4th channel into the patch embedding.             |
| `rgbd_dca`   | RGB + Depth       | Depth-guided cross-attention: depth tokens attend into image tokens. `--dca-iters K` runs `K` bidirectional refinement cycles after the initial depth→image step. |

> Earlier experimental/ablation variants (BiCMA, masked-DCA, multi-head DCA) are kept for
> reference under [`variant_archive/`](variant_archive/) and are **not** registered or reported.

---

## Repository structure

```
train.py                 # Training entry point (Detectron2 DefaultTrainer)
inference.py             # Run a trained variant on an image or the validation set
check_dependencies.py    # Pre-flight environment checker
environment.yml          # Conda environment (Python 3.10, PyTorch 2.5.1, CUDA 11.8)
slurm_train.sh           # SLURM job submission (GPU node)
slurm_infer.sh           # SLURM inference job
slurm_visualize.sh       # SLURM visualization job

variants/                # Registered input-layer variants
  __init__.py            #   VARIANTS registry (rgb, rgbd_early, rgbd_dca)
  rgb.py                 #   RGB baseline
  rgbd_early.py          #   4-channel early fusion (+ shared RGB-D data mapper)
  rgbd_dca.py            #   Depth-guided cross-attention
variant_archive/         # Archived experimental variants (not registered)

scripts/
  visualize_val_predictions.py  # Side-by-side prediction contact sheets (rob2pheno_val)
  cleanup_checkpoints.py        # Prune output dirs to best-by-AP + model_final.pth

Mask2Former/             # Vendored upstream Mask2Former (see Environment, not tracked here)
data/Rob2Pheno/          # Dataset (not included — see Dataset)
pretrained/              # COCO-pretrained Mask2Former weights (not included)
output/                  # Training runs: metrics, configs, logs (checkpoints are not tracked)
```

---

## Environment reproduction

Training and inference require **Detectron2** and Mask2Former's custom **MSDeformAttn CUDA
kernel**.

Target: Python 3.10, PyTorch 2.5.1 / torchvision 0.20.1, CUDA 11.8.

**Critical version pins** (newer releases break the stack):

| Package      | Pin            | Reason                                                          |
|--------------|----------------|----------------------------------------------------------------|
| `torch`      | `== 2.5.1`     | MSDeformAttn CUDA kernel compatibility.                        |
| `timm`       | `< 1.0`        | v1.0+ breaks the Mask2Former Swin backbone imports.            |
| `setuptools` | `< 81`         | v81 removed `pkg_resources`, which Detectron2 needs.          |

### 0. Obtain the vendored dependencies (not in this repo)

- **Mask2Former** — clone Facebook's repo into `Mask2Former/`:
  ```bash
  git clone https://github.com/facebookresearch/Mask2Former.git Mask2Former
  ```
- **COCO-pretrained weights** — download the Swin-Tiny instance-segmentation checkpoint
  from the Mask2Former model zoo into `pretrained/` (the path `train.py` expects):
  ```bash
  mkdir -p pretrained
  wget -O pretrained/mask2former_swin_tiny_coco_instance.pkl \
    https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl
  ```

### 1. Create the environment

This environment runs on the HPC ampere partitions, since the newer hopper partitions don't provide the older Python and module versions required—those would have to be installed separately.

```bash
conda create -p ~/conda_envs/tomato-seg python=3.10 && conda activate ~/conda_envs/tomato-seg
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install "setuptools<81" "timm<1.0" opencv-python scikit-image shapely h5py pycocotools pillow numpy scipy cython
```

### 2. Install Detectron2 from source

```bash
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

### 3. Compile the MSDeformAttn CUDA kernel (GPU node only)

```bash
cd Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh
cd -
```

### 4. Verify

```bash
python check_dependencies.py
```

---

## Dataset

This project uses the **Rob2Pheno** tomato RGB-D dataset (Afonso et al., 2020). The data is
**not included** in this repository.

> **Datasets are available at:**

Extract it so the layout matches what `train.py` expects:

```
data/Rob2Pheno/
  RGB/                  # RGB TIFFs
  Depth/                # Depth TIFFs (8-bit grayscale, aligned to RGB)
  train_2class.JSON     # Training annotations (83 images, COCO format, 2 classes)
  val_2class.JSON       # Validation annotations (40 images)
```

Classes: `0 = red (ripe)`, `1 = green (unripe)`.

---

## Training

> **Training requires a Linux GPU node** (the MSDeformAttn CUDA kernel). It is launched on the
> HPC via SLURM; configuration is passed through environment variables.

Single run (RGB baseline):
```bash
VARIANT=rgb sbatch slurm_train.sh
```

Depth-guided cross-attention with one refinement cycle:
```bash
VARIANT=rgbd_dca DCA_ITERS=1 sbatch slurm_train.sh
```

5-fold cross-validation (array job; `%2` caps concurrency to 2 nodes; the fold partition is
seeded so folds align across variants):
```bash
VARIANT=rgb FOLDS=5 sbatch --array=0-4%2 slurm_train.sh
```

`slurm_train.sh` forwards these environment variables to `train.py`. Key knobs:

| Env var          | `train.py` arg       | Default | Description                                              |
|------------------|----------------------|---------|----------------------------------------------------------|
| `VARIANT`        | `--variant`          | `rgb`   | `rgb`, `rgbd_early`, or `rgbd_dca`.                       |
| `BATCH_SIZE`     | `--batch-size`       | `2`     | Images per batch.                                        |
| `EPOCHS`         | `--epochs`           | —       | If set, derives `MAX_ITER` from train-set size.          |
| `EVAL_EPOCHS`    | `--eval-epochs`      | —       | Eval + checkpoint every N epochs.                        |
| `LR`             | `--lr`               | `1e-4`  | Base learning rate (backbone gets 0.1×).                 |
| `LR_SCHEDULER`   | `--lr-scheduler`     | `WarmupMultiStepLR` | `WarmupCosineLR` / `WarmupPolyLR` anneal smoothly. |
| `DCA_ITERS`      | `--dca-iters`        | `0`     | DCA refinement cycles (`rgbd_dca` only).                 |
| `DCA_LR_MULT`    | `--dca-lr-mult`      | `1.0`   | LR multiplier for the DCA fusion + depth path.           |
| `INPUT_MAX_SIZE` | `--input-max-size`   | `1280`  | Max image size for train + test.                         |
| `FOLDS`          | `--folds`            | —       | Enable k-fold CV (array job).                            |

Run `python train.py --help` for the full argument list. Outputs (metrics, config, logs) are
written under `output/<run-name>/`; checkpoints are kept locally but not committed.

---

## Inference & visualization

Run a trained variant on the validation set or a single image:
```bash
python inference.py --variant rgb --val-set
python inference.py --variant rgbd_dca --image data/Rob2Pheno/RGB/<image>.tiff
```

Render side-by-side prediction contact sheets across the variants for the 40-image
`rob2pheno_val` set (also a GPU job; see `scripts/visualize_val_predictions.py`):
```bash
sbatch slurm_visualize.sh
```

Each model entry points at a training run directory; the best-by-validation-AP checkpoint is
selected automatically (rather than the often-overfit final iteration).
