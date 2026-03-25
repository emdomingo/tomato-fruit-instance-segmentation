"""
Dependency check script for Mask2Former + Swin-Tiny
training on HPC. Run BEFORE submitting a training job.

Checks: Python version, package versions (with constraints),
CUDA availability, CUDA toolchain (nvcc, gcc),
MSDeformAttn compilation + functional test, data files.

Usage:
    python check_dependencies.py

Exit codes:
    0 -- all checks passed (or warnings only)
    1 -- one or more critical failures
"""

import sys
import os
import platform
import importlib
import shutil
import subprocess

# ── Constants ────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

ON_LINUX = platform.system() == "Linux"

# ANSI colors (auto-disabled when piped to a file)
USE_COLOR = (
    sys.stdout.isatty() and (ON_LINUX or os.name == "nt")
)
GREEN = "\033[32m" if USE_COLOR else ""
RED = "\033[31m" if USE_COLOR else ""
YELLOW = "\033[33m" if USE_COLOR else ""
RESET = "\033[0m" if USE_COLOR else ""

results = []


# ── Helpers ──────────────────────────────────────────────

def check(name, condition, msg_pass, msg_fail, level=FAIL):
    """Record a check result."""
    if condition:
        results.append((PASS, name, msg_pass))
    else:
        results.append((level, name, msg_fail))


def try_import(module_name):
    """Try to import a module."""
    try:
        mod = importlib.import_module(module_name)
        return mod, None
    except ImportError as exc:
        return None, str(exc)


def parse_version(ver_str):
    """Parse '2.5.1+cu118' -> (2, 5, 1)."""
    ver_str = (
        ver_str.split("+")[0]
        .split("a")[0]
        .split("b")[0]
        .split("rc")[0]
    )
    parts = ver_str.split(".")
    return tuple(int(p) for p in parts if p.isdigit())


def check_version(pkg_name, actual_str, op, target_str,
                   reason):
    """Compare version against constraint."""
    actual = parse_version(actual_str)
    target = parse_version(target_str)
    if op == "==":
        ok = actual == target
        label = f"=={target_str}"
    elif op == "<":
        ok = actual < target
        label = f"<{target_str}"
    elif op == ">=":
        ok = actual >= target
        label = f">={target_str}"
    else:
        return False, f"Unknown operator {op}"

    detail = f"{pkg_name} {actual_str} (need {label})"
    if not ok:
        detail += f" -- {reason}"
    return ok, detail


# ── 1. Python version ───────────────────────────────────
print("Checking dependencies...\n")

py_ver = sys.version_info
check(
    "Python version",
    (3, 8) <= py_ver < (3, 13),
    f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}",
    (
        f"Python {py_ver.major}.{py_ver.minor}"
        " -- need >= 3.8, < 3.13 (recommend 3.10)"
    ),
)

# ── 2. PyTorch + version constraint ─────────────────────
torch_mod, err = try_import("torch")
check(
    "torch",
    torch_mod is not None,
    f"torch {torch_mod.__version__}" if torch_mod else "",
    f"torch not installed: {err}",
)

if torch_mod is not None:
    ok, detail = check_version(
        "torch", torch_mod.__version__, "==", "2.5.1",
        ">=2.6 removed at::DeprecatedTypeProperties"
        " -- breaks MSDeformAttn CUDA kernel",
    )
    check("torch version", ok, detail, detail)

    cuda_available = torch_mod.cuda.is_available()
    check(
        "CUDA available",
        cuda_available,
        f"CUDA {torch_mod.version.cuda}",
        "No CUDA -- training will be extremely slow",
        level=FAIL,
    )

    if cuda_available:
        cuda_ver = parse_version(torch_mod.version.cuda)
        check(
            "CUDA version",
            cuda_ver >= (11, 8),
            f"CUDA {torch_mod.version.cuda}",
            f"CUDA {torch_mod.version.cuda}"
            " -- need >= 11.8 for A100 (sm_80)",
        )

        gpu_name = torch_mod.cuda.get_device_name(0)
        props = torch_mod.cuda.get_device_properties(0)
        gpu_mem = props.total_memory / (1024**3)
        check(
            "GPU", True,
            f"{gpu_name} ({gpu_mem:.1f} GB)", "",
        )

        cc = torch_mod.cuda.get_device_capability(0)
        check(
            "CUDA compute capability",
            cc >= (7, 0),
            f"Compute capability {cc[0]}.{cc[1]}",
            f"cc {cc[0]}.{cc[1]}"
            " -- MSDeformAttn needs >= 7.0",
        )
else:
    cuda_available = False

# ── 3. torchvision + version constraint ──────────────────
tv, err = try_import("torchvision")
check(
    "torchvision",
    tv is not None,
    f"torchvision {tv.__version__}" if tv else "",
    f"torchvision not installed: {err}",
)

if tv is not None:
    ok, detail = check_version(
        "torchvision", tv.__version__, "==", "0.20.1",
        "Must match PyTorch 2.5.1",
    )
    check("torchvision version", ok, detail, detail)

# ── 4. setuptools version constraint ────────────────────
st, err = try_import("setuptools")
if st is not None:
    ok, detail = check_version(
        "setuptools", st.__version__, "<", "81",
        ">=81 removed pkg_resources, breaks Detectron2",
    )
    check("setuptools version", ok, detail, detail)
else:
    check(
        "setuptools version", False, "",
        "setuptools not installed: " + str(err),
    )

# ── 5. CUDA toolchain (Linux only) ──────────────────────
if ON_LINUX:
    cuda_home = (
        os.environ.get("CUDA_HOME")
        or os.environ.get("CUDA_PATH")
    )
    if cuda_home and os.path.isdir(cuda_home):
        check(
            "CUDA_HOME", True,
            f"CUDA_HOME={cuda_home}", "",
        )
    elif cuda_home:
        check(
            "CUDA_HOME", False, "",
            f"CUDA_HOME={cuda_home} but dir missing",
        )
    else:
        check(
            "CUDA_HOME", False, "",
            "CUDA_HOME not set -- try: module load cuda",
        )

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        try:
            proc = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            lines = proc.stdout.splitlines()
            ver_lines = [
                line for line in lines
                if "release" in line.lower()
            ]
            info = ver_lines[0].strip() if ver_lines else "OK"
            check("nvcc", True, info, "")
        except Exception:
            check("nvcc", True, f"At {nvcc_path}", "")
    else:
        check(
            "nvcc", False, "",
            "nvcc not on PATH -- run: module load cuda",
        )

    gcc_path = shutil.which("gcc") or shutil.which("g++")
    check(
        "gcc/g++",
        gcc_path is not None,
        f"Found at {gcc_path}",
        "gcc/g++ not on PATH -- needed for CUDA kernels",
    )
else:
    check(
        "CUDA toolchain", True,
        "Skipped (not Linux)", "", level=WARN,
    )

# ── 6. Detectron2 ───────────────────────────────────────
d2, err = try_import("detectron2")
check(
    "detectron2",
    d2 is not None,
    f"detectron2 {d2.__version__}" if d2 else "",
    f"detectron2 not installed: {err}",
)

# ── 7. Mask2Former imports ───────────────────────────────
m2f_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Mask2Former",
)
if m2f_path not in sys.path:
    sys.path.insert(0, m2f_path)

m2f, err = try_import("mask2former")
check(
    "mask2former",
    m2f is not None,
    "mask2former package importable",
    f"Cannot import mask2former: {err}",
)

# ── 8. MSDeformAttn CUDA kernel ─────────────────────────
OPS_MOD = "mask2former.modeling.pixel_decoder.ops.functions"
MAKE_CMD = (
    "cd Mask2Former/mask2former/modeling/"
    "pixel_decoder/ops && sh make.sh"
)

msdeform_ok = False
msdeform, err = try_import(OPS_MOD)
if msdeform is not None:
    try:
        from mask2former.modeling.pixel_decoder.ops.functions import (  # noqa: F811
            MSDeformAttnFunction,
        )
        check(
            "MSDeformAttn import", True,
            "Compiled and importable", "",
        )
        msdeform_ok = True
    except Exception as exc:
        check(
            "MSDeformAttn import", False, "",
            f"Import failed: {exc}",
        )
else:
    check(
        "MSDeformAttn import", False, "",
        f"Not compiled -- run: {MAKE_CMD}",
    )

# Functional forward-pass test (Linux + CUDA only)
if msdeform_ok and cuda_available and ON_LINUX:
    try:
        import torch  # noqa: F811
        # Minimal tensors from Mask2Former/.../ops/test.py
        _N, _M, _D = 1, 2, 2
        _Lq, _Lvl, _P = 2, 2, 2
        shapes = torch.as_tensor(
            [(6, 4), (3, 2)], dtype=torch.long,
        ).cuda()
        lvl_start = torch.cat((
            shapes.new_zeros((1,)),
            shapes.prod(1).cumsum(0)[:-1],
        ))
        _S = sum(
            (H * W).item() for H, W in shapes
        )

        value = torch.rand(
            _N, _S, _M, _D,
        ).cuda() * 0.01
        samp_loc = torch.rand(
            _N, _Lq, _M, _Lvl, _P, 2,
        ).cuda()
        attn_w = (
            torch.rand(_N, _Lq, _M, _Lvl, _P).cuda()
            + 1e-5
        )
        attn_w /= (
            attn_w.sum(-1, keepdim=True)
            .sum(-2, keepdim=True)
        )

        with torch.no_grad():
            output = MSDeformAttnFunction.apply(
                value, shapes, lvl_start,
                samp_loc, attn_w, 2,
            )
        expected = (_N, _Lq, _M * _D)
        assert output.shape == expected, (
            f"Unexpected shape: {output.shape}"
        )
        check(
            "MSDeformAttn forward pass", True,
            "CUDA kernel executes correctly", "",
        )
    except Exception as exc:
        check(
            "MSDeformAttn forward pass", False, "",
            f"CUDA kernel failed at runtime: {exc}",
        )
elif msdeform_ok and not ON_LINUX:
    check(
        "MSDeformAttn forward pass", True,
        "Skipped (not Linux)", "", level=WARN,
    )
elif msdeform_ok and not cuda_available:
    check(
        "MSDeformAttn forward pass", True,
        "Skipped (no CUDA device)", "", level=WARN,
    )

# ── 9. Supporting packages ──────────────────────────────
supporting = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "scipy": "scipy",
    "skimage": "scikit-image",
    "shapely": "shapely",
    "h5py": "h5py",
    "pycocotools": "pycocotools",
    "PIL": "Pillow",
    "cython": "Cython",
}

for import_name, pip_name in supporting.items():
    mod, err = try_import(import_name)
    ver = getattr(mod, "__version__", "OK") if mod else ""
    check(
        pip_name,
        mod is not None,
        f"{pip_name} {ver}",
        f"{pip_name} not installed (pip install {pip_name})",
    )

# timm -- with version constraint
timm_mod, err = try_import("timm")
if timm_mod is not None:
    ok, detail = check_version(
        "timm", timm_mod.__version__, "<", "1.0",
        ">=1.0 breaks Mask2Former backbone imports",
    )
    check("timm", ok, detail, detail)
else:
    check(
        "timm", False, "",
        "timm not installed (pip install 'timm<1.0')",
    )

# ── 10. Data files ──────────────────────────────────────
project_root = os.path.dirname(os.path.abspath(__file__))

data_files = {
    "Train annotations":
        "data/Rob2Pheno/train_2class_fixed.JSON",
    "Val annotations":
        "data/Rob2Pheno/val_2class_fixed.JSON",
    "Swin-Tiny weights":
        "pretrained/mask2former_swin_tiny_coco_instance.pkl",
    "Swin config YAML": (
        "Mask2Former/configs/coco/"
        "instance-segmentation/swin/"
        "maskformer2_swin_tiny_bs16_50ep.yaml"
    ),
}

for label, rel_path in data_files.items():
    full_path = os.path.join(project_root, rel_path)
    exists = os.path.isfile(full_path)
    if exists:
        size_mb = os.path.getsize(full_path) / (1024**2)
        check(label, True, f"Found ({size_mb:.1f} MB)", "")
    else:
        check(label, False, "", f"Missing: {rel_path}")

rgb_dir = os.path.join(
    project_root, "data", "Rob2Pheno", "RGB",
)
if os.path.isdir(rgb_dir):
    n_img = len([
        f for f in os.listdir(rgb_dir)
        if f.endswith(".tiff")
    ])
    check(
        "RGB images",
        n_img >= 100,
        f"{n_img} TIFF images in data/Rob2Pheno/RGB/",
        f"Only {n_img} images (expected ~123)",
    )
else:
    check(
        "RGB images", False, "",
        "data/Rob2Pheno/RGB/ directory not found",
    )

# ── 11. Disk space ──────────────────────────────────────
try:
    _, _, free = shutil.disk_usage(project_root)
    free_gb = free / (1024**3)
    check(
        "Disk space",
        free_gb > 5,
        f"{free_gb:.1f} GB free",
        f"Only {free_gb:.1f} GB free -- need >5 GB",
        level=WARN,
    )
except Exception:
    check(
        "Disk space", False, "",
        "Could not check disk space", level=WARN,
    )

# ── 12. Platform ────────────────────────────────────────
check(
    "Platform",
    ON_LINUX,
    f"{platform.system()} (correct for HPC)",
    f"{platform.system()} -- expected Linux on HPC",
    level=WARN,
)

# ── Print results ────────────────────────────────────────
COLORS = {"PASS": GREEN, "FAIL": RED, "WARN": YELLOW}
SYMBOLS = {
    "PASS": "[OK]", "FAIL": "[FAIL]", "WARN": "[WARN]",
}

print(f"\n{'Status':<8} {'Check':<30} {'Details'}")
print("-" * 78)

n_fail = 0
n_warn = 0
for status, name, detail in results:
    color = COLORS[status]
    symbol = SYMBOLS[status]
    print(f"{color}{symbol:<8}{RESET} {name:<30} {detail}")
    if status == FAIL:
        n_fail += 1
    elif status == WARN:
        n_warn += 1

print("-" * 78)
total_pass = len(results) - n_fail - n_warn
print(
    f"\n{len(results)} checks: "
    f"{GREEN}{total_pass} passed{RESET}, "
    f"{RED}{n_fail} failed{RESET}, "
    f"{YELLOW}{n_warn} warnings{RESET}\n"
)

if n_fail > 0:
    print(
        f"{RED}Fix FAIL items before submitting"
        f" a training job.{RESET}"
    )
    sys.exit(1)
elif n_warn > 0:
    print(
        f"{YELLOW}Warnings present"
        f" -- review before training.{RESET}"
    )
    sys.exit(0)
else:
    print(
        f"{GREEN}All checks passed."
        f" Ready to train.{RESET}"
    )
    sys.exit(0)
