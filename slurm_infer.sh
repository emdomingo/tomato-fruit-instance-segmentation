#!/bin/bash
#SBATCH --job-name=tomato-infer-${VARIANT:-rgb}
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
#SBATCH --partition=ampere80
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Environment setup
module load anaconda/conda3 cuda/cuda-11.6
eval "$(conda shell.bash hook)"
conda activate ~/conda_envs/tomato-seg

export CUDA_HOME=/usr/local/cuda-11.6
export TORCH_CUDA_ARCH_LIST="8.0"

cd ~/tomato-seg
mkdir -p logs

# Install opencv if missing (headless for HPC)
python -c "import cv2" 2>/dev/null || pip install --quiet opencv-python-headless

# Build inference command
CMD="python inference.py \
    --variant ${VARIANT:-rgb} \
    --no-display \
    --threshold ${THRESHOLD:-0.3}"

if [ -n "${OUTPUT_DIR}" ]; then
    CMD="$CMD --output-dir ${OUTPUT_DIR}"
fi

if [ -n "${CHECKPOINT}" ]; then
    CMD="$CMD --checkpoint ${CHECKPOINT}"
fi

if [ -n "${IMAGE}" ]; then
    CMD="$CMD --image ${IMAGE}"
else
    CMD="$CMD --val-set"
fi

echo "Running: $CMD"
$CMD
