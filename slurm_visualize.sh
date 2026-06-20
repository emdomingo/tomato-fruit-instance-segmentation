#!/bin/bash
#SBATCH --job-name=inference-and-viz
#SBATCH --output=logs/viz_%j.out
#SBATCH --error=logs/viz_%j.err
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

python -W ignore::FutureWarning scripts/visualize_val_predictions.py \
    --out ${OUT:-viz_val} \
    --score-thr ${SCORE_THR:-0.5} \
    --input-max-size ${INPUT_MAX_SIZE:-1280}
