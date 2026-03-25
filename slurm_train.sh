#!/bin/bash
#SBATCH --job-name=tomato-${VARIANT:-rgb}
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=ampere80
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Environment setup (from guides/hpc_env_setup.md)
module load anaconda/conda3 cuda/cuda-11.6
eval "$(conda shell.bash hook)"
conda activate ~/conda_envs/tomato-seg

export CUDA_HOME=/usr/local/cuda-11.6
export TORCH_CUDA_ARCH_LIST="8.0"
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++

cd ~/tomato-seg
mkdir -p logs

# Dependency check
python check_dependencies.py
if [ $? -ne 0 ]; then
    echo "Dependency check failed. Aborting."
    exit 1
fi

# Train
python train.py \
    --variant ${VARIANT:-rgb} \
    --batch-size ${BATCH_SIZE:-2} \
    --lr ${LR:-1e-4} \
    --max-iter ${MAX_ITER:-5000} \
    --num-workers 4
