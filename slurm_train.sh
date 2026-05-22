#!/bin/bash
#SBATCH --job-name=tomato-${VARIANT:-rgb}
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=ampere80
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

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

# K-fold CV: when submitted as an array job (sbatch --array=0-4%2 ... FOLDS=5),
# use SLURM_ARRAY_TASK_ID as the fold index. Otherwise CV stays off.
# Cap concurrency to the cluster's 2-node availability with the %2 suffix.
# CV_SEED defaults to 42 so the same fold partition is reused across variants
# (rgb fold 0 and rgbd_early fold 0 see the same images). Override per-experiment
# by exporting CV_SEED before sbatch.
FOLD_ARGS=""
if [ -n "$FOLDS" ] && [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    FOLD_ARGS="--folds $FOLDS --fold-index $SLURM_ARRAY_TASK_ID --cv-seed ${CV_SEED:-42}"
fi

# Train
python -W ignore::FutureWarning train.py \
    --variant ${VARIANT:-rgb} \
    --batch-size ${BATCH_SIZE:-4} \
    --lr ${LR:-2e-4} \
    --max-iter ${MAX_ITER:-5000} \
    --num-workers 8 \
    ${DATASET:+--dataset $DATASET} \
    ${DCA_ITERS:+--dca-iters $DCA_ITERS} \
    ${DCA_HEADS:+--dca-heads $DCA_HEADS} \
    ${GREEN_WEIGHT:+--green-weight $GREEN_WEIGHT} \
    ${INPUT_MAX_SIZE:+--input-max-size $INPUT_MAX_SIZE} \
    ${RUN_TAG:+--run-tag $RUN_TAG} \
    $FOLD_ARGS
