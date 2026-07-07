#!/bin/bash
#SBATCH --job-name=01_run_hovernet
#SBATCH --output=/cluster/home/t144807uhn/logs/wsi-attention-composition/hovernet/%j.out
#SBATCH --error=/cluster/home/t144807uhn/logs/wsi-attention-composition/hovernet/%j.err
#SBATCH --account=kumargroup_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=12:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Make the project-specific logs directory
mkdir -p /cluster/home/t144807uhn/logs/wsi-attention-composition/hovernet

# Activate the virtual environment
source /cluster/home/t144807uhn/envs/wsi-env/bin/activate

# Ensure that all commands resolve back to the proper root directory
cd /cluster/home/t144807uhn/wsi-attention-composition

# Add the cloned HoVer-net repository to the Python path
export PYTHONPATH="/cluster/projects/kumargroup/sophia/hover_net:$PYTHONPATH"

echo "=========================================="
echo "Mini Sweep Job ID:  $SLURM_JOB_ID"
echo "Job Name:           $1"
echo "Node:               $SLURMD_NODENAME"
echo "GPU:                $CUDA_VISIBLE_DEVICES"
echo "Start time:         $(date)"
echo "=========================================="

srun python scripts/01_run_hovernet.py \
    --config config.yaml \
    --project TCGA-BLCA

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="