#!/bin/bash
#SBATCH --output=/cluster/home/t144807uhn/logs/wsi-attention-composition/composition/%j.out
#SBATCH --error=/cluster/home/t144807uhn/logs/wsi-attention-composition/composition/%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G

# Make the project-specific logs directory
mkdir -p /cluster/home/t144807uhn/logs/wsi-attention-composition/composition

# Activate the virtual environment
source /cluster/home/t144807uhn/envs/wsi-env/bin/activate

# Ensure that all commands resolve back to the proper root directory
cd /cluster/home/t144807uhn/wsi-attention-composition

# Add the cloned HoVer-net repository to the Python path
export PYTHONPATH="/cluster/projects/kumargroup/sophia/hover_net:$PYTHONPATH"

echo "=========================================="
echo "Mini Sweep Job ID:  $SLURM_JOB_ID"
echo "Job Name:           $SLURM_JOB_NAME"
echo "Node:               $SLURMD_NODENAME"
echo "Start time:         $(date)"
echo "=========================================="

srun python -u scripts/02_extract_composition.py \
    --config config.yaml \
    --project $1

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="