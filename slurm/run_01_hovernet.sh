#!/bin/bash
#SBATCH --output=/cluster/home/t144807uhn/logs/wsi-attention-composition/hovernet/%j.out
#SBATCH --error=/cluster/home/t144807uhn/logs/wsi-attention-composition/hovernet/%j.err
#SBATCH --account=kumargroup_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G




# Add the cloned HoVer-net repository to the Python path
export PYTHONPATH="/cluster/projects/kumargroup/sophia/hover_net:$PYTHONPATH"

...