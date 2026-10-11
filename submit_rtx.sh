#!/bin/bash

#SBATCH --job-name=rtx_job
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --time=00:00:30
#SBATCH --output=logs/report_%j.log

# sbatch submit.sh template
# sbatch submit.sh template data/camera.bmp

./"$@"
