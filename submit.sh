#!/bin/bash

#SBATCH --job-name=gpu_report
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --time=00:00:30
#SBATCH --output=logs/report_%j.log

# sbatch submit.sh template
# sbatch submit.sh template image.png

./"$@"
