#!/bin/bash

#SBATCH --job-name=gpu_report
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --time=00:00:30
#SBATCH --output=logs/report_%j.log

# sbatch submit.sh build/day01
# sbatch submit.sh build/day05 image.png

./"$@"
