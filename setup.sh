#!/bin/bash
# source setup.sh   (մեկ անգամ, repository-ի արմատից)

CUDA_BIN=/mnt/weka/shared-cache/miniforge3/envs/indoorolo/bin
LINE="export PATH=\"\$PATH\":$CUDA_BIN"

grep -qxF "$LINE" ~/.bashrc || echo "$LINE" >> ~/.bashrc
export PATH="$PATH":$CUDA_BIN

unset CUDA_BIN LINE

nvcc --version | grep release
