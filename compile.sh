#!/bin/bash
# ./compile.sh day01/template.cu  ->  ./template

# H100
# nvcc -arch=sm_90 "$1" -o "$(basename "${1%.cu}")" #-lcublas -lnppc -lnppif

# RTX A6000
nvcc -arch=sm_86 "$1" -o "$(basename "${1%.cu}")" #-lcublas -lnppc -lnppif
