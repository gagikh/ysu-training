#!/bin/bash
# ./compile.sh day01/template.cu  ->  ./template

nvcc -arch=sm_90 "$1" -o "$(basename "${1%.cu}")" -lcublas -lnppc -lnppif
