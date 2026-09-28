#!/bin/bash
# Build one day's program. On the login node, from the repository root:
#
#   bash compile.sh day01
#
# The executable is build/day01. The target is sm_90, the cluster's H100.
set -e

day="$1"
if [ -z "$day" ] || [ ! -f "$day/template.cu" ]; then
    echo "usage: bash compile.sh dayNN" >&2
    exit 1
fi

case "$day" in
    day05|day06|day07|day08) libs="$(pkg-config --cflags --libs opencv4)" ;;
    day09)                   libs="-lcublas -lnppc -lnppif $(pkg-config --cflags --libs opencv4)" ;;
    *)                       libs="" ;;
esac

mkdir -p build
nvcc -arch=sm_90 -lineinfo -o "build/$day" "$day/template.cu" $libs
echo "built build/$day"
