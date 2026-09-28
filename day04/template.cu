// Day 4: Memory Types and Host-Device Transfers
// Goal: compare pageable vs. pinned vs. unified memory for the same transfer/compute.
//
// Build:  ./compile.sh day04/template.cu      (login node, repository root)
// Run:    sbatch submit.sh template
//         under Nsight Systems, on a GPU node: nsys profile -o logs/day04 ./template

#include <cstdio>
#include <cuda_runtime.h>
#include "../common/cuda_check.h"

constexpr int N = 1 << 22;

__global__ void add_vectors(const double *a, const double *b, double *c, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}

void run_with_pageable_memory()
{
    // TODO: allocate host buffers with plain `new`/`malloc`, device buffers with
    // CUDA_CHECK(cudaMalloc(...)), CUDA_CHECK(cudaMemcpy(...)) in, launch
    // add_vectors, CUDA_CHECK_LAST_ERROR(), CUDA_CHECK(cudaMemcpy(...)) out,
    // time with cudaEvents.
}

void run_with_pinned_memory()
{
    // TODO: allocate host buffers with CUDA_CHECK(cudaMallocHost(...)) instead,
    // otherwise same as above. Compare the transfer time against
    // run_with_pageable_memory().
}

void run_with_unified_memory()
{
    // TODO: allocate a, b, c with CUDA_CHECK(cudaMallocManaged(...)), touch them
    // directly from host, launch add_vectors directly on them (no explicit
    // cudaMemcpy needed), CUDA_CHECK_LAST_ERROR(),
    // CUDA_CHECK(cudaDeviceSynchronize()) before reading results on host.
}

int main()
{
    run_with_pageable_memory();
    run_with_pinned_memory();
    run_with_unified_memory();

    // TODO (self-learning #2): try cudaHostRegister on a pageable buffer instead of
    // allocating pinned memory up front.

    return 0;
}
