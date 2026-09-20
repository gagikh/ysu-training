# Day 5: Shared Memory and Bank Conflicts

## Objectives
- Explain shared memory banking and how a conflict arises
- Synchronise access to shared memory correctly, and state what `__syncthreads()` does and does not guarantee
- Distinguish global, shared, constant and pitched memory and choose between them
- Implement a tiled 2D filter and remove its bank conflicts

## Key Concepts
- 32 banks; conflict-free access, broadcast, and n-way conflicts
- `__syncthreads()`: what it guarantees, and its cost in occupancy terms
- Shared memory and L1 share the same physical SRAM; the carveout is configurable
- Constant memory and the broadcast path
- Pitched memory and why `step` differs from `cols * elemSize()`
- Tiling as a reduction in global memory traffic

## Resources
- CUDA Programming Guide — Shared Memory
- CUDA C++ Best Practices Guide — Shared Memory
- Bank conflict diagrams and stride explorer: [`day05/`](https://github.com/gagikh/cuda/tree/main/day05)

## Hands-On Task
A shared-memory tiled 2D filter over a real image, then a Sobel filter.

## Self-Learning
1. Implement a tile-based 2D convolution (start with a box blur) with a halo region.
2. Create a shared-memory access pattern with bank conflicts on purpose, measure the cost, then remove it with padding.
3. Implement a 2D Sobel filter using shared memory.
4. For your tile, compute by hand how many bank conflicts a warp incurs, then confirm the number in Nsight Compute.
5. Vary the tile size and record the point at which shared memory per block, not arithmetic, limits occupancy.

## Self-Check
No answers given.

1. Why do 32 threads reading `tile[threadIdx.x][k]` for a fixed `k` collide on one bank?
2. A warp accesses shared memory with stride 3. How many transactions does that cost, and why is the answer not 3?
3. What does `__syncthreads()` explicitly not guarantee?
4. Padding a tile by one column wastes shared memory. Why is it usually still the right trade?

## Code Template
See [`template.cu`](template.cu).
