# Day 7: Warp Intrinsics, Reduction and Atomics

## Objectives
- Use warp shuffle functions for intra-warp communication without shared memory or barriers
- State what the lane mask asserts, and why a `_sync` intrinsic in a divergent branch is undefined behaviour rather than merely slow
- Implement warp-level reduction and extend it to a block with the warp to shared to warp pattern
- Implement an inclusive warp scan, and use `__ballot_sync` with `__popc` for binary predicates
- Explain why atomic contention serialises, and apply privatisation

## Key Concepts
- `__shfl_down_sync`, `__shfl_up_sync`, `__shfl_xor_sync`, `__shfl_sync`
- Lane mask, divergence, and identity values for out-of-range lanes
- Hierarchical reduction: warp, then shared, then warp
- Inclusive scan (Kogge-Stone) and stream compaction
- `__syncwarp`, `__activemask`, `__ballot_sync`
- Atomics execute at L2; contention on one address serialises
- Privatisation into shared memory; warp-aggregated atomics

## Why shuffles win
Shuffles move data register to register, touching no memory at all. That is why they beat a shared-memory reduction, and also why the advantage disappears once the data no longer fits in a warp's registers.

## Resources
- CUDA Programming Guide — Warp Shuffle Functions, Warp Vote Functions, Atomic Functions
- [`INTRINSICS.md`](https://github.com/gagikh/cuda/blob/main/INTRINSICS.md) — shuffle, vote, bit operations, atomics in one table
- CUB device-wide reduction and scan: https://nvidia.github.io/cccl/cub/

## Hands-On Task
Compute the mean of a real image using warp reduction plus `atomicAdd`, then a 256-bin histogram twice — once with global atomics, once privatised into shared memory.

## Self-Learning
1. Implement warp sum reduction with `__shfl_down_sync`. Verify against a host loop for `n = 1024` filled with ones; the result must be exactly `n`.
2. Extend to `block_reduce_sum` with the warp to shared to warp pattern, then reduce the grid by having thread 0 of each block `atomicAdd` its total. Count the `__syncthreads()` calls against a classic shared-memory tree reduction and time both.
3. Implement an inclusive scan within a warp and use it to compact the indices of pixels above a threshold.
4. Redo task 3 with `__ballot_sync` and `__popc`, and compare.
5. Implement the 256-bin histogram both ways. Time both on a large image, then on an image that is nearly one shade. Explain whether the gap widens or narrows.
6. Replace the shared-memory `atomicAdd` with `atomicAdd_block` and measure.

## Self-Check
No answers given.

1. Why does `__shfl_down_sync` not need `__syncthreads()`?
2. After five steps of the reduction, why is lane 0 specifically guaranteed to hold the total?
3. Passing `0xFFFFFFFF` when only some lanes reach the instruction is undefined behaviour. Why is a correct answer on your GPU not evidence of correctness?
4. Privatisation adds a zeroing pass, two barriers and a merge pass. Describe an input where that is a net loss.
5. Why is a shared-memory `atomicAdd` cheaper than a global one, when both serialise on collision?

## Code Template
See [`template.cu`](template.cu).
