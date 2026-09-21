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

## Definitions
**Lane** — A thread's position within its warp, 0 to 31.

**Warp shuffle** — The instruction family `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`, which lets a lane read a register of another lane in the same warp with no memory access and no barrier.

**Lane mask** — The 32-bit first argument of every `_sync` intrinsic, one bit per lane, naming the lanes that must take part. `0xffffffff` means the whole warp.

**`_sync` suffix** — Marks the intrinsics that require the named lanes to be converged at the instruction. If a lane in the mask does not reach it, the result is undefined rather than merely slow. The unsuffixed forms have been removed from the language.

**XOR (butterfly) exchange** — `__shfl_xor_sync(mask, v, k)`: lane `i` exchanges with lane `i ^ k`. Every lane both sends and receives in one instruction, which is why it is the form used when all lanes need the result.

**Reduction** — Combining N values into one with an associative operator. On a GPU it is done as a tree: within a warp by shuffles, across warps through shared memory, then by one warp again.

**Inclusive and exclusive scan** — Prefix sums. Element i of an inclusive scan is the combination of elements 0 to i; of an exclusive scan, elements 0 to i-1.

**Kogge-Stone** — The scan formulation used inside a warp: at step k every lane adds the value from the lane k positions below it, with k doubling each step. Five steps for a 32-lane warp.

**Stream compaction** — Removing the elements that fail a predicate and packing the rest. A scan over the predicate gives each surviving element its output index.

**`__ballot_sync`** — Returns a 32-bit mask with bit N set if lane N's predicate was true, delivered to every participating lane.

**`__popc`** — Counts the set bits of a 32-bit value. Applied to a ballot result it counts the lanes that satisfied the predicate.

**`__activemask`** — Returns which lanes are converged at this instruction. It reports what happens to be true, and is not a substitute for a mask the code determines itself.

**`__syncwarp`** — A warp-level barrier forcing the named lanes to converge. Needed where code depends on lanes being together and the compiler cannot prove that they are.

**Atomic operation** — A read-modify-write on one address that no other thread can interleave with: `atomicAdd`, `atomicCAS`, `atomicMax` and the rest.

**Atomic contention** — Several threads targeting the same address. The updates serialise at L2, so the cost grows with the number of colliding threads, not with the number of atomic instructions.

**Privatisation** — Giving each block, or each warp, a private copy of a contended accumulator, updating that copy locally, and merging once at the end. Turns one global atomic per input element into a few per block. The standard fix for atomic contention.

**Warp-aggregated atomics** — Having one lane perform a single atomic for the whole warp's contribution, computed first by a ballot and a warp reduction. Reduces the number of atomics by up to 32 times; the warp-scoped case of privatisation.

**Cooperative groups** — An API that makes the group a piece of code synchronises over explicit — the block, the currently converged lanes, the whole grid — instead of implicit in `__syncthreads()`.

## Why shuffles win
Shuffles move data register to register, touching no memory at all. That is why they beat a shared-memory reduction, and also why the advantage disappears once the data no longer fits in a warp's registers.

## Visual
![Warp shuffle reduction over 8 lanes, each step halving the offset (4, 2, 1) through __shfl_down_sync until lane 0 holds the total](warp_reduction.svg)

`__shfl_down_sync` lets a lane read a value straight out of another lane's register, with no shared memory and no `__syncthreads()`. Halving the offset each step — 16, 8, 4, 2, 1 for a full warp — sums 32 values in five steps, with lane 0 holding the result.

![__ballot_sync collecting a warp's 32 boolean predicates into a single 32-bit mask, one bit per lane](warp_ballot.svg)

`__ballot_sync` turns "which lanes satisfy this condition" into one 32-bit integer that every lane receives: bit N is set if and only if lane N's predicate was true. Combined with `__popc` it counts them in one instruction, and it is the same mechanism `__activemask` and `__syncwarp` use to know which lanes are still participating.

## Animated
![8 lanes cycling through three shuffle intrinsics: __shfl_down_sync where lane i reads from lane i+1, __shfl_up_sync where lane i reads from lane i-1, and __shfl_xor_sync where lanes swap in pairs](warp_shuffle_intrinsics.svg)

The same eight lanes under three intrinsics. `__shfl_down_sync` and `__shfl_up_sync` shift values one direction by a fixed offset; `__shfl_xor_sync` exchanges values between paired lanes (`i` and `i ^ mask`), so every lane both sends and receives in one instruction. That is what makes it the form used for butterfly reductions.

## Resources
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 7.22. Warp Shuffle Functions · 7.14. Atomic Functions · 8. Cooperative Groups
- [`INTRINSICS.md`](../INTRINSICS.md) — shuffle, vote, bit operations, atomics in one table
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
