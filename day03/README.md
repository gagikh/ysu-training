# Day 3: SIMT Execution, Warps, Divergence and Latency Hiding

## Objectives
- Explain SIMT: one instruction fetched and decoded once, issued to 32 lanes
- Describe the instruction pipeline and why register read and memory are separate stages
- Explain how the warp scheduler hides latency by switching between resident warps, and why this replaces the large caches a CPU uses
- Recognise and avoid branch divergence; explain why it serialises rather than parallelises
- Apply loop unrolling where it helps

## Key Concepts
- SIMT and the instruction pipeline: fetch, decode, warp scheduler, lockstep issue
- Warp formation from `threadIdx`; why block sizes that are not multiples of 32 waste lanes
- Eligible, active and stalled warps; what a stall reason means in a profiler
- Branch divergence and reconvergence
- Loop unrolling

## Why latency hiding is the centre of the architecture
A CPU reduces memory latency with deep cache hierarchies and speculation. A GPU largely does not: it tolerates the latency instead. When a warp stalls on a load, the scheduler issues an instruction from another resident warp in the same cycle. This single decision explains occupancy, why block size matters, why divergence is expensive, and why arithmetic intensity determines performance. Everything later in the course is a consequence of it.

## Resources
- CUDA Programming Guide — Hardware Implementation, Maximize Utilization
- Oxford CUDA course, lecture 3: https://people.maths.ox.ac.uk/~gilesm/cuda/lecs/lec3.pdf
- Using CUDA warp-level primitives: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
- Pipeline and warp scheduling diagrams: [`day03/`](https://github.com/gagikh/cuda/tree/main/day03), [`warp_animations.html`](https://github.com/gagikh/cuda/blob/main/day03/warp_animations.html)

## Hands-On Task
Vector addition timed against an equivalent CPU loop, then a BGR to grayscale conversion kernel.

## Self-Learning
1. Implement and time vector addition against a CPU loop.
2. Convert BGR to grayscale in a kernel (`gray = 0.114*B + 0.587*G + 0.299*R`).
3. Introduce divergence deliberately (`if (threadIdx.x % 2 == 0)`) and measure the cost against a divergence-free version.
4. Repeat task 3 with the branch taken on `threadIdx.x / 32` instead. Explain the difference in result.
5. Apply `#pragma unroll` to a small fixed-trip-count loop and compare the generated SASS as well as the timing.

## Self-Check
No answers given.

1. Why is divergence expensive even though every thread eventually does its useful work?
2. Why are register read and memory separate pipeline stages rather than folded into execute?
3. Half a warp takes the `if`, half the `else`. How does that warp's execution time compare with a divergence-free warp doing the same total work?
4. Task 4 branches on `threadIdx.x / 32` and costs nothing. Why?
5. If latency hiding depends on having other warps ready, what happens to a kernel that uses so many registers that only one warp is resident?

## Code Template
See [`template.cu`](template.cu).
