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

## Definitions
**Warp** — A group of 32 threads within a block that the hardware schedules and executes together in lockstep. The unit warp-level intrinsics operate on.

**SIMT (Single Instruction, Multiple Threads)** — NVIDIA's execution model: one instruction is fetched and decoded once and issued to all 32 threads of a warp at the same time.

**Instruction pipeline** — The stages an instruction passes through: fetch, decode, register read, execute, memory, writeback. Several instructions are in flight at once, one per stage.

**Register file** — The per-SM storage from which every thread's registers are allocated. Its size is fixed, so registers per thread and resident warps trade against each other.

**Eligible, active and stalled warp** — An active, that is resident, warp occupies a warp slot on the SM. It is eligible when its next instruction's operands and the required unit are ready, and stalled otherwise. The scheduler issues only from eligible warps.

**Stall reason** — The profiler's classification of why a warp was not eligible: a memory dependency, a barrier, a busy execution pipe, instruction fetch, and so on. It names what to fix.

**Latency hiding** — The GPU's performance strategy: when one warp stalls, the warp scheduler issues an instruction from a different, ready warp in the same cycle instead of leaving the pipeline idle. The reason GPUs favour many threads over few fast ones.

**Divergence (warp divergence)** — When threads within one warp take different paths through a branch. Because a warp executes in lockstep, the hardware runs each path separately with some lanes masked off, instead of in parallel.

**Reconvergence** — The point after a divergent branch where all lanes of the warp execute the same instruction again. From Volta on, lanes have independent program counters and reconvergence at the end of a branch is not guaranteed; `__syncwarp` makes it explicit.

**Loop unrolling** — Replacing a loop by repeated copies of its body, which removes branch and index instructions and exposes independent operations to the scheduler. `#pragma unroll` controls it.

## Why latency hiding is the centre of the architecture
A CPU reduces memory latency with deep cache hierarchies and speculation. A GPU largely does not: it tolerates the latency instead. When a warp stalls on a load, the scheduler issues an instruction from another resident warp in the same cycle. This single decision explains occupancy, why block size matters, why divergence is expensive, and why arithmetic intensity determines performance. Everything later in the course is a consequence of it.

## Visual
![SIMT instruction pipeline: fetch, decode, warp scheduler, then one instruction issued in lockstep to all 32 lanes of a warp](pipeline.svg)

One instruction is fetched and decoded once, then issued to all 32 threads of a warp at the same time. This is the reason divergence costs: when lanes disagree on a branch the hardware masks lanes off and runs each path in turn.

![A six-stage pipeline — fetch, decode, register read, execute, memory, writeback — with four instructions in flight, each one stage behind the previous](pipeline_timeline.svg)

The same pipeline seen over time. Register read and memory are separate stages because the register file and global memory are both limited resources with real access latency. When a warp stalls in the memory stage, the scheduler issues an instruction from a different warp in that cycle rather than leaving the stage idle.

## Animated
![A warp scheduler cycling through six resident warps: a token travels to the scheduler and on to the execution units, and a new warp is issued as soon as one goes idle](warp_scheduling.svg)

As soon as one warp becomes not ready, another resident warp takes its place. This is latency hiding, and occupancy is the measure of how much of it is available.

![A 32-thread warp splitting on a branch: threads 0-15 execute path A while 16-31 are masked off, then the reverse, then all 32 reconverge](warp_divergence.svg)

The two paths run one after the other. Divergence serialises a warp; it does not add parallelism.

A steppable version is in [`warp_animations.html`](warp_animations.html). Open it locally in a browser — GitHub's file viewer shows HTML as source rather than running it.

## Resources
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 4. Hardware Implementation · 5. Performance Guidelines
- Oxford CUDA course, lecture 3: https://people.maths.ox.ac.uk/~gilesm/cuda/lecs/lec3.pdf
- Using CUDA warp-level primitives: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
- Pipeline and warp scheduling diagrams: see `Visual` and `Animated` above; interactive version in [`warp_animations.html`](warp_animations.html)

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
