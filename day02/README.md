# Day 2: Thread Hierarchy, Indexing and Launch Configuration

## Objectives
- Divide work across threads, blocks and grids, for 1D and 2D data
- Write the global-index formula and explain why it has the shape it has
- Recognise coalesced and uncoalesced indexing, and index so that a warp's accesses are contiguous
- Choose a block size, and check the choice with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
- Write a grid-stride loop that is correct for any input size at a fixed launch configuration

## Key Concepts
- Threads, blocks, grids: structure and enumeration
- Launch configuration and kernel invocation
- 1D and 2D thread indexing
- Memory coalescing as a consequence of the index formula
- Occupancy and block-size choice (first pass; the hardware reason comes on Day 3 and Day 6)
- Grid-stride loops

## Definitions
**Thread** — The smallest unit of execution; identified within its block by `threadIdx`, within the grid by combining `threadIdx` with `blockIdx` and `blockDim`.

**Block** — A group of threads, up to `maxThreadsPerBlock`, that execute on the same SM and can cooperate through shared memory and `__syncthreads()`. A kernel launch creates a grid of blocks.

**Grid** — The full set of blocks launched by one kernel call, `<<<grid, block>>>`.

**Launch configuration** — The arguments of a kernel call: grid dimensions in blocks and block dimensions in threads, each up to three-dimensional, plus optional dynamic shared memory size and stream.

**`blockIdx`, `threadIdx`, `blockDim`, `gridDim`** — Built-in read-only variables in device code: the block's index in the grid, the thread's index in the block, the block's dimensions, the grid's dimensions.

**Resident blocks** — The blocks assigned to one SM at the same time. The count is the smallest of three limits: the hardware cap on blocks per SM, registers per SM divided by the block's register demand, and shared memory per SM divided by the block's shared memory demand.

**Occupancy** — How many warps are resident on an SM at once relative to the maximum it could hold. Limited by whichever resource runs out first: registers per thread, shared memory per block, or the thread-count cap. It is a means to **latency hiding**, not a goal — returns flatten past roughly 50 percent, and coarsened kernels trade it away deliberately.

**`cudaOccupancyMaxActiveBlocksPerMultiprocessor`** — The runtime call returning how many blocks of a given kernel and block size will be resident per SM, without running the kernel.

**Coalescing (memory coalescing)** — When the 32 lanes of a warp access consecutive addresses, the hardware serves them in a single 128-byte transaction instead of up to 32 separate ones. The property belongs to the warp, not the thread: what matters is the combined footprint of one instruction across all 32 lanes, not the pattern one thread traces over time.

**Grid-stride loop** — A launch pattern where a fixed number of threads each process several elements in a loop, striding by the total thread count, instead of sizing the grid to match the data. Correct for any input size without recomputing launch dimensions.

## How a block is placed
A block is assigned to one SM and stays there until it finishes. How many blocks fit on an SM at once is decided by three limits at the same time: registers per thread, shared memory per block, and the hardware cap on resident blocks. The lowest of the three wins. This is why block size is a hardware question, not a style question.

## Visual
![A grid of blocks, each block a 2D array of threads, with the global-index formula shown](thread_hierarchy.svg)

A launch creates a grid of blocks, and each block is itself a 1D, 2D or 3D array of threads. `blockIdx` says which block a thread is in, `threadIdx` which slot inside that block. The formula in the diagram is the one reused in almost every kernel that follows.

## Animated
![The global-index formula evaluated for four threads across three blocks, each cycling through blockIdx.x, blockDim.x and threadIdx.x to a concrete number](thread_indexing.svg)

One formula, four concrete threads. It is always `blockIdx.x * blockDim.x + threadIdx.x`; only the block and thread values change.

![Four fixed threads sweeping a 16-element array in stride-4 iterations, each thread taking a different element every iteration](grid_stride_loop.svg)

Four threads cover sixteen elements in four iterations. Doubling the array to thirty-two gives eight iterations at the same launch configuration. This is the grid-stride loop of the hands-on task.

## Resources
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 2. Programming Model · 5. Performance Guidelines
- CUDA C++ Best Practices Guide — Occupancy, Coalesced Access to Global Memory

## Hands-On Task
Vector addition, then the same kernel extended to 2D indexing over two grayscale images.

## Self-Learning
1. Implement 1D vector addition for several array sizes, with bounds checking for sizes that are not a multiple of the block size.
2. Extend to 2D indexing and add two grayscale images pixel by pixel.
3. Time block sizes of 32, 64, 128 and 256. Call `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for each and note where the timings stop tracking the occupancy numbers.
4. Implement the same kernel as a grid-stride loop. Verify it is still correct after increasing `n` far beyond `blocks * threads` without changing the launch configuration.
5. Write two copy kernels, one indexed `blockIdx.x * blockDim.x + threadIdx.x` and one indexed `threadIdx.x * gridDim.x + blockIdx.x`. Both are correct. Measure both.
6. For your chosen block size, compute by hand how many blocks are resident per SM, then confirm with the occupancy API.

## Self-Check
No answers given.

1. What goes wrong in the global-index formula if you omit `blockDim.x`?
2. Why does a grid-stride loop stay correct when `n` doubles, while a one-thread-per-element kernel does not?
3. Both index formulas in task 5 produce correct output. What measurement tells you which one to keep?
4. You launch `<<<100, 256>>>` over 20,000 elements with bounds checking. How many threads do no work?

## Code Template
See [`template.cu`](template.cu).
