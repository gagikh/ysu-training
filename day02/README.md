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

## How a block is placed
A block is assigned to one SM and stays there until it finishes. How many blocks fit on an SM at once is decided by three limits at the same time: registers per thread, shared memory per block, and the hardware cap on resident blocks. The lowest of the three wins. This is why block size is a hardware question, not a style question.

## Resources
- CUDA Programming Guide — Thread Hierarchy
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
