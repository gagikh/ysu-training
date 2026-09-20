# Day 6: Coalescing, Caches and Bandwidth

## Objectives
- Explain coalescing in terms of sectors and cache lines rather than as a style rule
- Describe L1 and L2 behaviour, including where atomics are executed
- Use `__ldg` and the per-instruction cache operators, and explain when each is justified
- Amortise per-thread fixed cost with thread coarsening, and describe what it trades against
- Express a kernel's speed as a percentage of theoretical peak bandwidth and use it to decide when to stop optimising

## Key Concepts
- Cache line is 128 bytes, sector is 32 bytes; a warp's request becomes some number of sectors
- Coalescing as minimisation of sectors per request
- L1 per SM, L2 chip-wide; atomics execute at L2
- LRU approximation and cache hints: `__ldg`, `__ldcs`, `__stcs`, `__ldlu`
- L2 persistence (`cudaAccessPolicyWindow`)
- Thread coarsening against occupancy
- Memory-bound and compute-bound; achieved bandwidth as a fraction of peak

## Method
Measure achieved bandwidth, divide by the peak figure computed on Day 1, and place the kernel on a roofline. Below roughly half of peak there is usually something structural to fix. Above roughly 80 percent the remaining work is algorithmic — fewer bytes moved, not faster movement.

## Resources
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 3.2.3. Device Memory L2 Access Management · 5. Performance Guidelines
- CUDA C++ Best Practices Guide — Coalesced Access to Global Memory, L2 Cache
- Nsight Compute — Profiling Guide, memory chart and sectors per request
- Williams S., Waterman A., Patterson D. Roofline: An Insightful Visual Performance Model. *CACM* 52(4), 2009
- [`PERFORMANCE.md`](https://github.com/gagikh/cuda/blob/main/PERFORMANCE.md) — which optimisation to try, in what order

## Hands-On Task
Optimise the Day 5 filter using everything above, and document before and after figures including percentage of peak bandwidth.

## Self-Learning
1. Measure the bandwidth of coalesced, strided and random access to the same buffer. Record sectors per request for each in Nsight Compute.
2. Add `__ldg()` to a read-heavy kernel and measure the effect.
3. Replace tile padding with `tile[row][col ^ row]` swizzling and compare against the padded version from Day 5.
4. Write a filter's output with `__stcs` (written once, never re-read) and read the halo with the default operator. Measure.
5. Apply thread coarsening to one kernel: process 2, 4 then 8 elements per thread. Record runtime and occupancy at each step.
6. Express your best version as a percentage of theoretical peak bandwidth.

## Self-Check
No answers given.

1. Why does `__ldg` help only for data the kernel treats as read-only?
2. Why does `col ^ row` swizzling require a power-of-two row width?
3. When can L2 persistence hints make performance worse?
4. A kernel runs at 88 percent of peak bandwidth and a 2x speedup is requested. What is still available, and what is not?
5. Coarsening raises work per thread and lowers occupancy. Why is there no single correct coarsening factor?

## Code Template
See [`template.cu`](template.cu).
