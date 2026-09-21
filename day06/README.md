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

## Definitions
**Cache line** — The 128-byte unit of L1 allocation.

**Sector** — The 32-byte unit in which memory is actually requested and moved. A warp's access is counted in sectors, and four sectors make a cache line. Coalescing is the minimisation of sectors per request.

**L1** — Per-SM cache, physically the same SRAM as shared memory, with a configurable split between the two.

**L2** — Chip-wide cache in front of device memory, shared by every SM. Atomics are executed here.

**Cache operator** — A per-instruction hint about how a load or store should use the caches: `__ldg` for the read-only path, `__ldcs` for a streaming load that is evicted first, `__stcs` for a streaming store, `__ldlu` for a last-use load whose line is then discarded.

**LRU** — Least recently used, the eviction order the caches approximate.

**Thread coarsening** — Giving each thread several output elements instead of one, so per-thread fixed costs — index arithmetic, bounds checks, shared-memory tile loads — are paid once and amortised. A grid-stride loop is the coalescing-safe way to write it. It trades against occupancy, so it has to be measured.

**Swizzling** — Scrambling a shared-memory index, for example `tile[row][col ^ row]`, so that a fixed logical column maps to a different physical bank on every row, removing bank conflicts without spending a padding column.

**Theoretical peak bandwidth** — Bus width times memory clock times transfers per clock, computed from the device numbers recorded on Day 1. The ceiling a kernel is measured against.

**Achieved bandwidth** — Bytes a kernel actually moves divided by its runtime, normally expressed as a percentage of theoretical peak.

**Arithmetic intensity** — FLOPs performed per byte of memory traffic. Decides which side of the roofline a kernel sits on: low intensity means memory-bound, which covers most kernels, high means compute-bound. Tiling raises it without changing the arithmetic.

**Compute-bound and memory-bound** — Whether a kernel's ceiling is instruction throughput or memory bandwidth. Determines which optimisations can possibly help.

**Roofline** — A plot of achievable performance against arithmetic intensity: a diagonal bandwidth ceiling that flattens into a horizontal compute ceiling. Which part of the roof a kernel sits under says whether memory or instruction optimisations can help it.

## Method
Measure achieved bandwidth, divide by the peak figure computed on Day 1, and place the kernel on a roofline. Below roughly half of peak there is usually something structural to fix. Above roughly 80 percent the remaining work is algorithmic — fewer bytes moved, not faster movement.

## Visual
![Memory hierarchy: registers at the top, then shared memory and L1, then L2, then global memory and VRAM at the bottom](cache_hierarchy.svg)

Every optimisation on this day is the same idea: keep frequently read data as high in this hierarchy as possible, for as long as possible. `__ldg`, shared-memory swizzling and L2 persistence hints are three tools for that one goal.

![XOR swizzling: in an 8x8 shared-memory tile, logical column 3 addressed without swizzling hits bank 3 on every row, while indexing with tile[row][col ^ row] spreads the same logical column across a different bank on each row, with no padding column](swizzling.svg)

Padding (Day 5) removes bank conflicts by spending a column so that the row stride is no longer a multiple of the bank count. Swizzling removes the same conflicts without spending memory: index as `tile[row][col ^ row]` instead of `tile[row][col]`. XOR is its own inverse, so writing and reading with the same formula stays correct, and each logical column is physically scattered across all banks instead of pinned to one.

## Animated
![Three requests travelling from the SM down to L1, L2 and global memory at different speeds, each lighting up the level it lands in](memory_traffic.svg)

Most traffic resolves close to the SM, some reaches L2, and a few requests pay the full global-memory round trip. Same hierarchy as the diagram above, now showing where requests actually land.

![Coalesced access completing in one transaction next to strided access needing 32 sequential transactions for the same 128 bytes](coalescing_strided.svg)

The measurement in the hands-on task is this picture in numbers.

A steppable version, where requests can be fired on demand, is in [`memory_animations.html`](memory_animations.html). Open it locally in a browser.

## Resources
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 3.2.3. Device Memory L2 Access Management · 5. Performance Guidelines
- CUDA C++ Best Practices Guide — Coalesced Access to Global Memory, L2 Cache
- Nsight Compute — Profiling Guide, memory chart and sectors per request
- Williams S., Waterman A., Patterson D. Roofline: An Insightful Visual Performance Model. *CACM* 52(4), 2009
- [`PERFORMANCE.md`](../PERFORMANCE.md) — which optimisation to try, in what order

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
