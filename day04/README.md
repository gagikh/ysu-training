# Day 4: Memory Types and Host-Device Transfers

## Objectives
- Distinguish paged, pinned, page-locked, mapped and unified memory, and name the API for each
- Explain why a pageable transfer is slower than a pinned one for the same bytes
- Measure transfer bandwidth and compare it against the device's own memory bandwidth
- Use Nsight Systems to see the effect of the memory choice on a timeline

## Key Concepts
- Paged memory (`malloc`); pinned memory (`cudaMallocHost`); page locking in place (`cudaHostRegister`)
- Mapped / zero-copy memory (`cudaHostAlloc` with `cudaHostAllocMapped`)
- Unified memory (`cudaMallocManaged`, `__managed__`) and page migration
- DMA and why it requires pinned pages
- PCIe and NVLink bandwidth compared with device memory bandwidth

## Definitions
**Pageable memory** — Ordinary host memory from `malloc` or `new`. The operating system may move or swap its pages, so the GPU cannot access it directly: a transfer first copies it into a page-locked staging buffer held by the driver.

**Page-locked memory** — Host memory whose pages the operating system may not move or swap out.

**Pinned memory** — Page-locked host memory allocated with `cudaMallocHost`, so the GPU can transfer it by DMA with no staging copy. Required for `cudaMemcpyAsync` to be genuinely asynchronous.

**`cudaHostRegister`** — Page-locks memory that was allocated normally, giving it the transfer properties of pinned memory without reallocating it. `cudaHostUnregister` reverses this.

**Mapped (zero-copy) memory** — Page-locked host memory that also has a device address, from `cudaHostAlloc` with `cudaHostAllocMapped`. A kernel reads and writes it directly across the link with no explicit copy, paying link latency on every access.

**Unified memory** — One allocation, `cudaMallocManaged`, addressable from both host and device, with the driver migrating pages between them on demand.

**Page migration** — The movement of a unified-memory page to the processor that faulted on it. Repeated migration in both directions is the usual reason unified memory is slow; `cudaMemPrefetchAsync` and `cudaMemAdvise` control it.

**DMA (Direct Memory Access)** — A transfer carried out by a copy engine without the CPU moving the data. It requires the host pages to be page-locked, which is why pinned transfers are faster.

**PCIe** — The bus connecting host and device on most systems. Its bandwidth is an order of magnitude below device memory bandwidth.

**NVLink** — NVIDIA's direct GPU-to-GPU link, and on some systems CPU-to-GPU, with several times the bandwidth of PCIe.

**Memory bandwidth** — Bytes per second between the SMs and device memory. Theoretical peak is bus width times memory clock times transfers per clock; the achieved figure is what a kernel actually reaches.

## The number that matters
Device memory bandwidth is measured in hundreds of GB/s to several TB/s. PCIe is an order of magnitude below that. A kernel that transfers its input, computes once over it, and transfers the result back is limited by the link, not by the GPU. This is the first thing to check when a ported kernel disappoints, and it is the reason streams exist (Day 8).

## Visual
![Four host-to-device transfer paths: pageable with a staging copy, pinned with direct DMA, mapped where the GPU reads host memory directly, unified where the runtime migrates pages](memory_types.svg)

Each memory type is a different answer to one question: how does data get from host RAM to device VRAM. Pageable memory needs a hidden staging copy; pinned memory does not; mapped memory removes the copy entirely and pays per-access latency instead; unified memory leaves the decision to the runtime.

## Resources
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 3.2.2. Device Memory · 3.2.6. Page-Locked Host Memory · 19. Unified Memory Programming
- CUDA C++ Best Practices Guide — Memory Optimizations, Pinned Memory
- Nsight Systems — User Guide

## Hands-On Task
Improve the Day 2 and Day 3 vector addition using pinned memory, and compare the two in Nsight Systems.

## Self-Learning
1. Benchmark `cudaMemcpy` with pageable and with pinned host memory, for several transfer sizes. Plot bandwidth against size.
2. Page-lock an existing pageable buffer with `cudaHostRegister` instead of allocating pinned memory up front; compare.
3. Rewrite the vector addition with `cudaMallocManaged` and compare both code complexity and performance.
4. Profile all three variants in Nsight Systems and compare the transfer timelines.
5. Compute the arithmetic intensity of your vector addition and determine, from the numbers gathered on Day 1, whether the kernel or the transfer is the limit.

## Self-Check
No answers given.

1. Why is a pageable-to-device copy slower than a pinned one, given that the same bytes move?
2. What is the system-wide cost of pinning large amounts of host memory?
3. When does zero-copy memory beat copying to the device first?
4. Unified memory removes the explicit copy from the source. Does it remove the transfer?

## Code Template
See [`template.cu`](template.cu).
