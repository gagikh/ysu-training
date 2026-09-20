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

## The number that matters
Device memory bandwidth is measured in hundreds of GB/s to several TB/s. PCIe is an order of magnitude below that. A kernel that transfers its input, computes once over it, and transfers the result back is limited by the link, not by the GPU. This is the first thing to check when a ported kernel disappoints, and it is the reason streams exist (Day 8).

## Resources
- CUDA Programming Guide — Device Memory, Unified Memory Programming
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
