# 100 CUDA Practice Tasks

A problem bank grouped by the day whose material a task depends on. Nothing here has an answer key.

Task numbers are stable identifiers, not an order: they were assigned when the bank was first written and are kept so that a task can still be referred to by number. They therefore do not run in sequence down the page. Lettered entries (`10a`, `40c`, ...) are later additions kept in place for the same reason.

See [GLOSSARY.md](GLOSSARY.md) if a term is unfamiliar, [PERFORMANCE.md](PERFORMANCE.md) for the optimisation tasks, [INTRINSICS.md](INTRINSICS.md) for the device functions, and the relevant `dayNN/README.md` for background before attempting that day's tasks.

## Day 1 — CUDA programming model and GPU architecture
1. Print block/thread identity from the device using `printf`, across several different launch configurations.
2. Write raw `blockIdx`/`threadIdx` values from the device into host-verifiable arrays.
3. Time a 1-thread launch vs. a many-thread launch with `<chrono>` and explain the difference.
4. Compile with `--keep` and read the generated `.ptx` file.
5. Run `report_device_capabilities()` and write down your GPU's warp size, max threads/block, and shared memory per SM.

## Day 2 — Thread hierarchy, indexing and launch configuration
6. Implement 1D vector addition for a few different array sizes.
7. Extend to 2D thread indexing and add two grayscale images pixel-by-pixel.
8. Compare timing across block sizes 32, 64, 128, and 256.
9. Make a kernel correct for array sizes that aren't an exact multiple of the block size.
10. Implement a grid-stride loop and verify it's still correct at 10x the original `n`, with the same launch configuration.
10a. Write a copy kernel two ways — indexed `blockIdx.x * blockDim.x + threadIdx.x` vs. `threadIdx.x * gridDim.x + blockIdx.x` — and measure the ratio. Both are correct; predict the gap before running.
10b. Swap the row/column roles of `threadIdx.x` and `threadIdx.y` in a 2D image kernel and measure the slowdown.
10c. Call `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for block sizes 32/64/128/256 and check where measured timing stops tracking occupancy.

## Day 3 — SIMT execution, warps, divergence and latency hiding
11. Implement large vector addition and time it against an equivalent CPU loop.
12. Convert a BGR image to grayscale in a kernel.
13. Deliberately introduce branch divergence and measure the performance hit.
14. Apply `#pragma unroll` to a fixed-trip-count loop and compare generated performance.
15. Sketch the fetch/decode/register-read/execute/memory/writeback pipeline for 4 instructions across 6 cycles, on paper.

## Day 4 — Memory types and host-device transfers
16. Benchmark `cudaMemcpy` with pageable vs. pinned host memory for a large transfer.
17. Page-lock an existing pageable buffer with `cudaHostRegister` instead of allocating pinned memory up front.
18. Rewrite vector-add to use `cudaMallocManaged` (unified memory).
19. Profile pageable/pinned/unified variants with Nsight Systems and compare the timelines.
20. Try mapped (zero-copy) memory and compare its transfer behavior to pinned.

## Day 5 — Shared memory and bank conflicts
25. Load a real image with `load_bmp`, write it back unchanged with `save_bmp`, and open the result, before writing any kernel logic.
21. Implement a shared-memory tile-based 2D box blur.
22. Deliberately create a bank-conflicting access pattern, measure the hit, then fix it with padding.
23. Implement a 2D Sobel filter using shared memory.
27. Reuse the tiling approach for a shared-memory convolution with a larger kernel than 3x3.
56. Implement matrix transpose using shared memory, padded to avoid bank conflicts.
34. Implement a small (32x32) matrix multiplication kernel.
46. Implement naive GPU matrix multiplication.
47. Optimize it with shared-memory tiling and compare timing against the naive version.

## Day 6 — Coalescing, caches and bandwidth
61. Add `__ldg()` to a read-heavy kernel from an earlier day and measure the effect.
62. Implement `col ^ row` swizzling to remove bank conflicts without a padding column.
63. Experiment with L2 persistence hints (`cudaAccessPolicyWindow`) on a repeatedly-read buffer.
64. Optimise the image transform kernel (task 28) using every technique from the course so far.
65. Benchmark `__ldg`, swizzling, and padding on the same kernel and rank them for your GPU.
65a. Add achieved-bandwidth and %-of-peak output to all three timings in `day06/template.cu`, and decide from the numbers whether the day’s optimisations ever had room to help.
65b. Coarsen the tiled box filter to 2, 4 and 8 outputs per thread; plot time against elements-per-thread and correlate the drop-off with register spilling from `-Xptxas -v`.
65c. Sweep the grid size of the Day 2 grid-stride vector add (64 → 4096 blocks) at fixed `n` and explain the curve as coarsening at one end, occupancy at the other.

## Day 7 — Warp intrinsics, reduction and atomics
36. Implement warp-level sum reduction using `__shfl_down_sync`.
40a. Extend the warp reduction to a full `block_reduce_sum` (warp → shared → warp), then reduce a whole grid via one `atomicAdd` per block. Count the `__syncthreads()` calls against a classic tree reduction.
40b. Rewrite the reduction with `__shfl_xor_sync` so every lane holds the total; confirm the cost is unchanged and say when you'd want it.
37. Implement an inclusive prefix sum (scan) within a single warp.
38. Use the scan result to compact indices of pixels above a threshold.
40c. Redo the compaction with `__ballot_sync` + `__popc` and one warp-aggregated atomic. Time it against the scan version at 5% and 95% pass rates.
39. Implement a 32-point FFT butterfly using warp shuffles.
40. Compare warp-shuffle reduction against a shared-memory reduction for the same problem size.
40d. Write the divergence bug deliberately — `warp_reduce_sum` inside `if (id < n)` with `n` not a multiple of 32 — and run it under `compute-sanitizer --tool synccheck`.
41. Compute an image's mean pixel value using warp reduction + `atomicAdd`.
42. Pack 32 binary pixel values into one 32-bit word using `__ballot_sync`.
43. Write the inverse "unzip" operation.
32. Compute vector mean and standard deviation on the GPU, compare against a CPU implementation.
45a. Build a 256-bin histogram twice — naive global `atomicAdd` per pixel vs. privatized into shared memory — and measure the ratio on a normal image and on a near-uniform one.
45b. Replace the shared-memory `atomicAdd` in the privatized histogram with `atomicAdd_block` and measure the difference.

## Day 8 — Streams, events, asynchrony and CUDA graphs
29. Time kernels precisely with `cudaEvent`s and compare against `<chrono>` measurements.
30. Split independent work across two CUDA streams and check whether they overlap.
26. Implement an image derivative (gradient) kernel — compute dx/dy per pixel.
28. Implement a simple image transform (rotate or scale) kernel.
31. Overlap an async H2D copy with kernel execution using two streams and `cudaMemcpyAsync`.
35. Chunk a real image into horizontal bands and pipeline copy-in/compute/copy-out across streams.
58. Capture a multi-kernel pipeline into a CUDA graph via stream capture.
59. Launch the captured graph 1000 times and compare total time against 1000 direct launches.
60. Add a memory operation (not just a kernel) into the same captured graph.

## Day 9 — Libraries, tensor cores and precision
66. Estimate π via Monte Carlo sampling with cuRAND.
67. Use cuBLAS for a matrix-vector multiply and compare against your Day 5 kernel.
68. Use cuFFT to compute an FFT and compare against your Day 7 32-point attempt.
69. Fill a device buffer with cuRAND-generated noise, download it and write it out with `save_bmp`.
98. Implement one kernel in half precision (FP16) and compare accuracy and speed against FP32.

## Day 10 — Discussion of proposals
100. Write a one-page performance report for any kernel from this course: measured throughput, theoretical peak from `report_device_capabilities()`, and the percentage of peak achieved.

## Bonus: image processing
Tasks that need only what the course covers, applied to problems it does not work through.

24. Extend the Sobel filter to process a video stream frame by frame.
33. Implement image dilation and/or erosion filters.
44. Implement `pyrDown` (blur + downsample by 2).
45. Implement `pyrUp` (upsample by 2 + blur).
48. Implement Hamming distance between binary descriptors using `__popc`.
49. Batch-match a query descriptor set against a reference set, finding each nearest neighbor.
50. Extract real ORB descriptors from an image and self-match them as a correctness sanity check.
77. Implement histogram equalization on the GPU.
78. Implement a Canny edge detector from scratch (gradient → non-max suppression → hysteresis).
79. Implement bilateral filtering (edge-preserving blur).
80. Implement a median filter using a small sorting network in shared memory.
81. Implement image thresholding — both fixed and adaptive — as a kernel.
82. Implement a simple optical flow estimator (Lucas-Kanade, small window).
83. Implement alpha blending of two images on the GPU.
84. Implement an RGB-to-HSV color-space conversion kernel.
86. Implement non-maximum suppression for corner detection.
87. Build a real-time webcam filter pipeline: `cv::VideoCapture` → GPU kernel → `cv::imshow`.
88. Implement a full Laplacian pyramid blend of two images.
89. Implement a separable box filter (horizontal pass, then vertical) and compare it to a single 2D tiled pass.
90. Implement template matching (normalized cross-correlation) on the GPU.
76. Implement a Gaussian blur kernel and compare it against NPP’s `nppiFilterGauss`.
85. Implement a perspective warp (homography) kernel using textures.

## Beyond this course
Topics this course does not teach: texture and surface objects, stream-ordered allocation, cooperative groups, multi-GPU, dynamic parallelism. Listed because a thesis project may need them.

51. Build a CUDA texture object bound to a real image.
52. Implement image zoom (upscale) using `tex2D` bilinear filtering.
53. Implement image rotation via inverse-mapped texture sampling.
54. Compare texture-based zoom against a manual shared-memory bilinear implementation.
55. Explain, in your own words, why `cudaAddressModeClamp` changes the result specifically at image borders.
57. Implement the same transpose using texture binding and compare performance.
70. Try a recursive/dynamic-parallelism kernel launch — have a kernel launch a child kernel.
71. Replace a `cudaMalloc`/`cudaFree` pair with `cudaMallocAsync`/`cudaFreeAsync` on a stream.
72. Benchmark allocation overhead: classic vs. stream-ordered, over many small allocations.
73. Create an explicit `cudaMemPool_t` and drive allocations on it from two different streams.
74. Combine stream-ordered allocation with a Day 8 CUDA graph capture.
75. Apply `cudaMallocAsync` to a real image-processing kernel end to end.
91. Implement a grid-wide reduction using cooperative groups (no host round-trip between blocks).
92. Split a vector-add workload across two GPUs with `cudaSetDevice(0)`/`cudaSetDevice(1)`.
93. Enable peer-to-peer memory access between two GPUs with `cudaDeviceEnablePeerAccess`, if you have more than one.
94. Profile a kernel with Nsight Compute and determine whether it's compute-bound or memory-bound.
95. Implement dynamic parallelism: a kernel that launches a child kernel based on data computed at runtime.
96. Implement a persistent-kernel pattern — a kernel that loops internally pulling work from a queue, instead of being relaunched per item.
97. Port one of your Day 5-7 kernels to cooperative groups’ tiled partitions instead of raw warp intrinsics.
99. Build a small CUDA unit-test harness that compares kernel output against a CPU reference for randomized inputs.

