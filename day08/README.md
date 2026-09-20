# Day 8: Streams, Events, Asynchrony and CUDA Graphs

## Objectives
- Distinguish synchronous and asynchronous `cudaMemcpy` and explain when the async form silently becomes synchronous
- Overlap transfer with computation across streams, and confirm the overlap with events and Nsight Systems
- Time on the device with events rather than on the host
- Capture a sequence of operations into a CUDA graph, and say when graph launch pays off

## Key Concepts
- Default stream semantics and why it serialises independent work
- `cudaMemcpyAsync`, stream dependencies, and the pinned-memory requirement
- Events: `cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`
- Chunked pipelines: copy, compute and copy-back in flight at once
- Graph capture, instantiation and launch; launch overhead amortisation

## Where this connects to Day 4
Day 4 established that the link is often the limit. Streams are the answer: while chunk *n* is computed, chunk *n+1* is being copied in and chunk *n-1* copied out. The ceiling is then the larger of the two rates, not their sum.

## Resources
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 3.2.8. Asynchronous Concurrent Execution · 3.2.8.7. CUDA Graphs
- CUDA C++ Best Practices Guide — Asynchronous Transfers and Overlapping Transfers with Computation
- Nsight Systems — User Guide, timeline view

## Hands-On Task
Process an image in row chunks across several streams, overlapping transfer and computation. Then capture the sequence into a graph and replay it.

## Self-Learning
1. Overlap an async host-to-device copy with kernel execution using two streams; confirm the overlap with events and in the Nsight Systems timeline.
2. Time a kernel with events and compare with your earlier host-side `<chrono>` measurement. Explain the difference.
3. Increase the number of streams from 2 to 4 to 8 and record where the overlap stops improving.
4. Capture the chunked pipeline into a CUDA graph. Launch it 1000 times and compare against 1000 sequential launches.
5. Remove the pinned allocation from the async path and measure what happens.

## Self-Check
No answers given.

1. Why does `cudaMemcpyAsync` behave synchronously if the host buffer is not pinned?
2. What goes wrong if you call `cudaEventElapsedTime` without `cudaEventSynchronize` first?
3. Why does adding more streams eventually stop helping?
4. Why does capturing kernels into a graph not make the GPU compute anything faster?
5. During `cudaStreamBeginCapture`, does the launched kernel execute?

## Code Template
See [`template.cu`](template.cu).
