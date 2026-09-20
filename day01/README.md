# Day 1: CUDA Programming Model and GPU Architecture

## Objectives
- Map the host/device split onto what you already know from MPI: two address spaces, explicit transfers, no coherence between them
- Describe the CUDA programming model: kernels, threads, blocks, grids
- Compile and run a `.cu` file with `nvcc`, and explain what nvcc produces (host object code, PTX, SASS)
- Read your own GPU's real numbers — SM count, registers per SM, shared memory per SM, warp size, memory bus width — and use them for the rest of the course
- Check every CUDA call, and explain why a kernel launch needs a different check than everything else

## Key Concepts
- Host vs device; why a GPU is a throughput machine and a CPU is a latency machine
- Kernels, threads, blocks, grids (overview only — detail on Day 2)
- Streaming multiprocessor: registers, ALUs, SFUs, tensor cores, warp schedulers, load/store units
- `nvcc`: host/device split, PTX vs SASS, virtual vs real architecture flags
- Error checking: `CUDA_CHECK`, `CUDA_CHECK_LAST_ERROR`
- `cudaGetDeviceProperties` and `report_device_capabilities()`

## Note for this audience
Participants already write OpenMP and MPI. Two contrasts are worth making explicitly and are the fastest way into the model:

- **Against OpenMP.** An OpenMP thread is scheduled by the OS and owns a full context. A CUDA thread is one lane of a warp; 32 of them share one instruction stream. Divergence is therefore a hardware cost, not a scheduling cost.
- **Against MPI.** Host and device memory are separate address spaces with explicit transfers, which is familiar. What is different is that the transfer cost is not a network cost but a PCIe or NVLink cost, and it is usually the first thing that limits a naive port.

## Resources
- CUDA Programming Guide — Programming Model, Hardware Implementation, Compute Capabilities
- CUDA C++ Best Practices Guide — Assess, Parallelize, Optimize, Deploy
- SM anatomy diagram and animations: [`sm_anatomy.svg`](https://github.com/gagikh/cuda/blob/main/sm_anatomy.svg), [`sm_animations.html`](https://github.com/gagikh/cuda/blob/main/sm_animations.html)
- [`ARCHITECTURE.md`](https://github.com/gagikh/cuda/blob/main/ARCHITECTURE.md) — what is inside an SM, in detail

## Hands-On Task
Run `report_device_capabilities()` on the cluster GPU and write down the numbers. Then write and launch a minimal kernel that reports its own block and thread index. The goal is a clean compile-and-run cycle plus a record of the hardware you will be measuring against all course.

## Self-Learning
1. Print `Hello from block X, thread Y` with device-side `printf`, launched as `<<<1,1>>>`, `<<<2,4>>>`, `<<<4,32>>>`.
2. Have each thread write its raw `blockIdx.x` and `threadIdx.x` into two arrays; copy back and verify against the launch configuration.
3. From `report_device_capabilities()` output, compute the theoretical peak FP32 throughput and peak memory bandwidth of the cluster GPU. Compare with NVIDIA's published figures and account for the difference.
4. Compile the same kernel with `-arch=sm_75` and with `-arch=native`; dump SASS for both with `cuobjdump --dump-sass` and diff the output.
5. Launch with 5000 threads per block once without `CUDA_CHECK_LAST_ERROR()` and once with it. Note what each run tells you.

## Self-Check
No answers given.

1. Why can a kernel launch not return a `cudaError_t` the way `cudaMalloc` does?
2. Why does nvcc emit PTX before machine code, and what does that buy you when the cluster is upgraded?
3. Your GPU reports N SMs and a peak bandwidth of B GB/s. Which of the two numbers will limit a vector addition, and how do you know before running it?
4. An OpenMP thread and a CUDA thread are both called "thread". Name two properties that do not carry over.

## Code Template
See [`template.cu`](template.cu).
