# CUDA for University Lecturers

**Title**: GPU Architecture and CUDA C/C++ for Teaching Staff
**University**: Yerevan State University — Informatics and Applied Mathematics
**Format**: 20 academic hours, 10 sessions of 2 hours, weekly
**Participants**: 6 lecturers, experienced with OpenMP, MPI and threads, new to GPU
**Instructor**: Gagik Hakobyan

This course is the architecture-led counterpart to the [15-day student course](https://github.com/gagikh/cuda).
Participants already know parallel programming, so the programming model is covered
quickly and the emphasis falls on what the hardware does: streaming multiprocessors,
warp scheduling, the memory hierarchy, caches, and how each of them shows up in a
measurement.

---

## Goals

1. Write, profile and optimise GPU code.
2. Supervise coursework and master's theses in AI and computer vision: estimate whether
   a topic is feasible on the available hardware, check speedup claims, and decide when
   a hand-written kernel is justified.

## Outline

| Day | Topic |
|---|---|
| [1](day01/README.md) | CUDA programming model and GPU architecture |
| [2](day02/README.md) | Thread hierarchy, indexing and launch configuration |
| [3](day03/README.md) | SIMT execution, warps, divergence and latency hiding |
| [4](day04/README.md) | Memory types and host-device transfers |
| [5](day05/README.md) | Shared memory and bank conflicts |
| [6](day06/README.md) | Coalescing, caches and bandwidth |
| [7](day07/README.md) | Warp intrinsics, reduction and atomics |
| [8](day08/README.md) | Streams, events, asynchrony and CUDA graphs |
| [9](day09/README.md) | Libraries, tensor cores and precision |
| [10](day10/README.md) | Profiling method, feasibility and thesis topics |

Each day folder contains that day's `README.md` and a `template.cu` to start from.
Shared helpers are in [`common/`](common): `cuda_check.h`, `device_info.h`, `timer.h`.

Not covered: textures and surfaces, stream-ordered memory allocation, multiple GPUs,
multi-node MPI. The [student course](https://github.com/gagikh/cuda) covers the first two.

## The thread that runs through the course

Day 1 produces the concrete numbers for the cluster GPU — SM count, registers and shared
memory per SM, warp size, bus width, peak FP32 throughput, peak bandwidth. Every later
day measures against those numbers rather than against adjectives. By Day 6 a kernel is
expressed as a percentage of peak bandwidth; by Day 10 that same method is used to decide
whether a student's proposed project is possible at all.

## Before the first session

- Cluster access, one pinned CUDA version and one container image for everyone.
- `report_device_capabilities()` runs and prints sensible numbers.
- A smoke test: clone, `make run`, see output.

## Reference material

- [Glossary](https://github.com/gagikh/cuda/blob/main/GLOSSARY.md), [Intrinsics cheat sheet](https://github.com/gagikh/cuda/blob/main/INTRINSICS.md)
- [Architecture deep dive](https://github.com/gagikh/cuda/blob/main/ARCHITECTURE.md), [SM animations](https://github.com/gagikh/cuda/blob/main/sm_animations.html)
- [Performance checklist](https://github.com/gagikh/cuda/blob/main/PERFORMANCE.md)
- [100 practice tasks](https://github.com/gagikh/cuda/blob/main/TASKS.md)

## Bibliography

**Primary**

- NVIDIA. *CUDA Programming Guide* — https://docs.nvidia.com/cuda/cuda-programming-guide/ (v13.4.2, 10 Sep 2026). This replaces the *CUDA C++ Programming Guide*, which the document itself marks as no longer updated since CUDA 13.0.
- NVIDIA. *CUDA C++ Best Practices Guide* — https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ (v13.4, current)
- Hwu W., Kirk D., El Hajj I. *Programming Massively Parallel Processors*, 5th ed., Elsevier, 2026

**Profiling**

- NVIDIA. *Nsight Compute* — https://docs.nvidia.com/nsight-compute/ (v2026.3.1)
- NVIDIA. *Nsight Systems* — https://docs.nvidia.com/nsight-systems/ (v2026.4)
- Williams S., Waterman A., Patterson D. Roofline: An Insightful Visual Performance Model. *CACM* 52(4), 2009

**Libraries and precision**

- cuBLAS, cuSOLVER, cuSPARSE, cuFFT, cuRAND — https://docs.nvidia.com/cuda/
- cuDNN — https://docs.nvidia.com/deeplearning/cudnn/
- CUB — https://nvidia.github.io/cccl/cub/ · Thrust — https://nvidia.github.io/cccl/thrust/
- NVIDIA. *Train With Mixed Precision* — https://docs.nvidia.com/deeplearning/performance/
- Micikevicius P. et al. Mixed Precision Training. *ICLR*, 2018. arXiv:1710.03740

**Computer vision**

- Szeliski R. *Computer Vision: Algorithms and Applications*, 2nd ed. Free PDF: https://szeliski.org/Book/

**Teaching material**

- NVIDIA DLI Teaching Kit — Accelerated Computing. https://developer.nvidia.com/teaching-kits
- Oxford CUDA course, Mike Giles — https://people.maths.ox.ac.uk/~gilesm/cuda/
