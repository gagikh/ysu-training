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
| [10](day10/README.md) | Discussion of proposals |

## Lessons and labs

| Day | Lesson | Lab |
|---|---|---|
| 1 | Programming model, SM, memory hierarchy | `report_device_capabilities()`, first kernel, record the GPU's own numbers |
| 2 | Thread, block, grid, indexing, occupancy | Vector addition, then 2D indexing over two images |
| 3 | SIMT pipeline, warp, divergence, latency hiding | Vector add timed against a CPU loop, then BGR to grayscale |
| 4 | Paged, pinned, mapped, unified memory; transfers | Pinned against pageable transfer, compared in Nsight Systems |
| 5 | Shared memory, banks, conflicts, tiling | Tiled 2D filter with a halo, then Sobel |
| 6 | Coalescing, sectors, L1/L2, coarsening | Optimise the Day 5 filter, report percentage of peak bandwidth |
| 7 | Warp shuffle, reduction, scan, atomics | Image mean by warp reduction; histogram, global against privatised |
| 8 | Streams, events, async copy, CUDA graphs | Chunked pipeline across streams, then captured as a graph |
| 9 | cuBLAS, cuDNN, tensor cores, precision | Replace hand-written kernels with library calls, enable tensor cores |
| 10 | Discussion | Participants present their proposals |

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

## Building and running

```
./compile.sh day01/template.cu
sbatch submit.sh template
sbatch submit.sh template image.png    # days 5-9
```

The output is in `logs/`. Target: H100 (`sm_90`), CUDA 12.8. Days 5 to 9 need OpenCV 4.

## Before the first session

- Cluster access, one CUDA version and one environment for everyone.
- OpenCV 4 on the login node: `pkg-config --modversion opencv4` prints a version.
- `./compile.sh day01/template.cu && sbatch submit.sh template` produces a log with sensible numbers.
- A test image on the cluster that every participant can read.

## Reference material

- [Glossary](GLOSSARY.md), [Intrinsics cheat sheet](INTRINSICS.md)
- [Architecture deep dive](ARCHITECTURE.md), [SM animations](sm_animations.html)
- [Performance checklist](PERFORMANCE.md)
- [100 practice tasks](TASKS.md)

## Bibliography

**Primary**

- NVIDIA. *CUDA C Programming Guide* (PDF) — https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
  The course's primary reading. Each day's `Resources` section names the exact chapters for that session.
- NVIDIA. *CUDA C++ Best Practices Guide* — https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ (v13.4, current)
- NVIDIA. *CUDA Programming Guide* (HTML, v13.4.2) — https://docs.nvidia.com/cuda/cuda-programming-guide/
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
