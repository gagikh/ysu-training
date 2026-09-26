# Day 9: Libraries, Tensor Cores and Precision

## Objectives
- Replace hand-written kernels with cuBLAS, cuFFT, cuRAND and cuDNN, and say when a hand-written kernel is still justified
- Explain what a tensor core does and what it requires of data layout and dimensions
- Compare fp64, fp32, tf32, bf16 and fp16 throughput on the cluster GPU and state the consequences for numerical work
- Explain why atomic accumulation is not reproducible run to run

## Key Concepts
- cuBLAS, cuSOLVER, cuSPARSE, cuFFT, cuRAND; CUB and Thrust
- cuDNN, NPP and nvJPEG — what a deep learning framework actually calls
- Tensor cores: matrix multiply-accumulate as one instruction; alignment and dimension requirements
- Precision: fp64, fp32, tf32, bf16, fp16; fused multiply-add
- Non-determinism of atomic accumulation and what reproducibility claims require
- Cooperative groups: `tiled_partition`, group-typed shuffles, grid-wide sync

## Definitions
**Tensor Core** — Specialised SM hardware, compute capability 7.0 and newer, for fast mixed-precision matrix multiply-accumulate. Used by cuBLAS and cuDNN, and reachable directly through the warp matrix functions.

**MMA (matrix multiply-accumulate)** — The tensor core operation `D = A * B + C` on small matrix fragments, issued as one instruction per warp. Fragment shapes and alignment are fixed by the hardware, which is why dimensions have to be multiples of the fragment size to use it.

**fp64, fp32, tf32, bf16, fp16** — Floating-point formats, given as exponent and mantissa bits. fp64: 11 and 52. fp32: 8 and 23. tf32: 8 and 10, a tensor core input format only. bf16: 8 and 7, the same range as fp32 with less precision. fp16: 5 and 10, narrower range and precision.

**Mixed precision** — Computing in a narrow format while accumulating in a wider one, typically fp16 or bf16 inputs with fp32 accumulation. This is what tensor cores do natively.

**FMA (fused multiply-add)** — `a * b + c` computed with a single rounding instead of two. One reason a GPU result and a CPU result can differ in the last bits for identical inputs in identical order.

**Non-determinism of atomic accumulation** — Atomics do not fix the order in which values are combined, and floating-point addition is not associative, so a kernel accumulating floats with `atomicAdd` can give different results run to run. A reproducibility claim requires a fixed reduction order.

**Grid-wide synchronisation** — A barrier across every block of a grid, available through cooperative groups, and only for kernels launched with `cudaLaunchCooperativeKernel` and sized so that all blocks are resident at once.

**cuBLAS, cuFFT, cuRAND, cuDNN, NPP, nvJPEG, CUB, Thrust** — NVIDIA's libraries: dense linear algebra; fast Fourier transforms; random number generation; deep learning primitives; image and signal processing; JPEG decode and encode; block- and device-level parallel primitives; and an STL-like algorithms layer built on CUB.

## Precision on your hardware
On consumer and inference-class cards fp64 runs at 1/32 to 1/64 of the fp32 rate; on datacentre cards it is closer to 1/2. Determine which case applies to the cluster GPU from the Day 1 numbers before designing any double-precision work.

## Visual
![Bit layout of fp64, fp32, tf32, bf16 and fp16, each split into sign, exponent and mantissa, drawn to scale](precision_formats.svg)

The exponent field sets the range, the mantissa the precision. bf16 and tf32
keep fp32's 8-bit exponent, so a value that fits in fp32 fits in them; fp16's
5-bit exponent does not, which is why training in fp16 needs loss scaling and
training in bf16 does not. tf32 is not a storage format: it exists only as a
tensor core input, which is why cuBLAS turns it on through a math mode rather
than through a data type.

## Resources
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 7.24. Warp Matrix Functions · 8. Cooperative Groups
- Train With Mixed Precision: https://docs.nvidia.com/deeplearning/performance/
- Micikevicius P. et al. Mixed Precision Training. *ICLR*, 2018. arXiv:1710.03740
- cuBLAS, cuFFT, cuRAND: https://docs.nvidia.com/cuda/ · cuDNN: https://docs.nvidia.com/deeplearning/cudnn/

## Hands-On Task
Replace two hand-written kernels with library calls: the Day 5 and Day 6 box filter with NPP, and a matrix multiply with cuBLAS. The naive matrix multiply is given in the template, so this does not depend on Day 5's extension task. Then enable tensor cores and compare both speed and result.

## Self-Learning
1. Use cuBLAS for a matrix multiply and compare against your own kernel, in time and in result.
2. Enable tf32 and then bf16 for the same multiply. Record speed and the difference in result against the fp32 baseline.
3. Confirm from the profiler that tensor cores are actually being used, rather than assuming.
4. Measure fp64 against fp32 throughput on the cluster GPU with a compute-bound kernel. Compare with the ratio published for that card.
5. Sum the same large array with `atomicAdd` ten times. Record whether the results are bit-identical, and explain.
6. Use cuRAND for a Monte Carlo estimate of pi.

## Self-Check
No answers given.

1. Why do GPUs older than Volta have no tensor cores, and what does that mean for cuBLAS on them?
2. A student reports a 40x speedup from tensor cores with no accuracy loss. What would you ask to see?
3. Why is `cg::tiled_partition<32>` preferable to a raw `__shfl_down_sync` even though they compile to the same instruction?
4. Under what circumstances is writing your own kernel still the right answer when a library function exists?

## Code Template
See [`template.cu`](template.cu).
