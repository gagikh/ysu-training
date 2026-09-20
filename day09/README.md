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

## Precision on your hardware
On consumer and inference-class cards fp64 runs at 1/32 to 1/64 of the fp32 rate; on datacentre cards it is closer to 1/2. Determine which case applies to the cluster GPU from the Day 1 numbers before designing any double-precision work.

## Resources
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 7.24. Warp Matrix Functions · 8. Cooperative Groups
- Train With Mixed Precision: https://docs.nvidia.com/deeplearning/performance/
- Micikevicius P. et al. Mixed Precision Training. *ICLR*, 2018. arXiv:1710.03740
- cuBLAS, cuFFT, cuRAND: https://docs.nvidia.com/cuda/ · cuDNN: https://docs.nvidia.com/deeplearning/cudnn/

## Hands-On Task
Replace your Day 5 filter and your matrix multiply with library calls, enable tensor cores, and compare both speed and result.

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
