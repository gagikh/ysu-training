// Day 9: Libraries, Tensor Cores and Precision
// Goal: replace two hand-written kernels with library calls -- a matrix
//       multiply with cuBLAS, the Day 5 box filter with NPP -- then turn on
//       tensor cores and measure what precision costs.
//
// Build:  ./compile.sh day09/template.cu      (login node, repository root)
// Run:    sbatch submit.sh template <image.bmp> [matrix size, default 2048]
//
// The naive matrix multiply is given here, so this day does not depend on
// Day 5's extension task. Seven TODOs.

#include <cstdio>
#include <cmath>
#include <vector>
#include <cublas_v2.h>
#include <nppi_filtering_functions.h>
#include "../common/cuda_check.h"
#include "../common/timer.h"
#include "../common/image_io.h"

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t s__ = (call);                                          \
        if (s__ != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n  in call: %s\n",     \
                    __FILE__, __LINE__, static_cast<int>(s__), #call);        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define NPP_CHECK(call)                                                       \
    do {                                                                      \
        NppStatus s__ = (call);                                               \
        if (s__ != NPP_SUCCESS) {                                             \
            fprintf(stderr, "NPP error at %s:%d: %d\n  in call: %s\n",        \
                    __FILE__, __LINE__, static_cast<int>(s__), #call);        \
            exit(EXIT_FAILURE);                                              \
        }                                                                     \
    } while (0)

// Given: one output element per thread, every input read from global memory.
// Not the tiled version -- the point is the distance between a straightforward
// kernel and a library, not between two hand-written kernels.
__global__ void matmul_naive(const float *A, const float *B, float *C, int n)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;

    float acc = 0.0f;
    for (int k = 0; k < n; ++k) acc += A[row * n + k] * B[k * n + col];
    C[row * n + col] = acc;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: %s <image.bmp> [n]\n", argv[0]);
        return 1;
    }
    const int n = (argc > 2) ? atoi(argv[2]) : 2048;

    // ============================================================== matmul
    const size_t elems = static_cast<size_t>(n) * n;
    std::vector<float> h_A(elems), h_B(elems);
    for (size_t i = 0; i < elems; ++i) {
        h_A[i] = static_cast<float>(drand48() - 0.5);
        h_B[i] = static_cast<float>(drand48() - 0.5);
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_C_ref = nullptr;
    for (float **p : {&d_A, &d_B, &d_C, &d_C_ref})
        CUDA_CHECK(cudaMalloc(p, elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), elems * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid(div_up(n, 16), div_up(n, 16));
    const double flops = 2.0 * n * n * n;

    kernel_timer_t t;
    for (int i = 0; i < 5; ++i) {
        t.start();
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C_ref, n);
        CUDA_CHECK_LAST_ERROR();
        t.stop();
    }
    t.report_tflops("matmul, hand-written", flops);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // TODO 1: the same product with cublasSgemm. cuBLAS is column-major and
    // the arrays above are row-major, so computing B * A in cuBLAS's order
    // gives (A * B) in ours without transposing anything. Work out the
    // argument order rather than copying it: this is the single most common
    // mistake when a hand-written kernel is first replaced by cuBLAS.
    //
    // TODO 2: check the result against d_C_ref. It will not match bit for bit.
    // State the largest absolute difference and say which of the two is more
    // nearly correct, and why the answer is not "the reference".
    t.reset();
    // TODO 1

    // TODO 3: allow tf32 on the tensor cores with
    //   cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    // and time the same call again. Then compare the error against the fp32
    // result: tf32 keeps fp32's exponent and 10 mantissa bits, so report how
    // much accuracy the speedup cost, in the same units as TODO 2.
    //
    // TODO 4: on this GPU, what is fp64's rate relative to fp32? Repeat the
    // cuBLAS call with cublasDgemm on double inputs and compare. The ratio
    // decides whether a student's double-precision thesis project is feasible
    // here at all, which is the question Day 10 asks.

    CUBLAS_CHECK(cublasDestroy(handle));
    for (float *p : {d_A, d_B, d_C, d_C_ref}) CUDA_CHECK(cudaFree(p));

    // ============================================================ the filter
    image_t h_img = load_bmp(argv[1]);
    image_t h_out(h_img.width, h_img.height);

    device_image_t<unsigned char> d_img(h_img.width, h_img.height);
    device_image_t<unsigned char> d_filtered(h_img.width, h_img.height);
    d_img.upload(h_img);

    // TODO 5: the Day 5 and Day 6 box filter as one NPP call:
    //   nppiFilterBoxBorder_8u_C1R with a 3x3 mask, NPP_BORDER_REPLICATE.
    // NPP takes the pitch as an int line step and the size as an NppiSize, so
    // the parameters map directly onto what device_image_t already holds.
    //
    // TODO 6: compare it with your Day 6 kernel, in time and pixel for pixel.
    // If the outputs differ, find out whether the difference is at the border
    // or everywhere, and decide which behaviour you want.
    //
    // TODO 7: the honest conclusion. For this filter, does the library win?
    // State the rule you would give a student for when a hand-written kernel
    // is still worth writing.

    d_filtered.download(h_out);
    save_bmp("day09_npp.bmp", h_out);
    return 0;
}
