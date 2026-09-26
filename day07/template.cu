// Day 7: Warp Intrinsics, Reduction and Atomics
// Goal: (a) the mean of an image by warp reduction plus one atomic per block,
//       (b) a 256-bin histogram, twice: global atomics, then privatised.
//
// Build:  cmake -B build && cmake --build build -j
// Run:    ./build/day07 <image>
//
// Scan, stream compaction and warp-aggregated atomics are in Self-Learning,
// not here. Eight TODOs.

#include <cstdio>
#include "../common/cuda_check.h"
#include "../common/timer.h"
#include "../common/image_io.h"

#define BLOCK 256
#define WARP 32
#define BINS 256

// ============================================================ part a: mean
// TODO 1: sum the 32 values held by one warp, leaving the total in lane 0.
// Five __shfl_down_sync steps, offset halving from 16. The mask is
// 0xffffffff: every lane takes part.
__device__ __forceinline__ int warp_sum(int v)
{
    // TODO 1
    return v;
}

// TODO 2: extend that to the block with the warp-shared-warp pattern. Each
// warp's lane 0 writes its total into a __shared__ array of BLOCK / WARP
// entries; after a __syncthreads(), the first warp reduces those entries with
// warp_sum again.
//
// TODO 3: one atomicAdd per block on the global total, from thread 0 only.
// Note what this costs: one atomic per block rather than one per pixel, which
// is privatisation at block scope before the term is used for histograms.
__global__ void image_sum(const unsigned char *in, size_t pitch,
                          int width, int height, unsigned long long *total)
{
    __shared__ int warp_totals[BLOCK / WARP];

    const int x = blockIdx.x * BLOCK + threadIdx.x;
    const int y = blockIdx.y;

    int v = 0;
    if (x < width && y < height)
        v = (in + static_cast<size_t>(y) * pitch)[x];

    // TODO 2: warp_sum, then across warps through warp_totals
    // TODO 3: thread 0 does one atomicAdd into *total
    (void)warp_totals; (void)v; (void)total;
}

// ======================================================= part b: histogram
// Given: the obvious version. Every thread does one global atomic, and every
// thread whose pixel has the same value collides with every other. On a
// photograph the collisions are severe, because real images are far from
// uniform.
__global__ void histogram_global(const unsigned char *in, size_t pitch,
                                 int width, int height, unsigned int *hist)
{
    const int x = blockIdx.x * BLOCK + threadIdx.x;
    const int y = blockIdx.y;
    if (x >= width || y >= height) return;

    const unsigned char v = (in + static_cast<size_t>(y) * pitch)[x];
    atomicAdd(&hist[v], 1u);
}

// TODO 4: the same histogram, privatised. Each block keeps a BINS-entry copy
// in shared memory.
//   - clear it cooperatively, then __syncthreads();
//   - accumulate into the shared copy with atomicAdd on shared memory;
//   - __syncthreads();
//   - merge the shared copy into the global one, one atomicAdd per bin per
//     block instead of one per pixel.
//
// TODO 5: shared-memory atomics still serialise on a contended bin. Say which
// part of the cost privatisation removes and which part it does not.
__global__ void histogram_private(const unsigned char *in, size_t pitch,
                                  int width, int height, unsigned int *hist)
{
    __shared__ unsigned int local[BINS];
    // TODO 4
    (void)local; (void)in; (void)pitch; (void)width; (void)height; (void)hist;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: %s <image>\n", argv[0]);
        return 1;
    }

    cv::Mat h_in = load_gray(argv[1]);
    const int width = h_in.cols, height = h_in.rows;
    const long long pixels = static_cast<long long>(width) * height;

    device_image_t<unsigned char> d_in(width, height);
    d_in.upload(h_in);

    unsigned long long *d_total = nullptr;
    unsigned int *d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_hist, BINS * sizeof(unsigned int)));

    const dim3 block(BLOCK);
    const dim3 grid(div_up(width, BLOCK), height);

    // ---- mean -----------------------------------------------------------
    CUDA_CHECK(cudaMemset(d_total, 0, sizeof(unsigned long long)));
    image_sum<<<grid, block>>>(d_in.ptr, d_in.pitch, width, height, d_total);
    CUDA_CHECK_LAST_ERROR();

    unsigned long long h_total = 0;
    CUDA_CHECK(cudaMemcpy(&h_total, d_total, sizeof(h_total), cudaMemcpyDeviceToHost));
    printf("GPU mean  %.6f\n", static_cast<double>(h_total) / pixels);

    // TODO 6: check it. cv::mean(h_in)[0] is the reference. They should agree
    // exactly here, because the accumulation is in integers. Repeat the
    // exercise with a float accumulator and explain why that one does not.
    printf("CPU mean  %.6f\n", cv::mean(h_in)[0]);

    // ---- histogram ------------------------------------------------------
    const double bytes = static_cast<double>(d_in.useful_bytes());   // read only

    kernel_timer_t t;
    for (int i = 0; i < 20; ++i) {
        CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));
        t.start();
        histogram_global<<<grid, block>>>(d_in.ptr, d_in.pitch, width, height, d_hist);
        CUDA_CHECK_LAST_ERROR();
        t.stop();
    }
    t.report_bandwidth("histogram, global", bytes);

    unsigned int h_ref[BINS];
    CUDA_CHECK(cudaMemcpy(h_ref, d_hist, sizeof(h_ref), cudaMemcpyDeviceToHost));

    t.reset();
    for (int i = 0; i < 20; ++i) {
        CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));
        t.start();
        histogram_private<<<grid, block>>>(d_in.ptr, d_in.pitch, width, height, d_hist);
        CUDA_CHECK_LAST_ERROR();
        t.stop();
    }
    t.report_bandwidth("histogram, privatised", bytes);

    unsigned int h_got[BINS];
    CUDA_CHECK(cudaMemcpy(h_got, d_hist, sizeof(h_got), cudaMemcpyDeviceToHost));

    // TODO 7: the two histograms must be identical, bin for bin. Compare them
    // and report the first mismatch rather than only a pass or fail.
    long long checked = 0;
    for (int b = 0; b < BINS; ++b) checked += h_got[b];
    printf("bins sum to %lld, pixels %lld  %s\n", checked, pixels,
           checked == pixels ? "ok" : "MISMATCH");

    // TODO 8: the speedup depends on the image. Run both versions on a
    // photograph and on a synthetic image where every pixel is the same value,
    // and explain the difference from the contention argument, not from the
    // code.

    CUDA_CHECK(cudaFree(d_total));
    CUDA_CHECK(cudaFree(d_hist));
    return 0;
}
