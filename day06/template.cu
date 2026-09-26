// Day 6: Coalescing, Caches and Bandwidth
// Goal: take Day 5's tiled filter to a stated percentage of peak bandwidth.
//
// Build:  cmake -B build && cmake --build build -j
// Run:    ./build/day06 <image>
//
// Day 5's working tiled filter is given below, so the whole session is spent
// on the optimisations and on the measurement. Five TODOs, three of them one
// line each.

#include <cstdio>
#include "../common/cuda_check.h"
#include "../common/timer.h"
#include "../common/image_io.h"

#define TILE 16
#define RADIUS 1
#define HALO (TILE + 2 * RADIUS)

__device__ __forceinline__ const unsigned char *row_of(const unsigned char *base,
                                                       size_t pitch, int y)
{
    return base + static_cast<size_t>(y) * pitch;
}

// --------------------------------------------------------------- baseline
// Day 5's answer, unchanged. This is the figure the rest of the day is
// measured against.
__global__ void box_tiled(const unsigned char *in, size_t in_pitch,
                          unsigned char *out, size_t out_pitch,
                          int width, int height)
{
    __shared__ unsigned char tile[HALO][HALO];

    const int x0 = blockIdx.x * TILE - RADIUS;
    const int y0 = blockIdx.y * TILE - RADIUS;

    for (int i = threadIdx.y * TILE + threadIdx.x; i < HALO * HALO; i += TILE * TILE) {
        const int ty = i / HALO, tx = i % HALO;
        const int sx = min(max(x0 + tx, 0), width - 1);
        const int sy = min(max(y0 + ty, 0), height - 1);
        tile[ty][tx] = row_of(in, in_pitch, sy)[sx];
    }
    __syncthreads();

    const int x = blockIdx.x * TILE + threadIdx.x;
    const int y = blockIdx.y * TILE + threadIdx.y;
    if (x >= width || y >= height) return;

    int sum = 0;
    for (int dy = 0; dy <= 2 * RADIUS; ++dy)
        for (int dx = 0; dx <= 2 * RADIUS; ++dx)
            sum += tile[threadIdx.y + dy][threadIdx.x + dx];

    reinterpret_cast<unsigned char *>(out + static_cast<size_t>(y) * out_pitch)[x] =
        static_cast<unsigned char>(sum / 9);
}

// -------------------------------------------------------------------- ldg
// TODO 1: same kernel, but read the input through __ldg(). The pointer is
// already marked __restrict__, which is what lets the compiler prove the data
// is read-only for the kernel's lifetime. One line changes.
//
// Before running it, predict the result. The input is read exactly once per
// block here, so ask yourself what __ldg() can still help with.
__global__ void box_tiled_ldg(const unsigned char *__restrict__ in, size_t in_pitch,
                              unsigned char *__restrict__ out, size_t out_pitch,
                              int width, int height)
{
    __shared__ unsigned char tile[HALO][HALO];
    // TODO 1: copy box_tiled and replace the load with __ldg(...)
    (void)tile; (void)in; (void)in_pitch; (void)out; (void)out_pitch; (void)width; (void)height;
}

// -------------------------------------------------------------- coarsened
// TODO 2: give each thread COARSEN output pixels stacked vertically, so one
// loaded halo row serves several outputs and the per-thread index arithmetic
// is paid once. The block covers TILE x (TILE * COARSEN) output pixels, so the
// shared tile grows to (TILE * COARSEN + 2 * RADIUS) rows and the grid's y
// dimension shrinks by COARSEN.
//
// TODO 3: registers per thread go up and resident blocks go down. Report the
// occupancy from cudaOccupancyMaxActiveBlocksPerMultiprocessor for COARSEN =
// 1, 2, 4 and 8 alongside the runtime, and say where the two curves part.
#define COARSEN 4
__global__ void box_tiled_coarsened(const unsigned char *__restrict__ in, size_t in_pitch,
                                    unsigned char *__restrict__ out, size_t out_pitch,
                                    int width, int height)
{
    __shared__ unsigned char tile[TILE * COARSEN + 2 * RADIUS][HALO];
    // TODO 2
    (void)tile; (void)in; (void)in_pitch; (void)out; (void)out_pitch; (void)width; (void)height;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: %s <image>\n", argv[0]);
        return 1;
    }

    cv::Mat h_in = load_gray(argv[1]);
    cv::Mat h_out(h_in.size(), h_in.type());

    device_image_t<unsigned char> d_in(h_in.cols, h_in.rows);
    device_image_t<unsigned char> d_out(h_in.cols, h_in.rows);
    d_in.upload(h_in);

    const dim3 block(TILE, TILE);
    const dim3 grid(div_up(h_in.cols, TILE), div_up(h_in.rows, TILE));

    // One read and one write of every pixel. This is the useful traffic, not
    // the traffic the hardware actually moves -- TODO 5 is about the gap.
    const double bytes = 2.0 * d_in.useful_bytes();

    printf("peak bandwidth for this GPU: %.0f GB/s\n", kernel_timer_t::peak_gb_per_s());

    kernel_timer_t t;
    for (int i = 0; i < 50; ++i) {
        t.start();
        box_tiled<<<grid, block>>>(d_in.ptr, d_in.pitch, d_out.ptr, d_out.pitch,
                                   h_in.cols, h_in.rows);
        CUDA_CHECK_LAST_ERROR();
        t.stop();
    }
    t.report_bandwidth("Day 5 tiled", bytes);
    d_out.download(h_out);
    save_and_show("day06_baseline.png", h_out);

    t.reset();
    for (int i = 0; i < 50; ++i) {
        t.start();
        box_tiled_ldg<<<grid, block>>>(d_in.ptr, d_in.pitch, d_out.ptr, d_out.pitch,
                                       h_in.cols, h_in.rows);
        CUDA_CHECK_LAST_ERROR();
        t.stop();
    }
    t.report_bandwidth("+ __ldg", bytes);

    const dim3 grid_c(div_up(h_in.cols, TILE), div_up(h_in.rows, TILE * COARSEN));
    t.reset();
    for (int i = 0; i < 50; ++i) {
        t.start();
        box_tiled_coarsened<<<grid_c, block>>>(d_in.ptr, d_in.pitch, d_out.ptr, d_out.pitch,
                                               h_in.cols, h_in.rows);
        CUDA_CHECK_LAST_ERROR();
        t.stop();
    }
    t.report_bandwidth("+ coarsened", bytes);
    d_out.download(h_out);
    save_and_show("day06_optimised.png", h_out);

    // TODO 4: write down the three percentages. Decide, from the numbers and
    // not from taste, whether this kernel is worth optimising further.
    //
    // TODO 5: run the best version under
    //   ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./build/day06 <image>
    // and compare what the hardware moved with the useful bytes above. Account
    // for the difference: sector granularity, the halo read by two blocks, the
    // pitch padding.
    return 0;
}
