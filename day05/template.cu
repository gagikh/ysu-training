// Day 5: Shared Memory and Bank Conflicts
// Goal: a tiled 3x3 box filter with a halo, then a Sobel magnitude, both
//       reading the image once into shared memory per block.
//
// Build:  bash compile.sh day05        (login node, repository root)
// Run:    sbatch submit.sh build/day05 <image>
//
// Time: five TODOs. TODO 1 and 2 are the lab; 3 is the same structure again
// and should be quick once 1 and 2 work; 4 and 5 are measurements, not code.

#include <cstdio>
#include "../common/cuda_check.h"
#include "../common/timer.h"
#include "../common/image_io.h"
#include "../common/device_info.h"

#define TILE 16
#define RADIUS 1
#define HALO (TILE + 2 * RADIUS)

// ---------------------------------------------------------------- baseline
// Reads every neighbour straight from global memory: nine loads per output
// pixel, eight of which a neighbouring thread also performs. Given as the
// figure to beat.
__global__ void box_filter_global(const unsigned char *in, size_t in_pitch,
                                  unsigned char *out, size_t out_pitch,
                                  int width, int height)
{
    const int x = blockIdx.x * TILE + threadIdx.x;
    const int y = blockIdx.y * TILE + threadIdx.y;
    if (x >= width || y >= height) return;

    int sum = 0;
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            const int sx = min(max(x + dx, 0), width - 1);   // clamp at the border
            const int sy = min(max(y + dy, 0), height - 1);
            sum += ((const unsigned char *)(in + sy * in_pitch))[sx];
        }
    }
    ((unsigned char *)(out + y * out_pitch))[x] = (unsigned char)(sum / 9);
}

// ------------------------------------------------------------------ tiled
// TODO 1: load a TILE x TILE region plus a one-pixel halo on every side into
// `tile`, so the block reads HALO x HALO pixels once instead of 9 * TILE^2
// times. The awkward part is that there are HALO^2 = 324 elements to load and
// only TILE^2 = 256 threads, so some threads load two. A flat loop over the
// shared tile, strided by the block size, handles it without special cases:
//
//     for (int i = threadIdx.y * TILE + threadIdx.x; i < HALO * HALO; i += TILE * TILE) {
//         const int ty = i / HALO, tx = i % HALO;
//         ... global coordinates are blockIdx * TILE - RADIUS + (tx, ty),
//             clamped to the image as in the baseline above ...
//     }
//
// TODO 2: put a __syncthreads() between the load and the compute, then average
// the nine neighbours out of `tile`. Note which indices shift by RADIUS.
__global__ void box_filter_tiled(const unsigned char *in, size_t in_pitch,
                                 unsigned char *out, size_t out_pitch,
                                 int width, int height)
{
    __shared__ unsigned char tile[HALO][HALO];

    // TODO 1: cooperative load into tile[][]
    // TODO 2: __syncthreads(), then the average from shared memory
    (void)tile; (void)in; (void)in_pitch; (void)out; (void)out_pitch; (void)width; (void)height;
}

// ------------------------------------------------------------------ sobel
// TODO 3: the same tile, different arithmetic. Gx and Gy are the two 3x3
// Sobel kernels; the output is min(255, sqrtf(gx*gx + gy*gy)). Copy the load
// from box_filter_tiled once it works -- the point of this one is that the
// shared tile is reused for a different operator, not that loading is hard.
__global__ void sobel_tiled(const unsigned char *in, size_t in_pitch,
                            unsigned char *out, size_t out_pitch,
                            int width, int height)
{
    __shared__ unsigned char tile[HALO][HALO];
    // TODO 3
    (void)tile; (void)in; (void)in_pitch; (void)out; (void)out_pitch; (void)width; (void)height;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: %s <image>\n", argv[0]);
        return 1;
    }

    printf("%s", report_device_capabilities().c_str());

    cv::Mat h_in = load_gray(argv[1]);
    cv::Mat h_out(h_in.size(), h_in.type());

    device_image_t<unsigned char> d_in(h_in.cols, h_in.rows);
    device_image_t<unsigned char> d_out(h_in.cols, h_in.rows);
    d_in.upload(h_in);

    // cudaMallocPitch padded each row. Worth printing once: this is the
    // difference between pitch and width * sizeof(T) that the README defines.
    printf("width %d bytes, pitch %zu bytes\n", h_in.cols, d_in.pitch);

    const dim3 block(TILE, TILE);
    const dim3 grid(div_up(h_in.cols, TILE), div_up(h_in.rows, TILE));

    // Read + write per launch. Both kernels touch every pixel twice in total,
    // so the same figure applies to all three and they are comparable.
    const double bytes = 2.0 * d_in.useful_bytes();

    kernel_timer_t t;
    for (int i = 0; i < 20; ++i) {
        t.start();
        box_filter_global<<<grid, block>>>(d_in.ptr, d_in.pitch, d_out.ptr, d_out.pitch,
                                           h_in.cols, h_in.rows);
        CUDA_CHECK_LAST_ERROR();
        t.stop();
    }
    t.report_bandwidth("box, global memory", bytes);

    t.reset();
    for (int i = 0; i < 20; ++i) {
        t.start();
        box_filter_tiled<<<grid, block>>>(d_in.ptr, d_in.pitch, d_out.ptr, d_out.pitch,
                                          h_in.cols, h_in.rows);
        CUDA_CHECK_LAST_ERROR();
        t.stop();
    }
    t.report_bandwidth("box, shared tile", bytes);
    d_out.download(h_out);
    save_and_show("day05_box.png", h_out);

    t.reset();
    for (int i = 0; i < 20; ++i) {
        t.start();
        sobel_tiled<<<grid, block>>>(d_in.ptr, d_in.pitch, d_out.ptr, d_out.pitch,
                                     h_in.cols, h_in.rows);
        CUDA_CHECK_LAST_ERROR();
        t.stop();
    }
    t.report_bandwidth("sobel, shared tile", bytes);
    d_out.download(h_out);
    save_and_show("day05_sobel.png", h_out);

    // TODO 4: the tiled version should beat the global one. By how much, and
    // is that the factor you predicted from the number of avoided loads?
    //
    // TODO 5: `tile[HALO][HALO]` is 18 bytes per row. Change it to
    // `tile[HALO][HALO + 14]` so a row is 32 bytes, run again, and explain the
    // result from the bank rule in the README. Then predict what
    // `tile[HALO][HALO + 1]` does before running it.
    return 0;
}
