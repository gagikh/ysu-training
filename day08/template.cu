// Day 8: Streams, Events, Asynchrony and CUDA Graphs
// Goal: process an image in row bands across several streams so that a band's
//       transfer overlaps another band's compute, then capture the whole
//       sequence into a graph and replay it.
//
// Build:  ./compile.sh day08/template.cu      (login node, repository root)
// Run:    sbatch submit.sh template <image.bmp>
//         under Nsight Systems, on a GPU node: nsys profile -o logs/day08 ./template <image.bmp>
//
// Seven TODOs. The kernel is given: today is about what surrounds it.

#include <cstdio>
#include <vector>
#include "../common/cuda_check.h"
#include "../common/timer.h"
#include "../common/image_io.h"

#define BLOCK 256
#define STREAMS 4

// Given. Deliberately cheap, so the transfer is the limit and the overlap is
// visible. Operates on a band of `rows` rows.
__global__ void invert_band(const unsigned char *in, unsigned char *out, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = 255 - in[i];
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: %s <image.bmp>\n", argv[0]);
        return 1;
    }

    image_t h_in_img = load_bmp(argv[1]);
    const int width = h_in_img.width, height = h_in_img.height;
    const size_t total = static_cast<size_t>(width) * height;

    // TODO 1: allocate pinned host buffers for the input and the output with
    // cudaMallocHost, and copy the image into the input buffer. Pageable
    // memory here makes cudaMemcpyAsync synchronous, and the whole lab then
    // shows no overlap at all -- which is worth seeing once, deliberately, in
    // TODO 7.
    unsigned char *h_in = nullptr, *h_out = nullptr;
    // TODO 1

    unsigned char *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, total));
    CUDA_CHECK(cudaMalloc(&d_out, total));

    // TODO 2: create STREAMS streams.
    std::vector<cudaStream_t> streams(STREAMS);
    // TODO 2

    // Bands of whole rows, so no band splits a row.
    const int rows_per_band = div_up(height, STREAMS);

    // ------------------------------------------------- the chunked pipeline
    kernel_timer_t t;
    for (int rep = 0; rep < 20; ++rep) {
        t.start();
        // TODO 3: for each band b, on stream b % STREAMS, issue in this order:
        //   cudaMemcpyAsync host -> device for that band's bytes,
        //   invert_band for that band's element count,
        //   cudaMemcpyAsync device -> host for that band.
        // Issue all of band 0's work, then all of band 1's, and so on. The
        // streams are independent, so band 1's copy-in overlaps band 0's
        // kernel without any explicit dependency.
        //
        // TODO 4: synchronise. cudaDeviceSynchronize() works but says nothing;
        // record an event in each stream instead and wait on those, and print
        // how far apart the four bands finished.
        t.stop();
    }
    t.report("pipelined, 4 streams");

    // ------------------------------------------------------- a single stream
    // TODO 5: the same work on the default stream only, as one copy-in, one
    // kernel, one copy-out. This is the figure the pipeline is compared
    // against. Predict the ratio before running: what is the ceiling when
    // transfer and compute overlap perfectly?

    // ---------------------------------------------------------------- graph
    // TODO 6: capture the pipeline into a graph and replay it.
    //   cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal);
    //   ... issue exactly what TODO 3 issues, with the other streams joined
    //       to streams[0] through events so the capture sees one graph ...
    //   cudaStreamEndCapture(streams[0], &graph);
    //   cudaGraphInstantiate(&exec, graph, 0);
    // then time cudaGraphLaunch(exec, streams[0]) in the same 20-iteration
    // loop. The work per iteration is identical, so any difference is the
    // launch overhead the graph removed.
    //
    // Expect little on four bands of a photograph. Then raise STREAMS to 32
    // and shrink the bands, and measure again: the graph wins when the launch
    // count per iteration is high and each launch is short.

    // TODO 7: rerun the pipeline with plain malloc instead of cudaMallocHost
    // and look at the Nsight Systems timeline. Name what changed on it, and
    // say why the copies no longer overlap.

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
