// Load a BMP, add 1 to every pixel on the GPU, save the result as output.bmp,
// and check it on the host. 255 stays 255 rather than wrapping to 0.
//
//   ./compile.sh common/bmp_example.cu
//   sbatch submit.sh bmp_example data/camera.bmp

#include <cstdio>
#include "cuda_check.h"
#include "image_io.h"

__global__ void add_one(unsigned char *img, size_t pitch, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned char *row = img + y * pitch;
    row[x] = min(row[x] + 1, 255);
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "data/camera.bmp";

    image_t in = load_bmp(path);
    image_t out(in.width, in.height);

    device_image_t<unsigned char> d(in.width, in.height);
    d.upload(in);

    const dim3 block(16, 16);
    const dim3 grid(div_up(in.width, 16), div_up(in.height, 16));
    add_one<<<grid, block>>>(d.ptr, d.pitch, in.width, in.height);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    d.download(out);
    save_bmp("output.bmp", out);

    size_t wrong = 0;
    for (size_t i = 0; i < in.data.size(); ++i)
        if (out.data[i] != (in.data[i] == 255 ? 255 : in.data[i] + 1)) ++wrong;

    printf("%s: %zu of %zu pixels wrong\n", wrong ? "FAILED" : "OK", wrong, in.data.size());
    return wrong ? 1 : 0;
}
