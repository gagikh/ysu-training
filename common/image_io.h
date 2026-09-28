// common/image_io.h
//
// A grayscale image on the device, and the copies between it and an image_t
// from load_bmp (bmp_utilities.h).
//
// cudaMallocPitch pads each row, so `pitch` -- bytes per row on the device --
// is generally larger than width. Index rows by pitch, never by width:
//
//     const unsigned char *row = ptr + y * pitch;
#pragma once

#include <cuda_runtime.h>
#include "cuda_check.h"
#include "bmp_utilities.h"

template <typename T>
struct device_image_t
{
    static_assert(sizeof(T) == 1, "images are 8-bit grayscale");

    T     *ptr    = nullptr;
    size_t pitch  = 0;   // bytes per row on the device
    int    width  = 0;
    int    height = 0;

    device_image_t(int w, int h) : width(w), height(h)
    {
        CUDA_CHECK(cudaMallocPitch(&ptr, &pitch, size_t(w), h));
    }
    ~device_image_t() { cudaFree(ptr); }

    device_image_t(const device_image_t &) = delete;
    device_image_t &operator=(const device_image_t &) = delete;

    void upload(const image_t &img)
    {
        CUDA_CHECK(cudaMemcpy2D(ptr, pitch, img.data.data(), img.width,
                                width, height, cudaMemcpyHostToDevice));
    }

    void download(image_t &img) const
    {
        CUDA_CHECK(cudaMemcpy2D(img.data.data(), img.width, ptr, pitch,
                                width, height, cudaMemcpyDeviceToHost));
    }

    // Bytes of actual pixels, not counting the row padding.
    size_t useful_bytes() const { return size_t(width) * height; }
};

inline int div_up(int a, int b) { return (a + b - 1) / b; }
