// common/image_io.h
//
// OpenCV does three things in this course: read an image from disk, write the
// result back, and show it when a display is available. Nothing else.
//
// Device memory is managed with the CUDA API directly rather than through
// cv::cuda::GpuMat, for two reasons. Allocation, pitch and transfer are the
// subject of Days 4, 5 and 6, so they belong in the lab code where they can
// be read and changed. And the cv::cuda modules require OpenCV built with
// CUDA and opencv_contrib, which a distribution package does not give you;
// core, imgcodecs and highgui always do.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include "cuda_check.h"

// Reads an 8-bit single-channel image. Exits on failure: a lab that carries
// on with an empty cv::Mat fails later at an unrelated line, and the real
// cause (a wrong path) is then hard to see.
inline cv::Mat load_gray(const char *path)
{
    cv::Mat m = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (m.empty()) {
        fprintf(stderr, "cannot read image: %s\n", path);
        exit(EXIT_FAILURE);
    }
    printf("loaded %s  %dx%d  1 channel\n", path, m.cols, m.rows);
    return m;
}

// Reads an 8-bit 3-channel image. OpenCV's channel order is B, G, R.
inline cv::Mat load_bgr(const char *path)
{
    cv::Mat m = cv::imread(path, cv::IMREAD_COLOR);
    if (m.empty()) {
        fprintf(stderr, "cannot read image: %s\n", path);
        exit(EXIT_FAILURE);
    }
    printf("loaded %s  %dx%d  3 channels (BGR)\n", path, m.cols, m.rows);
    return m;
}

// Writes the image, and shows it only if there is a display. A lab run over
// SSH on the cluster has no DISPLAY, and cv::imshow there aborts the process;
// the file is written either way and can be copied back.
inline void save_and_show(const std::string &name, const cv::Mat &img)
{
    if (!cv::imwrite(name, img)) {
        fprintf(stderr, "cannot write %s\n", name.c_str());
        return;
    }
    printf("wrote %s\n", name.c_str());

    const char *display = getenv("DISPLAY");
    if (display != nullptr && *display != '\0') {
        cv::imshow(name, img);
        cv::waitKey(0);
    }
}

// A 2D device allocation from cudaMallocPitch. `pitch` is the byte stride
// between rows and is generally larger than width * sizeof(T), because the
// runtime aligns each row so that a warp's access to the start of a row is
// coalesced (Day 5: pitch, Day 6: coalescing). Index as
//
//     T *row = (T *)((char *)ptr + y * pitch);
//     row[x] = ...;
//
// never as ptr[y * width + x].
template <typename T>
struct device_image_t
{
    T     *ptr    = nullptr;
    size_t pitch  = 0;   // bytes per row on the device
    int    width  = 0;   // in elements, not bytes
    int    height = 0;

    device_image_t() = default;
    device_image_t(int w, int h) { allocate(w, h); }

    ~device_image_t()
    {
        // A destructor must not exit() on failure, so this is not CUDA_CHECK'd.
        if (ptr != nullptr) cudaFree(ptr);
    }

    device_image_t(const device_image_t &) = delete;
    device_image_t &operator=(const device_image_t &) = delete;

    void allocate(int w, int h)
    {
        width  = w;
        height = h;
        CUDA_CHECK(cudaMallocPitch(&ptr, &pitch, static_cast<size_t>(w) * sizeof(T), h));
    }

    void upload(const cv::Mat &m)
    {
        CUDA_CHECK(cudaMemcpy2D(ptr, pitch, m.ptr(), m.step,
                                static_cast<size_t>(width) * sizeof(T), height,
                                cudaMemcpyHostToDevice));
    }

    void download(cv::Mat &m) const
    {
        CUDA_CHECK(cudaMemcpy2D(m.ptr(), m.step, ptr, pitch,
                                static_cast<size_t>(width) * sizeof(T), height,
                                cudaMemcpyDeviceToHost));
    }

    // Useful bytes, for a bandwidth figure. Not pitch * height: the padding
    // is allocated but never read, so counting it overstates the traffic.
    size_t useful_bytes() const { return static_cast<size_t>(width) * height * sizeof(T); }
};

// Integer ceiling division, for grid sizing.
inline int div_up(int a, int b) { return (a + b - 1) / b; }
