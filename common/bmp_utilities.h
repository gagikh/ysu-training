// common/bmp_utilities.h
//
// Load and save grayscale images as BMP files, with no external library.
//
//   image_t img = load_bmp("input.bmp");
//   save_bmp("output.bmp", img);
//
// load_bmp reads uncompressed 8-bit and 24-bit BMPs and converts them to
// grayscale. save_bmp writes an 8-bit grayscale BMP.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

// A grayscale image on the host: width * height bytes, one per pixel,
// row by row, top row first.
struct image_t
{
    int width  = 0;
    int height = 0;
    std::vector<unsigned char> data;

    image_t() = default;
    image_t(int w, int h) : width(w), height(h), data(size_t(w) * h) {}
};

inline image_t load_bmp(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(EXIT_FAILURE);
    }

    std::vector<unsigned char> file;
    unsigned char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) file.insert(file.end(), buf, buf + n);
    fclose(f);

    auto u16 = [&](size_t o) { uint16_t v; memcpy(&v, &file[o], 2); return v; };
    auto u32 = [&](size_t o) { uint32_t v; memcpy(&v, &file[o], 4); return v; };

    if (file.size() < 54 || file[0] != 'B' || file[1] != 'M') {
        fprintf(stderr, "%s is not a BMP file\n", path);
        exit(EXIT_FAILURE);
    }

    const uint32_t offset      = u32(10);           // start of the pixel rows
    const uint32_t header_size = u32(14);
    const int32_t  width       = int32_t(u32(18));
    const int32_t  height_raw  = int32_t(u32(22));  // positive: bottom row first
    const uint16_t bpp         = u16(28);
    const uint32_t compression = u32(30);
    const uint32_t colours     = u32(46);

    if (compression != 0 || (bpp != 8 && bpp != 24)) {
        fprintf(stderr, "%s: only uncompressed 8-bit and 24-bit BMP are supported\n", path);
        exit(EXIT_FAILURE);
    }

    const bool bottom_up = height_raw > 0;
    const int  height    = bottom_up ? height_raw : -height_raw;
    const size_t stride  = ((size_t(width) * bpp + 31) / 32) * 4;   // rows are padded to 4 bytes

    if (offset + stride * height > file.size()) {
        fprintf(stderr, "%s is truncated\n", path);
        exit(EXIT_FAILURE);
    }

    auto gray = [](int b, int g, int r) {
        return (unsigned char)((114 * b + 587 * g + 299 * r + 500) / 1000);
    };

    // An 8-bit BMP stores indices into a palette of B, G, R, 0 entries.
    unsigned char palette[256];
    if (bpp == 8) {
        const size_t count = colours ? colours : 256;
        for (size_t i = 0; i < 256; ++i) {
            const size_t p = 14 + header_size + 4 * i;
            palette[i] = (i < count && p + 2 < file.size()) ? gray(file[p], file[p + 1], file[p + 2])
                                                             : (unsigned char)i;
        }
    }

    image_t img(width, height);
    for (int y = 0; y < height; ++y) {
        const unsigned char *src = &file[offset + stride * (bottom_up ? height - 1 - y : y)];
        unsigned char *dst = &img.data[size_t(y) * width];
        for (int x = 0; x < width; ++x)
            dst[x] = (bpp == 8) ? palette[src[x]] : gray(src[3 * x], src[3 * x + 1], src[3 * x + 2]);
    }

    printf("loaded %s  %dx%d\n", path, width, height);
    return img;
}

inline void save_bmp(const char *path, const image_t &img)
{
    const uint32_t stride  = (uint32_t(img.width) + 3) & ~3u;
    const uint32_t offset  = 14 + 40 + 256 * 4;
    const uint32_t size    = offset + stride * uint32_t(img.height);

    unsigned char h[54] = {'B', 'M'};
    auto put16 = [&](int o, uint16_t v) { memcpy(h + o, &v, 2); };
    auto put32 = [&](int o, uint32_t v) { memcpy(h + o, &v, 4); };
    put32(2, size);
    put32(10, offset);
    put32(14, 40);                      // header size
    put32(18, uint32_t(img.width));
    put32(22, uint32_t(img.height));    // positive: bottom row first
    put16(26, 1);                       // planes
    put16(28, 8);                       // bits per pixel
    put32(34, stride * uint32_t(img.height));
    put32(46, 256);                     // palette entries

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path);
        return;
    }
    fwrite(h, 1, sizeof(h), f);
    for (int i = 0; i < 256; ++i) {
        const unsigned char entry[4] = {(unsigned char)i, (unsigned char)i, (unsigned char)i, 0};
        fwrite(entry, 1, 4, f);
    }
    std::vector<unsigned char> row(stride, 0);
    for (int y = img.height - 1; y >= 0; --y) {
        memcpy(row.data(), &img.data[size_t(y) * img.width], img.width);
        fwrite(row.data(), 1, stride, f);
    }
    fclose(f);
    printf("wrote %s\n", path);
}
