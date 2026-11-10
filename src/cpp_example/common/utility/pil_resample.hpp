/**
 * @file pil_resample.hpp
 * @brief Pillow's antialiased resize (Image.resize with BILINEAR / BICUBIC),
 *        byte for byte, for uint8 images with interleaved channels.
 *
 * The model zoo's compile JSON gives most classifiers the torchvision
 * evaluation transform: Resize(shorter side S) then CenterCrop. torchvision
 * resizes a PIL image with Pillow, and Pillow's resize is not OpenCV's: it is
 * a separable convolution whose kernel widens with the scale factor (it
 * antialiases when it shrinks), done in two passes -- horizontal, then
 * vertical -- with 22-bit fixed-point weights and a uint8 image in between.
 * This reproduces that arithmetic (Pillow src/libImaging/Resample.c):
 *
 *   scale = in / out;  filterscale = max(scale, 1)
 *   support = filter_support * filterscale    (bilinear 1, bicubic 2)
 *   for each output index i: centre = (i + 0.5) * scale
 *     xmin = max(int(centre - support + 0.5), 0)
 *     xmax = min(int(centre + support + 0.5), in) - xmin
 *     w[x] = filter((x + xmin - centre + 0.5) / filterscale), normalised
 *     weights to int: (int)(w * 2^22 +- 0.5)
 *   out = clip((2^21 + sum in * w) >> 22)
 *   the vertical pass reads only the rows the vertical kernels touch
 *
 * bicubic is Keys' cubic with a = -0.5.
 */
#ifndef PIL_RESAMPLE_HPP
#define PIL_RESAMPLE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace pil_resample {

enum class Filter { kBilinear, kBicubic };

namespace detail {

constexpr int kPrecisionBits = 32 - 8 - 2;

inline double bilinear(double x)
{
    if (x < 0.0) x = -x;
    return x < 1.0 ? 1.0 - x : 0.0;
}

inline double bicubic(double x)
{
    constexpr double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

struct Coeffs {
    int ksize{0};
    std::vector<int> bounds;      // xmin, count per output index
    std::vector<int32_t> kk;      // ksize fixed-point weights per output index
};

inline Coeffs precompute(int in_size, int out_size, Filter filter)
{
    const double support_1 = filter == Filter::kBicubic ? 2.0 : 1.0;
    const double scale = static_cast<double>(in_size) / out_size;
    double filterscale = scale;
    if (filterscale < 1.0) filterscale = 1.0;
    const double support = support_1 * filterscale;
    Coeffs c;
    c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    c.bounds.resize(static_cast<size_t>(out_size) * 2);
    c.kk.assign(static_cast<size_t>(out_size) * c.ksize, 0);
    std::vector<double> k(static_cast<size_t>(c.ksize));
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;
        const double ss = 1.0 / filterscale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        double ww = 0.0;
        for (int x = 0; x < xmax; ++x) {
            const double t = (x + xmin - center + 0.5) * ss;
            const double w = filter == Filter::kBicubic ? bicubic(t) : bilinear(t);
            k[static_cast<size_t>(x)] = w;
            ww += w;
        }
        for (int x = 0; x < xmax; ++x) {
            double v = k[static_cast<size_t>(x)];
            if (ww != 0.0) v /= ww;
            c.kk[static_cast<size_t>(xx) * c.ksize + x] = static_cast<int32_t>(
                v < 0 ? -0.5 + v * (1 << kPrecisionBits) : 0.5 + v * (1 << kPrecisionBits));
        }
        c.bounds[static_cast<size_t>(xx) * 2] = xmin;
        c.bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
    }
    return c;
}

inline uint8_t clip8(int64_t v)
{
    if (v >= (int64_t{1} << kPrecisionBits << 8)) return 255;
    if (v <= 0) return 0;
    return static_cast<uint8_t>(v >> kPrecisionBits);
}

}  // namespace detail

/// Resize `src` (src_w x src_h x channels, rows `src_stride` bytes apart) to
/// dst_w x dst_h, tightly packed. Same result as
/// PIL.Image.fromarray(src).resize((dst_w, dst_h), BILINEAR | BICUBIC).
inline std::vector<uint8_t> resize(const uint8_t* src, int src_w, int src_h, int channels,
                                   size_t src_stride, int dst_w, int dst_h, Filter filter)
{
    using detail::clip8;
    if (src_stride == 0) src_stride = static_cast<size_t>(src_w) * channels;
    const bool need_h = dst_w != src_w;
    const bool need_v = dst_h != src_h;
    const detail::Coeffs ch = detail::precompute(src_w, dst_w, filter);
    detail::Coeffs cv = detail::precompute(src_h, dst_h, filter);
    const int64_t half = int64_t{1} << (detail::kPrecisionBits - 1);

    // Rows the vertical pass will read (Pillow's ybox_first / ybox_last).
    const int y_first = need_v ? cv.bounds[0] : 0;
    const int y_last = need_v ? cv.bounds[static_cast<size_t>(dst_h) * 2 - 2] +
                                    cv.bounds[static_cast<size_t>(dst_h) * 2 - 1]
                              : src_h;

    // Horizontal pass into a uint8 image of the needed rows.
    std::vector<uint8_t> tmp;
    const uint8_t* mid = src;
    size_t mid_stride = src_stride;
    int mid_rows_offset = 0;  // the row of `mid` that is source row y_first
    if (need_h) {
        const int rows = y_last - y_first;
        tmp.resize(static_cast<size_t>(rows) * dst_w * channels);
        for (int y = 0; y < rows; ++y) {
            const uint8_t* in = src + static_cast<size_t>(y + y_first) * src_stride;
            uint8_t* out = tmp.data() + static_cast<size_t>(y) * dst_w * channels;
            for (int xx = 0; xx < dst_w; ++xx) {
                const int xmin = ch.bounds[static_cast<size_t>(xx) * 2];
                const int xn = ch.bounds[static_cast<size_t>(xx) * 2 + 1];
                const int32_t* k = &ch.kk[static_cast<size_t>(xx) * ch.ksize];
                for (int c = 0; c < channels; ++c) {
                    int64_t ss = half;
                    for (int x = 0; x < xn; ++x) {
                        ss += static_cast<int64_t>(in[(x + xmin) * channels + c]) * k[x];
                    }
                    out[xx * channels + c] = clip8(ss);
                }
            }
        }
        mid = tmp.data();
        mid_stride = static_cast<size_t>(dst_w) * channels;
        mid_rows_offset = y_first;
    }

    const int mid_w = need_h ? dst_w : src_w;
    std::vector<uint8_t> dst(static_cast<size_t>(dst_w) * dst_h * channels);
    if (!need_v) {
        for (int y = 0; y < dst_h; ++y) {
            std::copy_n(mid + static_cast<size_t>(y) * mid_stride,
                        static_cast<size_t>(mid_w) * channels,
                        dst.data() + static_cast<size_t>(y) * dst_w * channels);
        }
        return dst;
    }
    for (int yy = 0; yy < dst_h; ++yy) {
        const int ymin = cv.bounds[static_cast<size_t>(yy) * 2] - mid_rows_offset;
        const int yn = cv.bounds[static_cast<size_t>(yy) * 2 + 1];
        const int32_t* k = &cv.kk[static_cast<size_t>(yy) * cv.ksize];
        uint8_t* out = dst.data() + static_cast<size_t>(yy) * dst_w * channels;
        for (int i = 0; i < mid_w * channels; ++i) {
            int64_t ss = half;
            for (int y = 0; y < yn; ++y) {
                ss += static_cast<int64_t>(mid[static_cast<size_t>(ymin + y) * mid_stride + i]) * k[y];
            }
            out[i] = clip8(ss);
        }
    }
    return dst;
}

/// torchvision Resize(short) + CenterCrop(crop_w, crop_h) on an image, the
/// compile JSON's {resize: {mode: torchvision, size: short}} + {centercrop}:
/// the shorter side becomes `short_side`, the longer int(short * long / short);
/// the crop starts at round((resized - crop) / 2) (half to even, Python's).
inline std::vector<uint8_t> resize_short_center_crop(
    const uint8_t* src, int src_w, int src_h, int channels, size_t src_stride,
    int short_side, int crop_w, int crop_h, Filter filter)
{
    int nw, nh;
    if (src_w <= src_h) {
        nw = short_side;
        nh = static_cast<int>(static_cast<double>(short_side) * src_h / src_w);
    } else {
        nh = short_side;
        nw = static_cast<int>(static_cast<double>(short_side) * src_w / src_h);
    }
    const std::vector<uint8_t> resized = resize(src, src_w, src_h, channels, src_stride, nw, nh, filter);
    const int left = static_cast<int>(std::nearbyint((nw - crop_w) / 2.0));
    const int top = static_cast<int>(std::nearbyint((nh - crop_h) / 2.0));
    std::vector<uint8_t> out(static_cast<size_t>(crop_w) * crop_h * channels, 0);
    for (int y = 0; y < crop_h; ++y) {
        const int sy = top + y;
        if (sy < 0 || sy >= nh) continue;  // a crop larger than the image pads with 0, as torchvision
        for (int x = 0; x < crop_w; ++x) {
            const int sx = left + x;
            if (sx < 0 || sx >= nw) continue;
            std::copy_n(&resized[(static_cast<size_t>(sy) * nw + sx) * channels], channels,
                        &out[(static_cast<size_t>(y) * crop_w + x) * channels]);
        }
    }
    return out;
}

}  // namespace pil_resample

#endif  // PIL_RESAMPLE_HPP
