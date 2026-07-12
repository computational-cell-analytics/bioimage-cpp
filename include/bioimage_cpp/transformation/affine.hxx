#pragma once

#include "bioimage_cpp/array_view.hxx"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace bioimage_cpp::transformation {

namespace detail {

template <std::size_t D>
void require_matrix_shape(const ConstArrayView<double> &matrix) {
    if (matrix.ndim() != 2 ||
        matrix.shape[0] != static_cast<std::ptrdiff_t>(D) ||
        matrix.shape[1] != static_cast<std::ptrdiff_t>(D + 1)) {
        throw std::invalid_argument(
            std::string("matrix must have shape (") + std::to_string(D) + ", " +
            std::to_string(D + 1) + ")"
        );
    }
}

template <std::size_t D>
void require_starts_shape(const ConstArrayView<std::ptrdiff_t> &starts) {
    if (starts.ndim() != 1 ||
        starts.shape[0] != static_cast<std::ptrdiff_t>(D)) {
        throw std::invalid_argument(
            std::string("starts must have shape (") + std::to_string(D) + ",)"
        );
    }
}

template <std::size_t D, class T>
void require_views(const ConstArrayView<T> &input, const ArrayView<T> &output) {
    if (input.ndim() != static_cast<std::ptrdiff_t>(D)) {
        throw std::invalid_argument(
            "input must have ndim=" + std::to_string(D) +
            ", got ndim=" + std::to_string(input.ndim())
        );
    }
    if (output.ndim() != static_cast<std::ptrdiff_t>(D)) {
        throw std::invalid_argument(
            "output must have ndim=" + std::to_string(D) +
            ", got ndim=" + std::to_string(output.ndim())
        );
    }
    // Element-stride C-contiguity check for both views.
    std::ptrdiff_t expected_in = 1;
    std::ptrdiff_t expected_out = 1;
    for (std::ptrdiff_t axis = static_cast<std::ptrdiff_t>(D) - 1; axis >= 0; --axis) {
        const auto k = static_cast<std::size_t>(axis);
        if (input.strides[k] != expected_in) {
            throw std::invalid_argument("input must be C-contiguous");
        }
        if (output.strides[k] != expected_out) {
            throw std::invalid_argument("output must be C-contiguous");
        }
        expected_in *= input.shape[k];
        expected_out *= output.shape[k];
    }
}

// Catmull-Rom / Keys cubic convolution kernel with a = -0.5.
inline double cubic_weight(const double x) {
    const double ax = std::abs(x);
    if (ax < 1.0) {
        return (1.5 * ax - 2.5) * ax * ax + 1.0;
    }
    if (ax < 2.0) {
        return ((-0.5 * ax + 2.5) * ax - 4.0) * ax + 2.0;
    }
    return 0.0;
}

// Cast an interpolated double back to T. For integer T, round to nearest and
// clamp to the dtype range so that cubic overshoot (or out-of-range fill
// values) cannot produce undefined behaviour.
template <class T>
T to_output(const double value) {
    if constexpr (std::is_integral_v<T>) {
        using L = std::numeric_limits<T>;
        constexpr double lo = static_cast<double>(L::min());
        constexpr double hi = static_cast<double>(L::max());
        // `!(value > lo)` covers NaN as well.
        if (!(value > lo)) return L::min();
        if (value >= hi) return L::max();
        return static_cast<T>(std::round(value));
    } else {
        return static_cast<T>(value);
    }
}

// ----- B-spline weights (orders 2, 4, 5) -----------------------------------
//
// Cardinal B-spline kernels of order n. Without a prefilter pass on the input,
// these convolution kernels low-pass smooth the image; sampling them at
// integer grid points does NOT reproduce the input sample exactly. This
// matches `scipy.ndimage.affine_transform(..., prefilter=False)`. See
// PERFORMANCE_NOTES.md for a discussion of what an interpolating ("scipy
// prefilter=True") implementation would cost.
//
// Even orders (2, 4) center the kernel on `round(coord)`; odd orders (5) center
// on `floor(coord)`. The number of taps is `order + 1`.

template <int Order> struct SplineKernel;

template <> struct SplineKernel<2> {
    static constexpr int N = 3;
    static constexpr bool even = true;
    // x in [-0.5, 0.5]
    static inline void weights(const double x, double w[N]) {
        w[1] = 0.75 - x * x;
        const double y_minus = 0.5 - x;
        w[0] = 0.5 * y_minus * y_minus;
        w[2] = 1.0 - w[0] - w[1];
    }
};

template <> struct SplineKernel<4> {
    static constexpr int N = 5;
    static constexpr bool even = true;
    // x in [-0.5, 0.5]
    static inline void weights(const double x, double w[N]) {
        const double t = x * x;
        w[2] = t * (t * 0.25 - 0.625) + 115.0 / 192.0;
        const double y = 1.0 + x;
        w[1] = y * (y * (y * (5.0 - y) / 6.0 - 1.25) + 5.0 / 24.0) +
               55.0 / 96.0;
        const double z = 1.0 - x;
        w[3] = z * (z * (z * (5.0 - z) / 6.0 - 1.25) + 5.0 / 24.0) +
               55.0 / 96.0;
        const double y_neg = 0.5 - x;
        const double t2 = y_neg * y_neg;
        w[0] = t2 * t2 / 24.0;
        w[4] = 1.0 - w[0] - w[1] - w[2] - w[3];
    }
};

template <> struct SplineKernel<5> {
    static constexpr int N = 6;
    static constexpr bool even = false;
    // x in [0, 1)
    static inline void weights(const double x, double w[N]) {
        const double y0 = x;
        const double z0 = 1.0 - x;
        double t = y0 * y0;
        w[2] = t * (t * (0.25 - y0 / 12.0) - 0.5) + 0.55;
        t = z0 * z0;
        w[3] = t * (t * (0.25 - z0 / 12.0) - 0.5) + 0.55;
        const double y1 = 1.0 + x;
        w[1] = y1 * (y1 * (y1 * (y1 * (y1 / 24.0 - 0.375) + 1.25) - 1.75) +
                     0.625) + 0.425;
        const double z1 = 2.0 - x;
        w[4] = z1 * (z1 * (z1 * (z1 * (z1 / 24.0 - 0.375) + 1.25) - 1.75) +
                     0.625) + 0.425;
        const double y2 = 1.0 - x;
        const double t2 = y2 * y2;
        w[0] = y2 * t2 * t2 / 120.0;
        w[5] = 1.0 - w[0] - w[1] - w[2] - w[3] - w[4];
    }
};

template <int Order>
inline std::ptrdiff_t spline_base(const double c) {
    if constexpr (SplineKernel<Order>::even) {
        return static_cast<std::ptrdiff_t>(std::floor(c + 0.5)) -
               SplineKernel<Order>::N / 2;
    } else {
        return static_cast<std::ptrdiff_t>(std::floor(c)) -
               (SplineKernel<Order>::N - 1) / 2;
    }
}

template <int Order>
inline double spline_offset(const double c) {
    if constexpr (SplineKernel<Order>::even) {
        return c - std::floor(c + 0.5);
    } else {
        return c - std::floor(c);
    }
}

template <int Order, class T>
inline double bspline_2d(
    const T *data, std::ptrdiff_t in_h, std::ptrdiff_t in_w,
    double cy, double cx, double fill
) {
    // B-spline kernels are *smoothing* (not interpolating). We do not apply
    // the strict outer coord check used by nearest/linear/Keys-cubic; instead
    // we evaluate the kernel at every coordinate and let the per-tap bounds
    // check pull in `fill` for out-of-bounds taps. Matches
    // `scipy.ndimage.affine_transform(..., mode='grid-constant', cval=fill)`.
    constexpr int N = SplineKernel<Order>::N;
    const std::ptrdiff_t by = spline_base<Order>(cy);
    const std::ptrdiff_t bx = spline_base<Order>(cx);
    double wy[N];
    double wx[N];
    SplineKernel<Order>::weights(spline_offset<Order>(cy), wy);
    SplineKernel<Order>::weights(spline_offset<Order>(cx), wx);
    if (by >= 0 && by + (N - 1) < in_h &&
        bx >= 0 && bx + (N - 1) < in_w) {
        double value = 0.0;
        for (int dy = 0; dy < N; ++dy) {
            const T *row = data + (by + dy) * in_w;
            double rs = 0.0;
            for (int dx = 0; dx < N; ++dx) {
                rs += wx[dx] * static_cast<double>(row[bx + dx]);
            }
            value += wy[dy] * rs;
        }
        return value;
    }
    double value = 0.0;
    for (int dy = 0; dy < N; ++dy) {
        const std::ptrdiff_t y = by + dy;
        const bool y_in = (y >= 0 && y < in_h);
        for (int dx = 0; dx < N; ++dx) {
            const std::ptrdiff_t x = bx + dx;
            const double s = (y_in && x >= 0 && x < in_w)
                ? static_cast<double>(data[y * in_w + x])
                : fill;
            value += wy[dy] * wx[dx] * s;
        }
    }
    return value;
}

template <int Order, class T>
inline double bspline_3d(
    const T *data,
    std::ptrdiff_t in_d, std::ptrdiff_t in_h, std::ptrdiff_t in_w,
    double cz, double cy, double cx, double fill
) {
    // See bspline_2d: no outer coord check; per-tap fill handles boundary.
    constexpr int N = SplineKernel<Order>::N;
    const std::ptrdiff_t bz = spline_base<Order>(cz);
    const std::ptrdiff_t by = spline_base<Order>(cy);
    const std::ptrdiff_t bx = spline_base<Order>(cx);
    double wz[N];
    double wy[N];
    double wx[N];
    SplineKernel<Order>::weights(spline_offset<Order>(cz), wz);
    SplineKernel<Order>::weights(spline_offset<Order>(cy), wy);
    SplineKernel<Order>::weights(spline_offset<Order>(cx), wx);
    const std::ptrdiff_t plane = in_h * in_w;
    if (bz >= 0 && bz + (N - 1) < in_d &&
        by >= 0 && by + (N - 1) < in_h &&
        bx >= 0 && bx + (N - 1) < in_w) {
        double value = 0.0;
        for (int dz = 0; dz < N; ++dz) {
            const T *p = data + (bz + dz) * plane;
            for (int dy = 0; dy < N; ++dy) {
                const T *row = p + (by + dy) * in_w;
                double rs = 0.0;
                for (int dx = 0; dx < N; ++dx) {
                    rs += wx[dx] * static_cast<double>(row[bx + dx]);
                }
                value += wz[dz] * wy[dy] * rs;
            }
        }
        return value;
    }
    double value = 0.0;
    for (int dz = 0; dz < N; ++dz) {
        const std::ptrdiff_t z = bz + dz;
        const bool z_in = (z >= 0 && z < in_d);
        for (int dy = 0; dy < N; ++dy) {
            const std::ptrdiff_t y = by + dy;
            const bool y_in = z_in && (y >= 0 && y < in_h);
            for (int dx = 0; dx < N; ++dx) {
                const std::ptrdiff_t x = bx + dx;
                const double s = (y_in && x >= 0 && x < in_w)
                    ? static_cast<double>(data[z * plane + y * in_w + x])
                    : fill;
                value += wz[dz] * wy[dy] * wx[dx] * s;
            }
        }
    }
    return value;
}

// ----- 2D sampling kernels --------------------------------------------------

template <class T>
inline double nearest_2d(
    const T *data, std::ptrdiff_t in_h, std::ptrdiff_t in_w,
    double cy, double cx, double fill
) {
    if (cy < 0.0 || cy > static_cast<double>(in_h - 1) ||
        cx < 0.0 || cx > static_cast<double>(in_w - 1)) {
        return fill;
    }
    // Inside [0, shape-1] the rounded index lies in [0, shape-1].
    const auto y = static_cast<std::ptrdiff_t>(std::floor(cy + 0.5));
    const auto x = static_cast<std::ptrdiff_t>(std::floor(cx + 0.5));
    return static_cast<double>(data[y * in_w + x]);
}

template <class T>
inline double linear_2d(
    const T *data, std::ptrdiff_t in_h, std::ptrdiff_t in_w,
    double cy, double cx, double fill
) {
    if (cy < 0.0 || cy > static_cast<double>(in_h - 1) ||
        cx < 0.0 || cx > static_cast<double>(in_w - 1)) {
        return fill;
    }
    const auto y0 = static_cast<std::ptrdiff_t>(std::floor(cy));
    const auto x0 = static_cast<std::ptrdiff_t>(std::floor(cx));
    const double fy = cy - static_cast<double>(y0);
    const double fx = cx - static_cast<double>(x0);
    // At the upper boundary lower == shape - 1 and fraction == 0, so we can
    // safely clamp the upper neighbour to the lower index without changing the
    // interpolated value.
    const std::ptrdiff_t y1 = (y0 + 1 < in_h) ? y0 + 1 : y0;
    const std::ptrdiff_t x1 = (x0 + 1 < in_w) ? x0 + 1 : x0;
    const double v00 = static_cast<double>(data[y0 * in_w + x0]);
    const double v01 = static_cast<double>(data[y0 * in_w + x1]);
    const double v10 = static_cast<double>(data[y1 * in_w + x0]);
    const double v11 = static_cast<double>(data[y1 * in_w + x1]);
    return (1.0 - fy) * ((1.0 - fx) * v00 + fx * v01)
         +        fy  * ((1.0 - fx) * v10 + fx * v11);
}

template <class T>
inline double cubic_2d(
    const T *data, std::ptrdiff_t in_h, std::ptrdiff_t in_w,
    double cy, double cx, double fill
) {
    if (cy < 0.0 || cy > static_cast<double>(in_h - 1) ||
        cx < 0.0 || cx > static_cast<double>(in_w - 1)) {
        return fill;
    }
    const auto by = static_cast<std::ptrdiff_t>(std::floor(cy));
    const auto bx = static_cast<std::ptrdiff_t>(std::floor(cx));
    double wy[4];
    double wx[4];
    for (int k = 0; k < 4; ++k) {
        wy[k] = cubic_weight(cy - static_cast<double>(by + k - 1));
        wx[k] = cubic_weight(cx - static_cast<double>(bx + k - 1));
    }
    if (by >= 1 && by + 2 < in_h && bx >= 1 && bx + 2 < in_w) {
        double value = 0.0;
        for (int dy = 0; dy < 4; ++dy) {
            const T *row = data + (by + dy - 1) * in_w;
            value += wy[dy] * (
                wx[0] * static_cast<double>(row[bx - 1]) +
                wx[1] * static_cast<double>(row[bx    ]) +
                wx[2] * static_cast<double>(row[bx + 1]) +
                wx[3] * static_cast<double>(row[bx + 2])
            );
        }
        return value;
    }
    double value = 0.0;
    for (int dy = 0; dy < 4; ++dy) {
        const std::ptrdiff_t y = by + dy - 1;
        const bool y_in = (y >= 0 && y < in_h);
        for (int dx = 0; dx < 4; ++dx) {
            const std::ptrdiff_t x = bx + dx - 1;
            const double s = (y_in && x >= 0 && x < in_w)
                ? static_cast<double>(data[y * in_w + x])
                : fill;
            value += wy[dy] * wx[dx] * s;
        }
    }
    return value;
}

// ----- 3D sampling kernels --------------------------------------------------

template <class T>
inline double nearest_3d(
    const T *data,
    std::ptrdiff_t in_d, std::ptrdiff_t in_h, std::ptrdiff_t in_w,
    double cz, double cy, double cx, double fill
) {
    if (cz < 0.0 || cz > static_cast<double>(in_d - 1) ||
        cy < 0.0 || cy > static_cast<double>(in_h - 1) ||
        cx < 0.0 || cx > static_cast<double>(in_w - 1)) {
        return fill;
    }
    const auto z = static_cast<std::ptrdiff_t>(std::floor(cz + 0.5));
    const auto y = static_cast<std::ptrdiff_t>(std::floor(cy + 0.5));
    const auto x = static_cast<std::ptrdiff_t>(std::floor(cx + 0.5));
    return static_cast<double>(data[(z * in_h + y) * in_w + x]);
}

template <class T>
inline double linear_3d(
    const T *data,
    std::ptrdiff_t in_d, std::ptrdiff_t in_h, std::ptrdiff_t in_w,
    double cz, double cy, double cx, double fill
) {
    if (cz < 0.0 || cz > static_cast<double>(in_d - 1) ||
        cy < 0.0 || cy > static_cast<double>(in_h - 1) ||
        cx < 0.0 || cx > static_cast<double>(in_w - 1)) {
        return fill;
    }
    const auto z0 = static_cast<std::ptrdiff_t>(std::floor(cz));
    const auto y0 = static_cast<std::ptrdiff_t>(std::floor(cy));
    const auto x0 = static_cast<std::ptrdiff_t>(std::floor(cx));
    const double fz = cz - static_cast<double>(z0);
    const double fy = cy - static_cast<double>(y0);
    const double fx = cx - static_cast<double>(x0);
    const std::ptrdiff_t z1 = (z0 + 1 < in_d) ? z0 + 1 : z0;
    const std::ptrdiff_t y1 = (y0 + 1 < in_h) ? y0 + 1 : y0;
    const std::ptrdiff_t x1 = (x0 + 1 < in_w) ? x0 + 1 : x0;
    const std::ptrdiff_t plane = in_h * in_w;
    auto at = [&](std::ptrdiff_t z, std::ptrdiff_t y, std::ptrdiff_t x) {
        return static_cast<double>(data[z * plane + y * in_w + x]);
    };
    const double v000 = at(z0, y0, x0);
    const double v001 = at(z0, y0, x1);
    const double v010 = at(z0, y1, x0);
    const double v011 = at(z0, y1, x1);
    const double v100 = at(z1, y0, x0);
    const double v101 = at(z1, y0, x1);
    const double v110 = at(z1, y1, x0);
    const double v111 = at(z1, y1, x1);
    return (1.0 - fz) * (
              (1.0 - fy) * ((1.0 - fx) * v000 + fx * v001)
            +        fy  * ((1.0 - fx) * v010 + fx * v011)
           )
         +        fz  * (
              (1.0 - fy) * ((1.0 - fx) * v100 + fx * v101)
            +        fy  * ((1.0 - fx) * v110 + fx * v111)
           );
}

template <class T>
inline double cubic_3d(
    const T *data,
    std::ptrdiff_t in_d, std::ptrdiff_t in_h, std::ptrdiff_t in_w,
    double cz, double cy, double cx, double fill
) {
    if (cz < 0.0 || cz > static_cast<double>(in_d - 1) ||
        cy < 0.0 || cy > static_cast<double>(in_h - 1) ||
        cx < 0.0 || cx > static_cast<double>(in_w - 1)) {
        return fill;
    }
    const auto bz = static_cast<std::ptrdiff_t>(std::floor(cz));
    const auto by = static_cast<std::ptrdiff_t>(std::floor(cy));
    const auto bx = static_cast<std::ptrdiff_t>(std::floor(cx));
    double wz[4];
    double wy[4];
    double wx[4];
    for (int k = 0; k < 4; ++k) {
        wz[k] = cubic_weight(cz - static_cast<double>(bz + k - 1));
        wy[k] = cubic_weight(cy - static_cast<double>(by + k - 1));
        wx[k] = cubic_weight(cx - static_cast<double>(bx + k - 1));
    }
    const std::ptrdiff_t plane = in_h * in_w;
    if (bz >= 1 && bz + 2 < in_d &&
        by >= 1 && by + 2 < in_h &&
        bx >= 1 && bx + 2 < in_w) {
        double value = 0.0;
        for (int dz = 0; dz < 4; ++dz) {
            const T *p = data + (bz + dz - 1) * plane;
            for (int dy = 0; dy < 4; ++dy) {
                const T *row = p + (by + dy - 1) * in_w;
                value += wz[dz] * wy[dy] * (
                    wx[0] * static_cast<double>(row[bx - 1]) +
                    wx[1] * static_cast<double>(row[bx    ]) +
                    wx[2] * static_cast<double>(row[bx + 1]) +
                    wx[3] * static_cast<double>(row[bx + 2])
                );
            }
        }
        return value;
    }
    double value = 0.0;
    for (int dz = 0; dz < 4; ++dz) {
        const std::ptrdiff_t z = bz + dz - 1;
        const bool z_in = (z >= 0 && z < in_d);
        for (int dy = 0; dy < 4; ++dy) {
            const std::ptrdiff_t y = by + dy - 1;
            const bool y_in = z_in && (y >= 0 && y < in_h);
            for (int dx = 0; dx < 4; ++dx) {
                const std::ptrdiff_t x = bx + dx - 1;
                const double s = (y_in && x >= 0 && x < in_w)
                    ? static_cast<double>(data[z * plane + y * in_w + x])
                    : fill;
                value += wz[dz] * wy[dy] * wx[dx] * s;
            }
        }
    }
    return value;
}

// Dispatch to the per-voxel sampler for the requested interpolation order. Shared by the affine and
// map_coordinates kernels so the interpolation backend lives in exactly one place. `order` must be in
// 0..5 (validated by the public entry points before the sampling loop).
//
// `CheckFinite` gates a per-coordinate `std::isfinite` guard. map_coordinates
// passes user-supplied coordinates that may be non-finite and instantiates it
// with `true`. Affine coordinates are `matrix * output_grid` with a
// finite-validated matrix, so the affine callers use the default `false` and
// the check is elided at compile time — keeping the affine sampling hot loop
// identical to the pre-fix code.
template <bool CheckFinite = false, class T>
inline double sample_2d(
    const T *data, std::ptrdiff_t in_h, std::ptrdiff_t in_w,
    double cy, double cx, const int order, double fill
) {
    if constexpr (CheckFinite) {
        if (!std::isfinite(cy) || !std::isfinite(cx)) {
            return fill;
        }
    }
    switch (order) {
        case 0: return nearest_2d(data, in_h, in_w, cy, cx, fill);
        case 1: return linear_2d(data, in_h, in_w, cy, cx, fill);
        case 2: return bspline_2d<2>(data, in_h, in_w, cy, cx, fill);
        case 3: return cubic_2d(data, in_h, in_w, cy, cx, fill);
        case 4: return bspline_2d<4>(data, in_h, in_w, cy, cx, fill);
        default: return bspline_2d<5>(data, in_h, in_w, cy, cx, fill);  // 5
    }
}

template <bool CheckFinite = false, class T>
inline double sample_3d(
    const T *data,
    std::ptrdiff_t in_d, std::ptrdiff_t in_h, std::ptrdiff_t in_w,
    double cz, double cy, double cx, const int order, double fill
) {
    if constexpr (CheckFinite) {
        if (!std::isfinite(cz) || !std::isfinite(cy) || !std::isfinite(cx)) {
            return fill;
        }
    }
    switch (order) {
        case 0: return nearest_3d(data, in_d, in_h, in_w, cz, cy, cx, fill);
        case 1: return linear_3d(data, in_d, in_h, in_w, cz, cy, cx, fill);
        case 2: return bspline_3d<2>(data, in_d, in_h, in_w, cz, cy, cx, fill);
        case 3: return cubic_3d(data, in_d, in_h, in_w, cz, cy, cx, fill);
        case 4: return bspline_3d<4>(data, in_d, in_h, in_w, cz, cy, cx, fill);
        default: return bspline_3d<5>(data, in_d, in_h, in_w, cz, cy, cx, fill);  // 5
    }
}

} // namespace detail

// ----- 2D entry point -------------------------------------------------------

template <class T>
void affine_transform_2d(
    const ConstArrayView<T> &input,
    ArrayView<T> &output,
    const ConstArrayView<double> &matrix,
    const ConstArrayView<std::ptrdiff_t> &starts,
    const int order,
    const T fill_value
) {
    detail::require_views<2, T>(input, output);
    detail::require_matrix_shape<2>(matrix);
    detail::require_starts_shape<2>(starts);
    if (order < 0 || order > 5) {
        throw std::invalid_argument(
            "order must be in 0..5, got " + std::to_string(order)
        );
    }

    const auto out_h = output.shape[0];
    const auto out_w = output.shape[1];
    if (out_h < 0 || out_w < 0) {
        throw std::invalid_argument("output shape must be non-negative");
    }
    if (out_h == 0 || out_w == 0) return;

    const auto in_h = input.shape[0];
    const auto in_w = input.shape[1];

    // Hoist matrix and starts so the inner loop reads from local scalars.
    const auto m_row = matrix.strides[0];
    const auto m_col = matrix.strides[1];
    const double m00 = matrix.data[0 * m_row + 0 * m_col];
    const double m01 = matrix.data[0 * m_row + 1 * m_col];
    const double m02 = matrix.data[0 * m_row + 2 * m_col];
    const double m10 = matrix.data[1 * m_row + 0 * m_col];
    const double m11 = matrix.data[1 * m_row + 1 * m_col];
    const double m12 = matrix.data[1 * m_row + 2 * m_col];

    const auto sy = starts.data[0 * starts.strides[0]];
    const auto sx = starts.data[1 * starts.strides[0]];

    // Input coordinate at output index (0, 0). Walking +1 along output axis k
    // adds matrix column k to the input coordinate, so we only need scalar
    // increments inside the loop.
    double row_y = m00 * static_cast<double>(sy) + m01 * static_cast<double>(sx) + m02;
    double row_x = m10 * static_cast<double>(sy) + m11 * static_cast<double>(sx) + m12;

    const double fill = static_cast<double>(fill_value);
    const T *in_data = input.data;
    T *out_ptr = output.data;

    for (std::ptrdiff_t i = 0; i < out_h; ++i) {
        double cy = row_y;
        double cx = row_x;
        for (std::ptrdiff_t j = 0; j < out_w; ++j) {
            const double value = detail::sample_2d(in_data, in_h, in_w, cy, cx, order, fill);
            *out_ptr++ = detail::to_output<T>(value);
            cy += m01;
            cx += m11;
        }
        row_y += m00;
        row_x += m10;
    }
}

// ----- 3D entry point -------------------------------------------------------

template <class T>
void affine_transform_3d(
    const ConstArrayView<T> &input,
    ArrayView<T> &output,
    const ConstArrayView<double> &matrix,
    const ConstArrayView<std::ptrdiff_t> &starts,
    const int order,
    const T fill_value
) {
    detail::require_views<3, T>(input, output);
    detail::require_matrix_shape<3>(matrix);
    detail::require_starts_shape<3>(starts);
    if (order < 0 || order > 5) {
        throw std::invalid_argument(
            "order must be in 0..5, got " + std::to_string(order)
        );
    }

    const auto out_d = output.shape[0];
    const auto out_h = output.shape[1];
    const auto out_w = output.shape[2];
    if (out_d < 0 || out_h < 0 || out_w < 0) {
        throw std::invalid_argument("output shape must be non-negative");
    }
    if (out_d == 0 || out_h == 0 || out_w == 0) return;

    const auto in_d = input.shape[0];
    const auto in_h = input.shape[1];
    const auto in_w = input.shape[2];

    const auto m_row = matrix.strides[0];
    const auto m_col = matrix.strides[1];
    const double m00 = matrix.data[0 * m_row + 0 * m_col];
    const double m01 = matrix.data[0 * m_row + 1 * m_col];
    const double m02 = matrix.data[0 * m_row + 2 * m_col];
    const double m03 = matrix.data[0 * m_row + 3 * m_col];
    const double m10 = matrix.data[1 * m_row + 0 * m_col];
    const double m11 = matrix.data[1 * m_row + 1 * m_col];
    const double m12 = matrix.data[1 * m_row + 2 * m_col];
    const double m13 = matrix.data[1 * m_row + 3 * m_col];
    const double m20 = matrix.data[2 * m_row + 0 * m_col];
    const double m21 = matrix.data[2 * m_row + 1 * m_col];
    const double m22 = matrix.data[2 * m_row + 2 * m_col];
    const double m23 = matrix.data[2 * m_row + 3 * m_col];

    const auto sz = starts.data[0 * starts.strides[0]];
    const auto sy = starts.data[1 * starts.strides[0]];
    const auto sx = starts.data[2 * starts.strides[0]];

    double plane_z = m00 * static_cast<double>(sz) + m01 * static_cast<double>(sy)
                   + m02 * static_cast<double>(sx) + m03;
    double plane_y = m10 * static_cast<double>(sz) + m11 * static_cast<double>(sy)
                   + m12 * static_cast<double>(sx) + m13;
    double plane_x = m20 * static_cast<double>(sz) + m21 * static_cast<double>(sy)
                   + m22 * static_cast<double>(sx) + m23;

    const double fill = static_cast<double>(fill_value);
    const T *in_data = input.data;
    T *out_ptr = output.data;

    for (std::ptrdiff_t k = 0; k < out_d; ++k) {
        double row_z = plane_z;
        double row_y = plane_y;
        double row_x = plane_x;
        for (std::ptrdiff_t i = 0; i < out_h; ++i) {
            double cz = row_z;
            double cy = row_y;
            double cx = row_x;
            for (std::ptrdiff_t j = 0; j < out_w; ++j) {
                const double value = detail::sample_3d(in_data, in_d, in_h, in_w,
                                                       cz, cy, cx, order, fill);
                *out_ptr++ = detail::to_output<T>(value);
                cz += m02;
                cy += m12;
                cx += m22;
            }
            row_z += m01;
            row_y += m11;
            row_x += m21;
        }
        plane_z += m00;
        plane_y += m10;
        plane_x += m20;
    }
}

} // namespace bioimage_cpp::transformation
