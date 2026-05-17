#pragma once

#include "bioimage_cpp/filters/kernel.hxx"

#include <algorithm>
#include <cstddef>

namespace bioimage_cpp::filters {

namespace detail {

// Mirror reflection without edge-pixel repeat (scipy mode="mirror").
// Period is 2*(n-1). Handles arbitrary integer x.
inline std::ptrdiff_t mirror_index(std::ptrdiff_t x, std::ptrdiff_t n) {
    if (n <= 1) return 0;
    const std::ptrdiff_t period = 2 * (n - 1);
    std::ptrdiff_t r = x % period;
    if (r < 0) r += period;
    if (r >= n) r = period - r;
    return r;
}

// Convolve along the contiguous (innermost) axis. Specialised for compile-time
// radius R and symmetry.
template <int R, bool Symmetric>
void convolve_x_radius(
    const float *__restrict in,
    float *__restrict out,
    std::ptrdiff_t n_rows,
    std::ptrdiff_t n_cols,
    const float *__restrict h
) {
    if (n_cols <= 0 || n_rows <= 0) return;

    const std::ptrdiff_t prologue_end = std::min<std::ptrdiff_t>(R, n_cols);
    const std::ptrdiff_t epilogue_start = std::max<std::ptrdiff_t>(prologue_end, n_cols - R);

    for (std::ptrdiff_t row = 0; row < n_rows; ++row) {
        const float *__restrict in_row = in + row * n_cols;
        float *__restrict out_row = out + row * n_cols;

        // Border prologue
        for (std::ptrdiff_t x = 0; x < prologue_end; ++x) {
            float acc;
            if constexpr (Symmetric) {
                acc = h[0] * in_row[x];
                for (int k = 1; k <= R; ++k) {
                    const float left = in_row[mirror_index(x - k, n_cols)];
                    const float right = in_row[mirror_index(x + k, n_cols)];
                    acc += h[k] * (left + right);
                }
            } else {
                acc = 0.0f;
                for (int k = 1; k <= R; ++k) {
                    const float left = in_row[mirror_index(x - k, n_cols)];
                    const float right = in_row[mirror_index(x + k, n_cols)];
                    acc += h[k] * (right - left);
                }
            }
            out_row[x] = acc;
        }

        // Main loop (branchless): the compiler vectorizes the outer x loop.
        if constexpr (Symmetric) {
            for (std::ptrdiff_t x = prologue_end; x < epilogue_start; ++x) {
                float acc = h[0] * in_row[x];
                for (int k = 1; k <= R; ++k) {
                    acc += h[k] * (in_row[x + k] + in_row[x - k]);
                }
                out_row[x] = acc;
            }
        } else {
            for (std::ptrdiff_t x = prologue_end; x < epilogue_start; ++x) {
                float acc = 0.0f;
                for (int k = 1; k <= R; ++k) {
                    acc += h[k] * (in_row[x + k] - in_row[x - k]);
                }
                out_row[x] = acc;
            }
        }

        // Border epilogue
        for (std::ptrdiff_t x = epilogue_start; x < n_cols; ++x) {
            float acc;
            if constexpr (Symmetric) {
                acc = h[0] * in_row[x];
                for (int k = 1; k <= R; ++k) {
                    const float left = in_row[mirror_index(x - k, n_cols)];
                    const float right = in_row[mirror_index(x + k, n_cols)];
                    acc += h[k] * (left + right);
                }
            } else {
                acc = 0.0f;
                for (int k = 1; k <= R; ++k) {
                    const float left = in_row[mirror_index(x - k, n_cols)];
                    const float right = in_row[mirror_index(x + k, n_cols)];
                    acc += h[k] * (right - left);
                }
            }
            out_row[x] = acc;
        }
    }
}

// Same as above but with runtime radius (handles radii > the template cap).
template <bool Symmetric>
void convolve_x_runtime(
    const float *__restrict in,
    float *__restrict out,
    std::ptrdiff_t n_rows,
    std::ptrdiff_t n_cols,
    int radius,
    const float *__restrict h
) {
    if (n_cols <= 0 || n_rows <= 0) return;

    const std::ptrdiff_t R = radius;
    const std::ptrdiff_t prologue_end = std::min<std::ptrdiff_t>(R, n_cols);
    const std::ptrdiff_t epilogue_start = std::max<std::ptrdiff_t>(prologue_end, n_cols - R);

    for (std::ptrdiff_t row = 0; row < n_rows; ++row) {
        const float *__restrict in_row = in + row * n_cols;
        float *__restrict out_row = out + row * n_cols;

        for (std::ptrdiff_t x = 0; x < prologue_end; ++x) {
            float acc;
            if constexpr (Symmetric) {
                acc = h[0] * in_row[x];
                for (int k = 1; k <= R; ++k) {
                    acc += h[k] * (in_row[mirror_index(x - k, n_cols)] +
                                   in_row[mirror_index(x + k, n_cols)]);
                }
            } else {
                acc = 0.0f;
                for (int k = 1; k <= R; ++k) {
                    acc += h[k] * (in_row[mirror_index(x + k, n_cols)] -
                                   in_row[mirror_index(x - k, n_cols)]);
                }
            }
            out_row[x] = acc;
        }

        if constexpr (Symmetric) {
            for (std::ptrdiff_t x = prologue_end; x < epilogue_start; ++x) {
                float acc = h[0] * in_row[x];
                for (int k = 1; k <= R; ++k) {
                    acc += h[k] * (in_row[x + k] + in_row[x - k]);
                }
                out_row[x] = acc;
            }
        } else {
            for (std::ptrdiff_t x = prologue_end; x < epilogue_start; ++x) {
                float acc = 0.0f;
                for (int k = 1; k <= R; ++k) {
                    acc += h[k] * (in_row[x + k] - in_row[x - k]);
                }
                out_row[x] = acc;
            }
        }

        for (std::ptrdiff_t x = epilogue_start; x < n_cols; ++x) {
            float acc;
            if constexpr (Symmetric) {
                acc = h[0] * in_row[x];
                for (int k = 1; k <= R; ++k) {
                    acc += h[k] * (in_row[mirror_index(x - k, n_cols)] +
                                   in_row[mirror_index(x + k, n_cols)]);
                }
            } else {
                acc = 0.0f;
                for (int k = 1; k <= R; ++k) {
                    acc += h[k] * (in_row[mirror_index(x + k, n_cols)] -
                                   in_row[mirror_index(x - k, n_cols)]);
                }
            }
            out_row[x] = acc;
        }
    }
}

// X-strip block size for the strided pass. 64 floats = 256 bytes, fits in L1
// and large enough to amortise the kernel-coefficient broadcast.
inline constexpr std::ptrdiff_t kStripBlock = 64;

// Convolve along a strided (non-innermost) axis. Logical shape
// (n_outer, n_axis, n_inner) in C-order; we accumulate into strips of
// kStripBlock contiguous columns of the innermost axis, which gives the
// compiler something easy to vectorise.
template <int R, bool Symmetric>
void convolve_strided_radius(
    const float *__restrict in,
    float *__restrict out,
    std::ptrdiff_t n_outer,
    std::ptrdiff_t n_axis,
    std::ptrdiff_t n_inner,
    const float *__restrict h
) {
    if (n_axis <= 0 || n_inner <= 0 || n_outer <= 0) return;

    const std::ptrdiff_t outer_stride = n_axis * n_inner;
    const std::ptrdiff_t prologue_end = std::min<std::ptrdiff_t>(R, n_axis);
    const std::ptrdiff_t epilogue_start = std::max<std::ptrdiff_t>(prologue_end, n_axis - R);

    for (std::ptrdiff_t o = 0; o < n_outer; ++o) {
        const float *__restrict in_o = in + o * outer_stride;
        float *__restrict out_o = out + o * outer_stride;

        // Main rows (no border on the axis dimension).
        for (std::ptrdiff_t y = prologue_end; y < epilogue_start; ++y) {
            float *__restrict out_row = out_o + y * n_inner;
            const float *__restrict center = in_o + y * n_inner;

            for (std::ptrdiff_t xb = 0; xb < n_inner; xb += kStripBlock) {
                const std::ptrdiff_t strip = std::min(kStripBlock, n_inner - xb);
                float acc[kStripBlock];

                if constexpr (Symmetric) {
                    const float h0 = h[0];
                    for (std::ptrdiff_t i = 0; i < strip; ++i) {
                        acc[i] = h0 * center[xb + i];
                    }
                } else {
                    for (std::ptrdiff_t i = 0; i < strip; ++i) {
                        acc[i] = 0.0f;
                    }
                }

                for (int k = 1; k <= R; ++k) {
                    const float hk = h[k];
                    const float *__restrict up = in_o + (y - k) * n_inner + xb;
                    const float *__restrict dn = in_o + (y + k) * n_inner + xb;
                    if constexpr (Symmetric) {
                        for (std::ptrdiff_t i = 0; i < strip; ++i) {
                            acc[i] += hk * (up[i] + dn[i]);
                        }
                    } else {
                        for (std::ptrdiff_t i = 0; i < strip; ++i) {
                            acc[i] += hk * (dn[i] - up[i]);
                        }
                    }
                }

                for (std::ptrdiff_t i = 0; i < strip; ++i) {
                    out_row[xb + i] = acc[i];
                }
            }
        }

        // Border rows (axis prologue + epilogue), with mirror on y±k.
        auto run_border_row = [&](std::ptrdiff_t y) {
            float *__restrict out_row = out_o + y * n_inner;
            const float *__restrict center = in_o + y * n_inner;

            for (std::ptrdiff_t xb = 0; xb < n_inner; xb += kStripBlock) {
                const std::ptrdiff_t strip = std::min(kStripBlock, n_inner - xb);
                float acc[kStripBlock];

                if constexpr (Symmetric) {
                    const float h0 = h[0];
                    for (std::ptrdiff_t i = 0; i < strip; ++i) {
                        acc[i] = h0 * center[xb + i];
                    }
                } else {
                    for (std::ptrdiff_t i = 0; i < strip; ++i) {
                        acc[i] = 0.0f;
                    }
                }

                for (int k = 1; k <= R; ++k) {
                    const float hk = h[k];
                    const std::ptrdiff_t y_up = mirror_index(y - k, n_axis);
                    const std::ptrdiff_t y_dn = mirror_index(y + k, n_axis);
                    const float *__restrict up = in_o + y_up * n_inner + xb;
                    const float *__restrict dn = in_o + y_dn * n_inner + xb;
                    if constexpr (Symmetric) {
                        for (std::ptrdiff_t i = 0; i < strip; ++i) {
                            acc[i] += hk * (up[i] + dn[i]);
                        }
                    } else {
                        for (std::ptrdiff_t i = 0; i < strip; ++i) {
                            acc[i] += hk * (dn[i] - up[i]);
                        }
                    }
                }

                for (std::ptrdiff_t i = 0; i < strip; ++i) {
                    out_row[xb + i] = acc[i];
                }
            }
        };

        for (std::ptrdiff_t y = 0; y < prologue_end; ++y) run_border_row(y);
        for (std::ptrdiff_t y = epilogue_start; y < n_axis; ++y) run_border_row(y);
    }
}

// Runtime-radius fallback for the strided pass. Falls back to per-pixel
// indexing rather than the strip pattern; intended only for the rare case
// where the kernel radius exceeds the compile-time cap.
template <bool Symmetric>
void convolve_strided_runtime(
    const float *__restrict in,
    float *__restrict out,
    std::ptrdiff_t n_outer,
    std::ptrdiff_t n_axis,
    std::ptrdiff_t n_inner,
    int radius,
    const float *__restrict h
) {
    if (n_axis <= 0 || n_inner <= 0 || n_outer <= 0) return;

    const std::ptrdiff_t R = radius;
    const std::ptrdiff_t outer_stride = n_axis * n_inner;

    for (std::ptrdiff_t o = 0; o < n_outer; ++o) {
        const float *__restrict in_o = in + o * outer_stride;
        float *__restrict out_o = out + o * outer_stride;

        for (std::ptrdiff_t y = 0; y < n_axis; ++y) {
            float *__restrict out_row = out_o + y * n_inner;
            const float *__restrict center = in_o + y * n_inner;
            const bool needs_mirror = (y < R) || (y >= n_axis - R);

            for (std::ptrdiff_t xb = 0; xb < n_inner; xb += kStripBlock) {
                const std::ptrdiff_t strip = std::min(kStripBlock, n_inner - xb);
                float acc[kStripBlock];

                if constexpr (Symmetric) {
                    const float h0 = h[0];
                    for (std::ptrdiff_t i = 0; i < strip; ++i) acc[i] = h0 * center[xb + i];
                } else {
                    for (std::ptrdiff_t i = 0; i < strip; ++i) acc[i] = 0.0f;
                }

                for (int k = 1; k <= R; ++k) {
                    const float hk = h[k];
                    const std::ptrdiff_t y_up =
                        needs_mirror ? mirror_index(y - k, n_axis) : (y - k);
                    const std::ptrdiff_t y_dn =
                        needs_mirror ? mirror_index(y + k, n_axis) : (y + k);
                    const float *up = in_o + y_up * n_inner + xb;
                    const float *dn = in_o + y_dn * n_inner + xb;
                    if constexpr (Symmetric) {
                        for (std::ptrdiff_t i = 0; i < strip; ++i) {
                            acc[i] += hk * (up[i] + dn[i]);
                        }
                    } else {
                        for (std::ptrdiff_t i = 0; i < strip; ++i) {
                            acc[i] += hk * (dn[i] - up[i]);
                        }
                    }
                }

                for (std::ptrdiff_t i = 0; i < strip; ++i) out_row[xb + i] = acc[i];
            }
        }
    }
}

} // namespace detail

// Maximum compile-time-specialised kernel radius. Kernels with radius > this
// dispatch to the runtime-radius fallback. 12 covers sigma up to ~3.4 with the
// default window (3*sigma); larger sigma is supported but slower.
inline constexpr int kMaxSpecialisedRadius = 12;

// Convolve along the innermost (contiguous) axis. in and out must not alias.
inline void convolve_axis_x(
    const float *in,
    float *out,
    std::ptrdiff_t n_rows,
    std::ptrdiff_t n_cols,
    const Kernel1D &kernel
) {
    const int r = kernel.radius;
    const float *h = kernel.half_coefs.data();
    const bool sym = kernel.is_symmetric;

#define BIOIMAGE_FILTERS_DISPATCH_RADIUS_X(R)                                                      \
    case R:                                                                                        \
        if (sym) detail::convolve_x_radius<R, true>(in, out, n_rows, n_cols, h);                   \
        else detail::convolve_x_radius<R, false>(in, out, n_rows, n_cols, h);                      \
        return;

    switch (r) {
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_X(1)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_X(2)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_X(3)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_X(4)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_X(5)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_X(6)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_X(7)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_X(8)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_X(9)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_X(10)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_X(11)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_X(12)
        default:
            if (sym) detail::convolve_x_runtime<true>(in, out, n_rows, n_cols, r, h);
            else detail::convolve_x_runtime<false>(in, out, n_rows, n_cols, r, h);
            return;
    }
#undef BIOIMAGE_FILTERS_DISPATCH_RADIUS_X
}

// Convolve along a strided axis. Logical layout is (n_outer, n_axis, n_inner)
// in C-order; in and out must not alias.
inline void convolve_axis_strided(
    const float *in,
    float *out,
    std::ptrdiff_t n_outer,
    std::ptrdiff_t n_axis,
    std::ptrdiff_t n_inner,
    const Kernel1D &kernel
) {
    const int r = kernel.radius;
    const float *h = kernel.half_coefs.data();
    const bool sym = kernel.is_symmetric;

#define BIOIMAGE_FILTERS_DISPATCH_RADIUS_S(R)                                                      \
    case R:                                                                                        \
        if (sym)                                                                                   \
            detail::convolve_strided_radius<R, true>(in, out, n_outer, n_axis, n_inner, h);        \
        else                                                                                       \
            detail::convolve_strided_radius<R, false>(in, out, n_outer, n_axis, n_inner, h);       \
        return;

    switch (r) {
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_S(1)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_S(2)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_S(3)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_S(4)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_S(5)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_S(6)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_S(7)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_S(8)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_S(9)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_S(10)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_S(11)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_S(12)
        default:
            if (sym)
                detail::convolve_strided_runtime<true>(in, out, n_outer, n_axis, n_inner, r, h);
            else
                detail::convolve_strided_runtime<false>(in, out, n_outer, n_axis, n_inner, r, h);
            return;
    }
#undef BIOIMAGE_FILTERS_DISPATCH_RADIUS_S
}

} // namespace bioimage_cpp::filters
