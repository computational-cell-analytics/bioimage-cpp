#pragma once

#include "bioimage_cpp/filters/kernel.hxx"

#if defined(BIOIMAGE_FILTERS_AVX2_DISPATCH)
#include "bioimage_cpp/filters/dispatch.hxx"
#endif

#include <algorithm>
#include <array>
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

template <int R, bool Symmetric>
inline void convolve_x_border_range(
    const float *__restrict in_row,
    float *__restrict out_row,
    const std::ptrdiff_t begin,
    const std::ptrdiff_t end,
    const std::ptrdiff_t n_cols,
    const float *__restrict h
) {
    for (std::ptrdiff_t x = begin; x < end; ++x) {
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

template <int R, bool Symmetric>
inline void convolve_x_main_range(
    const float *__restrict in_row,
    float *__restrict out_row,
    const std::ptrdiff_t begin,
    const std::ptrdiff_t end,
    const float *__restrict h
) {
    if constexpr (Symmetric) {
        for (std::ptrdiff_t x = begin; x < end; ++x) {
            float acc = h[0] * in_row[x];
            for (int k = 1; k <= R; ++k) {
                acc += h[k] * (in_row[x + k] + in_row[x - k]);
            }
            out_row[x] = acc;
        }
    } else {
        for (std::ptrdiff_t x = begin; x < end; ++x) {
            float acc = 0.0f;
            for (int k = 1; k <= R; ++k) {
                acc += h[k] * (in_row[x + k] - in_row[x - k]);
            }
            out_row[x] = acc;
        }
    }
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

        convolve_x_border_range<R, Symmetric>(
            in_row, out_row, 0, prologue_end, n_cols, h
        );
        convolve_x_main_range<R, Symmetric>(
            in_row, out_row, prologue_end, epilogue_start, h
        );
        convolve_x_border_range<R, Symmetric>(
            in_row, out_row, epilogue_start, n_cols, n_cols, h
        );
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

template <int R, bool Symmetric>
inline void convolve_strided_border_row(
    const float *__restrict in_o,
    float *__restrict out_o,
    const std::ptrdiff_t y,
    const std::ptrdiff_t n_axis,
    const std::ptrdiff_t n_inner,
    const float *__restrict h
) {
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
}

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

        for (std::ptrdiff_t y = 0; y < prologue_end; ++y) {
            convolve_strided_border_row<R, Symmetric>(
                in_o, out_o, y, n_axis, n_inner, h
            );
        }
        for (std::ptrdiff_t y = epilogue_start; y < n_axis; ++y) {
            convolve_strided_border_row<R, Symmetric>(
                in_o, out_o, y, n_axis, n_inner, h
            );
        }
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

template <std::size_t D>
inline constexpr std::size_t kSymmetricComponents = D * (D + 1) / 2;

template <std::size_t D, int R>
void convolve_strided_outer_products_radius(
    const std::array<const float *, D> &gradients,
    const std::array<float *, kSymmetricComponents<D>> &out,
    std::ptrdiff_t n_outer,
    std::ptrdiff_t n_axis,
    std::ptrdiff_t n_inner,
    const float *__restrict h
) {
    static_assert(D == 2 || D == 3);
    if (n_axis <= 0 || n_inner <= 0 || n_outer <= 0) return;

    constexpr std::size_t C = kSymmetricComponents<D>;
    const std::ptrdiff_t outer_stride = n_axis * n_inner;
    const std::ptrdiff_t prologue_end = std::min<std::ptrdiff_t>(R, n_axis);
    const std::ptrdiff_t epilogue_start =
        std::max<std::ptrdiff_t>(prologue_end, n_axis - R);

    for (std::ptrdiff_t o = 0; o < n_outer; ++o) {
        std::array<const float *, D> in_o{};
        std::array<float *, C> out_o{};
        for (std::size_t d = 0; d < D; ++d) {
            in_o[d] = gradients[d] + o * outer_stride;
        }
        for (std::size_t c = 0; c < C; ++c) {
            out_o[c] = out[c] + o * outer_stride;
        }

        auto run_row = [&]<bool Mirror>(const std::ptrdiff_t y) {
            for (std::ptrdiff_t xb = 0; xb < n_inner; xb += kStripBlock) {
                const std::ptrdiff_t strip = std::min(kStripBlock, n_inner - xb);
                float acc[C][kStripBlock];

                for (std::ptrdiff_t i = 0; i < strip; ++i) {
                    const std::ptrdiff_t index = y * n_inner + xb + i;
                    const float g0 = in_o[0][index];
                    const float g1 = in_o[1][index];
                    acc[0][i] = h[0] * g0 * g0;
                    acc[1][i] = h[0] * g0 * g1;
                    if constexpr (D == 2) {
                        acc[2][i] = h[0] * g1 * g1;
                    } else {
                        const float g2 = in_o[2][index];
                        acc[2][i] = h[0] * g0 * g2;
                        acc[3][i] = h[0] * g1 * g1;
                        acc[4][i] = h[0] * g1 * g2;
                        acc[5][i] = h[0] * g2 * g2;
                    }
                }

                for (int k = 1; k <= R; ++k) {
                    const float hk = h[k];
                    const std::ptrdiff_t y_up =
                        Mirror ? mirror_index(y - k, n_axis) : (y - k);
                    const std::ptrdiff_t y_dn =
                        Mirror ? mirror_index(y + k, n_axis) : (y + k);
                    const std::ptrdiff_t up_base = y_up * n_inner + xb;
                    const std::ptrdiff_t dn_base = y_dn * n_inner + xb;

                    for (std::ptrdiff_t i = 0; i < strip; ++i) {
                        const float g0u = in_o[0][up_base + i];
                        const float g0d = in_o[0][dn_base + i];
                        const float g1u = in_o[1][up_base + i];
                        const float g1d = in_o[1][dn_base + i];
                        acc[0][i] += hk * (g0u * g0u + g0d * g0d);
                        acc[1][i] += hk * (g0u * g1u + g0d * g1d);
                        if constexpr (D == 2) {
                            acc[2][i] += hk * (g1u * g1u + g1d * g1d);
                        } else {
                            const float g2u = in_o[2][up_base + i];
                            const float g2d = in_o[2][dn_base + i];
                            acc[2][i] += hk * (g0u * g2u + g0d * g2d);
                            acc[3][i] += hk * (g1u * g1u + g1d * g1d);
                            acc[4][i] += hk * (g1u * g2u + g1d * g2d);
                            acc[5][i] += hk * (g2u * g2u + g2d * g2d);
                        }
                    }
                }

                for (std::size_t c = 0; c < C; ++c) {
                    float *__restrict out_row = out_o[c] + y * n_inner + xb;
                    for (std::ptrdiff_t i = 0; i < strip; ++i) {
                        out_row[i] = acc[c][i];
                    }
                }
            }
        };

        for (std::ptrdiff_t y = prologue_end; y < epilogue_start; ++y) {
            run_row.template operator()<false>(y);
        }
        for (std::ptrdiff_t y = 0; y < prologue_end; ++y) {
            run_row.template operator()<true>(y);
        }
        for (std::ptrdiff_t y = epilogue_start; y < n_axis; ++y) {
            run_row.template operator()<true>(y);
        }
    }
}

template <std::size_t D>
void convolve_strided_outer_products_runtime(
    const std::array<const float *, D> &gradients,
    const std::array<float *, kSymmetricComponents<D>> &out,
    std::ptrdiff_t n_outer,
    std::ptrdiff_t n_axis,
    std::ptrdiff_t n_inner,
    int radius,
    const float *__restrict h
) {
    static_assert(D == 2 || D == 3);
    if (n_axis <= 0 || n_inner <= 0 || n_outer <= 0) return;

    constexpr std::size_t C = kSymmetricComponents<D>;
    const std::ptrdiff_t outer_stride = n_axis * n_inner;
    const std::ptrdiff_t R = radius;

    for (std::ptrdiff_t o = 0; o < n_outer; ++o) {
        std::array<const float *, D> in_o{};
        std::array<float *, C> out_o{};
        for (std::size_t d = 0; d < D; ++d) {
            in_o[d] = gradients[d] + o * outer_stride;
        }
        for (std::size_t c = 0; c < C; ++c) {
            out_o[c] = out[c] + o * outer_stride;
        }

        for (std::ptrdiff_t y = 0; y < n_axis; ++y) {
            const bool needs_mirror = y < R || y >= n_axis - R;
            for (std::ptrdiff_t xb = 0; xb < n_inner; xb += kStripBlock) {
                const std::ptrdiff_t strip = std::min(kStripBlock, n_inner - xb);
                float acc[C][kStripBlock];

                for (std::ptrdiff_t i = 0; i < strip; ++i) {
                    const std::ptrdiff_t index = y * n_inner + xb + i;
                    const float g0 = in_o[0][index];
                    const float g1 = in_o[1][index];
                    acc[0][i] = h[0] * g0 * g0;
                    acc[1][i] = h[0] * g0 * g1;
                    if constexpr (D == 2) {
                        acc[2][i] = h[0] * g1 * g1;
                    } else {
                        const float g2 = in_o[2][index];
                        acc[2][i] = h[0] * g0 * g2;
                        acc[3][i] = h[0] * g1 * g1;
                        acc[4][i] = h[0] * g1 * g2;
                        acc[5][i] = h[0] * g2 * g2;
                    }
                }

                for (int k = 1; k <= R; ++k) {
                    const float hk = h[k];
                    const std::ptrdiff_t y_up =
                        needs_mirror ? mirror_index(y - k, n_axis) : (y - k);
                    const std::ptrdiff_t y_dn =
                        needs_mirror ? mirror_index(y + k, n_axis) : (y + k);
                    const std::ptrdiff_t up_base = y_up * n_inner + xb;
                    const std::ptrdiff_t dn_base = y_dn * n_inner + xb;

                    for (std::ptrdiff_t i = 0; i < strip; ++i) {
                        const float g0u = in_o[0][up_base + i];
                        const float g0d = in_o[0][dn_base + i];
                        const float g1u = in_o[1][up_base + i];
                        const float g1d = in_o[1][dn_base + i];
                        acc[0][i] += hk * (g0u * g0u + g0d * g0d);
                        acc[1][i] += hk * (g0u * g1u + g0d * g1d);
                        if constexpr (D == 2) {
                            acc[2][i] += hk * (g1u * g1u + g1d * g1d);
                        } else {
                            const float g2u = in_o[2][up_base + i];
                            const float g2d = in_o[2][dn_base + i];
                            acc[2][i] += hk * (g0u * g2u + g0d * g2d);
                            acc[3][i] += hk * (g1u * g1u + g1d * g1d);
                            acc[4][i] += hk * (g1u * g2u + g1d * g2d);
                            acc[5][i] += hk * (g2u * g2u + g2d * g2d);
                        }
                    }
                }

                for (std::size_t c = 0; c < C; ++c) {
                    float *__restrict out_row = out_o[c] + y * n_inner + xb;
                    for (std::ptrdiff_t i = 0; i < strip; ++i) {
                        out_row[i] = acc[c][i];
                    }
                }
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

#if defined(BIOIMAGE_FILTERS_AVX2_DISPATCH)
    if (dispatch::try_convolve_axis_x(in, out, n_rows, n_cols, r, sym, h)) {
        return;
    }
#endif

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

#if defined(BIOIMAGE_FILTERS_AVX2_DISPATCH)
    if (dispatch::try_convolve_axis_strided(
            in, out, n_outer, n_axis, n_inner, r, sym, h
        )) {
        return;
    }
#endif

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

template <std::size_t D>
inline void convolve_axis_strided_outer_products(
    const std::array<const float *, D> &gradients,
    const std::array<float *, detail::kSymmetricComponents<D>> &out,
    std::ptrdiff_t n_outer,
    std::ptrdiff_t n_axis,
    std::ptrdiff_t n_inner,
    const Kernel1D &kernel
) {
    const int r = kernel.radius;
    const float *h = kernel.half_coefs.data();

#if defined(BIOIMAGE_FILTERS_AVX2_DISPATCH)
    if constexpr (D == 2) {
        if (dispatch::try_convolve_outer_products_2d(
                gradients, out, n_outer, n_axis, n_inner, r, h
            )) {
            return;
        }
    } else {
        if (dispatch::try_convolve_outer_products_3d(
                gradients, out, n_outer, n_axis, n_inner, r, h
            )) {
            return;
        }
    }
#endif

#define BIOIMAGE_FILTERS_DISPATCH_RADIUS_OP(R)                                                    \
    case R:                                                                                        \
        detail::convolve_strided_outer_products_radius<D, R>(                                     \
            gradients, out, n_outer, n_axis, n_inner, h                                           \
        );                                                                                         \
        return;

    switch (r) {
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_OP(1)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_OP(2)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_OP(3)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_OP(4)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_OP(5)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_OP(6)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_OP(7)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_OP(8)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_OP(9)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_OP(10)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_OP(11)
        BIOIMAGE_FILTERS_DISPATCH_RADIUS_OP(12)
        default:
            detail::convolve_strided_outer_products_runtime<D>(
                gradients, out, n_outer, n_axis, n_inner, r, h
            );
            return;
    }
#undef BIOIMAGE_FILTERS_DISPATCH_RADIUS_OP
}

inline const char *convolution_backend() {
#if defined(BIOIMAGE_FILTERS_AVX2_DISPATCH)
    return dispatch::convolution_backend();
#else
    return "scalar";
#endif
}

} // namespace bioimage_cpp::filters
