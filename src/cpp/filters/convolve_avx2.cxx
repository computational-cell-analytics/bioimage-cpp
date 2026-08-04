#include "bioimage_cpp/filters/convolve.hxx"

#if !defined(BIOIMAGE_FILTERS_AVX2_DISPATCH)
#error "convolve_avx2.cxx requires BIOIMAGE_FILTERS_AVX2_DISPATCH"
#endif

#include <algorithm>
#include <immintrin.h>

namespace bioimage_cpp::filters::avx2 {
namespace {

template <int R, bool Symmetric>
void convolve_x_radius_avx2(
    const float *__restrict in,
    float *__restrict out,
    const std::ptrdiff_t n_rows,
    const std::ptrdiff_t n_cols,
    const float *__restrict half_coefs
) {
    if (n_cols <= 0 || n_rows <= 0) return;

    const std::ptrdiff_t prologue_end = std::min<std::ptrdiff_t>(R, n_cols);
    const std::ptrdiff_t epilogue_start =
        std::max<std::ptrdiff_t>(prologue_end, n_cols - R);
    __m256 h[R + 1];
    for (int k = 0; k <= R; ++k) {
        h[k] = _mm256_set1_ps(half_coefs[k]);
    }

    for (std::ptrdiff_t row = 0; row < n_rows; ++row) {
        const float *__restrict in_row = in + row * n_cols;
        float *__restrict out_row = out + row * n_cols;
        detail::convolve_x_border_range<R, Symmetric>(
            in_row, out_row, 0, prologue_end, n_cols, half_coefs
        );

        std::ptrdiff_t x = prologue_end;
        for (; x + 8 <= epilogue_start; x += 8) {
            __m256 acc;
            if constexpr (Symmetric) {
                acc = _mm256_mul_ps(_mm256_loadu_ps(in_row + x), h[0]);
            } else {
                acc = _mm256_setzero_ps();
            }
            for (int k = 1; k <= R; ++k) {
                const __m256 left = _mm256_loadu_ps(in_row + x - k);
                const __m256 right = _mm256_loadu_ps(in_row + x + k);
                const __m256 pair = Symmetric
                    ? _mm256_add_ps(left, right)
                    : _mm256_sub_ps(right, left);
                acc = _mm256_fmadd_ps(h[k], pair, acc);
            }
            _mm256_storeu_ps(out_row + x, acc);
        }
        detail::convolve_x_main_range<R, Symmetric>(
            in_row, out_row, x, epilogue_start, half_coefs
        );
        detail::convolve_x_border_range<R, Symmetric>(
            in_row, out_row, epilogue_start, n_cols, n_cols, half_coefs
        );
    }
}

template <bool Symmetric>
void convolve_x(
    const float *in,
    float *out,
    const std::ptrdiff_t n_rows,
    const std::ptrdiff_t n_cols,
    const int radius,
    const float *half_coefs
) {
#define BIOIMAGE_FILTERS_AVX2_RADIUS_X(R)                                                          \
    case R:                                                                                        \
        convolve_x_radius_avx2<R, Symmetric>(                                                      \
            in, out, n_rows, n_cols, half_coefs                                                    \
        );                                                                                         \
        return;

    switch (radius) {
        BIOIMAGE_FILTERS_AVX2_RADIUS_X(1)
        BIOIMAGE_FILTERS_AVX2_RADIUS_X(2)
        BIOIMAGE_FILTERS_AVX2_RADIUS_X(3)
        BIOIMAGE_FILTERS_AVX2_RADIUS_X(4)
        BIOIMAGE_FILTERS_AVX2_RADIUS_X(5)
        BIOIMAGE_FILTERS_AVX2_RADIUS_X(6)
        BIOIMAGE_FILTERS_AVX2_RADIUS_X(7)
        BIOIMAGE_FILTERS_AVX2_RADIUS_X(8)
        BIOIMAGE_FILTERS_AVX2_RADIUS_X(9)
        BIOIMAGE_FILTERS_AVX2_RADIUS_X(10)
        BIOIMAGE_FILTERS_AVX2_RADIUS_X(11)
        BIOIMAGE_FILTERS_AVX2_RADIUS_X(12)
        default:
            return;
    }
#undef BIOIMAGE_FILTERS_AVX2_RADIUS_X
}

template <int R, bool Symmetric>
void convolve_strided_radius_avx2(
    const float *__restrict in,
    float *__restrict out,
    const std::ptrdiff_t n_outer,
    const std::ptrdiff_t n_axis,
    const std::ptrdiff_t n_inner,
    const float *__restrict half_coefs
) {
    if (n_axis <= 0 || n_inner <= 0 || n_outer <= 0) return;

    const std::ptrdiff_t outer_stride = n_axis * n_inner;
    const std::ptrdiff_t prologue_end = std::min<std::ptrdiff_t>(R, n_axis);
    const std::ptrdiff_t epilogue_start =
        std::max<std::ptrdiff_t>(prologue_end, n_axis - R);
    __m256 h[R + 1];
    for (int k = 0; k <= R; ++k) {
        h[k] = _mm256_set1_ps(half_coefs[k]);
    }

    for (std::ptrdiff_t o = 0; o < n_outer; ++o) {
        const float *__restrict in_o = in + o * outer_stride;
        float *__restrict out_o = out + o * outer_stride;

        for (std::ptrdiff_t y = prologue_end; y < epilogue_start; ++y) {
            const float *__restrict center = in_o + y * n_inner;
            float *__restrict out_row = out_o + y * n_inner;
            std::ptrdiff_t i = 0;
            for (; i + 8 <= n_inner; i += 8) {
                __m256 acc;
                if constexpr (Symmetric) {
                    acc = _mm256_mul_ps(_mm256_loadu_ps(center + i), h[0]);
                } else {
                    acc = _mm256_setzero_ps();
                }
                for (int k = 1; k <= R; ++k) {
                    const __m256 up = _mm256_loadu_ps(
                        in_o + (y - k) * n_inner + i
                    );
                    const __m256 dn = _mm256_loadu_ps(
                        in_o + (y + k) * n_inner + i
                    );
                    const __m256 pair = Symmetric
                        ? _mm256_add_ps(up, dn)
                        : _mm256_sub_ps(dn, up);
                    acc = _mm256_fmadd_ps(h[k], pair, acc);
                }
                _mm256_storeu_ps(out_row + i, acc);
            }
            for (; i < n_inner; ++i) {
                float acc = Symmetric ? half_coefs[0] * center[i] : 0.0f;
                for (int k = 1; k <= R; ++k) {
                    const float up = in_o[(y - k) * n_inner + i];
                    const float dn = in_o[(y + k) * n_inner + i];
                    acc += half_coefs[k] * (Symmetric ? up + dn : dn - up);
                }
                out_row[i] = acc;
            }
        }

        for (std::ptrdiff_t y = 0; y < prologue_end; ++y) {
            detail::convolve_strided_border_row<R, Symmetric>(
                in_o, out_o, y, n_axis, n_inner, half_coefs
            );
        }
        for (std::ptrdiff_t y = epilogue_start; y < n_axis; ++y) {
            detail::convolve_strided_border_row<R, Symmetric>(
                in_o, out_o, y, n_axis, n_inner, half_coefs
            );
        }
    }
}

template <bool Symmetric>
void convolve_strided(
    const float *in,
    float *out,
    const std::ptrdiff_t n_outer,
    const std::ptrdiff_t n_axis,
    const std::ptrdiff_t n_inner,
    const int radius,
    const float *half_coefs
) {
#define BIOIMAGE_FILTERS_AVX2_RADIUS_S(R)                                                          \
    case R:                                                                                        \
        convolve_strided_radius_avx2<R, Symmetric>(                                                \
            in, out, n_outer, n_axis, n_inner, half_coefs                                          \
        );                                                                                         \
        return;

    switch (radius) {
        BIOIMAGE_FILTERS_AVX2_RADIUS_S(1)
        BIOIMAGE_FILTERS_AVX2_RADIUS_S(2)
        BIOIMAGE_FILTERS_AVX2_RADIUS_S(3)
        BIOIMAGE_FILTERS_AVX2_RADIUS_S(4)
        BIOIMAGE_FILTERS_AVX2_RADIUS_S(5)
        BIOIMAGE_FILTERS_AVX2_RADIUS_S(6)
        BIOIMAGE_FILTERS_AVX2_RADIUS_S(7)
        BIOIMAGE_FILTERS_AVX2_RADIUS_S(8)
        BIOIMAGE_FILTERS_AVX2_RADIUS_S(9)
        BIOIMAGE_FILTERS_AVX2_RADIUS_S(10)
        BIOIMAGE_FILTERS_AVX2_RADIUS_S(11)
        BIOIMAGE_FILTERS_AVX2_RADIUS_S(12)
        default:
            return;
    }
#undef BIOIMAGE_FILTERS_AVX2_RADIUS_S
}

} // namespace

void convolve_axis_x(
    const float *in,
    float *out,
    const std::ptrdiff_t n_rows,
    const std::ptrdiff_t n_cols,
    const int radius,
    const bool symmetric,
    const float *half_coefs
) {
    if (symmetric) {
        convolve_x<true>(in, out, n_rows, n_cols, radius, half_coefs);
    } else {
        convolve_x<false>(in, out, n_rows, n_cols, radius, half_coefs);
    }
}

void convolve_axis_strided(
    const float *in,
    float *out,
    const std::ptrdiff_t n_outer,
    const std::ptrdiff_t n_axis,
    const std::ptrdiff_t n_inner,
    const int radius,
    const bool symmetric,
    const float *half_coefs
) {
    if (symmetric) {
        convolve_strided<true>(
            in, out, n_outer, n_axis, n_inner, radius, half_coefs
        );
    } else {
        convolve_strided<false>(
            in, out, n_outer, n_axis, n_inner, radius, half_coefs
        );
    }
}

template <std::size_t D, int R>
void convolve_outer_products_radius_avx2(
    const std::array<const float *, D> &gradients,
    const std::array<float *, detail::kSymmetricComponents<D>> &out,
    const std::ptrdiff_t n_outer,
    const std::ptrdiff_t n_axis,
    const std::ptrdiff_t n_inner,
    const float *half_coefs
) {
    static_assert(D == 2 || D == 3);
    if (n_axis <= 0 || n_inner <= 0 || n_outer <= 0) return;

    constexpr std::size_t C = detail::kSymmetricComponents<D>;
    const std::ptrdiff_t outer_stride = n_axis * n_inner;
    __m256 h[R + 1];
    for (int k = 0; k <= R; ++k) {
        h[k] = _mm256_set1_ps(half_coefs[k]);
    }

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
            std::ptrdiff_t i = 0;
            for (; i + 8 <= n_inner; i += 8) {
                const std::ptrdiff_t center_index = y * n_inner + i;
                const __m256 g0 = _mm256_loadu_ps(in_o[0] + center_index);
                const __m256 g1 = _mm256_loadu_ps(in_o[1] + center_index);
                __m256 acc0 = _mm256_mul_ps(h[0], _mm256_mul_ps(g0, g0));
                __m256 acc1 = _mm256_mul_ps(h[0], _mm256_mul_ps(g0, g1));
                __m256 acc2;
                __m256 acc3;
                __m256 acc4;
                __m256 acc5;
                if constexpr (D == 2) {
                    acc2 = _mm256_mul_ps(h[0], _mm256_mul_ps(g1, g1));
                } else {
                    const __m256 g2 = _mm256_loadu_ps(in_o[2] + center_index);
                    acc2 = _mm256_mul_ps(h[0], _mm256_mul_ps(g0, g2));
                    acc3 = _mm256_mul_ps(h[0], _mm256_mul_ps(g1, g1));
                    acc4 = _mm256_mul_ps(h[0], _mm256_mul_ps(g1, g2));
                    acc5 = _mm256_mul_ps(h[0], _mm256_mul_ps(g2, g2));
                }

                for (int k = 1; k <= R; ++k) {
                    const std::ptrdiff_t y_up = detail::mirror_index(y - k, n_axis);
                    const std::ptrdiff_t y_dn = detail::mirror_index(y + k, n_axis);
                    const std::ptrdiff_t up_index = y_up * n_inner + i;
                    const std::ptrdiff_t dn_index = y_dn * n_inner + i;
                    const __m256 g0u = _mm256_loadu_ps(in_o[0] + up_index);
                    const __m256 g0d = _mm256_loadu_ps(in_o[0] + dn_index);
                    const __m256 g1u = _mm256_loadu_ps(in_o[1] + up_index);
                    const __m256 g1d = _mm256_loadu_ps(in_o[1] + dn_index);
                    acc0 = _mm256_fmadd_ps(
                        h[k],
                        _mm256_add_ps(
                            _mm256_mul_ps(g0u, g0u), _mm256_mul_ps(g0d, g0d)
                        ),
                        acc0
                    );
                    acc1 = _mm256_fmadd_ps(
                        h[k],
                        _mm256_add_ps(
                            _mm256_mul_ps(g0u, g1u), _mm256_mul_ps(g0d, g1d)
                        ),
                        acc1
                    );
                    if constexpr (D == 2) {
                        acc2 = _mm256_fmadd_ps(
                            h[k],
                            _mm256_add_ps(
                                _mm256_mul_ps(g1u, g1u), _mm256_mul_ps(g1d, g1d)
                            ),
                            acc2
                        );
                    } else {
                        const __m256 g2u = _mm256_loadu_ps(in_o[2] + up_index);
                        const __m256 g2d = _mm256_loadu_ps(in_o[2] + dn_index);
                        acc2 = _mm256_fmadd_ps(
                            h[k],
                            _mm256_add_ps(
                                _mm256_mul_ps(g0u, g2u), _mm256_mul_ps(g0d, g2d)
                            ),
                            acc2
                        );
                        acc3 = _mm256_fmadd_ps(
                            h[k],
                            _mm256_add_ps(
                                _mm256_mul_ps(g1u, g1u), _mm256_mul_ps(g1d, g1d)
                            ),
                            acc3
                        );
                        acc4 = _mm256_fmadd_ps(
                            h[k],
                            _mm256_add_ps(
                                _mm256_mul_ps(g1u, g2u), _mm256_mul_ps(g1d, g2d)
                            ),
                            acc4
                        );
                        acc5 = _mm256_fmadd_ps(
                            h[k],
                            _mm256_add_ps(
                                _mm256_mul_ps(g2u, g2u), _mm256_mul_ps(g2d, g2d)
                            ),
                            acc5
                        );
                    }
                }

                _mm256_storeu_ps(out_o[0] + center_index, acc0);
                _mm256_storeu_ps(out_o[1] + center_index, acc1);
                _mm256_storeu_ps(out_o[2] + center_index, acc2);
                if constexpr (D == 3) {
                    _mm256_storeu_ps(out_o[3] + center_index, acc3);
                    _mm256_storeu_ps(out_o[4] + center_index, acc4);
                    _mm256_storeu_ps(out_o[5] + center_index, acc5);
                }
            }

            for (; i < n_inner; ++i) {
                const std::ptrdiff_t center_index = y * n_inner + i;
                const float g0 = in_o[0][center_index];
                const float g1 = in_o[1][center_index];
                float acc[C];
                acc[0] = half_coefs[0] * g0 * g0;
                acc[1] = half_coefs[0] * g0 * g1;
                if constexpr (D == 2) {
                    acc[2] = half_coefs[0] * g1 * g1;
                } else {
                    const float g2 = in_o[2][center_index];
                    acc[2] = half_coefs[0] * g0 * g2;
                    acc[3] = half_coefs[0] * g1 * g1;
                    acc[4] = half_coefs[0] * g1 * g2;
                    acc[5] = half_coefs[0] * g2 * g2;
                }

                for (int k = 1; k <= R; ++k) {
                    const std::ptrdiff_t y_up = detail::mirror_index(y - k, n_axis);
                    const std::ptrdiff_t y_dn = detail::mirror_index(y + k, n_axis);
                    const std::ptrdiff_t up_index = y_up * n_inner + i;
                    const std::ptrdiff_t dn_index = y_dn * n_inner + i;
                    const float g0u = in_o[0][up_index];
                    const float g0d = in_o[0][dn_index];
                    const float g1u = in_o[1][up_index];
                    const float g1d = in_o[1][dn_index];
                    const float hk = half_coefs[k];
                    acc[0] += hk * (g0u * g0u + g0d * g0d);
                    acc[1] += hk * (g0u * g1u + g0d * g1d);
                    if constexpr (D == 2) {
                        acc[2] += hk * (g1u * g1u + g1d * g1d);
                    } else {
                        const float g2u = in_o[2][up_index];
                        const float g2d = in_o[2][dn_index];
                        acc[2] += hk * (g0u * g2u + g0d * g2d);
                        acc[3] += hk * (g1u * g1u + g1d * g1d);
                        acc[4] += hk * (g1u * g2u + g1d * g2d);
                        acc[5] += hk * (g2u * g2u + g2d * g2d);
                    }
                }
                for (std::size_t c = 0; c < C; ++c) {
                    out_o[c][center_index] = acc[c];
                }
            }
        }
    }
}

template <std::size_t D>
void convolve_outer_products(
    const std::array<const float *, D> &gradients,
    const std::array<float *, detail::kSymmetricComponents<D>> &out,
    const std::ptrdiff_t n_outer,
    const std::ptrdiff_t n_axis,
    const std::ptrdiff_t n_inner,
    const int radius,
    const float *half_coefs
) {
#define BIOIMAGE_FILTERS_AVX2_RADIUS_OP(R)                                                         \
    case R:                                                                                        \
        convolve_outer_products_radius_avx2<D, R>(                                                 \
            gradients, out, n_outer, n_axis, n_inner, half_coefs                                  \
        );                                                                                         \
        return;

    switch (radius) {
        BIOIMAGE_FILTERS_AVX2_RADIUS_OP(1)
        BIOIMAGE_FILTERS_AVX2_RADIUS_OP(2)
        BIOIMAGE_FILTERS_AVX2_RADIUS_OP(3)
        BIOIMAGE_FILTERS_AVX2_RADIUS_OP(4)
        BIOIMAGE_FILTERS_AVX2_RADIUS_OP(5)
        BIOIMAGE_FILTERS_AVX2_RADIUS_OP(6)
        BIOIMAGE_FILTERS_AVX2_RADIUS_OP(7)
        BIOIMAGE_FILTERS_AVX2_RADIUS_OP(8)
        BIOIMAGE_FILTERS_AVX2_RADIUS_OP(9)
        BIOIMAGE_FILTERS_AVX2_RADIUS_OP(10)
        BIOIMAGE_FILTERS_AVX2_RADIUS_OP(11)
        BIOIMAGE_FILTERS_AVX2_RADIUS_OP(12)
        default:
            return;
    }
#undef BIOIMAGE_FILTERS_AVX2_RADIUS_OP
}

void convolve_outer_products_2d(
    const std::array<const float *, 2> &gradients,
    const std::array<float *, 3> &out,
    const std::ptrdiff_t n_outer,
    const std::ptrdiff_t n_axis,
    const std::ptrdiff_t n_inner,
    const int radius,
    const float *half_coefs
) {
    convolve_outer_products<2>(
        gradients, out, n_outer, n_axis, n_inner, radius, half_coefs
    );
}

void convolve_outer_products_3d(
    const std::array<const float *, 3> &gradients,
    const std::array<float *, 6> &out,
    const std::ptrdiff_t n_outer,
    const std::ptrdiff_t n_axis,
    const std::ptrdiff_t n_inner,
    const int radius,
    const float *half_coefs
) {
    convolve_outer_products<3>(
        gradients, out, n_outer, n_axis, n_inner, radius, half_coefs
    );
}

} // namespace bioimage_cpp::filters::avx2
