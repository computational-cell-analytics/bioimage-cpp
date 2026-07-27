#include "bioimage_cpp/filters/eigenvalues.hxx"

#if !defined(BIOIMAGE_FILTERS_AVX2_DISPATCH)
#error "eigenvalues_avx2.cxx requires BIOIMAGE_FILTERS_AVX2_DISPATCH"
#endif

#include <cstddef>
#include <immintrin.h>
#include <limits>

namespace bioimage_cpp::filters::avx2 {
namespace {

inline __m256 abs_ps(const __m256 value) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    return _mm256_andnot_ps(sign, value);
}

inline __m256 approximate_acos(const __m256 value) {
    constexpr float coefficients[] = {
        1.4866664409637451f,
        -0.07770606875419617f,
        0.005770874209702015f,
        -0.0005768424598500133f,
        0.00006643808592343703f,
        -0.000008315640116052236f,
        0.0000010878551393034286f,
        -0.00000014935945102934056f,
    };

    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 t = abs_ps(value);
    const __m256 z = _mm256_sub_ps(_mm256_add_ps(t, t), one);
    const __m256 two_z = _mm256_add_ps(z, z);

    __m256 b1 = _mm256_setzero_ps();
    __m256 b2 = _mm256_setzero_ps();
    for (int k = 7; k >= 1; --k) {
        const __m256 ck = _mm256_set1_ps(coefficients[k]);
        const __m256 b0 = _mm256_fmadd_ps(two_z, b1, _mm256_sub_ps(ck, b2));
        b2 = b1;
        b1 = b0;
    }
    const __m256 q = _mm256_fmadd_ps(
        z, b1, _mm256_sub_ps(_mm256_set1_ps(coefficients[0]), b2)
    );
    const __m256 base = _mm256_mul_ps(
        _mm256_sqrt_ps(_mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(one, t))),
        q
    );
    const __m256 reflected = _mm256_sub_ps(_mm256_set1_ps(3.14159265358979323846f), base);
    const __m256 negative = _mm256_cmp_ps(value, _mm256_setzero_ps(), _CMP_LT_OQ);
    return _mm256_blendv_ps(base, reflected, negative);
}

inline __m256 approximate_cos_phi(const __m256 phi) {
    constexpr float coefficients[] = {
        1.0f,
        -0.4999999701976776f,
        0.041666433215141296f,
        -0.0013882536441087723f,
        0.000024095119442790747f,
    };

    const __m256 u = _mm256_mul_ps(phi, phi);
    __m256 result = _mm256_set1_ps(coefficients[4]);
    result = _mm256_fmadd_ps(result, u, _mm256_set1_ps(coefficients[3]));
    result = _mm256_fmadd_ps(result, u, _mm256_set1_ps(coefficients[2]));
    result = _mm256_fmadd_ps(result, u, _mm256_set1_ps(coefficients[1]));
    return _mm256_fmadd_ps(result, u, _mm256_set1_ps(coefficients[0]));
}

inline __m256 approximate_sin_phi(const __m256 phi) {
    constexpr float coefficients[] = {
        1.0f,
        -0.1666666567325592f,
        0.008333305828273296f,
        -0.0001983467664103955f,
        0.0000026878346943703946f,
    };

    const __m256 u = _mm256_mul_ps(phi, phi);
    __m256 sinc = _mm256_set1_ps(coefficients[4]);
    sinc = _mm256_fmadd_ps(sinc, u, _mm256_set1_ps(coefficients[3]));
    sinc = _mm256_fmadd_ps(sinc, u, _mm256_set1_ps(coefficients[2]));
    sinc = _mm256_fmadd_ps(sinc, u, _mm256_set1_ps(coefficients[1]));
    sinc = _mm256_fmadd_ps(sinc, u, _mm256_set1_ps(coefficients[0]));
    return _mm256_mul_ps(phi, sinc);
}

inline void store_four_triples(
    __m128 e0,
    __m128 e1,
    __m128 e2,
    float *out
) {
    __m128 zero = _mm_setzero_ps();
    _MM_TRANSPOSE4_PS(e0, e1, e2, zero);

    const __m128 first_next = _mm_shuffle_ps(e1, e1, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 out0 = _mm_blend_ps(e0, first_next, 0b1000);
    const __m128 out1 = _mm_shuffle_ps(e1, e2, _MM_SHUFFLE(1, 0, 2, 1));
    const __m128 out2_candidate =
        _mm_shuffle_ps(e2, zero, _MM_SHUFFLE(2, 1, 0, 2));
    const __m128 fourth_first =
        _mm_shuffle_ps(zero, zero, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 out2 = _mm_blend_ps(out2_candidate, fourth_first, 0b0010);

    _mm_storeu_ps(out, out0);
    _mm_storeu_ps(out + 4, out1);
    _mm_storeu_ps(out + 8, out2);
}

inline void store_eight_triples(
    const __m256 e0,
    const __m256 e1,
    const __m256 e2,
    float *out
) {
    store_four_triples(
        _mm256_castps256_ps128(e0),
        _mm256_castps256_ps128(e1),
        _mm256_castps256_ps128(e2),
        out
    );
    store_four_triples(
        _mm256_extractf128_ps(e0, 1),
        _mm256_extractf128_ps(e1, 1),
        _mm256_extractf128_ps(e2, 1),
        out + 12
    );
}

} // namespace

void ev3_symmetric_descending_interleaved(
    const float *__restrict a00,
    const float *__restrict a01,
    const float *__restrict a02,
    const float *__restrict a11,
    const float *__restrict a12,
    const float *__restrict a22,
    float *__restrict out,
    const std::ptrdiff_t n
) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 one_third = _mm256_set1_ps(1.0f / 3.0f);
    const __m256 one_sixth = _mm256_set1_ps(1.0f / 6.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 minus_one = _mm256_set1_ps(-1.0f);
    const __m256 minus_half = _mm256_set1_ps(-0.5f);
    const __m256 minus_sqrt_three_over_two =
        _mm256_set1_ps(-0.8660254037844386f);

    const std::ptrdiff_t vector_end = n - n % 8;
    std::ptrdiff_t i = 0;
    for (; i < vector_end; i += 8) {
        __m256 v00 = _mm256_loadu_ps(a00 + i);
        __m256 v01 = _mm256_loadu_ps(a01 + i);
        __m256 v02 = _mm256_loadu_ps(a02 + i);
        __m256 v11 = _mm256_loadu_ps(a11 + i);
        __m256 v12 = _mm256_loadu_ps(a12 + i);
        __m256 v22 = _mm256_loadu_ps(a22 + i);

        __m256 scale = abs_ps(v00);
        scale = _mm256_max_ps(scale, abs_ps(v01));
        scale = _mm256_max_ps(scale, abs_ps(v02));
        scale = _mm256_max_ps(scale, abs_ps(v11));
        scale = _mm256_max_ps(scale, abs_ps(v12));
        scale = _mm256_max_ps(scale, abs_ps(v22));

        const __m256 zero_scale = _mm256_cmp_ps(scale, zero, _CMP_EQ_OQ);
        const __m256 positive_scale = _mm256_cmp_ps(scale, zero, _CMP_GT_OQ);
        const __m256 subnormal_scale = _mm256_and_ps(
            positive_scale,
            _mm256_cmp_ps(
                scale,
                _mm256_set1_ps(std::numeric_limits<float>::min()),
                _CMP_LT_OQ
            )
        );
        if (_mm256_movemask_ps(subnormal_scale) != 0) {
            for (std::ptrdiff_t j = i; j < i + 8; ++j) {
                detail::ev3_one_descending(
                    a00[j], a01[j], a02[j], a11[j], a12[j], a22[j],
                    out[3 * j], out[3 * j + 1], out[3 * j + 2]
                );
            }
            continue;
        }
        const __m256 safe_scale = _mm256_blendv_ps(scale, one, zero_scale);
        const __m256 inv_scale = _mm256_div_ps(one, safe_scale);
        v00 = _mm256_mul_ps(v00, inv_scale);
        v01 = _mm256_mul_ps(v01, inv_scale);
        v02 = _mm256_mul_ps(v02, inv_scale);
        v11 = _mm256_mul_ps(v11, inv_scale);
        v12 = _mm256_mul_ps(v12, inv_scale);
        v22 = _mm256_mul_ps(v22, inv_scale);

        const __m256 mean = _mm256_mul_ps(
            _mm256_add_ps(_mm256_add_ps(v00, v11), v22), one_third
        );
        const __m256 b00 = _mm256_sub_ps(v00, mean);
        const __m256 b11 = _mm256_sub_ps(v11, mean);
        const __m256 b22 = _mm256_sub_ps(v22, mean);

        __m256 trace_b2 = _mm256_mul_ps(b00, b00);
        trace_b2 = _mm256_fmadd_ps(b11, b11, trace_b2);
        trace_b2 = _mm256_fmadd_ps(b22, b22, trace_b2);
        __m256 off_diagonal = _mm256_mul_ps(v01, v01);
        off_diagonal = _mm256_fmadd_ps(v02, v02, off_diagonal);
        off_diagonal = _mm256_fmadd_ps(v12, v12, off_diagonal);
        trace_b2 = _mm256_fmadd_ps(two, off_diagonal, trace_b2);
        const __m256 p2 = _mm256_mul_ps(trace_b2, one_sixth);

        const __m256 valid_p2 = _mm256_cmp_ps(p2, zero, _CMP_GT_OQ);
        const __m256 safe_p2 = _mm256_blendv_ps(one, p2, valid_p2);
        const __m256 p = _mm256_sqrt_ps(safe_p2);

        const __m256 minor0 = _mm256_fnmadd_ps(v12, v12, _mm256_mul_ps(b11, b22));
        const __m256 minor1 = _mm256_fnmadd_ps(v12, v02, _mm256_mul_ps(v01, b22));
        const __m256 minor2 = _mm256_fnmadd_ps(b11, v02, _mm256_mul_ps(v01, v12));
        __m256 det_b = _mm256_mul_ps(b00, minor0);
        det_b = _mm256_fnmadd_ps(v01, minor1, det_b);
        det_b = _mm256_fmadd_ps(v02, minor2, det_b);

        const __m256 denominator = _mm256_mul_ps(two, _mm256_mul_ps(safe_p2, p));
        const __m256 valid_denominator =
            _mm256_cmp_ps(denominator, zero, _CMP_GT_OQ);
        const __m256 valid = _mm256_and_ps(valid_p2, valid_denominator);
        const __m256 safe_denominator =
            _mm256_blendv_ps(one, denominator, valid);
        __m256 r = _mm256_div_ps(det_b, safe_denominator);
        r = _mm256_min_ps(one, _mm256_max_ps(minus_one, r));

        const __m256 phi = _mm256_mul_ps(approximate_acos(r), one_third);
        const __m256 cos_phi = approximate_cos_phi(phi);
        const __m256 sin_phi = approximate_sin_phi(phi);
        const __m256 two_p = _mm256_add_ps(p, p);

        __m256 root_large = _mm256_fmadd_ps(two_p, cos_phi, mean);
        const __m256 small_angle = _mm256_fmadd_ps(
            minus_sqrt_three_over_two, sin_phi,
            _mm256_mul_ps(minus_half, cos_phi)
        );
        __m256 root_small = _mm256_fmadd_ps(two_p, small_angle, mean);
        __m256 root_mid = _mm256_sub_ps(
            _mm256_mul_ps(_mm256_set1_ps(3.0f), mean),
            _mm256_add_ps(root_large, root_small)
        );

        const __m256 equal_root = _mm256_mul_ps(mean, scale);
        root_large = _mm256_mul_ps(root_large, scale);
        root_mid = _mm256_mul_ps(root_mid, scale);
        root_small = _mm256_mul_ps(root_small, scale);
        const __m256 invalid = _mm256_or_ps(
            zero_scale, _mm256_cmp_ps(valid, zero, _CMP_EQ_OQ)
        );
        root_large = _mm256_blendv_ps(root_large, equal_root, invalid);
        root_mid = _mm256_blendv_ps(root_mid, equal_root, invalid);
        root_small = _mm256_blendv_ps(root_small, equal_root, invalid);

        const __m256 high01 = _mm256_max_ps(root_large, root_mid);
        const __m256 low01 = _mm256_min_ps(root_large, root_mid);
        const __m256 high = _mm256_max_ps(high01, root_small);
        const __m256 middle_candidate = _mm256_min_ps(high01, root_small);
        const __m256 middle = _mm256_max_ps(low01, middle_candidate);
        const __m256 low = _mm256_min_ps(low01, middle_candidate);
        store_eight_triples(high, middle, low, out + 3 * i);
    }

    for (; i < n; ++i) {
        detail::ev3_one_descending(
            a00[i], a01[i], a02[i], a11[i], a12[i], a22[i],
            out[3 * i], out[3 * i + 1], out[3 * i + 2]
        );
    }
}

} // namespace bioimage_cpp::filters::avx2
