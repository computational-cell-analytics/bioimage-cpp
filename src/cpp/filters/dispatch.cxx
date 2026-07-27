#include "bioimage_cpp/filters/dispatch.hxx"

#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#endif

namespace bioimage_cpp::filters::avx2 {

void convolve_axis_x(
    const float *in,
    float *out,
    std::ptrdiff_t n_rows,
    std::ptrdiff_t n_cols,
    int radius,
    bool symmetric,
    const float *half_coefs
);

void convolve_axis_strided(
    const float *in,
    float *out,
    std::ptrdiff_t n_outer,
    std::ptrdiff_t n_axis,
    std::ptrdiff_t n_inner,
    int radius,
    bool symmetric,
    const float *half_coefs
);

void convolve_outer_products_2d(
    const std::array<const float *, 2> &gradients,
    const std::array<float *, 3> &out,
    std::ptrdiff_t n_outer,
    std::ptrdiff_t n_axis,
    std::ptrdiff_t n_inner,
    int radius,
    const float *half_coefs
);

void convolve_outer_products_3d(
    const std::array<const float *, 3> &gradients,
    const std::array<float *, 6> &out,
    std::ptrdiff_t n_outer,
    std::ptrdiff_t n_axis,
    std::ptrdiff_t n_inner,
    int radius,
    const float *half_coefs
);

void ev3_symmetric_descending_interleaved(
    const float *__restrict a00,
    const float *__restrict a01,
    const float *__restrict a02,
    const float *__restrict a11,
    const float *__restrict a12,
    const float *__restrict a22,
    float *__restrict out,
    std::ptrdiff_t n
);

} // namespace bioimage_cpp::filters::avx2

namespace bioimage_cpp::filters::dispatch {
namespace {

bool force_scalar_requested() {
    const char *value = std::getenv("BIOIMAGE_CPP_FILTERS_FORCE_SCALAR");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool runtime_avx2_fma_supported() noexcept {
    static const bool supported = []() noexcept {
#if defined(_MSC_VER)
        int registers[4]{};
        __cpuid(registers, 1);
        constexpr int fma_bit = 1 << 12;
        constexpr int osxsave_bit = 1 << 27;
        constexpr int avx_bit = 1 << 28;
        if ((registers[2] & (fma_bit | osxsave_bit | avx_bit)) !=
            (fma_bit | osxsave_bit | avx_bit)) {
            return false;
        }
        if ((_xgetbv(0) & 0x6) != 0x6) {
            return false;
        }
        __cpuidex(registers, 7, 0);
        constexpr int avx2_bit = 1 << 5;
        return (registers[1] & avx2_bit) != 0;
#elif defined(__GNUC__) || defined(__clang__)
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
        return false;
#endif
    }();
    return supported;
}

bool use_avx2() {
    return !force_scalar_requested() && runtime_avx2_fma_supported();
}

} // namespace

bool try_convolve_axis_x(
    const float *in,
    float *out,
    const std::ptrdiff_t n_rows,
    const std::ptrdiff_t n_cols,
    const int radius,
    const bool symmetric,
    const float *half_coefs
) {
    if (!use_avx2() || radius < 1 || radius > 12) {
        return false;
    }
    avx2::convolve_axis_x(
        in, out, n_rows, n_cols, radius, symmetric, half_coefs
    );
    return true;
}

bool try_convolve_axis_strided(
    const float *in,
    float *out,
    const std::ptrdiff_t n_outer,
    const std::ptrdiff_t n_axis,
    const std::ptrdiff_t n_inner,
    const int radius,
    const bool symmetric,
    const float *half_coefs
) {
    if (!use_avx2() || radius < 1 || radius > 12) {
        return false;
    }
    avx2::convolve_axis_strided(
        in, out, n_outer, n_axis, n_inner, radius, symmetric, half_coefs
    );
    return true;
}

bool try_convolve_outer_products_2d(
    const std::array<const float *, 2> &gradients,
    const std::array<float *, 3> &out,
    const std::ptrdiff_t n_outer,
    const std::ptrdiff_t n_axis,
    const std::ptrdiff_t n_inner,
    const int radius,
    const float *half_coefs
) {
    if (!use_avx2() || radius < 1 || radius > 12) {
        return false;
    }
    avx2::convolve_outer_products_2d(
        gradients, out, n_outer, n_axis, n_inner, radius, half_coefs
    );
    return true;
}

bool try_convolve_outer_products_3d(
    const std::array<const float *, 3> &gradients,
    const std::array<float *, 6> &out,
    const std::ptrdiff_t n_outer,
    const std::ptrdiff_t n_axis,
    const std::ptrdiff_t n_inner,
    const int radius,
    const float *half_coefs
) {
    if (!use_avx2() || radius < 1 || radius > 12) {
        return false;
    }
    avx2::convolve_outer_products_3d(
        gradients, out, n_outer, n_axis, n_inner, radius, half_coefs
    );
    return true;
}

bool try_ev3_symmetric_descending_interleaved(
    const float *a00,
    const float *a01,
    const float *a02,
    const float *a11,
    const float *a12,
    const float *a22,
    float *out,
    const std::ptrdiff_t n
) {
    if (!use_avx2() || n < 8) {
        return false;
    }
    avx2::ev3_symmetric_descending_interleaved(
        a00, a01, a02, a11, a12, a22, out, n
    );
    return true;
}

const char *convolution_backend() {
    return use_avx2() ? "avx2" : "scalar";
}

const char *eigenvalue_backend() {
    return use_avx2() ? "avx2" : "scalar";
}

} // namespace bioimage_cpp::filters::dispatch
