#pragma once

#include <array>
#include <cstddef>

namespace bioimage_cpp::filters::dispatch {

bool try_convolve_axis_x(
    const float *in,
    float *out,
    std::ptrdiff_t n_rows,
    std::ptrdiff_t n_cols,
    int radius,
    bool symmetric,
    const float *half_coefs
);

bool try_convolve_axis_strided(
    const float *in,
    float *out,
    std::ptrdiff_t n_outer,
    std::ptrdiff_t n_axis,
    std::ptrdiff_t n_inner,
    int radius,
    bool symmetric,
    const float *half_coefs
);

bool try_convolve_outer_products_2d(
    const std::array<const float *, 2> &gradients,
    const std::array<float *, 3> &out,
    std::ptrdiff_t n_outer,
    std::ptrdiff_t n_axis,
    std::ptrdiff_t n_inner,
    int radius,
    const float *half_coefs
);

bool try_convolve_outer_products_3d(
    const std::array<const float *, 3> &gradients,
    const std::array<float *, 6> &out,
    std::ptrdiff_t n_outer,
    std::ptrdiff_t n_axis,
    std::ptrdiff_t n_inner,
    int radius,
    const float *half_coefs
);

bool try_ev3_symmetric_descending_interleaved(
    const float *a00,
    const float *a01,
    const float *a02,
    const float *a11,
    const float *a12,
    const float *a22,
    float *out,
    std::ptrdiff_t n
);

const char *convolution_backend();
const char *eigenvalue_backend();

} // namespace bioimage_cpp::filters::dispatch
