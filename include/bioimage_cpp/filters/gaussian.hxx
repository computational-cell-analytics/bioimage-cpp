#pragma once

#include "bioimage_cpp/filters/convolve.hxx"
#include "bioimage_cpp/filters/eigenvalues.hxx"
#include "bioimage_cpp/filters/kernel.hxx"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace bioimage_cpp::filters {

// ---------------------------------------------------------------------------
// Separable Gaussian (derivative) along each axis.
// ---------------------------------------------------------------------------
//
// `in`, `out`, `workspace` are C-contiguous buffers of the same total size.
// `out` and `workspace` must NOT alias `in` (the binding ensures this by
// allocating fresh output and scratch). `out` and `workspace` may not alias
// each other. After the call, `out` holds the filter response and `workspace`
// is left in an unspecified state.

inline void gaussian_separable_2d(
    const float *in,
    float *out,
    float *workspace,
    std::ptrdiff_t ny,
    std::ptrdiff_t nx,
    const Kernel1D &ky,
    const Kernel1D &kx
) {
    // Axis 0 (Y), strided: (n_outer=1, n_axis=ny, n_inner=nx).
    convolve_axis_strided(in, workspace, 1, ny, nx, ky);
    // Axis 1 (X), contiguous: (n_rows=ny, n_cols=nx).
    convolve_axis_x(workspace, out, ny, nx, kx);
}

inline void gaussian_separable_3d(
    const float *in,
    float *out,
    float *workspace,
    std::ptrdiff_t nz,
    std::ptrdiff_t ny,
    std::ptrdiff_t nx,
    const Kernel1D &kz,
    const Kernel1D &ky,
    const Kernel1D &kx
) {
    // Axis 0 (Z), strided: (1, nz, ny*nx). in -> out.
    convolve_axis_strided(in, out, 1, nz, ny * nx, kz);
    // Axis 1 (Y), strided: (nz, ny, nx). out -> workspace.
    convolve_axis_strided(out, workspace, nz, ny, nx, ky);
    // Axis 2 (X), contiguous: (nz*ny, nx). workspace -> out.
    convolve_axis_x(workspace, out, nz * ny, nx, kx);
}

// ---------------------------------------------------------------------------
// Public composite filters. All operate on float32 C-contiguous buffers in
// NumPy axis order: 2D = (ny, nx), 3D = (nz, ny, nx). `sigma_*` and `order_*`
// are per-axis; `window_ratio = 0` selects the default kernel radius
// (ceil((3 + 0.5 * order) * sigma) per axis).
// ---------------------------------------------------------------------------

inline void gaussian_smoothing_2d(
    const float *in,
    float *out,
    std::ptrdiff_t ny,
    std::ptrdiff_t nx,
    double sigma_y,
    double sigma_x,
    double window_ratio
) {
    const auto ky = gaussian_kernel(sigma_y, 0, window_ratio);
    const auto kx = gaussian_kernel(sigma_x, 0, window_ratio);
    std::vector<float> workspace(static_cast<std::size_t>(ny * nx));
    gaussian_separable_2d(in, out, workspace.data(), ny, nx, ky, kx);
}

inline void gaussian_smoothing_3d(
    const float *in,
    float *out,
    std::ptrdiff_t nz,
    std::ptrdiff_t ny,
    std::ptrdiff_t nx,
    double sigma_z,
    double sigma_y,
    double sigma_x,
    double window_ratio
) {
    const auto kz = gaussian_kernel(sigma_z, 0, window_ratio);
    const auto ky = gaussian_kernel(sigma_y, 0, window_ratio);
    const auto kx = gaussian_kernel(sigma_x, 0, window_ratio);
    std::vector<float> workspace(static_cast<std::size_t>(nz * ny * nx));
    gaussian_separable_3d(in, out, workspace.data(), nz, ny, nx, kz, ky, kx);
}

inline void gaussian_derivative_2d(
    const float *in,
    float *out,
    std::ptrdiff_t ny,
    std::ptrdiff_t nx,
    double sigma_y,
    double sigma_x,
    int order_y,
    int order_x,
    double window_ratio
) {
    const auto ky = gaussian_kernel(sigma_y, order_y, window_ratio);
    const auto kx = gaussian_kernel(sigma_x, order_x, window_ratio);
    std::vector<float> workspace(static_cast<std::size_t>(ny * nx));
    gaussian_separable_2d(in, out, workspace.data(), ny, nx, ky, kx);
}

inline void gaussian_derivative_3d(
    const float *in,
    float *out,
    std::ptrdiff_t nz,
    std::ptrdiff_t ny,
    std::ptrdiff_t nx,
    double sigma_z,
    double sigma_y,
    double sigma_x,
    int order_z,
    int order_y,
    int order_x,
    double window_ratio
) {
    const auto kz = gaussian_kernel(sigma_z, order_z, window_ratio);
    const auto ky = gaussian_kernel(sigma_y, order_y, window_ratio);
    const auto kx = gaussian_kernel(sigma_x, order_x, window_ratio);
    std::vector<float> workspace(static_cast<std::size_t>(nz * ny * nx));
    gaussian_separable_3d(in, out, workspace.data(), nz, ny, nx, kz, ky, kx);
}

inline void gaussian_gradient_magnitude_2d(
    const float *in,
    float *out,
    std::ptrdiff_t ny,
    std::ptrdiff_t nx,
    double sigma_y,
    double sigma_x,
    double window_ratio
) {
    const std::ptrdiff_t n = ny * nx;
    const auto ky0 = gaussian_kernel(sigma_y, 0, window_ratio);
    const auto kx0 = gaussian_kernel(sigma_x, 0, window_ratio);
    const auto ky1 = gaussian_kernel(sigma_y, 1, window_ratio);
    const auto kx1 = gaussian_kernel(sigma_x, 1, window_ratio);

    std::vector<float> work1(static_cast<std::size_t>(n));
    std::vector<float> work2(static_cast<std::size_t>(n));

    // d/dy
    gaussian_separable_2d(in, work1.data(), work2.data(), ny, nx, ky1, kx0);
    for (std::ptrdiff_t i = 0; i < n; ++i) out[i] = work1[i] * work1[i];

    // d/dx
    gaussian_separable_2d(in, work1.data(), work2.data(), ny, nx, ky0, kx1);
    for (std::ptrdiff_t i = 0; i < n; ++i) out[i] += work1[i] * work1[i];

    for (std::ptrdiff_t i = 0; i < n; ++i) out[i] = std::sqrt(out[i]);
}

inline void gaussian_gradient_magnitude_3d(
    const float *in,
    float *out,
    std::ptrdiff_t nz,
    std::ptrdiff_t ny,
    std::ptrdiff_t nx,
    double sigma_z,
    double sigma_y,
    double sigma_x,
    double window_ratio
) {
    const std::ptrdiff_t n = nz * ny * nx;
    const auto kz0 = gaussian_kernel(sigma_z, 0, window_ratio);
    const auto ky0 = gaussian_kernel(sigma_y, 0, window_ratio);
    const auto kx0 = gaussian_kernel(sigma_x, 0, window_ratio);
    const auto kz1 = gaussian_kernel(sigma_z, 1, window_ratio);
    const auto ky1 = gaussian_kernel(sigma_y, 1, window_ratio);
    const auto kx1 = gaussian_kernel(sigma_x, 1, window_ratio);

    std::vector<float> work1(static_cast<std::size_t>(n));
    std::vector<float> work2(static_cast<std::size_t>(n));

    gaussian_separable_3d(in, work1.data(), work2.data(), nz, ny, nx, kz1, ky0, kx0);
    for (std::ptrdiff_t i = 0; i < n; ++i) out[i] = work1[i] * work1[i];

    gaussian_separable_3d(in, work1.data(), work2.data(), nz, ny, nx, kz0, ky1, kx0);
    for (std::ptrdiff_t i = 0; i < n; ++i) out[i] += work1[i] * work1[i];

    gaussian_separable_3d(in, work1.data(), work2.data(), nz, ny, nx, kz0, ky0, kx1);
    for (std::ptrdiff_t i = 0; i < n; ++i) out[i] += work1[i] * work1[i];

    for (std::ptrdiff_t i = 0; i < n; ++i) out[i] = std::sqrt(out[i]);
}

inline void laplacian_of_gaussian_2d(
    const float *in,
    float *out,
    std::ptrdiff_t ny,
    std::ptrdiff_t nx,
    double sigma_y,
    double sigma_x,
    double window_ratio
) {
    const std::ptrdiff_t n = ny * nx;
    const auto ky0 = gaussian_kernel(sigma_y, 0, window_ratio);
    const auto kx0 = gaussian_kernel(sigma_x, 0, window_ratio);
    const auto ky2 = gaussian_kernel(sigma_y, 2, window_ratio);
    const auto kx2 = gaussian_kernel(sigma_x, 2, window_ratio);

    std::vector<float> work1(static_cast<std::size_t>(n));
    std::vector<float> work2(static_cast<std::size_t>(n));

    // d²/dy²
    gaussian_separable_2d(in, work1.data(), work2.data(), ny, nx, ky2, kx0);
    for (std::ptrdiff_t i = 0; i < n; ++i) out[i] = work1[i];

    // d²/dx²
    gaussian_separable_2d(in, work1.data(), work2.data(), ny, nx, ky0, kx2);
    for (std::ptrdiff_t i = 0; i < n; ++i) out[i] += work1[i];
}

inline void laplacian_of_gaussian_3d(
    const float *in,
    float *out,
    std::ptrdiff_t nz,
    std::ptrdiff_t ny,
    std::ptrdiff_t nx,
    double sigma_z,
    double sigma_y,
    double sigma_x,
    double window_ratio
) {
    const std::ptrdiff_t n = nz * ny * nx;
    const auto kz0 = gaussian_kernel(sigma_z, 0, window_ratio);
    const auto ky0 = gaussian_kernel(sigma_y, 0, window_ratio);
    const auto kx0 = gaussian_kernel(sigma_x, 0, window_ratio);
    const auto kz2 = gaussian_kernel(sigma_z, 2, window_ratio);
    const auto ky2 = gaussian_kernel(sigma_y, 2, window_ratio);
    const auto kx2 = gaussian_kernel(sigma_x, 2, window_ratio);

    std::vector<float> work1(static_cast<std::size_t>(n));
    std::vector<float> work2(static_cast<std::size_t>(n));

    gaussian_separable_3d(in, work1.data(), work2.data(), nz, ny, nx, kz2, ky0, kx0);
    for (std::ptrdiff_t i = 0; i < n; ++i) out[i] = work1[i];

    gaussian_separable_3d(in, work1.data(), work2.data(), nz, ny, nx, kz0, ky2, kx0);
    for (std::ptrdiff_t i = 0; i < n; ++i) out[i] += work1[i];

    gaussian_separable_3d(in, work1.data(), work2.data(), nz, ny, nx, kz0, ky0, kx2);
    for (std::ptrdiff_t i = 0; i < n; ++i) out[i] += work1[i];
}

// ---------------------------------------------------------------------------
// Hessian-of-Gaussian eigenvalues. `out` has trailing-axis layout: in 2D, the
// output buffer holds ny*nx*2 floats laid out so out[2*i + 0] = largest
// eigenvalue, out[2*i + 1] = smallest. In 3D similarly with stride 3.
// ---------------------------------------------------------------------------

inline void hessian_of_gaussian_eigenvalues_2d(
    const float *in,
    float *out,
    std::ptrdiff_t ny,
    std::ptrdiff_t nx,
    double sigma_y,
    double sigma_x,
    double window_ratio
) {
    const std::ptrdiff_t n = ny * nx;
    const auto ky0 = gaussian_kernel(sigma_y, 0, window_ratio);
    const auto kx0 = gaussian_kernel(sigma_x, 0, window_ratio);
    const auto ky1 = gaussian_kernel(sigma_y, 1, window_ratio);
    const auto kx1 = gaussian_kernel(sigma_x, 1, window_ratio);
    const auto ky2 = gaussian_kernel(sigma_y, 2, window_ratio);
    const auto kx2 = gaussian_kernel(sigma_x, 2, window_ratio);

    std::vector<float> work(static_cast<std::size_t>(n));
    std::vector<float> hyy(static_cast<std::size_t>(n));
    std::vector<float> hyx(static_cast<std::size_t>(n));
    std::vector<float> hxx(static_cast<std::size_t>(n));

    // d²/dy²
    gaussian_separable_2d(in, hyy.data(), work.data(), ny, nx, ky2, kx0);
    // d²/(dy dx)
    gaussian_separable_2d(in, hyx.data(), work.data(), ny, nx, ky1, kx1);
    // d²/dx²
    gaussian_separable_2d(in, hxx.data(), work.data(), ny, nx, ky0, kx2);

    // ev2 sorted descending into interleaved output.
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const float a = hyy[i];
        const float b = hyx[i];
        const float c = hxx[i];
        const float half_tr = 0.5f * (a + c);
        const float half_diff = 0.5f * (a - c);
        const float disc = std::sqrt(half_diff * half_diff + b * b);
        out[2 * i + 0] = half_tr + disc;
        out[2 * i + 1] = half_tr - disc;
    }
}

inline void hessian_of_gaussian_eigenvalues_3d(
    const float *in,
    float *out,
    std::ptrdiff_t nz,
    std::ptrdiff_t ny,
    std::ptrdiff_t nx,
    double sigma_z,
    double sigma_y,
    double sigma_x,
    double window_ratio
) {
    const std::ptrdiff_t n = nz * ny * nx;
    const auto kz0 = gaussian_kernel(sigma_z, 0, window_ratio);
    const auto ky0 = gaussian_kernel(sigma_y, 0, window_ratio);
    const auto kx0 = gaussian_kernel(sigma_x, 0, window_ratio);
    const auto kz1 = gaussian_kernel(sigma_z, 1, window_ratio);
    const auto ky1 = gaussian_kernel(sigma_y, 1, window_ratio);
    const auto kx1 = gaussian_kernel(sigma_x, 1, window_ratio);
    const auto kz2 = gaussian_kernel(sigma_z, 2, window_ratio);
    const auto ky2 = gaussian_kernel(sigma_y, 2, window_ratio);
    const auto kx2 = gaussian_kernel(sigma_x, 2, window_ratio);

    std::vector<float> work(static_cast<std::size_t>(n));
    std::vector<float> hzz(static_cast<std::size_t>(n));
    std::vector<float> hzy(static_cast<std::size_t>(n));
    std::vector<float> hzx(static_cast<std::size_t>(n));
    std::vector<float> hyy(static_cast<std::size_t>(n));
    std::vector<float> hyx(static_cast<std::size_t>(n));
    std::vector<float> hxx(static_cast<std::size_t>(n));

    gaussian_separable_3d(in, hzz.data(), work.data(), nz, ny, nx, kz2, ky0, kx0);
    gaussian_separable_3d(in, hzy.data(), work.data(), nz, ny, nx, kz1, ky1, kx0);
    gaussian_separable_3d(in, hzx.data(), work.data(), nz, ny, nx, kz1, ky0, kx1);
    gaussian_separable_3d(in, hyy.data(), work.data(), nz, ny, nx, kz0, ky2, kx0);
    gaussian_separable_3d(in, hyx.data(), work.data(), nz, ny, nx, kz0, ky1, kx1);
    gaussian_separable_3d(in, hxx.data(), work.data(), nz, ny, nx, kz0, ky0, kx2);

    // ev3 sorted descending into interleaved output.
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        float e0;
        float e1;
        float e2;
        detail::ev3_one_descending(
            hzz[i], hzy[i], hzx[i], hyy[i], hyx[i], hxx[i],
            e0, e1, e2
        );
        out[3 * i + 0] = e0;
        out[3 * i + 1] = e1;
        out[3 * i + 2] = e2;
    }
}

// ---------------------------------------------------------------------------
// Structure-tensor eigenvalues. Two-scale: first take first-order Gaussian
// derivatives at sigma_inner, form the outer products, smooth them with
// sigma_outer, then compute eigenvalues of the resulting symmetric tensor.
// Output layout matches the Hessian variants (trailing axis size N).
// ---------------------------------------------------------------------------

inline void structure_tensor_eigenvalues_2d(
    const float *in,
    float *out,
    std::ptrdiff_t ny,
    std::ptrdiff_t nx,
    double sigma_inner_y,
    double sigma_inner_x,
    double sigma_outer_y,
    double sigma_outer_x,
    double window_ratio
) {
    const std::ptrdiff_t n = ny * nx;

    const auto kiy0 = gaussian_kernel(sigma_inner_y, 0, window_ratio);
    const auto kix0 = gaussian_kernel(sigma_inner_x, 0, window_ratio);
    const auto kiy1 = gaussian_kernel(sigma_inner_y, 1, window_ratio);
    const auto kix1 = gaussian_kernel(sigma_inner_x, 1, window_ratio);
    const auto koy0 = gaussian_kernel(sigma_outer_y, 0, window_ratio);
    const auto kox0 = gaussian_kernel(sigma_outer_x, 0, window_ratio);

    std::vector<float> work(static_cast<std::size_t>(n));
    std::vector<float> gy(static_cast<std::size_t>(n));
    std::vector<float> gx(static_cast<std::size_t>(n));
    std::vector<float> tmp(static_cast<std::size_t>(n));
    std::vector<float> syy(static_cast<std::size_t>(n));
    std::vector<float> syx(static_cast<std::size_t>(n));
    std::vector<float> sxx(static_cast<std::size_t>(n));

    // First-order partials at sigma_inner.
    gaussian_separable_2d(in, gy.data(), work.data(), ny, nx, kiy1, kix0);
    gaussian_separable_2d(in, gx.data(), work.data(), ny, nx, kiy0, kix1);

    // Outer products, smoothed with sigma_outer.
    for (std::ptrdiff_t i = 0; i < n; ++i) tmp[i] = gy[i] * gy[i];
    gaussian_separable_2d(tmp.data(), syy.data(), work.data(), ny, nx, koy0, kox0);

    for (std::ptrdiff_t i = 0; i < n; ++i) tmp[i] = gy[i] * gx[i];
    gaussian_separable_2d(tmp.data(), syx.data(), work.data(), ny, nx, koy0, kox0);

    for (std::ptrdiff_t i = 0; i < n; ++i) tmp[i] = gx[i] * gx[i];
    gaussian_separable_2d(tmp.data(), sxx.data(), work.data(), ny, nx, koy0, kox0);

    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const float a = syy[i];
        const float b = syx[i];
        const float c = sxx[i];
        const float half_tr = 0.5f * (a + c);
        const float half_diff = 0.5f * (a - c);
        const float disc = std::sqrt(half_diff * half_diff + b * b);
        out[2 * i + 0] = half_tr + disc;
        out[2 * i + 1] = half_tr - disc;
    }
}

inline void structure_tensor_eigenvalues_3d(
    const float *in,
    float *out,
    std::ptrdiff_t nz,
    std::ptrdiff_t ny,
    std::ptrdiff_t nx,
    double sigma_inner_z,
    double sigma_inner_y,
    double sigma_inner_x,
    double sigma_outer_z,
    double sigma_outer_y,
    double sigma_outer_x,
    double window_ratio
) {
    const std::ptrdiff_t n = nz * ny * nx;

    const auto kiz0 = gaussian_kernel(sigma_inner_z, 0, window_ratio);
    const auto kiy0 = gaussian_kernel(sigma_inner_y, 0, window_ratio);
    const auto kix0 = gaussian_kernel(sigma_inner_x, 0, window_ratio);
    const auto kiz1 = gaussian_kernel(sigma_inner_z, 1, window_ratio);
    const auto kiy1 = gaussian_kernel(sigma_inner_y, 1, window_ratio);
    const auto kix1 = gaussian_kernel(sigma_inner_x, 1, window_ratio);
    const auto koz0 = gaussian_kernel(sigma_outer_z, 0, window_ratio);
    const auto koy0 = gaussian_kernel(sigma_outer_y, 0, window_ratio);
    const auto kox0 = gaussian_kernel(sigma_outer_x, 0, window_ratio);

    std::vector<float> work(static_cast<std::size_t>(n));
    std::vector<float> gz(static_cast<std::size_t>(n));
    std::vector<float> gy(static_cast<std::size_t>(n));
    std::vector<float> gx(static_cast<std::size_t>(n));
    std::vector<float> tmp(static_cast<std::size_t>(n));
    std::vector<float> szz(static_cast<std::size_t>(n));
    std::vector<float> szy(static_cast<std::size_t>(n));
    std::vector<float> szx(static_cast<std::size_t>(n));
    std::vector<float> syy(static_cast<std::size_t>(n));
    std::vector<float> syx(static_cast<std::size_t>(n));
    std::vector<float> sxx(static_cast<std::size_t>(n));

    gaussian_separable_3d(in, gz.data(), work.data(), nz, ny, nx, kiz1, kiy0, kix0);
    gaussian_separable_3d(in, gy.data(), work.data(), nz, ny, nx, kiz0, kiy1, kix0);
    gaussian_separable_3d(in, gx.data(), work.data(), nz, ny, nx, kiz0, kiy0, kix1);

    for (std::ptrdiff_t i = 0; i < n; ++i) tmp[i] = gz[i] * gz[i];
    gaussian_separable_3d(tmp.data(), szz.data(), work.data(), nz, ny, nx, koz0, koy0, kox0);

    for (std::ptrdiff_t i = 0; i < n; ++i) tmp[i] = gz[i] * gy[i];
    gaussian_separable_3d(tmp.data(), szy.data(), work.data(), nz, ny, nx, koz0, koy0, kox0);

    for (std::ptrdiff_t i = 0; i < n; ++i) tmp[i] = gz[i] * gx[i];
    gaussian_separable_3d(tmp.data(), szx.data(), work.data(), nz, ny, nx, koz0, koy0, kox0);

    for (std::ptrdiff_t i = 0; i < n; ++i) tmp[i] = gy[i] * gy[i];
    gaussian_separable_3d(tmp.data(), syy.data(), work.data(), nz, ny, nx, koz0, koy0, kox0);

    for (std::ptrdiff_t i = 0; i < n; ++i) tmp[i] = gy[i] * gx[i];
    gaussian_separable_3d(tmp.data(), syx.data(), work.data(), nz, ny, nx, koz0, koy0, kox0);

    for (std::ptrdiff_t i = 0; i < n; ++i) tmp[i] = gx[i] * gx[i];
    gaussian_separable_3d(tmp.data(), sxx.data(), work.data(), nz, ny, nx, koz0, koy0, kox0);

    for (std::ptrdiff_t i = 0; i < n; ++i) {
        float e0;
        float e1;
        float e2;
        detail::ev3_one_descending(
            szz[i], szy[i], szx[i], syy[i], syx[i], sxx[i],
            e0, e1, e2
        );
        out[3 * i + 0] = e0;
        out[3 * i + 1] = e1;
        out[3 * i + 2] = e2;
    }
}

} // namespace bioimage_cpp::filters
