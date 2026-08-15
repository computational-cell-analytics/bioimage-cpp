#pragma once

#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/filters/convolve.hxx"
#include "bioimage_cpp/filters/eigenvalues.hxx"
#include "bioimage_cpp/filters/kernel.hxx"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace bioimage_cpp::filters {

namespace detail {

inline std::unique_ptr<float[]> allocate_scratch(
    const std::ptrdiff_t n,
    const std::size_t slots
) {
    if (n < 0) {
        throw std::invalid_argument("filter scratch size must be non-negative");
    }
    const auto size = static_cast<std::size_t>(n);
    if (slots != 0 && size > std::numeric_limits<std::size_t>::max() / slots) {
        throw std::length_error("filter scratch size overflow");
    }
    return std::make_unique_for_overwrite<float[]>(size * slots);
}

inline float *scratch_slot(
    const std::unique_ptr<float[]> &scratch,
    const std::ptrdiff_t n,
    const std::size_t slot
) {
    if (n == 0) {
        return scratch.get();
    }
    return scratch.get() + static_cast<std::size_t>(n) * slot;
}

template <class Profiler>
inline void gaussian_separable_2d_profiled(
    const float *in,
    float *out,
    float *workspace,
    const std::ptrdiff_t ny,
    const std::ptrdiff_t nx,
    const Kernel1D &ky,
    const Kernel1D &kx,
    Profiler &profile
) {
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "axis_y");
        convolve_axis_strided(in, workspace, 1, ny, nx, ky);
    }
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "axis_x");
        convolve_axis_x(workspace, out, ny, nx, kx);
    }
}

template <class Profiler>
inline void gaussian_first_axis_3d_profiled(
    const float *in,
    float *partial,
    const std::ptrdiff_t nz,
    const std::ptrdiff_t ny,
    const std::ptrdiff_t nx,
    const Kernel1D &kz,
    Profiler &profile
) {
    BIOIMAGE_PROFILE_SCOPE(profile, "axis_z");
    convolve_axis_strided(in, partial, 1, nz, ny * nx, kz);
}

template <class Profiler>
inline void gaussian_remaining_axes_3d_profiled(
    const float *partial,
    float *out,
    float *workspace,
    const std::ptrdiff_t nz,
    const std::ptrdiff_t ny,
    const std::ptrdiff_t nx,
    const Kernel1D &ky,
    const Kernel1D &kx,
    Profiler &profile
) {
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "axis_y");
        convolve_axis_strided(partial, workspace, nz, ny, nx, ky);
    }
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "axis_x");
        convolve_axis_x(workspace, out, nz * ny, nx, kx);
    }
}

template <class Profiler>
inline void gaussian_separable_3d_profiled(
    const float *in,
    float *out,
    float *workspace,
    const std::ptrdiff_t nz,
    const std::ptrdiff_t ny,
    const std::ptrdiff_t nx,
    const Kernel1D &kz,
    const Kernel1D &ky,
    const Kernel1D &kx,
    Profiler &profile
) {
    gaussian_first_axis_3d_profiled(in, out, nz, ny, nx, kz, profile);
    gaussian_remaining_axes_3d_profiled(out, out, workspace, nz, ny, nx, ky, kx, profile);
}

} // namespace detail

inline void gaussian_separable_2d(
    const float *in,
    float *out,
    float *workspace,
    std::ptrdiff_t ny,
    std::ptrdiff_t nx,
    const Kernel1D &ky,
    const Kernel1D &kx
) {
    bioimage_cpp::detail::NullProfiler profile;
    detail::gaussian_separable_2d_profiled(
        in, out, workspace, ny, nx, ky, kx, profile
    );
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
    bioimage_cpp::detail::NullProfiler profile;
    detail::gaussian_separable_3d_profiled(
        in, out, workspace, nz, ny, nx, kz, ky, kx, profile
    );
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
    BIOIMAGE_PROFILE_INIT(profile);
    const std::ptrdiff_t n = ny * nx;
    const auto ky = gaussian_kernel(sigma_y, 0, window_ratio);
    const auto kx = gaussian_kernel(sigma_x, 0, window_ratio);
    std::unique_ptr<float[]> scratch;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "scratch_alloc");
        scratch = detail::allocate_scratch(n, 1);
    }
    detail::gaussian_separable_2d_profiled(
        in, out, scratch.get(), ny, nx, ky, kx, profile
    );
    BIOIMAGE_PROFILE_REPORT(profile);
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
    BIOIMAGE_PROFILE_INIT(profile);
    const std::ptrdiff_t n = nz * ny * nx;
    const auto kz = gaussian_kernel(sigma_z, 0, window_ratio);
    const auto ky = gaussian_kernel(sigma_y, 0, window_ratio);
    const auto kx = gaussian_kernel(sigma_x, 0, window_ratio);
    std::unique_ptr<float[]> scratch;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "scratch_alloc");
        scratch = detail::allocate_scratch(n, 1);
    }
    detail::gaussian_separable_3d_profiled(
        in, out, scratch.get(), nz, ny, nx, kz, ky, kx, profile
    );
    BIOIMAGE_PROFILE_REPORT(profile);
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
    BIOIMAGE_PROFILE_INIT(profile);
    const std::ptrdiff_t n = ny * nx;
    const auto ky = gaussian_kernel(sigma_y, order_y, window_ratio);
    const auto kx = gaussian_kernel(sigma_x, order_x, window_ratio);
    std::unique_ptr<float[]> scratch;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "scratch_alloc");
        scratch = detail::allocate_scratch(n, 1);
    }
    detail::gaussian_separable_2d_profiled(
        in, out, scratch.get(), ny, nx, ky, kx, profile
    );
    BIOIMAGE_PROFILE_REPORT(profile);
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
    BIOIMAGE_PROFILE_INIT(profile);
    const std::ptrdiff_t n = nz * ny * nx;
    const auto kz = gaussian_kernel(sigma_z, order_z, window_ratio);
    const auto ky = gaussian_kernel(sigma_y, order_y, window_ratio);
    const auto kx = gaussian_kernel(sigma_x, order_x, window_ratio);
    std::unique_ptr<float[]> scratch;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "scratch_alloc");
        scratch = detail::allocate_scratch(n, 1);
    }
    detail::gaussian_separable_3d_profiled(
        in, out, scratch.get(), nz, ny, nx, kz, ky, kx, profile
    );
    BIOIMAGE_PROFILE_REPORT(profile);
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
    BIOIMAGE_PROFILE_INIT(profile);
    const std::ptrdiff_t n = ny * nx;
    const auto ky0 = gaussian_kernel(sigma_y, 0, window_ratio);
    const auto kx0 = gaussian_kernel(sigma_x, 0, window_ratio);
    const auto ky1 = gaussian_kernel(sigma_y, 1, window_ratio);
    const auto kx1 = gaussian_kernel(sigma_x, 1, window_ratio);

    std::unique_ptr<float[]> scratch;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "scratch_alloc");
        scratch = detail::allocate_scratch(n, 2);
    }
    float *work1 = detail::scratch_slot(scratch, n, 0);
    float *work2 = detail::scratch_slot(scratch, n, 1);

    detail::gaussian_separable_2d_profiled(
        in, work1, work2, ny, nx, ky1, kx0, profile
    );
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "combine");
        for (std::ptrdiff_t i = 0; i < n; ++i) out[i] = work1[i] * work1[i];
    }

    detail::gaussian_separable_2d_profiled(
        in, work1, work2, ny, nx, ky0, kx1, profile
    );
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "combine");
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            out[i] = std::sqrt(out[i] + work1[i] * work1[i]);
        }
    }
    BIOIMAGE_PROFILE_REPORT(profile);
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
    BIOIMAGE_PROFILE_INIT(profile);
    const std::ptrdiff_t n = nz * ny * nx;
    const auto kz0 = gaussian_kernel(sigma_z, 0, window_ratio);
    const auto ky0 = gaussian_kernel(sigma_y, 0, window_ratio);
    const auto kx0 = gaussian_kernel(sigma_x, 0, window_ratio);
    const auto kz1 = gaussian_kernel(sigma_z, 1, window_ratio);
    const auto ky1 = gaussian_kernel(sigma_y, 1, window_ratio);
    const auto kx1 = gaussian_kernel(sigma_x, 1, window_ratio);

    std::unique_ptr<float[]> scratch;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "scratch_alloc");
        scratch = detail::allocate_scratch(n, 2);
    }
    float *work1 = detail::scratch_slot(scratch, n, 0);
    float *work2 = detail::scratch_slot(scratch, n, 1);

    detail::gaussian_first_axis_3d_profiled(in, out, nz, ny, nx, kz0, profile);
    detail::gaussian_remaining_axes_3d_profiled(
        out, work2, work1, nz, ny, nx, ky1, kx0, profile
    );
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "combine");
        for (std::ptrdiff_t i = 0; i < n; ++i) work1[i] = work2[i] * work2[i];
    }

    detail::gaussian_remaining_axes_3d_profiled(
        out, out, work2, nz, ny, nx, ky0, kx1, profile
    );
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "combine");
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            work1[i] += out[i] * out[i];
        }
    }

    detail::gaussian_first_axis_3d_profiled(in, work2, nz, ny, nx, kz1, profile);
    detail::gaussian_remaining_axes_3d_profiled(
        work2, work2, out, nz, ny, nx, ky0, kx0, profile
    );
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "combine");
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            out[i] = std::sqrt(work1[i] + work2[i] * work2[i]);
        }
    }
    BIOIMAGE_PROFILE_REPORT(profile);
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
    BIOIMAGE_PROFILE_INIT(profile);
    const std::ptrdiff_t n = ny * nx;
    const auto ky0 = gaussian_kernel(sigma_y, 0, window_ratio);
    const auto kx0 = gaussian_kernel(sigma_x, 0, window_ratio);
    const auto ky2 = gaussian_kernel(sigma_y, 2, window_ratio);
    const auto kx2 = gaussian_kernel(sigma_x, 2, window_ratio);

    std::unique_ptr<float[]> scratch;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "scratch_alloc");
        scratch = detail::allocate_scratch(n, 2);
    }
    float *work1 = detail::scratch_slot(scratch, n, 0);
    float *work2 = detail::scratch_slot(scratch, n, 1);

    detail::gaussian_separable_2d_profiled(
        in, work1, work2, ny, nx, ky2, kx0, profile
    );
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "combine");
        for (std::ptrdiff_t i = 0; i < n; ++i) out[i] = work1[i];
    }

    detail::gaussian_separable_2d_profiled(
        in, work1, work2, ny, nx, ky0, kx2, profile
    );
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "combine");
        for (std::ptrdiff_t i = 0; i < n; ++i) out[i] += work1[i];
    }
    BIOIMAGE_PROFILE_REPORT(profile);
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
    BIOIMAGE_PROFILE_INIT(profile);
    const std::ptrdiff_t n = nz * ny * nx;
    const auto kz0 = gaussian_kernel(sigma_z, 0, window_ratio);
    const auto ky0 = gaussian_kernel(sigma_y, 0, window_ratio);
    const auto kx0 = gaussian_kernel(sigma_x, 0, window_ratio);
    const auto kz2 = gaussian_kernel(sigma_z, 2, window_ratio);
    const auto ky2 = gaussian_kernel(sigma_y, 2, window_ratio);
    const auto kx2 = gaussian_kernel(sigma_x, 2, window_ratio);

    std::unique_ptr<float[]> scratch;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "scratch_alloc");
        scratch = detail::allocate_scratch(n, 2);
    }
    float *work1 = detail::scratch_slot(scratch, n, 0);
    float *work2 = detail::scratch_slot(scratch, n, 1);

    detail::gaussian_first_axis_3d_profiled(in, out, nz, ny, nx, kz0, profile);
    detail::gaussian_remaining_axes_3d_profiled(
        out, work2, work1, nz, ny, nx, ky2, kx0, profile
    );
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "combine");
        for (std::ptrdiff_t i = 0; i < n; ++i) work1[i] = work2[i];
    }

    detail::gaussian_remaining_axes_3d_profiled(
        out, out, work2, nz, ny, nx, ky0, kx2, profile
    );
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "combine");
        for (std::ptrdiff_t i = 0; i < n; ++i) out[i] += work1[i];
    }

    detail::gaussian_first_axis_3d_profiled(in, work2, nz, ny, nx, kz2, profile);
    detail::gaussian_remaining_axes_3d_profiled(
        work2, work2, work1, nz, ny, nx, ky0, kx0, profile
    );
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "combine");
        for (std::ptrdiff_t i = 0; i < n; ++i) out[i] += work2[i];
    }
    BIOIMAGE_PROFILE_REPORT(profile);
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
    BIOIMAGE_PROFILE_INIT(profile);
    const std::ptrdiff_t n = ny * nx;
    const auto ky0 = gaussian_kernel(sigma_y, 0, window_ratio);
    const auto kx0 = gaussian_kernel(sigma_x, 0, window_ratio);
    const auto ky1 = gaussian_kernel(sigma_y, 1, window_ratio);
    const auto kx1 = gaussian_kernel(sigma_x, 1, window_ratio);
    const auto ky2 = gaussian_kernel(sigma_y, 2, window_ratio);
    const auto kx2 = gaussian_kernel(sigma_x, 2, window_ratio);

    std::unique_ptr<float[]> scratch;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "scratch_alloc");
        scratch = detail::allocate_scratch(n, 4);
    }
    float *work = detail::scratch_slot(scratch, n, 0);
    float *hyy = detail::scratch_slot(scratch, n, 1);
    float *hyx = detail::scratch_slot(scratch, n, 2);
    float *hxx = detail::scratch_slot(scratch, n, 3);

    detail::gaussian_separable_2d_profiled(in, hyy, work, ny, nx, ky2, kx0, profile);
    detail::gaussian_separable_2d_profiled(in, hyx, work, ny, nx, ky1, kx1, profile);
    detail::gaussian_separable_2d_profiled(in, hxx, work, ny, nx, ky0, kx2, profile);

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "eigenvalues");
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
    BIOIMAGE_PROFILE_REPORT(profile);
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
    BIOIMAGE_PROFILE_INIT(profile);
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

    std::unique_ptr<float[]> scratch;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "scratch_alloc");
        scratch = detail::allocate_scratch(n, 7);
    }
    float *work = detail::scratch_slot(scratch, n, 0);
    float *hzz = detail::scratch_slot(scratch, n, 1);
    float *hzy = detail::scratch_slot(scratch, n, 2);
    float *hzx = detail::scratch_slot(scratch, n, 3);
    float *hyy = detail::scratch_slot(scratch, n, 4);
    float *hyx = detail::scratch_slot(scratch, n, 5);
    float *hxx = detail::scratch_slot(scratch, n, 6);

    detail::gaussian_first_axis_3d_profiled(in, hxx, nz, ny, nx, kz0, profile);
    detail::gaussian_remaining_axes_3d_profiled(
        hxx, hyy, work, nz, ny, nx, ky2, kx0, profile
    );
    detail::gaussian_remaining_axes_3d_profiled(
        hxx, hyx, work, nz, ny, nx, ky1, kx1, profile
    );
    detail::gaussian_remaining_axes_3d_profiled(
        hxx, hxx, work, nz, ny, nx, ky0, kx2, profile
    );

    detail::gaussian_first_axis_3d_profiled(in, hzx, nz, ny, nx, kz1, profile);
    detail::gaussian_remaining_axes_3d_profiled(
        hzx, hzy, work, nz, ny, nx, ky1, kx0, profile
    );
    detail::gaussian_remaining_axes_3d_profiled(
        hzx, hzx, work, nz, ny, nx, ky0, kx1, profile
    );

    detail::gaussian_first_axis_3d_profiled(in, hzz, nz, ny, nx, kz2, profile);
    detail::gaussian_remaining_axes_3d_profiled(
        hzz, hzz, work, nz, ny, nx, ky0, kx0, profile
    );

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "eigenvalues");
        ev3_symmetric_descending_interleaved(
            hzz, hzy, hzx, hyy, hyx, hxx, out, n
        );
    }
    BIOIMAGE_PROFILE_REPORT(profile);
}

// ---------------------------------------------------------------------------
// Structure tensor. The leading output axis stores the upper triangle in
// NumPy axis order: (yy, yx, xx) or (zz, zy, zx, yy, yx, xx).
// ---------------------------------------------------------------------------

namespace detail {

template <class Profiler>
inline void structure_tensor_components_2d_profiled(
    const float *in,
    const std::array<float *, 3> &components,
    const std::array<float *, 3> &workspace,
    const std::ptrdiff_t ny,
    const std::ptrdiff_t nx,
    const double sigma_inner_y,
    const double sigma_inner_x,
    const double sigma_outer_y,
    const double sigma_outer_x,
    const double window_ratio,
    Profiler &profile
) {
    const auto kiy0 = gaussian_kernel(sigma_inner_y, 0, window_ratio);
    const auto kix0 = gaussian_kernel(sigma_inner_x, 0, window_ratio);
    const auto kiy1 = gaussian_kernel(sigma_inner_y, 1, window_ratio);
    const auto kix1 = gaussian_kernel(sigma_inner_x, 1, window_ratio);
    const auto koy0 = gaussian_kernel(sigma_outer_y, 0, window_ratio);
    const auto kox0 = gaussian_kernel(sigma_outer_x, 0, window_ratio);

    float *gy = components[0];
    float *gx = components[1];
    float *work = components[2];
    gaussian_separable_2d_profiled(in, gy, work, ny, nx, kiy1, kix0, profile);
    gaussian_separable_2d_profiled(in, gx, work, ny, nx, kiy0, kix1, profile);

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "outer_products");
        const std::array<const float *, 2> gradients{gy, gx};
        convolve_axis_strided_outer_products<2>(
            gradients, workspace, 1, ny, nx, koy0
        );
    }

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "axis_x");
        convolve_axis_x(workspace[0], components[0], ny, nx, kox0);
        convolve_axis_x(workspace[1], components[1], ny, nx, kox0);
        convolve_axis_x(workspace[2], components[2], ny, nx, kox0);
    }
}

template <class Profiler>
inline void structure_tensor_components_3d_profiled(
    const float *in,
    const std::array<float *, 6> &components,
    const std::array<float *, 4> &workspace,
    const std::ptrdiff_t nz,
    const std::ptrdiff_t ny,
    const std::ptrdiff_t nx,
    const double sigma_inner_z,
    const double sigma_inner_y,
    const double sigma_inner_x,
    const double sigma_outer_z,
    const double sigma_outer_y,
    const double sigma_outer_x,
    const double window_ratio,
    Profiler &profile
) {
    const auto kiz0 = gaussian_kernel(sigma_inner_z, 0, window_ratio);
    const auto kiy0 = gaussian_kernel(sigma_inner_y, 0, window_ratio);
    const auto kix0 = gaussian_kernel(sigma_inner_x, 0, window_ratio);
    const auto kiz1 = gaussian_kernel(sigma_inner_z, 1, window_ratio);
    const auto kiy1 = gaussian_kernel(sigma_inner_y, 1, window_ratio);
    const auto kix1 = gaussian_kernel(sigma_inner_x, 1, window_ratio);
    const auto koz0 = gaussian_kernel(sigma_outer_z, 0, window_ratio);
    const auto koy0 = gaussian_kernel(sigma_outer_y, 0, window_ratio);
    const auto kox0 = gaussian_kernel(sigma_outer_x, 0, window_ratio);

    float *work = workspace[0];
    float *gz = workspace[1];
    float *gy = workspace[2];
    float *gx = workspace[3];

    gaussian_first_axis_3d_profiled(in, gx, nz, ny, nx, kiz0, profile);
    gaussian_remaining_axes_3d_profiled(
        gx, gy, work, nz, ny, nx, kiy1, kix0, profile
    );
    gaussian_remaining_axes_3d_profiled(
        gx, gx, work, nz, ny, nx, kiy0, kix1, profile
    );
    gaussian_first_axis_3d_profiled(in, gz, nz, ny, nx, kiz1, profile);
    gaussian_remaining_axes_3d_profiled(
        gz, gz, work, nz, ny, nx, kiy0, kix0, profile
    );

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "outer_products");
        const std::array<const float *, 3> gradients{gz, gy, gx};
        convolve_axis_strided_outer_products<3>(
            gradients, components, 1, nz, ny * nx, koz0
        );
    }

    for (float *component : components) {
        gaussian_remaining_axes_3d_profiled(
            component, component, work, nz, ny, nx, koy0, kox0, profile
        );
    }
}

} // namespace detail

inline void structure_tensor_2d(
    const float *in,
    float *out,
    const std::ptrdiff_t ny,
    const std::ptrdiff_t nx,
    const double sigma_inner_y,
    const double sigma_inner_x,
    const double sigma_outer_y,
    const double sigma_outer_x,
    const double window_ratio
) {
    BIOIMAGE_PROFILE_INIT(profile);
    const std::ptrdiff_t n = ny * nx;
    const std::array<float *, 3> components{out, out + n, out + 2 * n};
    std::unique_ptr<float[]> scratch;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "scratch_alloc");
        scratch = detail::allocate_scratch(n, 3);
    }
    const std::array<float *, 3> workspace{
        detail::scratch_slot(scratch, n, 0),
        detail::scratch_slot(scratch, n, 1),
        detail::scratch_slot(scratch, n, 2),
    };
    detail::structure_tensor_components_2d_profiled(
        in, components, workspace, ny, nx,
        sigma_inner_y, sigma_inner_x, sigma_outer_y, sigma_outer_x,
        window_ratio, profile
    );
    BIOIMAGE_PROFILE_REPORT(profile);
}

inline void structure_tensor_3d(
    const float *in,
    float *out,
    const std::ptrdiff_t nz,
    const std::ptrdiff_t ny,
    const std::ptrdiff_t nx,
    const double sigma_inner_z,
    const double sigma_inner_y,
    const double sigma_inner_x,
    const double sigma_outer_z,
    const double sigma_outer_y,
    const double sigma_outer_x,
    const double window_ratio
) {
    BIOIMAGE_PROFILE_INIT(profile);
    const std::ptrdiff_t n = nz * ny * nx;
    const std::array<float *, 6> components{
        out, out + n, out + 2 * n, out + 3 * n, out + 4 * n, out + 5 * n,
    };
    std::unique_ptr<float[]> scratch;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "scratch_alloc");
        scratch = detail::allocate_scratch(n, 4);
    }
    const std::array<float *, 4> workspace{
        detail::scratch_slot(scratch, n, 0),
        detail::scratch_slot(scratch, n, 1),
        detail::scratch_slot(scratch, n, 2),
        detail::scratch_slot(scratch, n, 3),
    };
    detail::structure_tensor_components_3d_profiled(
        in, components, workspace, nz, ny, nx,
        sigma_inner_z, sigma_inner_y, sigma_inner_x,
        sigma_outer_z, sigma_outer_y, sigma_outer_x,
        window_ratio, profile
    );
    BIOIMAGE_PROFILE_REPORT(profile);
}

inline void structure_tensor_eigenvalues_2d(
    const float *in,
    float *out,
    const std::ptrdiff_t ny,
    const std::ptrdiff_t nx,
    const double sigma_inner_y,
    const double sigma_inner_x,
    const double sigma_outer_y,
    const double sigma_outer_x,
    const double window_ratio
) {
    BIOIMAGE_PROFILE_INIT(profile);
    const std::ptrdiff_t n = ny * nx;
    std::unique_ptr<float[]> scratch;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "scratch_alloc");
        scratch = detail::allocate_scratch(n, 6);
    }
    const std::array<float *, 3> components{
        detail::scratch_slot(scratch, n, 0),
        detail::scratch_slot(scratch, n, 1),
        detail::scratch_slot(scratch, n, 2),
    };
    const std::array<float *, 3> workspace{
        detail::scratch_slot(scratch, n, 3),
        detail::scratch_slot(scratch, n, 4),
        detail::scratch_slot(scratch, n, 5),
    };
    detail::structure_tensor_components_2d_profiled(
        in, components, workspace, ny, nx,
        sigma_inner_y, sigma_inner_x, sigma_outer_y, sigma_outer_x,
        window_ratio, profile
    );

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "eigenvalues");
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            const float half_trace = 0.5f * (components[0][i] + components[2][i]);
            const float half_difference =
                0.5f * (components[0][i] - components[2][i]);
            const float discriminant = std::sqrt(
                half_difference * half_difference + components[1][i] * components[1][i]
            );
            out[2 * i] = half_trace + discriminant;
            out[2 * i + 1] = half_trace - discriminant;
        }
    }
    BIOIMAGE_PROFILE_REPORT(profile);
}

inline void structure_tensor_eigenvalues_3d(
    const float *in,
    float *out,
    const std::ptrdiff_t nz,
    const std::ptrdiff_t ny,
    const std::ptrdiff_t nx,
    const double sigma_inner_z,
    const double sigma_inner_y,
    const double sigma_inner_x,
    const double sigma_outer_z,
    const double sigma_outer_y,
    const double sigma_outer_x,
    const double window_ratio
) {
    BIOIMAGE_PROFILE_INIT(profile);
    const std::ptrdiff_t n = nz * ny * nx;
    std::unique_ptr<float[]> scratch;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "scratch_alloc");
        scratch = detail::allocate_scratch(n, 10);
    }
    const std::array<float *, 4> workspace{
        detail::scratch_slot(scratch, n, 0),
        detail::scratch_slot(scratch, n, 1),
        detail::scratch_slot(scratch, n, 2),
        detail::scratch_slot(scratch, n, 3),
    };
    const std::array<float *, 6> components{
        detail::scratch_slot(scratch, n, 4),
        detail::scratch_slot(scratch, n, 5),
        detail::scratch_slot(scratch, n, 6),
        detail::scratch_slot(scratch, n, 7),
        detail::scratch_slot(scratch, n, 8),
        detail::scratch_slot(scratch, n, 9),
    };
    detail::structure_tensor_components_3d_profiled(
        in, components, workspace, nz, ny, nx,
        sigma_inner_z, sigma_inner_y, sigma_inner_x,
        sigma_outer_z, sigma_outer_y, sigma_outer_x,
        window_ratio, profile
    );

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "eigenvalues");
        ev3_symmetric_descending_interleaved(
            components[0], components[1], components[2], components[3],
            components[4], components[5], out, n
        );
    }
    BIOIMAGE_PROFILE_REPORT(profile);
}

} // namespace bioimage_cpp::filters
