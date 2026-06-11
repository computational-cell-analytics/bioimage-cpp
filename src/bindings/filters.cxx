#include "filters.hxx"

#include "bioimage_cpp/filters/gaussian.hxx"

#include <nanobind/ndarray.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

using ConstImage = nb::ndarray<nb::numpy, const float, nb::c_contig>;
using Image = nb::ndarray<nb::numpy, float, nb::c_contig>;

void require_ndim(const ConstImage &image, int expected, const char *function) {
    if (static_cast<int>(image.ndim()) != expected) {
        throw std::invalid_argument(
            std::string(function) + ": image must have ndim=" + std::to_string(expected) +
            ", got ndim=" + std::to_string(image.ndim())
        );
    }
}

void require_positive_sigma(double sigma, const char *name, const char *function) {
    if (!(sigma > 0.0)) {
        throw std::invalid_argument(
            std::string(function) + ": " + name + " must be positive, got " +
            std::to_string(sigma)
        );
    }
}

void require_order(int order, const char *name, const char *function) {
    if (order < 0 || order > 2) {
        throw std::invalid_argument(
            std::string(function) + ": " + name + " must be 0, 1 or 2, got " +
            std::to_string(order)
        );
    }
}

void require_non_negative_window(double window_size, const char *function) {
    if (window_size < 0.0) {
        throw std::invalid_argument(
            std::string(function) + ": window_size must be >= 0 (0 selects the "
            "default), got " + std::to_string(window_size)
        );
    }
}

Image allocate_image(const std::size_t *shape, std::size_t ndim) {
    std::size_t total = 1;
    for (std::size_t i = 0; i < ndim; ++i) total *= shape[i];
    auto *data = new float[total]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<float *>(p); });
    return Image(data, ndim, shape, owner);
}

// ---------------------------------------------------------------------------
// gaussian_smoothing
// ---------------------------------------------------------------------------

Image gaussian_smoothing_2d(
    ConstImage image, double sigma_y, double sigma_x, double window_ratio
) {
    const char *fn = "gaussian_smoothing_2d";
    require_ndim(image, 2, fn);
    require_positive_sigma(sigma_y, "sigma_y", fn);
    require_positive_sigma(sigma_x, "sigma_x", fn);
    require_non_negative_window(window_ratio, fn);

    const std::size_t ny = image.shape(0);
    const std::size_t nx = image.shape(1);
    const std::size_t shape[2] = {ny, nx};
    Image out = allocate_image(shape, 2);

    const float *in_ptr = image.data();
    float *out_ptr = out.data();
    {
        nb::gil_scoped_release release;
        filters::gaussian_smoothing_2d(
            in_ptr, out_ptr,
            static_cast<std::ptrdiff_t>(ny),
            static_cast<std::ptrdiff_t>(nx),
            sigma_y, sigma_x, window_ratio
        );
    }
    return out;
}

Image gaussian_smoothing_3d(
    ConstImage image,
    double sigma_z, double sigma_y, double sigma_x, double window_ratio
) {
    const char *fn = "gaussian_smoothing_3d";
    require_ndim(image, 3, fn);
    require_positive_sigma(sigma_z, "sigma_z", fn);
    require_positive_sigma(sigma_y, "sigma_y", fn);
    require_positive_sigma(sigma_x, "sigma_x", fn);
    require_non_negative_window(window_ratio, fn);

    const std::size_t nz = image.shape(0);
    const std::size_t ny = image.shape(1);
    const std::size_t nx = image.shape(2);
    const std::size_t shape[3] = {nz, ny, nx};
    Image out = allocate_image(shape, 3);

    const float *in_ptr = image.data();
    float *out_ptr = out.data();
    {
        nb::gil_scoped_release release;
        filters::gaussian_smoothing_3d(
            in_ptr, out_ptr,
            static_cast<std::ptrdiff_t>(nz),
            static_cast<std::ptrdiff_t>(ny),
            static_cast<std::ptrdiff_t>(nx),
            sigma_z, sigma_y, sigma_x, window_ratio
        );
    }
    return out;
}

// ---------------------------------------------------------------------------
// gaussian_derivative
// ---------------------------------------------------------------------------

Image gaussian_derivative_2d(
    ConstImage image,
    double sigma_y, double sigma_x,
    int order_y, int order_x,
    double window_ratio
) {
    const char *fn = "gaussian_derivative_2d";
    require_ndim(image, 2, fn);
    require_positive_sigma(sigma_y, "sigma_y", fn);
    require_positive_sigma(sigma_x, "sigma_x", fn);
    require_order(order_y, "order_y", fn);
    require_order(order_x, "order_x", fn);
    require_non_negative_window(window_ratio, fn);

    const std::size_t ny = image.shape(0);
    const std::size_t nx = image.shape(1);
    const std::size_t shape[2] = {ny, nx};
    Image out = allocate_image(shape, 2);

    const float *in_ptr = image.data();
    float *out_ptr = out.data();
    {
        nb::gil_scoped_release release;
        filters::gaussian_derivative_2d(
            in_ptr, out_ptr,
            static_cast<std::ptrdiff_t>(ny),
            static_cast<std::ptrdiff_t>(nx),
            sigma_y, sigma_x, order_y, order_x, window_ratio
        );
    }
    return out;
}

Image gaussian_derivative_3d(
    ConstImage image,
    double sigma_z, double sigma_y, double sigma_x,
    int order_z, int order_y, int order_x,
    double window_ratio
) {
    const char *fn = "gaussian_derivative_3d";
    require_ndim(image, 3, fn);
    require_positive_sigma(sigma_z, "sigma_z", fn);
    require_positive_sigma(sigma_y, "sigma_y", fn);
    require_positive_sigma(sigma_x, "sigma_x", fn);
    require_order(order_z, "order_z", fn);
    require_order(order_y, "order_y", fn);
    require_order(order_x, "order_x", fn);
    require_non_negative_window(window_ratio, fn);

    const std::size_t nz = image.shape(0);
    const std::size_t ny = image.shape(1);
    const std::size_t nx = image.shape(2);
    const std::size_t shape[3] = {nz, ny, nx};
    Image out = allocate_image(shape, 3);

    const float *in_ptr = image.data();
    float *out_ptr = out.data();
    {
        nb::gil_scoped_release release;
        filters::gaussian_derivative_3d(
            in_ptr, out_ptr,
            static_cast<std::ptrdiff_t>(nz),
            static_cast<std::ptrdiff_t>(ny),
            static_cast<std::ptrdiff_t>(nx),
            sigma_z, sigma_y, sigma_x,
            order_z, order_y, order_x,
            window_ratio
        );
    }
    return out;
}

// ---------------------------------------------------------------------------
// gradient magnitude
// ---------------------------------------------------------------------------

Image gaussian_gradient_magnitude_2d(
    ConstImage image, double sigma_y, double sigma_x, double window_ratio
) {
    const char *fn = "gaussian_gradient_magnitude_2d";
    require_ndim(image, 2, fn);
    require_positive_sigma(sigma_y, "sigma_y", fn);
    require_positive_sigma(sigma_x, "sigma_x", fn);
    require_non_negative_window(window_ratio, fn);

    const std::size_t ny = image.shape(0);
    const std::size_t nx = image.shape(1);
    const std::size_t shape[2] = {ny, nx};
    Image out = allocate_image(shape, 2);

    const float *in_ptr = image.data();
    float *out_ptr = out.data();
    {
        nb::gil_scoped_release release;
        filters::gaussian_gradient_magnitude_2d(
            in_ptr, out_ptr,
            static_cast<std::ptrdiff_t>(ny),
            static_cast<std::ptrdiff_t>(nx),
            sigma_y, sigma_x, window_ratio
        );
    }
    return out;
}

Image gaussian_gradient_magnitude_3d(
    ConstImage image,
    double sigma_z, double sigma_y, double sigma_x, double window_ratio
) {
    const char *fn = "gaussian_gradient_magnitude_3d";
    require_ndim(image, 3, fn);
    require_positive_sigma(sigma_z, "sigma_z", fn);
    require_positive_sigma(sigma_y, "sigma_y", fn);
    require_positive_sigma(sigma_x, "sigma_x", fn);
    require_non_negative_window(window_ratio, fn);

    const std::size_t nz = image.shape(0);
    const std::size_t ny = image.shape(1);
    const std::size_t nx = image.shape(2);
    const std::size_t shape[3] = {nz, ny, nx};
    Image out = allocate_image(shape, 3);

    const float *in_ptr = image.data();
    float *out_ptr = out.data();
    {
        nb::gil_scoped_release release;
        filters::gaussian_gradient_magnitude_3d(
            in_ptr, out_ptr,
            static_cast<std::ptrdiff_t>(nz),
            static_cast<std::ptrdiff_t>(ny),
            static_cast<std::ptrdiff_t>(nx),
            sigma_z, sigma_y, sigma_x, window_ratio
        );
    }
    return out;
}

// ---------------------------------------------------------------------------
// Laplacian of Gaussian
// ---------------------------------------------------------------------------

Image laplacian_of_gaussian_2d(
    ConstImage image, double sigma_y, double sigma_x, double window_ratio
) {
    const char *fn = "laplacian_of_gaussian_2d";
    require_ndim(image, 2, fn);
    require_positive_sigma(sigma_y, "sigma_y", fn);
    require_positive_sigma(sigma_x, "sigma_x", fn);
    require_non_negative_window(window_ratio, fn);

    const std::size_t ny = image.shape(0);
    const std::size_t nx = image.shape(1);
    const std::size_t shape[2] = {ny, nx};
    Image out = allocate_image(shape, 2);

    const float *in_ptr = image.data();
    float *out_ptr = out.data();
    {
        nb::gil_scoped_release release;
        filters::laplacian_of_gaussian_2d(
            in_ptr, out_ptr,
            static_cast<std::ptrdiff_t>(ny),
            static_cast<std::ptrdiff_t>(nx),
            sigma_y, sigma_x, window_ratio
        );
    }
    return out;
}

Image laplacian_of_gaussian_3d(
    ConstImage image,
    double sigma_z, double sigma_y, double sigma_x, double window_ratio
) {
    const char *fn = "laplacian_of_gaussian_3d";
    require_ndim(image, 3, fn);
    require_positive_sigma(sigma_z, "sigma_z", fn);
    require_positive_sigma(sigma_y, "sigma_y", fn);
    require_positive_sigma(sigma_x, "sigma_x", fn);
    require_non_negative_window(window_ratio, fn);

    const std::size_t nz = image.shape(0);
    const std::size_t ny = image.shape(1);
    const std::size_t nx = image.shape(2);
    const std::size_t shape[3] = {nz, ny, nx};
    Image out = allocate_image(shape, 3);

    const float *in_ptr = image.data();
    float *out_ptr = out.data();
    {
        nb::gil_scoped_release release;
        filters::laplacian_of_gaussian_3d(
            in_ptr, out_ptr,
            static_cast<std::ptrdiff_t>(nz),
            static_cast<std::ptrdiff_t>(ny),
            static_cast<std::ptrdiff_t>(nx),
            sigma_z, sigma_y, sigma_x, window_ratio
        );
    }
    return out;
}

// ---------------------------------------------------------------------------
// Hessian-of-Gaussian eigenvalues. Output shape: input shape + (N,) trailing.
// ---------------------------------------------------------------------------

Image hessian_of_gaussian_eigenvalues_2d(
    ConstImage image, double sigma_y, double sigma_x, double window_ratio
) {
    const char *fn = "hessian_of_gaussian_eigenvalues_2d";
    require_ndim(image, 2, fn);
    require_positive_sigma(sigma_y, "sigma_y", fn);
    require_positive_sigma(sigma_x, "sigma_x", fn);
    require_non_negative_window(window_ratio, fn);

    const std::size_t ny = image.shape(0);
    const std::size_t nx = image.shape(1);
    const std::size_t shape[3] = {ny, nx, 2};
    Image out = allocate_image(shape, 3);

    const float *in_ptr = image.data();
    float *out_ptr = out.data();
    {
        nb::gil_scoped_release release;
        filters::hessian_of_gaussian_eigenvalues_2d(
            in_ptr, out_ptr,
            static_cast<std::ptrdiff_t>(ny),
            static_cast<std::ptrdiff_t>(nx),
            sigma_y, sigma_x, window_ratio
        );
    }
    return out;
}

Image hessian_of_gaussian_eigenvalues_3d(
    ConstImage image,
    double sigma_z, double sigma_y, double sigma_x, double window_ratio
) {
    const char *fn = "hessian_of_gaussian_eigenvalues_3d";
    require_ndim(image, 3, fn);
    require_positive_sigma(sigma_z, "sigma_z", fn);
    require_positive_sigma(sigma_y, "sigma_y", fn);
    require_positive_sigma(sigma_x, "sigma_x", fn);
    require_non_negative_window(window_ratio, fn);

    const std::size_t nz = image.shape(0);
    const std::size_t ny = image.shape(1);
    const std::size_t nx = image.shape(2);
    const std::size_t shape[4] = {nz, ny, nx, 3};
    Image out = allocate_image(shape, 4);

    const float *in_ptr = image.data();
    float *out_ptr = out.data();
    {
        nb::gil_scoped_release release;
        filters::hessian_of_gaussian_eigenvalues_3d(
            in_ptr, out_ptr,
            static_cast<std::ptrdiff_t>(nz),
            static_cast<std::ptrdiff_t>(ny),
            static_cast<std::ptrdiff_t>(nx),
            sigma_z, sigma_y, sigma_x, window_ratio
        );
    }
    return out;
}

// ---------------------------------------------------------------------------
// Structure-tensor eigenvalues. Output shape: input shape + (N,) trailing.
// ---------------------------------------------------------------------------

Image structure_tensor_eigenvalues_2d(
    ConstImage image,
    double sigma_inner_y, double sigma_inner_x,
    double sigma_outer_y, double sigma_outer_x,
    double window_ratio
) {
    const char *fn = "structure_tensor_eigenvalues_2d";
    require_ndim(image, 2, fn);
    require_positive_sigma(sigma_inner_y, "sigma_inner_y", fn);
    require_positive_sigma(sigma_inner_x, "sigma_inner_x", fn);
    require_positive_sigma(sigma_outer_y, "sigma_outer_y", fn);
    require_positive_sigma(sigma_outer_x, "sigma_outer_x", fn);
    require_non_negative_window(window_ratio, fn);

    const std::size_t ny = image.shape(0);
    const std::size_t nx = image.shape(1);
    const std::size_t shape[3] = {ny, nx, 2};
    Image out = allocate_image(shape, 3);

    const float *in_ptr = image.data();
    float *out_ptr = out.data();
    {
        nb::gil_scoped_release release;
        filters::structure_tensor_eigenvalues_2d(
            in_ptr, out_ptr,
            static_cast<std::ptrdiff_t>(ny),
            static_cast<std::ptrdiff_t>(nx),
            sigma_inner_y, sigma_inner_x,
            sigma_outer_y, sigma_outer_x,
            window_ratio
        );
    }
    return out;
}

Image structure_tensor_eigenvalues_3d(
    ConstImage image,
    double sigma_inner_z, double sigma_inner_y, double sigma_inner_x,
    double sigma_outer_z, double sigma_outer_y, double sigma_outer_x,
    double window_ratio
) {
    const char *fn = "structure_tensor_eigenvalues_3d";
    require_ndim(image, 3, fn);
    require_positive_sigma(sigma_inner_z, "sigma_inner_z", fn);
    require_positive_sigma(sigma_inner_y, "sigma_inner_y", fn);
    require_positive_sigma(sigma_inner_x, "sigma_inner_x", fn);
    require_positive_sigma(sigma_outer_z, "sigma_outer_z", fn);
    require_positive_sigma(sigma_outer_y, "sigma_outer_y", fn);
    require_positive_sigma(sigma_outer_x, "sigma_outer_x", fn);
    require_non_negative_window(window_ratio, fn);

    const std::size_t nz = image.shape(0);
    const std::size_t ny = image.shape(1);
    const std::size_t nx = image.shape(2);
    const std::size_t shape[4] = {nz, ny, nx, 3};
    Image out = allocate_image(shape, 4);

    const float *in_ptr = image.data();
    float *out_ptr = out.data();
    {
        nb::gil_scoped_release release;
        filters::structure_tensor_eigenvalues_3d(
            in_ptr, out_ptr,
            static_cast<std::ptrdiff_t>(nz),
            static_cast<std::ptrdiff_t>(ny),
            static_cast<std::ptrdiff_t>(nx),
            sigma_inner_z, sigma_inner_y, sigma_inner_x,
            sigma_outer_z, sigma_outer_y, sigma_outer_x,
            window_ratio
        );
    }
    return out;
}

} // namespace

void bind_filters(nb::module_ &m) {
    m.def(
        "_gaussian_smoothing_2d_float32", &gaussian_smoothing_2d,
        nb::arg("image"), nb::arg("sigma_y"), nb::arg("sigma_x"),
        nb::arg("window_size") = 0.0,
        "2D Gaussian smoothing on a float32 (ny, nx) image with anisotropic sigma."
    );
    m.def(
        "_gaussian_smoothing_3d_float32", &gaussian_smoothing_3d,
        nb::arg("image"),
        nb::arg("sigma_z"), nb::arg("sigma_y"), nb::arg("sigma_x"),
        nb::arg("window_size") = 0.0,
        "3D Gaussian smoothing on a float32 (nz, ny, nx) image with anisotropic sigma."
    );
    m.def(
        "_gaussian_derivative_2d_float32", &gaussian_derivative_2d,
        nb::arg("image"), nb::arg("sigma_y"), nb::arg("sigma_x"),
        nb::arg("order_y"), nb::arg("order_x"),
        nb::arg("window_size") = 0.0,
        "2D Gaussian derivative on a float32 (ny, nx) image with per-axis order."
    );
    m.def(
        "_gaussian_derivative_3d_float32", &gaussian_derivative_3d,
        nb::arg("image"),
        nb::arg("sigma_z"), nb::arg("sigma_y"), nb::arg("sigma_x"),
        nb::arg("order_z"), nb::arg("order_y"), nb::arg("order_x"),
        nb::arg("window_size") = 0.0,
        "3D Gaussian derivative on a float32 (nz, ny, nx) image with per-axis order."
    );
    m.def(
        "_gaussian_gradient_magnitude_2d_float32", &gaussian_gradient_magnitude_2d,
        nb::arg("image"), nb::arg("sigma_y"), nb::arg("sigma_x"),
        nb::arg("window_size") = 0.0,
        "Gradient magnitude of a Gaussian-smoothed 2D float32 image."
    );
    m.def(
        "_gaussian_gradient_magnitude_3d_float32", &gaussian_gradient_magnitude_3d,
        nb::arg("image"),
        nb::arg("sigma_z"), nb::arg("sigma_y"), nb::arg("sigma_x"),
        nb::arg("window_size") = 0.0,
        "Gradient magnitude of a Gaussian-smoothed 3D float32 image."
    );
    m.def(
        "_laplacian_of_gaussian_2d_float32", &laplacian_of_gaussian_2d,
        nb::arg("image"), nb::arg("sigma_y"), nb::arg("sigma_x"),
        nb::arg("window_size") = 0.0,
        "Laplacian of Gaussian on a 2D float32 image."
    );
    m.def(
        "_laplacian_of_gaussian_3d_float32", &laplacian_of_gaussian_3d,
        nb::arg("image"),
        nb::arg("sigma_z"), nb::arg("sigma_y"), nb::arg("sigma_x"),
        nb::arg("window_size") = 0.0,
        "Laplacian of Gaussian on a 3D float32 image."
    );
    m.def(
        "_hessian_of_gaussian_eigenvalues_2d_float32", &hessian_of_gaussian_eigenvalues_2d,
        nb::arg("image"), nb::arg("sigma_y"), nb::arg("sigma_x"),
        nb::arg("window_size") = 0.0,
        "Eigenvalues of the Hessian of Gaussian on a 2D float32 image. "
        "Output shape: (ny, nx, 2), sorted descending along the trailing axis."
    );
    m.def(
        "_hessian_of_gaussian_eigenvalues_3d_float32", &hessian_of_gaussian_eigenvalues_3d,
        nb::arg("image"),
        nb::arg("sigma_z"), nb::arg("sigma_y"), nb::arg("sigma_x"),
        nb::arg("window_size") = 0.0,
        "Eigenvalues of the Hessian of Gaussian on a 3D float32 image. "
        "Output shape: (nz, ny, nx, 3), sorted descending along the trailing axis."
    );
    m.def(
        "_structure_tensor_eigenvalues_2d_float32", &structure_tensor_eigenvalues_2d,
        nb::arg("image"),
        nb::arg("sigma_inner_y"), nb::arg("sigma_inner_x"),
        nb::arg("sigma_outer_y"), nb::arg("sigma_outer_x"),
        nb::arg("window_size") = 0.0,
        "Eigenvalues of the structure tensor on a 2D float32 image. "
        "Output shape: (ny, nx, 2), sorted descending along the trailing axis."
    );
    m.def(
        "_structure_tensor_eigenvalues_3d_float32", &structure_tensor_eigenvalues_3d,
        nb::arg("image"),
        nb::arg("sigma_inner_z"), nb::arg("sigma_inner_y"), nb::arg("sigma_inner_x"),
        nb::arg("sigma_outer_z"), nb::arg("sigma_outer_y"), nb::arg("sigma_outer_x"),
        nb::arg("window_size") = 0.0,
        "Eigenvalues of the structure tensor on a 3D float32 image. "
        "Output shape: (nz, ny, nx, 3), sorted descending along the trailing axis."
    );
}

} // namespace bioimage_cpp::bindings
