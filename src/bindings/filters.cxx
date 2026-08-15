#include "filters.hxx"
#include "ndarray.hxx"

#include "bioimage_cpp/filters/eigenvectors.hxx"
#include "bioimage_cpp/filters/gaussian.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

using ConstImage = nb::ndarray<nb::numpy, const float, nb::c_contig>;
using Image = nb::ndarray<nb::numpy, float, nb::c_contig>;
using ConstMask = nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig>;

template <class T>
using ConstTypedArray = nb::ndarray<nb::numpy, const T, nb::c_contig>;

template <class T>
using TypedArray = nb::ndarray<nb::numpy, T, nb::c_contig>;

void require_ndim(const ConstImage &image, int expected, const char *function) {
    if (static_cast<int>(image.ndim()) != expected) {
        throw std::invalid_argument(
            std::string(function) + ": image must have ndim=" + std::to_string(expected) +
            ", got ndim=" + std::to_string(image.ndim())
        );
    }
}

void require_positive_sigma(double sigma, const char *name, const char *function) {
    if (!(std::isfinite(sigma) && sigma > 0.0)) {
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
    if (!std::isfinite(window_size) || window_size < 0.0) {
        throw std::invalid_argument(
            std::string(function) + ": window_size must be >= 0 (0 selects the "
            "default), got " + std::to_string(window_size)
        );
    }
}

Image allocate_image(const std::size_t *shape, std::size_t ndim) {
    return detail::make_array<float>(std::span<const std::size_t>(shape, ndim));
}

Image allocate_image_for_overwrite(const std::size_t *shape, std::size_t ndim) {
    return detail::make_array_for_overwrite<float>(
        std::span<const std::size_t>(shape, ndim)
    );
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
// Structure tensor. Output has a leading upper-triangle component axis.
// ---------------------------------------------------------------------------

Image structure_tensor_2d(
    ConstImage image,
    double sigma_inner_y, double sigma_inner_x,
    double sigma_outer_y, double sigma_outer_x,
    double window_ratio
) {
    const char *fn = "structure_tensor_2d";
    require_ndim(image, 2, fn);
    require_positive_sigma(sigma_inner_y, "sigma_inner_y", fn);
    require_positive_sigma(sigma_inner_x, "sigma_inner_x", fn);
    require_positive_sigma(sigma_outer_y, "sigma_outer_y", fn);
    require_positive_sigma(sigma_outer_x, "sigma_outer_x", fn);
    require_non_negative_window(window_ratio, fn);

    const std::size_t ny = image.shape(0);
    const std::size_t nx = image.shape(1);
    const std::size_t shape[3] = {3, ny, nx};
    Image out = allocate_image_for_overwrite(shape, 3);

    const float *in_ptr = image.data();
    float *out_ptr = out.data();
    {
        nb::gil_scoped_release release;
        filters::structure_tensor_2d(
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

Image structure_tensor_3d(
    ConstImage image,
    double sigma_inner_z, double sigma_inner_y, double sigma_inner_x,
    double sigma_outer_z, double sigma_outer_y, double sigma_outer_x,
    double window_ratio
) {
    const char *fn = "structure_tensor_3d";
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
    const std::size_t shape[4] = {6, nz, ny, nx};
    Image out = allocate_image_for_overwrite(shape, 4);

    const float *in_ptr = image.data();
    float *out_ptr = out.data();
    {
        nb::gil_scoped_release release;
        filters::structure_tensor_3d(
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

void ev3_symmetric_float32(ConstImage components, Image out) {
    const char *fn = "_filters_ev3_symmetric_float32";
    require_ndim(components, 2, fn);
    if (components.shape(0) != 6) {
        throw std::invalid_argument(
            std::string(fn) + ": components must have shape (6, n), got first "
            "dimension=" + std::to_string(components.shape(0))
        );
    }

    if (out.ndim() != 2 || out.shape(0) != components.shape(1) ||
        out.shape(1) != 3) {
        throw std::invalid_argument(
            std::string(fn) + ": out must have shape (n, 3), got ndim=" +
            std::to_string(out.ndim()) +
            (out.ndim() == 2
                 ? ", shape=(" + std::to_string(out.shape(0)) + ", " +
                       std::to_string(out.shape(1)) + ")"
                 : "")
        );
    }

    const std::size_t n = components.shape(1);
    if (n > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::invalid_argument(
            std::string(fn) + ": n is too large"
        );
    }
    const float *components_ptr = components.data();
    for (std::size_t i = 0; i < 6 * n; ++i) {
        if (!std::isfinite(components_ptr[i])) {
            throw std::invalid_argument(
                std::string(fn) + ": components must contain only finite values"
            );
        }
    }

    float *out_ptr = out.data();
    {
        nb::gil_scoped_release release;
        filters::ev3_symmetric_descending_interleaved(
            components_ptr,
            components_ptr + n,
            components_ptr + 2 * n,
            components_ptr + 3 * n,
            components_ptr + 4 * n,
            components_ptr + 5 * n,
            out_ptr,
            static_cast<std::ptrdiff_t>(n)
        );
    }
}

template <class T>
TypedArray<T> symmetric_eigenvector(
    ConstTypedArray<T> components,
    const std::size_t index,
    std::optional<ConstMask> mask
) {
    const char *fn = "symmetric_eigenvector";
    if (components.ndim() < 1) {
        throw std::invalid_argument(
            std::string(fn) + ": components must have ndim >= 1"
        );
    }

    const std::size_t component_count = components.shape(0);
    std::size_t matrix_dimension = 0;
    if (component_count == 3) {
        matrix_dimension = 2;
    } else if (component_count == 6) {
        matrix_dimension = 3;
    } else {
        throw std::invalid_argument(
            std::string(fn) + ": components.shape[0] must be 3 or 6, got " +
            std::to_string(component_count)
        );
    }
    if (index >= matrix_dimension) {
        throw std::invalid_argument(
            std::string(fn) + ": index must be in [0, " +
            std::to_string(matrix_dimension) + "), got " + std::to_string(index)
        );
    }

    std::vector<std::size_t> batch_shape(components.ndim() - 1);
    for (std::size_t axis = 1; axis < components.ndim(); ++axis) {
        batch_shape[axis - 1] = components.shape(axis);
    }
    const std::size_t n = detail::checked_array_size(batch_shape);
    if (n > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::invalid_argument(std::string(fn) + ": batch is too large");
    }
    if (n != 0 && component_count > std::numeric_limits<std::size_t>::max() / n) {
        throw std::invalid_argument(std::string(fn) + ": component size overflows size_t");
    }

    const std::uint8_t *mask_ptr = nullptr;
    if (mask.has_value()) {
        if (mask->ndim() != batch_shape.size()) {
            throw std::invalid_argument(
                std::string(fn) + ": mask shape must match components batch shape"
            );
        }
        for (std::size_t axis = 0; axis < batch_shape.size(); ++axis) {
            if (mask->shape(axis) != batch_shape[axis]) {
                throw std::invalid_argument(
                    std::string(fn) + ": mask shape must match components batch shape"
                );
            }
        }
        mask_ptr = mask->data();
    }

    const T *components_ptr = components.data();
    for (std::size_t i = 0; i < component_count * n; ++i) {
        if (!std::isfinite(components_ptr[i])) {
            throw std::invalid_argument(
                std::string(fn) + ": components must contain only finite values"
            );
        }
    }

    std::vector<std::size_t> output_shape = batch_shape;
    output_shape.push_back(matrix_dimension);
    auto output = detail::make_array_for_overwrite<T>(output_shape);
    T *output_ptr = output.data();
    {
        nb::gil_scoped_release release;
        if (matrix_dimension == 2) {
            filters::ev2_symmetric_eigenvector_descending(
                components_ptr,
                components_ptr + n,
                components_ptr + 2 * n,
                index,
                mask_ptr,
                output_ptr,
                static_cast<std::ptrdiff_t>(n)
            );
        } else {
            filters::ev3_symmetric_eigenvector_descending(
                components_ptr,
                components_ptr + n,
                components_ptr + 2 * n,
                components_ptr + 3 * n,
                components_ptr + 4 * n,
                components_ptr + 5 * n,
                index,
                mask_ptr,
                output_ptr,
                static_cast<std::ptrdiff_t>(n)
            );
        }
    }
    return output;
}

} // namespace

void bind_filters(nb::module_ &m) {
    m.def(
        "_filters_convolution_backend", &filters::convolution_backend,
        "Return the selected internal filter convolution backend."
    );
    m.def(
        "_filters_eigenvalue_backend", &filters::eigenvalue_backend,
        "Return the selected internal filter eigenvalue backend."
    );
    m.def(
        "_filters_ev3_symmetric_float32", &ev3_symmetric_float32,
        nb::arg("components"), nb::arg("out"),
        "Compute sorted eigenvalues for symmetric 3x3 float32 matrices."
    );
    m.def(
        "_symmetric_eigenvector_float32", &symmetric_eigenvector<float>,
        nb::arg("components"), nb::arg("index"), nb::arg("mask") = nb::none(),
        "Return one descending-order eigenvector from packed float32 matrices."
    );
    m.def(
        "_symmetric_eigenvector_float64", &symmetric_eigenvector<double>,
        nb::arg("components"), nb::arg("index"), nb::arg("mask") = nb::none(),
        "Return one descending-order eigenvector from packed float64 matrices."
    );
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
        "_structure_tensor_2d_float32", &structure_tensor_2d,
        nb::arg("image"),
        nb::arg("sigma_inner_y"), nb::arg("sigma_inner_x"),
        nb::arg("sigma_outer_y"), nb::arg("sigma_outer_x"),
        nb::arg("window_size") = 0.0,
        "Structure tensor of a 2D float32 image. "
        "Output shape: (3, ny, nx), ordered as (Jyy, Jyx, Jxx)."
    );
    m.def(
        "_structure_tensor_3d_float32", &structure_tensor_3d,
        nb::arg("image"),
        nb::arg("sigma_inner_z"), nb::arg("sigma_inner_y"), nb::arg("sigma_inner_x"),
        nb::arg("sigma_outer_z"), nb::arg("sigma_outer_y"), nb::arg("sigma_outer_x"),
        nb::arg("window_size") = 0.0,
        "Structure tensor of a 3D float32 image. "
        "Output shape: (6, nz, ny, nx), ordered as "
        "(Jzz, Jzy, Jzx, Jyy, Jyx, Jxx)."
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
