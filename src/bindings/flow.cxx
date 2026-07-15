#include "flow.hxx"
#include "ndarray.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/flow/flow_density.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

using FlowArray = nb::ndarray<nb::numpy, const float, nb::c_contig>;
using MaskArray = nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig>;
using DensityArray = nb::ndarray<nb::numpy, float, nb::c_contig>;

template <class Array>
std::vector<std::ptrdiff_t> shape_of(const Array &array) {
    std::vector<std::ptrdiff_t> shape(array.ndim());
    for (std::size_t axis = 0; axis < array.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(array.shape(axis));
    }
    return shape;
}

template <std::size_t D>
DensityArray compute_flow_density_t(
    FlowArray flow,
    MaskArray fg_mask,
    const std::int64_t n_iter,
    const double dt,
    const double tol,
    const std::string &method,
    const bool restrict_to_mask,
    const std::int64_t number_of_threads
) {
    if (flow.ndim() != D + 1) {
        throw std::invalid_argument(
            "flow must have ndim=" + std::to_string(D + 1) +
            ", got ndim=" + std::to_string(flow.ndim())
        );
    }
    if (fg_mask.ndim() != D) {
        throw std::invalid_argument(
            "fg_mask must have ndim=" + std::to_string(D) +
            ", got ndim=" + std::to_string(fg_mask.ndim())
        );
    }
    if (flow.shape(0) != D) {
        throw std::invalid_argument(
            "flow first axis must match spatial ndim=" + std::to_string(D) +
            ", got " + std::to_string(flow.shape(0))
        );
    }
    for (std::size_t axis = 0; axis < D; ++axis) {
        if (flow.shape(axis + 1) != fg_mask.shape(axis)) {
            throw std::invalid_argument("flow spatial shape must match fg_mask shape");
        }
    }
    if (n_iter < 0) {
        throw std::invalid_argument("n_iter must be >= 0");
    }
    if (!std::isfinite(dt) || dt < 0.0) {
        throw std::invalid_argument("dt must be finite and >= 0");
    }
    if (!std::isfinite(tol) || tol < 0.0) {
        throw std::invalid_argument("tol must be finite and >= 0");
    }
    flow::IntegrationMethod integration_method;
    if (method == "euler") {
        integration_method = flow::IntegrationMethod::Euler;
    } else if (method == "rk2") {
        integration_method = flow::IntegrationMethod::RK2;
    } else {
        throw std::invalid_argument(
            "method must be 'euler' or 'rk2', got '" + method + "'"
        );
    }
    if (number_of_threads < 1) {
        throw std::invalid_argument("number_of_threads must be >= 1");
    }
    // Single authoritative finiteness check over the (contiguous) flow buffer;
    // the Python wrapper deliberately does not repeat this scan.
    std::size_t flow_size = D;
    for (std::size_t axis = 0; axis < D; ++axis) {
        flow_size *= flow.shape(axis + 1);
    }
    for (std::size_t index = 0; index < flow_size; ++index) {
        if (!std::isfinite(flow.data()[index])) {
            throw std::invalid_argument("flow must contain only finite values");
        }
    }

    std::vector<std::size_t> out_shape(D);
    std::vector<std::ptrdiff_t> view_shape(D);
    for (std::size_t axis = 0; axis < D; ++axis) {
        out_shape[axis] = fg_mask.shape(axis);
        view_shape[axis] = static_cast<std::ptrdiff_t>(fg_mask.shape(axis));
    }
    auto density = detail::make_array<float>(out_shape);

    const auto flow_shape = shape_of(flow);
    const auto flow_strides = bioimage_cpp::detail::c_order_strides(flow_shape);
    const auto mask_strides = bioimage_cpp::detail::c_order_strides(view_shape);
    ConstArrayView<float> flow_view{flow.data(), flow_shape, flow_strides};
    ConstArrayView<std::uint8_t> mask_view{fg_mask.data(), view_shape, mask_strides};
    ArrayView<float> density_view{density.data(), view_shape, mask_strides};

    {
        nb::gil_scoped_release release;
        if constexpr (D == 2) {
            flow::compute_flow_density_2d(
                flow_view,
                mask_view,
                density_view,
                static_cast<std::size_t>(n_iter),
                static_cast<float>(dt),
                static_cast<float>(tol),
                integration_method,
                restrict_to_mask,
                static_cast<std::size_t>(number_of_threads)
            );
        } else {
            flow::compute_flow_density_3d(
                flow_view,
                mask_view,
                density_view,
                static_cast<std::size_t>(n_iter),
                static_cast<float>(dt),
                static_cast<float>(tol),
                integration_method,
                restrict_to_mask,
                static_cast<std::size_t>(number_of_threads)
            );
        }
    }
    return density;
}

} // namespace

void bind_flow(nb::module_ &m) {
    m.def(
        "_compute_flow_density_2d_float32",
        &compute_flow_density_t<2>,
        nb::arg("flow"),
        nb::arg("fg_mask"),
        nb::arg("n_iter"),
        nb::arg("dt"),
        nb::arg("tol"),
        nb::arg("method"),
        nb::arg("restrict_to_mask"),
        nb::arg("number_of_threads"),
        "Compute a flow convergence density map for a 2D float32 flow field."
    );
    m.def(
        "_compute_flow_density_3d_float32",
        &compute_flow_density_t<3>,
        nb::arg("flow"),
        nb::arg("fg_mask"),
        nb::arg("n_iter"),
        nb::arg("dt"),
        nb::arg("tol"),
        nb::arg("method"),
        nb::arg("restrict_to_mask"),
        nb::arg("number_of_threads"),
        "Compute a flow convergence density map for a 3D float32 flow field."
    );
}

} // namespace bioimage_cpp::bindings
