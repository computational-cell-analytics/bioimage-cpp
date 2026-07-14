#include "skeleton.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/skeleton/teasar.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

using UInt8Input = nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig>;
using DoubleArray = nb::ndarray<nb::numpy, double, nb::c_contig>;
using UInt64Array = nb::ndarray<nb::numpy, std::uint64_t, nb::c_contig>;
using FloatArray = nb::ndarray<nb::numpy, float, nb::c_contig>;

template <class T, class Array>
Array make_array(const std::vector<std::size_t> &shape) {
    std::size_t size = 1;
    for (const auto extent : shape) {
        size *= extent;
    }
    auto *data = new T[size]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<T *>(p); });
    return Array(data, shape.size(), shape.data(), owner);
}

nb::tuple teasar_uint8_impl(
    UInt8Input mask,
    const std::vector<double> &spacing,
    const double scale,
    const double constant,
    const double pdrf_scale,
    const double pdrf_exponent,
    const skeleton::TeasarBackend backend,
    const std::size_t n_threads
) {
    if (spacing.size() != 3) {
        throw std::invalid_argument("spacing must contain exactly three values");
    }
    std::vector<std::ptrdiff_t> shape(mask.ndim());
    for (std::size_t axis = 0; axis < mask.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(mask.shape(axis));
    }
    ConstArrayView<std::uint8_t> mask_view{mask.data(), shape, {}};
    skeleton::SkeletonGraph result;
    {
        nb::gil_scoped_release release;
        result = skeleton::teasar_with_backend(
            mask_view,
            {{spacing[0], spacing[1], spacing[2]},
             scale,
             constant,
             pdrf_scale,
             pdrf_exponent,
             n_threads},
            backend
        );
    }

    auto vertices = make_array<double, DoubleArray>({result.vertices.size(), 3});
    for (std::size_t vertex = 0; vertex < result.vertices.size(); ++vertex) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            vertices.data()[vertex * 3 + axis] = result.vertices[vertex][axis];
        }
    }

    auto edges = make_array<std::uint64_t, UInt64Array>({result.edges.size(), 2});
    for (std::size_t edge = 0; edge < result.edges.size(); ++edge) {
        edges.data()[edge * 2] = result.edges[edge][0];
        edges.data()[edge * 2 + 1] = result.edges[edge][1];
    }

    auto radii = make_array<float, FloatArray>({result.radii.size()});
    std::copy(result.radii.begin(), result.radii.end(), radii.data());
    return nb::make_tuple(vertices, edges, radii);
}

nb::tuple teasar_uint8(
    UInt8Input mask,
    const std::vector<double> &spacing,
    const double scale,
    const double constant,
    const double pdrf_scale,
    const double pdrf_exponent,
    const std::size_t n_threads
) {
    return teasar_uint8_impl(
        mask, spacing, scale, constant, pdrf_scale, pdrf_exponent,
        skeleton::TeasarBackend::Auto, n_threads
    );
}

nb::tuple teasar_uint8_backend(
    UInt8Input mask,
    const std::vector<double> &spacing,
    const double scale,
    const double constant,
    const double pdrf_scale,
    const double pdrf_exponent,
    const std::string &backend,
    const std::size_t n_threads
) {
    skeleton::TeasarBackend selected;
    if (backend == "dense-fp64") {
        selected = skeleton::TeasarBackend::DenseFloat64;
    } else if (backend == "compact-on-the-fly-fp64") {
        selected = skeleton::TeasarBackend::CompactOnTheFlyFloat64;
    } else if (backend == "compact-csr-fp64") {
        selected = skeleton::TeasarBackend::CompactCsrFloat64;
    } else if (backend == "compact-csr-fp32") {
        selected = skeleton::TeasarBackend::CompactCsrFloat32;
    } else {
        throw std::invalid_argument("unknown TEASAR development backend: " + backend);
    }
    return teasar_uint8_impl(
        mask, spacing, scale, constant, pdrf_scale, pdrf_exponent, selected,
        n_threads
    );
}

} // namespace

void bind_skeleton(nb::module_ &m) {
    m.def(
        "_teasar_uint8",
        &teasar_uint8,
        nb::arg("mask"),
        nb::arg("spacing"),
        nb::arg("scale"),
        nb::arg("constant"),
        nb::arg("pdrf_scale"),
        nb::arg("pdrf_exponent"),
        nb::arg("n_threads"),
        "Core binary 3D TEASAR skeletonization."
    );
    m.def(
        "_teasar_uint8_backend",
        &teasar_uint8_backend,
        nb::arg("mask"),
        nb::arg("spacing"),
        nb::arg("scale"),
        nb::arg("constant"),
        nb::arg("pdrf_scale"),
        nb::arg("pdrf_exponent"),
        nb::arg("backend"),
        nb::arg("n_threads") = 1,
        "Development-only TEASAR backend selector."
    );
}

} // namespace bioimage_cpp::bindings
