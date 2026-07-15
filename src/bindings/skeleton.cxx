#include "skeleton.hxx"
#include "ndarray.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/skeleton/teasar.hxx"
#include "bioimage_cpp/skeleton/teasar_labels.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

using UInt8Input = nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig>;
using DoubleArray = nb::ndarray<nb::numpy, double, nb::c_contig>;
using UInt64Array = nb::ndarray<nb::numpy, std::uint64_t, nb::c_contig>;
using FloatArray = nb::ndarray<nb::numpy, float, nb::c_contig>;

nb::tuple skeleton_graph_to_tuple(const skeleton::SkeletonGraph &result) {
    auto vertices = detail::make_array<double>({result.vertices.size(), 3});
    for (std::size_t vertex = 0; vertex < result.vertices.size(); ++vertex) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            vertices.data()[vertex * 3 + axis] = result.vertices[vertex][axis];
        }
    }

    auto edges = detail::make_array<std::uint64_t>({result.edges.size(), 2});
    for (std::size_t edge = 0; edge < result.edges.size(); ++edge) {
        edges.data()[edge * 2] = result.edges[edge][0];
        edges.data()[edge * 2 + 1] = result.edges[edge][1];
    }

    auto radii = detail::copy_vector_to_array(result.radii);
    return nb::make_tuple(vertices, edges, radii);
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
    if (mask.ndim() != 3) {
        throw std::invalid_argument(
            "mask must have ndim 3, got ndim=" + std::to_string(mask.ndim())
        );
    }
    if (spacing.size() != 3) {
        throw std::invalid_argument(
            "spacing must contain exactly three values, got " +
            std::to_string(spacing.size())
        );
    }
    std::vector<std::ptrdiff_t> shape(mask.ndim());
    for (std::size_t axis = 0; axis < mask.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(mask.shape(axis));
    }
    ConstArrayView<std::uint8_t> mask_view{mask.data(), shape, {}};
    skeleton::SkeletonGraph result;
    {
        nb::gil_scoped_release release;
        const skeleton::TeasarOptions options{
            {spacing[0], spacing[1], spacing[2]},
            scale,
            constant,
            pdrf_scale,
            pdrf_exponent,
            n_threads,
        };
        if (backend == skeleton::TeasarBackend::Auto) {
            result = skeleton::teasar(mask_view, options);
        } else {
            result = skeleton::teasar_with_backend(mask_view, options, backend);
        }
    }
    return skeleton_graph_to_tuple(result);
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
    } else {
        throw std::invalid_argument("unknown TEASAR development backend: " + backend);
    }
    return teasar_uint8_impl(
        mask, spacing, scale, constant, pdrf_scale, pdrf_exponent, selected,
        n_threads
    );
}

template <class LabelT>
nb::dict teasar_labels_impl(
    nb::ndarray<nb::numpy, const LabelT, nb::c_contig> labels,
    const LabelT background,
    const std::vector<double> &spacing,
    const double scale,
    const double constant,
    const double pdrf_scale,
    const double pdrf_exponent,
    const std::size_t n_threads
) {
    if (labels.ndim() != 3) {
        throw std::invalid_argument(
            "labels must have ndim 3, got ndim=" +
            std::to_string(labels.ndim())
        );
    }
    if (spacing.size() != 3) {
        throw std::invalid_argument(
            "spacing must contain exactly three values, got " +
            std::to_string(spacing.size())
        );
    }
    std::vector<std::ptrdiff_t> shape(labels.ndim());
    for (std::size_t axis = 0; axis < labels.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(labels.shape(axis));
    }
    ConstArrayView<LabelT> labels_view{labels.data(), shape, {}};
    std::vector<skeleton::LabeledSkeleton<LabelT>> skeletons;
    {
        nb::gil_scoped_release release;
        skeletons = skeleton::teasar_labels(
            labels_view,
            background,
            {{spacing[0], spacing[1], spacing[2]},
             scale,
             constant,
             pdrf_scale,
             pdrf_exponent,
             n_threads}
        );
    }

    nb::dict output;
    for (const auto &entry : skeletons) {
        if constexpr (std::is_signed_v<LabelT>) {
            output[nb::int_(static_cast<long long>(entry.label))] =
                skeleton_graph_to_tuple(entry.skeleton);
        } else {
            output[nb::int_(static_cast<unsigned long long>(entry.label))] =
                skeleton_graph_to_tuple(entry.skeleton);
        }
    }
    return output;
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

#define BIC_BIND_TEASAR_LABELS(name, type)                                      \
    m.def(                                                                       \
        "_teasar_labels_" name,                                                 \
        &teasar_labels_impl<type>,                                               \
        nb::arg("labels"),                                                      \
        nb::arg("background"),                                                  \
        nb::arg("spacing"),                                                     \
        nb::arg("scale"),                                                       \
        nb::arg("constant"),                                                    \
        nb::arg("pdrf_scale"),                                                  \
        nb::arg("pdrf_exponent"),                                               \
        nb::arg("n_threads"),                                                   \
        "Core multi-label 3D TEASAR skeletonization."                           \
    )

    BIC_BIND_TEASAR_LABELS("uint8", std::uint8_t);
    BIC_BIND_TEASAR_LABELS("uint16", std::uint16_t);
    BIC_BIND_TEASAR_LABELS("uint32", std::uint32_t);
    BIC_BIND_TEASAR_LABELS("uint64", std::uint64_t);
    BIC_BIND_TEASAR_LABELS("int32", std::int32_t);
    BIC_BIND_TEASAR_LABELS("int64", std::int64_t);

#undef BIC_BIND_TEASAR_LABELS
}

} // namespace bioimage_cpp::bindings
