#include "segmentation.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/segmentation/mutex_watershed.hxx"
#include "bioimage_cpp/segmentation/semantic_mutex_watershed.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

template <class T>
using AffinityArray = nb::ndarray<nb::numpy, const T, nb::c_contig>;

using ValidEdgeArray = nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig>;
using LabelArray = nb::ndarray<nb::numpy, std::uint64_t, nb::c_contig>;
using SemanticLabelArray = nb::ndarray<nb::numpy, std::int64_t, nb::c_contig>;

template <class T>
LabelArray mutex_watershed_grid_t(
    AffinityArray<T> affinities,
    ValidEdgeArray valid_edges,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::size_t number_of_attractive_channels
) {
    std::vector<std::ptrdiff_t> affinity_shape(affinities.ndim());
    for (std::size_t axis = 0; axis < affinities.ndim(); ++axis) {
        affinity_shape[axis] = static_cast<std::ptrdiff_t>(affinities.shape(axis));
    }
    std::vector<std::ptrdiff_t> valid_edges_shape(valid_edges.ndim());
    for (std::size_t axis = 0; axis < valid_edges.ndim(); ++axis) {
        valid_edges_shape[axis] = static_cast<std::ptrdiff_t>(valid_edges.shape(axis));
    }

    std::vector<std::size_t> label_shape;
    label_shape.reserve(affinities.ndim() > 0 ? affinities.ndim() - 1 : 0);
    std::vector<std::ptrdiff_t> label_view_shape;
    label_view_shape.reserve(affinities.ndim() > 0 ? affinities.ndim() - 1 : 0);
    for (std::size_t axis = 1; axis < affinities.ndim(); ++axis) {
        label_shape.push_back(affinities.shape(axis));
        label_view_shape.push_back(static_cast<std::ptrdiff_t>(affinities.shape(axis)));
    }

    const auto number_of_nodes = std::accumulate(
        label_view_shape.begin(),
        label_view_shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    );
    auto *data = new std::uint64_t[static_cast<std::size_t>(number_of_nodes)]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<std::uint64_t *>(p); });

    ConstArrayView<T> affinities_view{
        affinities.data(),
        affinity_shape,
        {},
    };
    ConstArrayView<std::uint8_t> valid_edges_view{
        valid_edges.data(),
        valid_edges_shape,
        {},
    };
    ArrayView<std::uint64_t> out_view{
        data,
        label_view_shape,
        {},
    };

    {
        nb::gil_scoped_release release;
        mutex_watershed_grid<T>(
            affinities_view,
            valid_edges_view,
            offsets,
            number_of_attractive_channels,
            out_view
        );
    }

    return LabelArray(data, label_shape.size(), label_shape.data(), owner);
}

template <class T>
std::pair<LabelArray, SemanticLabelArray> semantic_mutex_watershed_grid_t(
    AffinityArray<T> affinities,
    ValidEdgeArray valid_edges,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::size_t number_of_attractive_channels,
    const std::size_t number_of_offsets
) {
    std::vector<std::ptrdiff_t> affinity_shape(affinities.ndim());
    for (std::size_t axis = 0; axis < affinities.ndim(); ++axis) {
        affinity_shape[axis] = static_cast<std::ptrdiff_t>(affinities.shape(axis));
    }
    std::vector<std::ptrdiff_t> valid_edges_shape(valid_edges.ndim());
    for (std::size_t axis = 0; axis < valid_edges.ndim(); ++axis) {
        valid_edges_shape[axis] = static_cast<std::ptrdiff_t>(valid_edges.shape(axis));
    }

    std::vector<std::size_t> label_shape;
    label_shape.reserve(affinities.ndim() > 0 ? affinities.ndim() - 1 : 0);
    std::vector<std::ptrdiff_t> label_view_shape;
    label_view_shape.reserve(affinities.ndim() > 0 ? affinities.ndim() - 1 : 0);
    for (std::size_t axis = 1; axis < affinities.ndim(); ++axis) {
        label_shape.push_back(affinities.shape(axis));
        label_view_shape.push_back(static_cast<std::ptrdiff_t>(affinities.shape(axis)));
    }

    const auto number_of_nodes = std::accumulate(
        label_view_shape.begin(),
        label_view_shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    );
    auto *node_data = new std::uint64_t[static_cast<std::size_t>(number_of_nodes)]();
    nb::capsule node_owner(
        node_data,
        [](void *p) noexcept { delete[] static_cast<std::uint64_t *>(p); }
    );
    auto *semantic_data = new std::int64_t[static_cast<std::size_t>(number_of_nodes)]();
    nb::capsule semantic_owner(
        semantic_data,
        [](void *p) noexcept { delete[] static_cast<std::int64_t *>(p); }
    );

    ConstArrayView<T> affinities_view{
        affinities.data(),
        affinity_shape,
        {},
    };
    ConstArrayView<std::uint8_t> valid_edges_view{
        valid_edges.data(),
        valid_edges_shape,
        {},
    };
    ArrayView<std::uint64_t> node_out_view{
        node_data,
        label_view_shape,
        {},
    };
    ArrayView<std::int64_t> semantic_out_view{
        semantic_data,
        label_view_shape,
        {},
    };

    {
        nb::gil_scoped_release release;
        semantic_mutex_watershed_grid<T>(
            affinities_view,
            valid_edges_view,
            offsets,
            number_of_attractive_channels,
            number_of_offsets,
            node_out_view,
            semantic_out_view
        );
    }

    return std::make_pair(
        LabelArray(node_data, label_shape.size(), label_shape.data(), node_owner),
        SemanticLabelArray(semantic_data, label_shape.size(), label_shape.data(), semantic_owner)
    );
}

} // namespace

void bind_segmentation(nb::module_ &m) {
    m.def(
        "_mutex_watershed_grid_float32",
        &mutex_watershed_grid_t<float>,
        nb::arg("affinities"),
        nb::arg("valid_edges"),
        nb::arg("offsets"),
        nb::arg("number_of_attractive_channels"),
        "Run mutex watershed on a 2D or 3D image-derived grid graph with float32 affinities."
    );
    m.def(
        "_mutex_watershed_grid_float64",
        &mutex_watershed_grid_t<double>,
        nb::arg("affinities"),
        nb::arg("valid_edges"),
        nb::arg("offsets"),
        nb::arg("number_of_attractive_channels"),
        "Run mutex watershed on a 2D or 3D image-derived grid graph with float64 affinities."
    );
    m.def(
        "_semantic_mutex_watershed_grid_float32",
        &semantic_mutex_watershed_grid_t<float>,
        nb::arg("affinities"),
        nb::arg("valid_edges"),
        nb::arg("offsets"),
        nb::arg("number_of_attractive_channels"),
        nb::arg("number_of_offsets"),
        "Run semantic mutex watershed on a 2D or 3D image-derived grid graph with float32 affinities."
    );
    m.def(
        "_semantic_mutex_watershed_grid_float64",
        &semantic_mutex_watershed_grid_t<double>,
        nb::arg("affinities"),
        nb::arg("valid_edges"),
        nb::arg("offsets"),
        nb::arg("number_of_attractive_channels"),
        nb::arg("number_of_offsets"),
        "Run semantic mutex watershed on a 2D or 3D image-derived grid graph with float64 affinities."
    );
}

} // namespace bioimage_cpp::bindings
