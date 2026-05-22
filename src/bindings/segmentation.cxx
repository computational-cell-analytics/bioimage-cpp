#include "segmentation.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/segmentation/mutex_watershed.hxx"
#include "bioimage_cpp/segmentation/relabel_sequential.hxx"
#include "bioimage_cpp/segmentation/semantic_mutex_watershed.hxx"
#include "bioimage_cpp/segmentation/watershed.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
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

template <class HeightT, class LabelT>
nb::ndarray<nb::numpy, LabelT, nb::c_contig> watershed_t(
    nb::ndarray<nb::numpy, const HeightT, nb::c_contig> image,
    nb::ndarray<nb::numpy, const LabelT, nb::c_contig> markers,
    std::optional<nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig>> mask
) {
    if (markers.ndim() != image.ndim()) {
        throw std::invalid_argument("markers shape must match image shape");
    }
    std::vector<std::ptrdiff_t> image_shape(image.ndim());
    std::vector<std::size_t> out_shape(image.ndim());
    for (std::size_t axis = 0; axis < image.ndim(); ++axis) {
        image_shape[axis] = static_cast<std::ptrdiff_t>(image.shape(axis));
        out_shape[axis] = image.shape(axis);
        if (markers.shape(axis) != image.shape(axis)) {
            throw std::invalid_argument("markers shape must match image shape");
        }
    }

    std::vector<std::ptrdiff_t> mask_shape;
    const std::uint8_t *mask_data = nullptr;
    if (mask.has_value()) {
        if (mask->ndim() != image.ndim()) {
            throw std::invalid_argument("mask shape must match image shape");
        }
        mask_shape.resize(mask->ndim());
        for (std::size_t axis = 0; axis < mask->ndim(); ++axis) {
            mask_shape[axis] = static_cast<std::ptrdiff_t>(mask->shape(axis));
            if (mask->shape(axis) != image.shape(axis)) {
                throw std::invalid_argument("mask shape must match image shape");
            }
        }
        mask_data = mask->data();
    }

    const auto number_of_nodes = std::accumulate(
        image_shape.begin(),
        image_shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    );
    auto *data = new LabelT[static_cast<std::size_t>(number_of_nodes)]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<LabelT *>(p); });

    ConstArrayView<HeightT> image_view{image.data(), image_shape, {}};
    ConstArrayView<LabelT> markers_view{markers.data(), image_shape, {}};
    ConstArrayView<std::uint8_t> mask_view{mask_data, mask_shape, {}};
    ArrayView<LabelT> out_view{data, image_shape, {}};

    {
        nb::gil_scoped_release release;
        watershed<HeightT, LabelT>(image_view, markers_view, mask_view, out_view);
    }

    return nb::ndarray<nb::numpy, LabelT, nb::c_contig>(
        data,
        out_shape.size(),
        out_shape.data(),
        owner
    );
}

template <class AffT, class LabelT>
nb::ndarray<nb::numpy, LabelT, nb::c_contig> watershed_from_affinities_t(
    nb::ndarray<nb::numpy, const AffT, nb::c_contig> affinities,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    nb::ndarray<nb::numpy, const LabelT, nb::c_contig> markers,
    std::optional<nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig>> mask
) {
    if (affinities.ndim() != 3 && affinities.ndim() != 4) {
        throw std::invalid_argument(
            "affinities must have ndim 3 or 4, got ndim=" +
            std::to_string(affinities.ndim())
        );
    }
    std::vector<std::ptrdiff_t> affinity_shape(affinities.ndim());
    for (std::size_t axis = 0; axis < affinities.ndim(); ++axis) {
        affinity_shape[axis] = static_cast<std::ptrdiff_t>(affinities.shape(axis));
    }

    const std::size_t spatial_ndim = affinities.ndim() - 1;
    std::vector<std::ptrdiff_t> spatial_shape(spatial_ndim);
    std::vector<std::size_t> out_shape(spatial_ndim);
    for (std::size_t axis = 0; axis < spatial_ndim; ++axis) {
        spatial_shape[axis] =
            static_cast<std::ptrdiff_t>(affinities.shape(axis + 1));
        out_shape[axis] = affinities.shape(axis + 1);
    }

    if (markers.ndim() != spatial_ndim) {
        throw std::invalid_argument("markers shape must match affinities spatial shape");
    }
    for (std::size_t axis = 0; axis < spatial_ndim; ++axis) {
        if (markers.shape(axis) != affinities.shape(axis + 1)) {
            throw std::invalid_argument("markers shape must match affinities spatial shape");
        }
    }

    std::vector<std::ptrdiff_t> mask_shape;
    const std::uint8_t *mask_data = nullptr;
    if (mask.has_value()) {
        if (mask->ndim() != spatial_ndim) {
            throw std::invalid_argument("mask shape must match affinities spatial shape");
        }
        mask_shape.resize(mask->ndim());
        for (std::size_t axis = 0; axis < spatial_ndim; ++axis) {
            mask_shape[axis] = static_cast<std::ptrdiff_t>(mask->shape(axis));
            if (mask->shape(axis) != affinities.shape(axis + 1)) {
                throw std::invalid_argument("mask shape must match affinities spatial shape");
            }
        }
        mask_data = mask->data();
    }

    const auto number_of_nodes = std::accumulate(
        spatial_shape.begin(),
        spatial_shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    );
    auto *data = new LabelT[static_cast<std::size_t>(number_of_nodes)]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<LabelT *>(p); });

    ConstArrayView<AffT> affinities_view{affinities.data(), affinity_shape, {}};
    ConstArrayView<LabelT> markers_view{markers.data(), spatial_shape, {}};
    ConstArrayView<std::uint8_t> mask_view{mask_data, mask_shape, {}};
    ArrayView<LabelT> out_view{data, spatial_shape, {}};

    {
        nb::gil_scoped_release release;
        watershed_from_affinities<AffT, LabelT>(
            affinities_view, offsets, markers_view, mask_view, out_view
        );
    }

    return nb::ndarray<nb::numpy, LabelT, nb::c_contig>(
        data,
        out_shape.size(),
        out_shape.data(),
        owner
    );
}

template <class T>
nb::tuple relabel_sequential_t(
    nb::ndarray<nb::numpy, const T, nb::c_contig> input,
    const T offset
) {
    std::vector<std::size_t> ndarray_shape(input.ndim());
    std::vector<std::ptrdiff_t> view_shape(input.ndim());
    for (std::size_t axis = 0; axis < input.ndim(); ++axis) {
        ndarray_shape[axis] = input.shape(axis);
        view_shape[axis] = static_cast<std::ptrdiff_t>(input.shape(axis));
    }

    const auto n = std::accumulate(
        view_shape.begin(),
        view_shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    );

    auto *relabeled_data = new T[static_cast<std::size_t>(n)]();
    nb::capsule relabeled_owner(relabeled_data, [](void *p) noexcept {
        delete[] static_cast<T *>(p);
    });

    ConstArrayView<T> input_view{input.data(), view_shape, {}};
    ArrayView<T> out_view{relabeled_data, view_shape, {}};

    segmentation::RelabelSequentialMaps<T> maps;
    {
        nb::gil_scoped_release release;
        maps = segmentation::relabel_sequential<T>(input_view, offset, out_view);
    }

    const std::size_t forward_size = maps.forward_map.size();
    auto *forward_data = new T[forward_size];
    std::copy(maps.forward_map.begin(), maps.forward_map.end(), forward_data);
    nb::capsule forward_owner(forward_data, [](void *p) noexcept {
        delete[] static_cast<T *>(p);
    });

    const std::size_t inverse_size = maps.inverse_map.size();
    auto *inverse_data = new T[inverse_size];
    std::copy(maps.inverse_map.begin(), maps.inverse_map.end(), inverse_data);
    nb::capsule inverse_owner(inverse_data, [](void *p) noexcept {
        delete[] static_cast<T *>(p);
    });

    auto relabeled_array = nb::ndarray<nb::numpy, T, nb::c_contig>(
        relabeled_data,
        ndarray_shape.size(),
        ndarray_shape.data(),
        relabeled_owner
    );
    std::size_t forward_shape[1] = {forward_size};
    auto forward_array = nb::ndarray<nb::numpy, T, nb::c_contig>(
        forward_data,
        1,
        forward_shape,
        forward_owner
    );
    std::size_t inverse_shape[1] = {inverse_size};
    auto inverse_array = nb::ndarray<nb::numpy, T, nb::c_contig>(
        inverse_data,
        1,
        inverse_shape,
        inverse_owner
    );

    return nb::make_tuple(
        nb::cast(relabeled_array),
        nb::cast(forward_array),
        nb::cast(inverse_array)
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

    m.def(
        "_watershed_float32_uint32",
        &watershed_t<float, std::uint32_t>,
        nb::arg("image"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Marker-controlled watershed on a 2D or 3D float32 image with uint32 markers."
    );
    m.def(
        "_watershed_float32_uint64",
        &watershed_t<float, std::uint64_t>,
        nb::arg("image"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Marker-controlled watershed on a 2D or 3D float32 image with uint64 markers."
    );
    m.def(
        "_watershed_float32_int32",
        &watershed_t<float, std::int32_t>,
        nb::arg("image"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Marker-controlled watershed on a 2D or 3D float32 image with int32 markers."
    );
    m.def(
        "_watershed_float32_int64",
        &watershed_t<float, std::int64_t>,
        nb::arg("image"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Marker-controlled watershed on a 2D or 3D float32 image with int64 markers."
    );
    m.def(
        "_watershed_float64_uint32",
        &watershed_t<double, std::uint32_t>,
        nb::arg("image"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Marker-controlled watershed on a 2D or 3D float64 image with uint32 markers."
    );
    m.def(
        "_watershed_float64_uint64",
        &watershed_t<double, std::uint64_t>,
        nb::arg("image"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Marker-controlled watershed on a 2D or 3D float64 image with uint64 markers."
    );
    m.def(
        "_watershed_float64_int32",
        &watershed_t<double, std::int32_t>,
        nb::arg("image"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Marker-controlled watershed on a 2D or 3D float64 image with int32 markers."
    );
    m.def(
        "_watershed_float64_int64",
        &watershed_t<double, std::int64_t>,
        nb::arg("image"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Marker-controlled watershed on a 2D or 3D float64 image with int64 markers."
    );

    m.def(
        "_watershed_from_affinities_float32_uint32",
        &watershed_from_affinities_t<float, std::uint32_t>,
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Affinity-based watershed on 2D or 3D nearest-neighbour float32 affinities with uint32 markers."
    );
    m.def(
        "_watershed_from_affinities_float32_uint64",
        &watershed_from_affinities_t<float, std::uint64_t>,
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Affinity-based watershed on 2D or 3D nearest-neighbour float32 affinities with uint64 markers."
    );
    m.def(
        "_watershed_from_affinities_float32_int32",
        &watershed_from_affinities_t<float, std::int32_t>,
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Affinity-based watershed on 2D or 3D nearest-neighbour float32 affinities with int32 markers."
    );
    m.def(
        "_watershed_from_affinities_float32_int64",
        &watershed_from_affinities_t<float, std::int64_t>,
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Affinity-based watershed on 2D or 3D nearest-neighbour float32 affinities with int64 markers."
    );
    m.def(
        "_watershed_from_affinities_float64_uint32",
        &watershed_from_affinities_t<double, std::uint32_t>,
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Affinity-based watershed on 2D or 3D nearest-neighbour float64 affinities with uint32 markers."
    );
    m.def(
        "_watershed_from_affinities_float64_uint64",
        &watershed_from_affinities_t<double, std::uint64_t>,
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Affinity-based watershed on 2D or 3D nearest-neighbour float64 affinities with uint64 markers."
    );
    m.def(
        "_watershed_from_affinities_float64_int32",
        &watershed_from_affinities_t<double, std::int32_t>,
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Affinity-based watershed on 2D or 3D nearest-neighbour float64 affinities with int32 markers."
    );
    m.def(
        "_watershed_from_affinities_float64_int64",
        &watershed_from_affinities_t<double, std::int64_t>,
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("markers"),
        nb::arg("mask") = nb::none(),
        "Affinity-based watershed on 2D or 3D nearest-neighbour float64 affinities with int64 markers."
    );

    m.def(
        "_relabel_sequential_uint32",
        &relabel_sequential_t<std::uint32_t>,
        nb::arg("input"),
        nb::arg("offset"),
        "Relabel a contiguous uint32 array to consecutive labels starting at offset."
    );
    m.def(
        "_relabel_sequential_uint64",
        &relabel_sequential_t<std::uint64_t>,
        nb::arg("input"),
        nb::arg("offset"),
        "Relabel a contiguous uint64 array to consecutive labels starting at offset."
    );
    m.def(
        "_relabel_sequential_int32",
        &relabel_sequential_t<std::int32_t>,
        nb::arg("input"),
        nb::arg("offset"),
        "Relabel a contiguous int32 array to consecutive labels starting at offset."
    );
    m.def(
        "_relabel_sequential_int64",
        &relabel_sequential_t<std::int64_t>,
        nb::arg("input"),
        nb::arg("offset"),
        "Relabel a contiguous int64 array to consecutive labels starting at offset."
    );
}

} // namespace bioimage_cpp::bindings
