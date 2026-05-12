#include "segmentation.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/segmentation/mutex_watershed.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

template <class T>
using AffinityArray = nb::ndarray<nb::numpy, const T, nb::c_contig>;

using LabelArray = nb::ndarray<nb::numpy, std::uint64_t, nb::c_contig>;

template <class T>
LabelArray mutex_watershed_grid_t(
    AffinityArray<T> affinities,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::size_t number_of_attractive_channels
) {
    std::vector<std::ptrdiff_t> affinity_shape(affinities.ndim());
    for (std::size_t axis = 0; axis < affinities.ndim(); ++axis) {
        affinity_shape[axis] = static_cast<std::ptrdiff_t>(affinities.shape(axis));
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
    ArrayView<std::uint64_t> out_view{
        data,
        label_view_shape,
        {},
    };

    {
        nb::gil_scoped_release release;
        mutex_watershed_grid<T>(
            affinities_view,
            offsets,
            number_of_attractive_channels,
            out_view
        );
    }

    return LabelArray(data, label_shape.size(), label_shape.data(), owner);
}

} // namespace

void bind_segmentation(nb::module_ &m) {
    m.def(
        "_mutex_watershed_grid_float32",
        &mutex_watershed_grid_t<float>,
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("number_of_attractive_channels"),
        "Run mutex watershed on a 2D or 3D image-derived grid graph with float32 affinities."
    );
    m.def(
        "_mutex_watershed_grid_float64",
        &mutex_watershed_grid_t<double>,
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("number_of_attractive_channels"),
        "Run mutex watershed on a 2D or 3D image-derived grid graph with float64 affinities."
    );
}

} // namespace bioimage_cpp::bindings
