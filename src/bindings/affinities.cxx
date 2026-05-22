#include "affinities.hxx"

#include "bioimage_cpp/affinities/compute_affinities.hxx"
#include "bioimage_cpp/affinities/compute_embedding_distances.hxx"
#include "bioimage_cpp/array_view.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

template <class T>
using LabelArray = nb::ndarray<nb::numpy, const T, nb::c_contig>;
using FloatArray = nb::ndarray<nb::numpy, float, nb::c_contig>;
using ConstFloatArray = nb::ndarray<nb::numpy, const float, nb::c_contig>;
using UInt8Array = nb::ndarray<nb::numpy, std::uint8_t, nb::c_contig>;

template <class T>
std::vector<std::ptrdiff_t> ndarray_shape(const T &array) {
    std::vector<std::ptrdiff_t> shape(array.ndim());
    for (std::size_t axis = 0; axis < array.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(array.shape(axis));
    }
    return shape;
}

FloatArray make_float_array(const std::vector<std::size_t> &shape) {
    std::size_t size = 1;
    for (const auto axis_size : shape) {
        size *= axis_size;
    }
    auto *data = new float[size]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<float *>(p); });
    return FloatArray(data, shape.size(), shape.data(), owner);
}

UInt8Array make_uint8_array(const std::vector<std::size_t> &shape) {
    std::size_t size = 1;
    for (const auto axis_size : shape) {
        size *= axis_size;
    }
    auto *data = new std::uint8_t[size]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<std::uint8_t *>(p); });
    return UInt8Array(data, shape.size(), shape.data(), owner);
}

template <std::size_t D>
std::vector<std::array<std::ptrdiff_t, D>> validate_offsets(
    const std::vector<std::vector<std::ptrdiff_t>> &offsets
) {
    std::vector<std::array<std::ptrdiff_t, D>> result;
    result.reserve(offsets.size());
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        if (offsets[i].size() != D) {
            throw std::invalid_argument(
                "each offset must have length matching the spatial ndim, got "
                "spatial ndim=" + std::to_string(D) +
                ", offset[" + std::to_string(i) + "].length=" +
                std::to_string(offsets[i].size())
            );
        }
        std::array<std::ptrdiff_t, D> entry{};
        for (std::size_t axis = 0; axis < D; ++axis) {
            entry[axis] = offsets[i][axis];
        }
        result.push_back(entry);
    }
    return result;
}

template <class LabelT, std::size_t D>
std::pair<FloatArray, UInt8Array> compute_affinities_nd_t(
    LabelArray<LabelT> labels,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::optional<LabelT> ignore_label,
    const bool return_mask,
    const std::size_t number_of_threads
) {
    if (labels.ndim() != D) {
        throw std::invalid_argument(
            "labels must have ndim=" + std::to_string(D) +
            ", got ndim=" + std::to_string(labels.ndim())
        );
    }
    if (offsets.empty()) {
        throw std::invalid_argument("offsets must not be empty");
    }
    auto offsets_typed = validate_offsets<D>(offsets);

    const auto labels_shape = ndarray_shape(labels);

    std::vector<std::size_t> out_shape;
    out_shape.reserve(D + 1);
    out_shape.push_back(offsets.size());
    for (std::size_t axis = 0; axis < D; ++axis) {
        out_shape.push_back(static_cast<std::size_t>(labels_shape[axis]));
    }

    std::vector<std::ptrdiff_t> out_view_shape(out_shape.size());
    for (std::size_t axis = 0; axis < out_shape.size(); ++axis) {
        out_view_shape[axis] = static_cast<std::ptrdiff_t>(out_shape[axis]);
    }

    auto affs = make_float_array(out_shape);
    UInt8Array mask = return_mask
        ? make_uint8_array(out_shape)
        : UInt8Array(nullptr, 0, nullptr);

    ConstArrayView<LabelT> labels_view{
        labels.data(),
        labels_shape,
        {},
    };
    ArrayView<float> affs_view{
        affs.data(),
        out_view_shape,
        {},
    };
    ArrayView<std::uint8_t> mask_view{
        return_mask ? mask.data() : nullptr,
        out_view_shape,
        {},
    };
    const ArrayView<std::uint8_t> *mask_ptr = return_mask ? &mask_view : nullptr;

    {
        nb::gil_scoped_release release;
        if constexpr (D == 2) {
            affinities::compute_affinities_2d<LabelT, float>(
                labels_view, offsets_typed, affs_view, mask_ptr,
                ignore_label, number_of_threads
            );
        } else if constexpr (D == 3) {
            affinities::compute_affinities_3d<LabelT, float>(
                labels_view, offsets_typed, affs_view, mask_ptr,
                ignore_label, number_of_threads
            );
        }
    }

    return {std::move(affs), std::move(mask)};
}

affinities::EmbeddingNorm parse_embedding_norm(const std::string &norm) {
    if (norm == "l1") {
        return affinities::EmbeddingNorm::L1;
    }
    if (norm == "l2") {
        return affinities::EmbeddingNorm::L2;
    }
    if (norm == "cosine") {
        return affinities::EmbeddingNorm::Cosine;
    }
    throw std::invalid_argument(
        "norm must be one of (\"l1\", \"l2\", \"cosine\"), got \"" + norm + "\""
    );
}

template <std::size_t D>
FloatArray compute_embedding_distances_nd_t(
    ConstFloatArray values,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::string &norm,
    const std::size_t number_of_threads
) {
    if (values.ndim() != D + 1) {
        throw std::invalid_argument(
            "values must have ndim=" + std::to_string(D + 1) +
            " for spatial ndim=" + std::to_string(D) +
            ", got ndim=" + std::to_string(values.ndim())
        );
    }
    if (offsets.empty()) {
        throw std::invalid_argument("offsets must not be empty");
    }
    auto offsets_typed = validate_offsets<D>(offsets);
    const auto norm_enum = parse_embedding_norm(norm);

    const auto values_shape = ndarray_shape(values);

    std::vector<std::size_t> out_shape;
    out_shape.reserve(D + 1);
    out_shape.push_back(offsets.size());
    for (std::size_t axis = 0; axis < D; ++axis) {
        out_shape.push_back(static_cast<std::size_t>(values_shape[axis + 1]));
    }

    std::vector<std::ptrdiff_t> out_view_shape(out_shape.size());
    for (std::size_t axis = 0; axis < out_shape.size(); ++axis) {
        out_view_shape[axis] = static_cast<std::ptrdiff_t>(out_shape[axis]);
    }

    auto distances = make_float_array(out_shape);

    ConstArrayView<float> values_view{
        values.data(),
        values_shape,
        {},
    };
    ArrayView<float> distances_view{
        distances.data(),
        out_view_shape,
        {},
    };

    {
        nb::gil_scoped_release release;
        if constexpr (D == 2) {
            affinities::compute_embedding_distances_2d<float>(
                values_view, offsets_typed, distances_view,
                norm_enum, number_of_threads
            );
        } else if constexpr (D == 3) {
            affinities::compute_embedding_distances_3d<float>(
                values_view, offsets_typed, distances_view,
                norm_enum, number_of_threads
            );
        }
    }

    return distances;
}

} // namespace

void bind_affinities(nb::module_ &m) {
    m.def(
        "_compute_affinities_2d_uint32",
        &compute_affinities_nd_t<std::uint32_t, 2>,
        nb::arg("labels"),
        nb::arg("offsets"),
        nb::arg("ignore_label"),
        nb::arg("return_mask"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_compute_affinities_2d_uint64",
        &compute_affinities_nd_t<std::uint64_t, 2>,
        nb::arg("labels"),
        nb::arg("offsets"),
        nb::arg("ignore_label"),
        nb::arg("return_mask"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_compute_affinities_2d_int32",
        &compute_affinities_nd_t<std::int32_t, 2>,
        nb::arg("labels"),
        nb::arg("offsets"),
        nb::arg("ignore_label"),
        nb::arg("return_mask"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_compute_affinities_2d_int64",
        &compute_affinities_nd_t<std::int64_t, 2>,
        nb::arg("labels"),
        nb::arg("offsets"),
        nb::arg("ignore_label"),
        nb::arg("return_mask"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_compute_affinities_3d_uint32",
        &compute_affinities_nd_t<std::uint32_t, 3>,
        nb::arg("labels"),
        nb::arg("offsets"),
        nb::arg("ignore_label"),
        nb::arg("return_mask"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_compute_affinities_3d_uint64",
        &compute_affinities_nd_t<std::uint64_t, 3>,
        nb::arg("labels"),
        nb::arg("offsets"),
        nb::arg("ignore_label"),
        nb::arg("return_mask"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_compute_affinities_3d_int32",
        &compute_affinities_nd_t<std::int32_t, 3>,
        nb::arg("labels"),
        nb::arg("offsets"),
        nb::arg("ignore_label"),
        nb::arg("return_mask"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_compute_affinities_3d_int64",
        &compute_affinities_nd_t<std::int64_t, 3>,
        nb::arg("labels"),
        nb::arg("offsets"),
        nb::arg("ignore_label"),
        nb::arg("return_mask"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_compute_embedding_distances_2d",
        &compute_embedding_distances_nd_t<2>,
        nb::arg("values"),
        nb::arg("offsets"),
        nb::arg("norm"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_compute_embedding_distances_3d",
        &compute_embedding_distances_nd_t<3>,
        nb::arg("values"),
        nb::arg("offsets"),
        nb::arg("norm"),
        nb::arg("number_of_threads")
    );
}

} // namespace bioimage_cpp::bindings
