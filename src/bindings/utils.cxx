#include "utils.hxx"
#include "ndarray.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/run_length.hxx"
#include "bioimage_cpp/take_dict.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

template <class T>
using InputArray = nb::ndarray<nb::numpy, const T, nb::c_contig>;

template <class T>
using OutputArray = nb::ndarray<nb::numpy, T, nb::c_contig>;

template <class T>
std::unordered_map<T, T> dict_to_map(const nb::dict &relabeling) {
    std::unordered_map<T, T> result;
    result.reserve(static_cast<std::size_t>(nb::len(relabeling)));

    for (auto item : relabeling) {
        const T key = nb::cast<T>(item.first);
        const T value = nb::cast<T>(item.second);
        result.emplace(key, value);
    }
    return result;
}

template <class T>
OutputArray<T> take_dict_t(const nb::dict &relabeling, InputArray<T> to_relabel) {
    std::vector<std::size_t> ndarray_shape(to_relabel.ndim());
    std::vector<std::ptrdiff_t> view_shape(to_relabel.ndim());
    for (std::size_t axis = 0; axis < to_relabel.ndim(); ++axis) {
        ndarray_shape[axis] = to_relabel.shape(axis);
        view_shape[axis] = static_cast<std::ptrdiff_t>(to_relabel.shape(axis));
    }

    auto output_array = detail::make_array<T>(ndarray_shape);
    auto *data = output_array.data();

    ConstArrayView<T> input{
        to_relabel.data(),
        view_shape,
        {},
    };
    ArrayView<T> output{
        data,
        view_shape,
        {},
    };
    const auto map = dict_to_map<T>(relabeling);

    {
        nb::gil_scoped_release release;
        take_dict<T>(map, input, output);
    }

    return output_array;
}


template <class T>
OutputArray<std::int64_t> compute_rle_t(InputArray<T> mask) {
    std::vector<std::ptrdiff_t> view_shape(mask.ndim());
    for (std::size_t axis = 0; axis < mask.ndim(); ++axis) {
        view_shape[axis] = static_cast<std::ptrdiff_t>(mask.shape(axis));
    }

    ConstArrayView<T> input{
        mask.data(),
        view_shape,
        {},
    };

    std::vector<std::int64_t> counts;
    {
        nb::gil_scoped_release release;
        counts = compute_rle<T>(input);
    }

    return detail::copy_vector_to_array(counts);
}

} // namespace

void bind_utils(nb::module_ &m) {
    m.def(
        "_take_dict_uint32",
        &take_dict_t<std::uint32_t>,
        nb::arg("relabeling"),
        nb::arg("to_relabel"),
        "Map a contiguous uint32 array through an integer dictionary."
    );
    m.def(
        "_take_dict_uint64",
        &take_dict_t<std::uint64_t>,
        nb::arg("relabeling"),
        nb::arg("to_relabel"),
        "Map a contiguous uint64 array through an integer dictionary."
    );
    m.def(
        "_take_dict_int32",
        &take_dict_t<std::int32_t>,
        nb::arg("relabeling"),
        nb::arg("to_relabel"),
        "Map a contiguous int32 array through an integer dictionary."
    );
    m.def(
        "_take_dict_int64",
        &take_dict_t<std::int64_t>,
        nb::arg("relabeling"),
        nb::arg("to_relabel"),
        "Map a contiguous int64 array through an integer dictionary."
    );
    m.def("_compute_rle_bool", &compute_rle_t<bool>, nb::arg("mask"),
          "COCO-style binary run-length encoding of a contiguous bool array.");
    m.def("_compute_rle_uint8", &compute_rle_t<std::uint8_t>, nb::arg("mask"),
          "COCO-style binary run-length encoding of a contiguous uint8 array.");
    m.def("_compute_rle_uint16", &compute_rle_t<std::uint16_t>, nb::arg("mask"),
          "COCO-style binary run-length encoding of a contiguous uint16 array.");
    m.def("_compute_rle_uint32", &compute_rle_t<std::uint32_t>, nb::arg("mask"),
          "COCO-style binary run-length encoding of a contiguous uint32 array.");
    m.def("_compute_rle_uint64", &compute_rle_t<std::uint64_t>, nb::arg("mask"),
          "COCO-style binary run-length encoding of a contiguous uint64 array.");
    m.def("_compute_rle_int32", &compute_rle_t<std::int32_t>, nb::arg("mask"),
          "COCO-style binary run-length encoding of a contiguous int32 array.");
    m.def("_compute_rle_int64", &compute_rle_t<std::int64_t>, nb::arg("mask"),
          "COCO-style binary run-length encoding of a contiguous int64 array.");
}

} // namespace bioimage_cpp::bindings
