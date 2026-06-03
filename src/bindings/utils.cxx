#include "utils.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/mesh_smoothing.hxx"
#include "bioimage_cpp/run_length.hxx"
#include "bioimage_cpp/take_dict.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
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

    const auto n = std::accumulate(
        view_shape.begin(),
        view_shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    );
    auto *data = new T[static_cast<std::size_t>(n)]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<T *>(p); });

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

    return OutputArray<T>(data, ndarray_shape.size(), ndarray_shape.data(), owner);
}

using FacesArray = nb::ndarray<nb::numpy, const std::int64_t, nb::c_contig>;

template <class V>
std::pair<OutputArray<V>, OutputArray<V>> smooth_mesh_t(
    InputArray<V> verts,
    InputArray<V> normals,
    FacesArray faces,
    std::size_t iterations,
    int n_threads
) {
    if (verts.ndim() != 2) {
        throw std::invalid_argument(
            "verts must have ndim=2, got ndim=" + std::to_string(verts.ndim())
        );
    }
    if (normals.ndim() != 2 || normals.shape(0) != verts.shape(0)
        || normals.shape(1) != verts.shape(1)) {
        throw std::invalid_argument("normals must have the same shape as verts");
    }
    if (faces.ndim() != 2 || faces.shape(1) != 3) {
        throw std::invalid_argument("faces must have shape (n_faces, 3)");
    }

    const std::size_t n_verts_size = verts.shape(0);
    const std::size_t dim_size = verts.shape(1);
    const std::size_t n_faces_size = faces.shape(0);
    const std::size_t n_total = n_verts_size * dim_size;

    std::vector<std::size_t> verts_ndarray_shape{n_verts_size, dim_size};
    std::vector<std::ptrdiff_t> verts_view_shape{
        static_cast<std::ptrdiff_t>(n_verts_size),
        static_cast<std::ptrdiff_t>(dim_size),
    };
    std::vector<std::ptrdiff_t> faces_view_shape{
        static_cast<std::ptrdiff_t>(n_faces_size),
        std::ptrdiff_t{3},
    };

    auto *verts_data = new V[n_total]();
    nb::capsule verts_owner(verts_data, [](void *p) noexcept { delete[] static_cast<V *>(p); });
    auto *normals_data = new V[n_total]();
    nb::capsule normals_owner(normals_data, [](void *p) noexcept { delete[] static_cast<V *>(p); });

    ConstArrayView<V> verts_view{verts.data(), verts_view_shape, {}};
    ConstArrayView<V> normals_view{normals.data(), verts_view_shape, {}};
    ConstArrayView<std::int64_t> faces_view{faces.data(), faces_view_shape, {}};
    ArrayView<V> out_verts_view{verts_data, verts_view_shape, {}};
    ArrayView<V> out_normals_view{normals_data, verts_view_shape, {}};

    {
        nb::gil_scoped_release release;
        smooth_mesh<V, std::int64_t>(
            verts_view,
            normals_view,
            faces_view,
            iterations,
            n_threads,
            out_verts_view,
            out_normals_view
        );
    }

    OutputArray<V> out_verts(
        verts_data, verts_ndarray_shape.size(), verts_ndarray_shape.data(), verts_owner
    );
    OutputArray<V> out_normals(
        normals_data, verts_ndarray_shape.size(), verts_ndarray_shape.data(), normals_owner
    );
    return {std::move(out_verts), std::move(out_normals)};
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

    auto *data = new std::int64_t[counts.size()];
    std::copy(counts.begin(), counts.end(), data);
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] static_cast<std::int64_t *>(p);
    });

    std::size_t shape[1] = {counts.size()};
    return OutputArray<std::int64_t>(data, 1, shape, owner);
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
    m.def(
        "_smooth_mesh_float32",
        &smooth_mesh_t<float>,
        nb::arg("verts"),
        nb::arg("normals"),
        nb::arg("faces"),
        nb::arg("iterations"),
        nb::arg("n_threads"),
        "Laplacian smoothing of a triangular mesh with float32 vertices and normals."
    );
    m.def(
        "_smooth_mesh_float64",
        &smooth_mesh_t<double>,
        nb::arg("verts"),
        nb::arg("normals"),
        nb::arg("faces"),
        nb::arg("iterations"),
        nb::arg("n_threads"),
        "Laplacian smoothing of a triangular mesh with float64 vertices and normals."
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
