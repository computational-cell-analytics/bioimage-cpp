#include "mesh.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/mesh/marching_cubes.hxx"
#include "bioimage_cpp/mesh/smoothing.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

using FloatInput = nb::ndarray<nb::numpy, const float, nb::c_contig>;
using UInt8Input = nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig>;
using FloatOutput = nb::ndarray<nb::numpy, float, nb::c_contig>;
using Int32Output = nb::ndarray<nb::numpy, std::int32_t, nb::c_contig>;

template <class T>
using InputArray = nb::ndarray<nb::numpy, const T, nb::c_contig>;

template <class T>
using OutputArray = nb::ndarray<nb::numpy, T, nb::c_contig>;

template <class T, class Output>
Output output_array(std::vector<T> &&values, const std::vector<std::size_t> &shape) {
    if (values.empty()) {
        // A zero-size allocation keeps the data pointer valid for nanobind's
        // ndarray constructor without paying an elementwise copy.
        auto allocation = std::make_unique<T[]>(0);
        auto *data = allocation.get();
        nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<T *>(p); });
        allocation.release();
        return Output(data, shape.size(), shape.data(), owner);
    }

    // NumPy owns the heap-allocated vector through the capsule. The vector is
    // never resized after this point, so data() remains valid for the full
    // lifetime of the returned ndarray.
    auto allocation = std::make_unique<std::vector<T>>(std::move(values));
    auto *data = allocation->data();
    nb::capsule owner(allocation.get(), [](void *p) noexcept {
        delete static_cast<std::vector<T> *>(p);
    });
    allocation.release();
    return Output(data, shape.size(), shape.data(), owner);
}

std::vector<std::ptrdiff_t> shape_of(const FloatInput &array) {
    std::vector<std::ptrdiff_t> shape(array.ndim());
    for (std::size_t axis = 0; axis < array.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(array.shape(axis));
    }
    return shape;
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
        mesh::smooth_mesh<V, std::int64_t>(
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

nb::tuple marching_cubes_float32(
    FloatInput volume,
    const double level,
    const int step_size,
    const bool classic,
    const bool descent,
    std::optional<UInt8Input> mask,
    const bool allow_degenerate
) {
    BIOIMAGE_PROFILE_INIT(profiler);
    if (volume.ndim() != 3) {
        throw std::invalid_argument(
            "volume must have ndim=3, got ndim=" + std::to_string(volume.ndim())
        );
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (volume.shape(axis) < 2) {
            throw std::invalid_argument("volume dimensions must all be at least 2");
        }
    }
    if (step_size < 1) {
        throw std::invalid_argument("step_size must be at least 1");
    }
    if (mask.has_value()) {
        if (mask->ndim() != 3) {
            throw std::invalid_argument(
                "mask must have ndim=3, got ndim=" + std::to_string(mask->ndim())
            );
        }
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (mask->shape(axis) != volume.shape(axis)) {
                throw std::invalid_argument("mask must have the same shape as volume");
            }
        }
    }

    const auto shape = shape_of(volume);
    ConstArrayView<float> volume_view{volume.data(), shape, {}};
    ConstArrayView<std::uint8_t> mask_view;
    const ConstArrayView<std::uint8_t> *mask_ptr = nullptr;
    if (mask.has_value()) {
        std::vector<std::ptrdiff_t> mask_shape(shape.begin(), shape.end());
        mask_view = ConstArrayView<std::uint8_t>{mask->data(), std::move(mask_shape), {}};
        mask_ptr = &mask_view;
    }

    mesh::MarchingCubesResult result;
    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "core_call");
        nb::gil_scoped_release release;
        result = mesh::marching_cubes(
            volume_view,
            level,
            step_size,
            classic ? mesh::MarchingCubesMethod::Lorensen : mesh::MarchingCubesMethod::Lewiner,
            descent ? mesh::GradientDirection::Descent : mesh::GradientDirection::Ascent,
            mask_ptr,
            allow_degenerate
        );
    }
    if (result.values.empty()) {
        throw std::runtime_error("No surface found at the given iso value.");
    }

    const std::size_t n_vertices = result.values.size();
    const std::size_t n_faces = result.faces.size() / 3;
    FloatOutput vertices;
    Int32Output faces;
    FloatOutput normals;
    FloatOutput values;
    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "numpy_handoff");
        vertices = output_array<float, FloatOutput>(
            std::move(result.vertices), {n_vertices, 3}
        );
        faces = output_array<std::int32_t, Int32Output>(
            std::move(result.faces), {n_faces, 3}
        );
        normals = output_array<float, FloatOutput>(
            std::move(result.normals), {n_vertices, 3}
        );
        values = output_array<float, FloatOutput>(
            std::move(result.values), {n_vertices}
        );
    }
    BIOIMAGE_PROFILE_REPORT(profiler);
    return nb::make_tuple(std::move(vertices), std::move(faces), std::move(normals), std::move(values));
}

} // namespace

void bind_mesh(nb::module_ &m) {
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
    m.def(
        "_marching_cubes_float32",
        &marching_cubes_float32,
        nb::arg("volume"),
        nb::arg("level"),
        nb::arg("step_size"),
        nb::arg("classic"),
        nb::arg("descent"),
        nb::arg("mask") = nb::none(),
        nb::arg("allow_degenerate") = true,
        "Marching Cubes 33/Lorensen extraction from a contiguous float32 3-D volume."
    );
}

} // namespace bioimage_cpp::bindings
