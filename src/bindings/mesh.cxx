#include "mesh.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/mesh/marching_cubes.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

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
            mask_ptr,
            allow_degenerate
        );
    }
    if (result.values.empty()) {
        throw std::runtime_error("No surface found at the given iso value.");
    }

    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "orient_output");
        // The core mirrors the reference kernel's x/y/z convention. Public arrays
        // use NumPy's z/y/x axis order, matching skimage.measure.marching_cubes.
        for (std::size_t vertex = 0; vertex < result.values.size(); ++vertex) {
            const std::size_t base = vertex * 3;
            std::swap(result.vertices[base], result.vertices[base + 2]);
            std::swap(result.normals[base], result.normals[base + 2]);
        }
        if (descent) {
            for (std::size_t face = 0; face < result.faces.size(); face += 3) {
                std::swap(result.faces[face], result.faces[face + 2]);
            }
        }
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
