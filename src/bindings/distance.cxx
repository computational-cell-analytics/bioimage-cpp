#include "distance.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/distance/distance_transform.hxx"
#include "bioimage_cpp/distance/grid_dijkstra.hxx"
#include "bioimage_cpp/distance/non_maximum_distance_suppression.hxx"
#include "bioimage_cpp/distance/geodesic_mask.hxx"
#include "bioimage_cpp/distance/geodesic_mesh.hxx"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

using UInt8Input = nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig>;
using FloatArray = nb::ndarray<nb::numpy, float, nb::c_contig>;
using Int32Array = nb::ndarray<nb::numpy, std::int32_t, nb::c_contig>;
using Int64Input = nb::ndarray<nb::numpy, const std::int64_t, nb::c_contig>;
using Int64Array = nb::ndarray<nb::numpy, std::int64_t, nb::c_contig>;
using DoubleInput = nb::ndarray<nb::numpy, const double, nb::c_contig>;
using DoubleArray = nb::ndarray<nb::numpy, double, nb::c_contig>;

template <class Array>
std::vector<std::ptrdiff_t> ndarray_shape(const Array &array) {
    std::vector<std::ptrdiff_t> shape(array.ndim());
    for (std::size_t axis = 0; axis < array.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(array.shape(axis));
    }
    return shape;
}

std::vector<std::size_t> size_t_shape(const std::vector<std::ptrdiff_t> &shape) {
    std::vector<std::size_t> out(shape.size());
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        out[axis] = static_cast<std::size_t>(shape[axis]);
    }
    return out;
}

template <class T, class Array>
Array make_array(const std::vector<std::size_t> &shape) {
    std::size_t size = 1;
    for (const auto axis_size : shape) {
        size *= axis_size;
    }
    auto *data = new T[size]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<T *>(p); });
    return Array(data, shape.size(), shape.data(), owner);
}

void check_buffer_shape(
    const char *name,
    const std::vector<std::ptrdiff_t> &actual,
    const std::vector<std::size_t> &expected
) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument(
            std::string(name) + ": buffer ndim mismatch, expected " +
            std::to_string(expected.size()) + ", got " + std::to_string(actual.size())
        );
    }
    for (std::size_t axis = 0; axis < expected.size(); ++axis) {
        if (static_cast<std::size_t>(actual[axis]) != expected[axis]) {
            throw std::invalid_argument(
                std::string(name) + ": buffer shape mismatch at axis " +
                std::to_string(axis) + ", expected " + std::to_string(expected[axis]) +
                ", got " + std::to_string(actual[axis])
            );
        }
    }
}

nb::tuple distance_transform_uint8(
    UInt8Input input,
    const std::vector<double> &sampling,
    const bool return_distances,
    const bool return_indices,
    const bool return_vectors,
    std::optional<FloatArray> distances_buf,
    std::optional<Int32Array> indices_buf,
    std::optional<FloatArray> vectors_buf,
    const std::size_t n_threads
) {
    if (input.ndim() == 0) {
        throw std::invalid_argument("input must have ndim >= 1, got ndim=0");
    }
    if (sampling.size() != input.ndim()) {
        throw std::invalid_argument(
            "sampling must have length matching input ndim, got ndim=" +
            std::to_string(input.ndim()) + ", sampling length=" +
            std::to_string(sampling.size())
        );
    }

    const auto shape = ndarray_shape(input);
    const auto distances_shape = size_t_shape(shape);
    std::vector<std::size_t> indices_shape;
    indices_shape.reserve(shape.size() + 1);
    indices_shape.push_back(shape.size());
    for (const auto axis_size : shape) {
        indices_shape.push_back(static_cast<std::size_t>(axis_size));
    }
    std::vector<std::size_t> vectors_shape = distances_shape;
    vectors_shape.push_back(shape.size());

    // Pre-allocated buffers: validate shape; otherwise allocate new arrays.
    bool distances_user_provided = false;
    bool indices_user_provided = false;
    bool vectors_user_provided = false;

    FloatArray distances_array;
    if (return_distances) {
        if (distances_buf.has_value()) {
            check_buffer_shape("distances", ndarray_shape(*distances_buf), distances_shape);
            distances_array = *distances_buf;
            distances_user_provided = true;
        } else {
            distances_array = make_array<float, FloatArray>(distances_shape);
        }
    }

    Int32Array indices_array;
    if (return_indices) {
        if (indices_buf.has_value()) {
            check_buffer_shape("indices", ndarray_shape(*indices_buf), indices_shape);
            indices_array = *indices_buf;
            indices_user_provided = true;
        } else {
            indices_array = make_array<std::int32_t, Int32Array>(indices_shape);
        }
    }

    FloatArray vectors_array;
    if (return_vectors) {
        if (vectors_buf.has_value()) {
            check_buffer_shape("vectors", ndarray_shape(*vectors_buf), vectors_shape);
            vectors_array = *vectors_buf;
            vectors_user_provided = true;
        } else {
            vectors_array = make_array<float, FloatArray>(vectors_shape);
        }
    }

    ConstArrayView<std::uint8_t> input_view{input.data(), shape, {}};
    ArrayView<float> distances_view{
        return_distances ? distances_array.data() : nullptr,
        shape,
        {},
    };
    std::vector<std::ptrdiff_t> indices_view_shape(indices_shape.begin(), indices_shape.end());
    ArrayView<std::int32_t> indices_view{
        return_indices ? indices_array.data() : nullptr,
        indices_view_shape,
        {},
    };
    std::vector<std::ptrdiff_t> vectors_view_shape(vectors_shape.begin(), vectors_shape.end());
    ArrayView<float> vectors_view{
        return_vectors ? vectors_array.data() : nullptr,
        vectors_view_shape,
        {},
    };

    {
        nb::gil_scoped_release release;
        distance::distance_transform(
            input_view,
            sampling,
            {distances_view, indices_view, vectors_view},
            n_threads
        );
    }

    auto distances_result = (!return_distances || distances_user_provided)
        ? nb::object(nb::none())
        : nb::cast(distances_array);
    auto indices_result = (!return_indices || indices_user_provided)
        ? nb::object(nb::none())
        : nb::cast(indices_array);
    auto vectors_result = (!return_vectors || vectors_user_provided)
        ? nb::object(nb::none())
        : nb::cast(vectors_array);

    return nb::make_tuple(distances_result, indices_result, vectors_result);
}

template <class PointT>
nb::ndarray<nb::numpy, PointT, nb::c_contig> non_maximum_distance_suppression_impl(
    nb::ndarray<nb::numpy, const float, nb::c_contig> distance_map,
    nb::ndarray<nb::numpy, const PointT, nb::c_contig> points,
    const std::size_t n_threads
) {
    if (distance_map.ndim() == 0) {
        throw std::invalid_argument("distance_map must have ndim >= 1, got ndim=0");
    }
    if (points.ndim() != 2) {
        throw std::invalid_argument(
            "points must have ndim == 2, got ndim=" + std::to_string(points.ndim())
        );
    }
    const auto coord_ndim = static_cast<std::size_t>(points.shape(1));
    if (coord_ndim != distance_map.ndim()) {
        throw std::invalid_argument(
            "points.shape[1] must match distance_map ndim, got points.shape[1]=" +
            std::to_string(coord_ndim) + ", distance_map.ndim()=" +
            std::to_string(distance_map.ndim())
        );
    }

    const auto map_shape = ndarray_shape(distance_map);
    const auto n_points = static_cast<std::size_t>(points.shape(0));

    // Bounds-check every coordinate before dropping the GIL.
    const PointT *points_data = points.data();
    for (std::size_t i = 0; i < n_points; ++i) {
        for (std::size_t d = 0; d < coord_ndim; ++d) {
            const PointT coord = points_data[i * coord_ndim + d];
            if constexpr (std::is_signed_v<PointT>) {
                if (coord < 0) {
                    throw std::invalid_argument(
                        "points coordinate out of bounds: points[" + std::to_string(i) +
                        ", " + std::to_string(d) + "]=" + std::to_string(coord) +
                        " is negative"
                    );
                }
            }
            if (static_cast<std::ptrdiff_t>(coord) >= map_shape[d]) {
                throw std::invalid_argument(
                    "points coordinate out of bounds: points[" + std::to_string(i) +
                    ", " + std::to_string(d) + "]=" + std::to_string(coord) +
                    " >= distance_map.shape[" + std::to_string(d) + "]=" +
                    std::to_string(map_shape[d])
                );
            }
        }
    }

    ConstArrayView<float> map_view{distance_map.data(), map_shape, {}};
    ConstArrayView<PointT> points_view{points_data, ndarray_shape(points), {}};

    std::vector<std::size_t> kept_indices;
    {
        nb::gil_scoped_release release;
        distance::non_maximum_distance_suppression(
            map_view, points_view, kept_indices, n_threads
        );
    }

    const std::size_t n_kept = kept_indices.size();
    std::vector<std::size_t> out_shape{n_kept, coord_ndim};
    auto output =
        make_array<PointT, nb::ndarray<nb::numpy, PointT, nb::c_contig>>(out_shape);
    PointT *out_data = output.data();
    for (std::size_t k = 0; k < n_kept; ++k) {
        const std::size_t i = kept_indices[k];
        for (std::size_t d = 0; d < coord_ndim; ++d) {
            out_data[k * coord_ndim + d] = points_data[i * coord_ndim + d];
        }
    }
    return output;
}

// ---------------------------------------------------------------------------
// Grid Dijkstra and geodesic distances (masks + meshes).
// ---------------------------------------------------------------------------

// Build a ConstArrayView<double> over an optional speed ndarray, checking its
// shape against `expected`. Returns nullptr when no speed was supplied.
struct OptionalSpeed {
    ConstArrayView<double> view;
    const ConstArrayView<double> *ptr = nullptr;
};

OptionalSpeed make_optional_speed(
    const std::optional<DoubleInput> &speed,
    const std::vector<std::ptrdiff_t> &shape,
    const char *name = "speed"
) {
    OptionalSpeed result;
    if (speed.has_value()) {
        check_buffer_shape(name, ndarray_shape(*speed), size_t_shape(shape));
        result.view = ConstArrayView<double>{speed->data(), shape, {}};
        result.ptr = &result.view;
    }
    return result;
}

distance::DijkstraCostMode dijkstra_cost_mode_from_int(const int value) {
    switch (value) {
        case 0:
            return distance::DijkstraCostMode::Physical;
        case 1:
            return distance::DijkstraCostMode::Node;
        case 2:
            return distance::DijkstraCostMode::NodeTimesPhysical;
        default:
            throw std::invalid_argument("invalid Dijkstra cost mode");
    }
}

nb::tuple dijkstra_distance_field_mask(
    UInt8Input mask,
    Int64Input sources,
    const int connectivity,
    const std::vector<double> &spacing,
    std::optional<DoubleInput> costs,
    const int cost_mode,
    const bool return_predecessors
) {
    if (sources.ndim() != 2 || sources.shape(1) != mask.ndim()) {
        throw std::invalid_argument(
            "sources must have shape (n_sources, mask.ndim), got ndim=" +
            std::to_string(sources.ndim())
        );
    }
    const auto shape = ndarray_shape(mask);
    const auto strides = bioimage_cpp::detail::c_order_strides(shape);
    ConstArrayView<std::int64_t> sources_view{
        sources.data(), ndarray_shape(sources), {}
    };
    const auto source_indices = distance::detail::linear_indices_from_coords(
        sources_view, shape, strides, "sources"
    );

    ConstArrayView<std::uint8_t> mask_view{mask.data(), shape, {}};
    const OptionalSpeed costs_opt = make_optional_speed(costs, shape, "costs");
    distance::DijkstraResult result;
    {
        nb::gil_scoped_release release;
        result = distance::dijkstra_distance_field(
            mask_view,
            source_indices,
            {connectivity, spacing, dijkstra_cost_mode_from_int(cost_mode)},
            costs_opt.ptr,
            return_predecessors
        );
    }

    auto distances_array = make_array<double, DoubleArray>(size_t_shape(shape));
    std::copy(result.distances.begin(), result.distances.end(), distances_array.data());

    nb::object predecessors_result = nb::none();
    if (return_predecessors) {
        auto predecessors_array = make_array<std::int64_t, Int64Array>(size_t_shape(shape));
        std::copy(
            result.predecessors.begin(),
            result.predecessors.end(),
            predecessors_array.data()
        );
        predecessors_result = nb::cast(predecessors_array);
    }
    return nb::make_tuple(nb::cast(distances_array), predecessors_result);
}

Int64Array dijkstra_path_mask(
    UInt8Input mask,
    Int64Input source,
    Int64Input targets,
    const int connectivity,
    const std::vector<double> &spacing,
    std::optional<DoubleInput> costs,
    const int cost_mode
) {
    if (source.ndim() != 2 || source.shape(0) != 1 || source.shape(1) != mask.ndim()) {
        throw std::invalid_argument("source must have shape (1, mask.ndim)");
    }
    if (targets.ndim() != 2 || targets.shape(1) != mask.ndim()) {
        throw std::invalid_argument("targets must have shape (n_targets, mask.ndim)");
    }
    const auto shape = ndarray_shape(mask);
    const auto strides = bioimage_cpp::detail::c_order_strides(shape);
    ConstArrayView<std::int64_t> source_view{source.data(), ndarray_shape(source), {}};
    ConstArrayView<std::int64_t> targets_view{targets.data(), ndarray_shape(targets), {}};
    const auto source_indices = distance::detail::linear_indices_from_coords(
        source_view, shape, strides, "source"
    );
    const auto target_indices = distance::detail::linear_indices_from_coords(
        targets_view, shape, strides, "targets"
    );

    ConstArrayView<std::uint8_t> mask_view{mask.data(), shape, {}};
    const OptionalSpeed costs_opt = make_optional_speed(costs, shape, "costs");
    std::vector<std::size_t> path;
    {
        nb::gil_scoped_release release;
        path = distance::dijkstra_path(
            mask_view,
            source_indices.front(),
            target_indices,
            {connectivity, spacing, dijkstra_cost_mode_from_int(cost_mode)},
            costs_opt.ptr
        );
    }

    const auto ndim = shape.size();
    auto output = make_array<std::int64_t, Int64Array>({path.size(), ndim});
    std::vector<std::ptrdiff_t> coords(ndim, 0);
    for (std::size_t i = 0; i < path.size(); ++i) {
        bioimage_cpp::detail::coords_from_index(
            static_cast<std::uint64_t>(path[i]), strides, ndim, coords.data()
        );
        for (std::size_t axis = 0; axis < ndim; ++axis) {
            output.data()[i * ndim + axis] = static_cast<std::int64_t>(coords[axis]);
        }
    }
    return output;
}

nb::tuple geodesic_distance_field_mask(
    UInt8Input mask,
    Int64Input sources,
    const std::vector<double> &sampling,
    std::optional<DoubleInput> speed,
    const bool return_gradient,
    const std::size_t n_threads
) {
    if (mask.ndim() == 0) {
        throw std::invalid_argument("mask must have ndim >= 1, got ndim=0");
    }
    if (sampling.size() != mask.ndim()) {
        throw std::invalid_argument(
            "sampling must have length matching mask ndim, got ndim=" +
            std::to_string(mask.ndim()) + ", sampling length=" +
            std::to_string(sampling.size())
        );
    }
    if (sources.ndim() != 2) {
        throw std::invalid_argument(
            "sources must have ndim == 2, got ndim=" + std::to_string(sources.ndim())
        );
    }
    if (static_cast<std::size_t>(sources.shape(1)) != mask.ndim()) {
        throw std::invalid_argument(
            "sources.shape[1] must match mask ndim, got sources.shape[1]=" +
            std::to_string(sources.shape(1)) + ", mask.ndim()=" +
            std::to_string(mask.ndim())
        );
    }

    const auto shape = ndarray_shape(mask);
    const auto ndim = shape.size();
    auto distances_array = make_array<double, DoubleArray>(size_t_shape(shape));

    ConstArrayView<std::uint8_t> mask_view{mask.data(), shape, {}};
    ConstArrayView<std::int64_t> sources_view{sources.data(), ndarray_shape(sources), {}};
    ArrayView<double> distances_view{distances_array.data(), shape, {}};
    const OptionalSpeed speed_opt = make_optional_speed(speed, shape);

    // Optional per-axis gradient output, shape (*mask.shape, ndim), float32.
    FloatArray gradient_array;
    ArrayView<float> gradient_view;
    ArrayView<float> *gradient_ptr = nullptr;
    if (return_gradient) {
        auto gradient_shape = size_t_shape(shape);
        gradient_shape.push_back(ndim);
        gradient_array = make_array<float, FloatArray>(gradient_shape);
        auto gradient_view_shape = shape;
        gradient_view_shape.push_back(static_cast<std::ptrdiff_t>(ndim));
        gradient_view = ArrayView<float>{gradient_array.data(), gradient_view_shape, {}};
        gradient_ptr = &gradient_view;
    }

    {
        nb::gil_scoped_release release;
        distance::geodesic_distance_field(
            mask_view, sources_view, sampling, speed_opt.ptr, distances_view, n_threads,
            gradient_ptr
        );
    }

    auto gradient_result =
        return_gradient ? nb::cast(gradient_array) : nb::object(nb::none());
    return nb::make_tuple(nb::cast(distances_array), gradient_result);
}

DoubleArray geodesic_distances_mask(
    UInt8Input mask,
    Int64Input points,
    const std::vector<double> &sampling,
    std::optional<DoubleInput> speed,
    const std::size_t n_threads
) {
    if (mask.ndim() == 0) {
        throw std::invalid_argument("mask must have ndim >= 1, got ndim=0");
    }
    if (sampling.size() != mask.ndim()) {
        throw std::invalid_argument(
            "sampling must have length matching mask ndim, got ndim=" +
            std::to_string(mask.ndim()) + ", sampling length=" +
            std::to_string(sampling.size())
        );
    }
    if (points.ndim() != 2) {
        throw std::invalid_argument(
            "points must have ndim == 2, got ndim=" + std::to_string(points.ndim())
        );
    }
    if (static_cast<std::size_t>(points.shape(1)) != mask.ndim()) {
        throw std::invalid_argument(
            "points.shape[1] must match mask ndim, got points.shape[1]=" +
            std::to_string(points.shape(1)) + ", mask.ndim()=" +
            std::to_string(mask.ndim())
        );
    }

    const auto shape = ndarray_shape(mask);
    const auto n_points = static_cast<std::size_t>(points.shape(0));
    auto distances_array = make_array<double, DoubleArray>({n_points, n_points});

    ConstArrayView<std::uint8_t> mask_view{mask.data(), shape, {}};
    ConstArrayView<std::int64_t> points_view{points.data(), ndarray_shape(points), {}};
    ArrayView<double> distances_view{
        distances_array.data(),
        {static_cast<std::ptrdiff_t>(n_points), static_cast<std::ptrdiff_t>(n_points)},
        {},
    };
    const OptionalSpeed speed_opt = make_optional_speed(speed, shape);

    {
        nb::gil_scoped_release release;
        distance::geodesic_distances(
            mask_view, points_view, sampling, speed_opt.ptr, distances_view, n_threads
        );
    }
    return distances_array;
}

// Validate a triangle mesh (vertices (V, 3) float64, faces (F, 3) int64) and
// return the vertex count.
std::size_t check_mesh(const DoubleInput &vertices, const Int64Input &faces) {
    if (vertices.ndim() != 2 || vertices.shape(1) != 3) {
        throw std::invalid_argument(
            "vertices must have shape (n_vertices, 3), got ndim=" +
            std::to_string(vertices.ndim())
        );
    }
    if (faces.ndim() != 2 || faces.shape(1) != 3) {
        throw std::invalid_argument(
            "faces must have shape (n_faces, 3), got ndim=" + std::to_string(faces.ndim())
        );
    }
    return static_cast<std::size_t>(vertices.shape(0));
}

DoubleArray geodesic_distance_field_mesh(
    DoubleInput vertices,
    Int64Input faces,
    Int64Input sources,
    std::optional<DoubleInput> speed,
    const std::size_t n_threads
) {
    const auto n_vertices = check_mesh(vertices, faces);
    if (sources.ndim() != 1) {
        throw std::invalid_argument(
            "sources must have ndim == 1 (vertex indices), got ndim=" +
            std::to_string(sources.ndim())
        );
    }

    auto distances_array = make_array<double, DoubleArray>({n_vertices});

    ConstArrayView<double> vertices_view{vertices.data(), ndarray_shape(vertices), {}};
    ConstArrayView<std::int64_t> faces_view{faces.data(), ndarray_shape(faces), {}};
    ConstArrayView<std::int64_t> sources_view{sources.data(), ndarray_shape(sources), {}};
    ArrayView<double> distances_view{
        distances_array.data(), {static_cast<std::ptrdiff_t>(n_vertices)}, {}
    };
    const OptionalSpeed speed_opt =
        make_optional_speed(speed, {static_cast<std::ptrdiff_t>(n_vertices)});

    {
        nb::gil_scoped_release release;
        distance::geodesic_distance_field_mesh(
            vertices_view, faces_view, sources_view, speed_opt.ptr, distances_view, n_threads
        );
    }
    return distances_array;
}

DoubleArray geodesic_distances_mesh(
    DoubleInput vertices,
    Int64Input faces,
    Int64Input points,
    std::optional<DoubleInput> speed,
    const std::size_t n_threads
) {
    const auto n_vertices = check_mesh(vertices, faces);
    if (points.ndim() != 1) {
        throw std::invalid_argument(
            "points must have ndim == 1 (vertex indices), got ndim=" +
            std::to_string(points.ndim())
        );
    }

    const auto n_points = static_cast<std::size_t>(points.shape(0));
    auto distances_array = make_array<double, DoubleArray>({n_points, n_points});

    ConstArrayView<double> vertices_view{vertices.data(), ndarray_shape(vertices), {}};
    ConstArrayView<std::int64_t> faces_view{faces.data(), ndarray_shape(faces), {}};
    ConstArrayView<std::int64_t> points_view{points.data(), ndarray_shape(points), {}};
    ArrayView<double> distances_view{
        distances_array.data(),
        {static_cast<std::ptrdiff_t>(n_points), static_cast<std::ptrdiff_t>(n_points)},
        {},
    };
    const OptionalSpeed speed_opt =
        make_optional_speed(speed, {static_cast<std::ptrdiff_t>(n_vertices)});

    {
        nb::gil_scoped_release release;
        distance::geodesic_distances_mesh(
            vertices_view, faces_view, points_view, speed_opt.ptr, distances_view, n_threads
        );
    }
    return distances_array;
}

} // namespace

void bind_distance(nb::module_ &m) {
    m.def(
        "_distance_transform_uint8",
        &distance_transform_uint8,
        nb::arg("input"),
        nb::arg("sampling"),
        nb::arg("return_distances"),
        nb::arg("return_indices"),
        nb::arg("return_vectors"),
        nb::arg("distances_buf").none(),
        nb::arg("indices_buf").none(),
        nb::arg("vectors_buf").none(),
        nb::arg("n_threads"),
        "Distance transform for a C-contiguous uint8 binary array. Computes any\n"
        "combination of (distances, indices, vectors) in a single separable F&H\n"
        "sweep. Pre-allocated output buffers are written into directly."
    );

    const char *nms_doc =
        "Non-maximum distance suppression of candidate points by a float32\n"
        "distance map. For each point p_i, keeps the point with the largest\n"
        "distance value within Euclidean distance distance_map[p_i] of p_i.\n"
        "Returns the unique selected points (shape (K, ndim)) in ascending\n"
        "input-index order. O(N^2) time and memory.";
    m.def(
        "_non_maximum_distance_suppression_int64",
        &non_maximum_distance_suppression_impl<std::int64_t>,
        nb::arg("distance_map"), nb::arg("points"), nb::arg("n_threads"), nms_doc
    );
    m.def(
        "_non_maximum_distance_suppression_uint64",
        &non_maximum_distance_suppression_impl<std::uint64_t>,
        nb::arg("distance_map"), nb::arg("points"), nb::arg("n_threads"), nms_doc
    );
    m.def(
        "_non_maximum_distance_suppression_int32",
        &non_maximum_distance_suppression_impl<std::int32_t>,
        nb::arg("distance_map"), nb::arg("points"), nb::arg("n_threads"), nms_doc
    );
    m.def(
        "_non_maximum_distance_suppression_uint32",
        &non_maximum_distance_suppression_impl<std::uint32_t>,
        nb::arg("distance_map"), nb::arg("points"), nb::arg("n_threads"), nms_doc
    );

    m.def(
        "_dijkstra_distance_field_mask",
        &dijkstra_distance_field_mask,
        nb::arg("mask"), nb::arg("sources"), nb::arg("connectivity"),
        nb::arg("spacing"), nb::arg("costs").none(), nb::arg("cost_mode"),
        nb::arg("return_predecessors"),
        "Grid Dijkstra distance field. Returns (distances, predecessors-or-None)."
    );
    m.def(
        "_dijkstra_path_mask",
        &dijkstra_path_mask,
        nb::arg("mask"), nb::arg("source"), nb::arg("targets"),
        nb::arg("connectivity"), nb::arg("spacing"), nb::arg("costs").none(),
        nb::arg("cost_mode"),
        "Early-stopping one-source/multi-target grid Dijkstra path."
    );

    m.def(
        "_geodesic_distance_field_mask",
        &geodesic_distance_field_mask,
        nb::arg("mask"), nb::arg("sources"), nb::arg("sampling"),
        nb::arg("speed").none(), nb::arg("return_gradient"), nb::arg("n_threads"),
        "Geodesic distance field within a mask from a set of source coordinates.\n"
        "mask nonzero = inside the domain. sources is (n_sources, ndim) int64.\n"
        "Returns (field, gradient): field is float64 of mask.shape (unreachable\n"
        "voxels +inf); gradient is float32 (*mask.shape, ndim) when\n"
        "return_gradient else None."
    );
    m.def(
        "_geodesic_distances_mask",
        &geodesic_distances_mask,
        nb::arg("mask"), nb::arg("points"), nb::arg("sampling"),
        nb::arg("speed").none(), nb::arg("n_threads"),
        "Full pairwise geodesic distance matrix between points within a mask.\n"
        "points is (n_points, ndim) int64. Returns a symmetric (n_points,\n"
        "n_points) float64 matrix; +inf where two points are not connected."
    );
    m.def(
        "_geodesic_distance_field_mesh",
        &geodesic_distance_field_mesh,
        nb::arg("vertices"), nb::arg("faces"), nb::arg("sources"),
        nb::arg("speed").none(), nb::arg("n_threads"),
        "Geodesic distance field on a triangle mesh from a set of source\n"
        "vertices. vertices (n_vertices, 3) float64, faces (n_faces, 3) int64,\n"
        "sources (n_sources,) int64 vertex indices. Returns (n_vertices,) float64."
    );
    m.def(
        "_geodesic_distances_mesh",
        &geodesic_distances_mesh,
        nb::arg("vertices"), nb::arg("faces"), nb::arg("points"),
        nb::arg("speed").none(), nb::arg("n_threads"),
        "Full pairwise geodesic distance matrix between mesh vertices.\n"
        "points is (n_points,) int64 vertex indices. Returns a symmetric\n"
        "(n_points, n_points) float64 matrix."
    );
}

} // namespace bioimage_cpp::bindings
