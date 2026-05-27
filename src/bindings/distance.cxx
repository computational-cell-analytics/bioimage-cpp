#include "distance.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/distance_transform.hxx"
#include "bioimage_cpp/non_maximum_distance_suppression.hxx"

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
    (void)n_threads;  // Reserved for future parallelization; single-threaded.

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
        distance::non_maximum_distance_suppression(map_view, points_view, kept_indices);
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
}

} // namespace bioimage_cpp::bindings
