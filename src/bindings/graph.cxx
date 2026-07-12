#include "graph.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/graph/breadth_first_search.hxx"
#include "bioimage_cpp/graph/connected_components.hxx"
#include "bioimage_cpp/graph/edge_weighted_watershed.hxx"
#include "bioimage_cpp/graph/feature_accumulation.hxx"
#include "bioimage_cpp/graph/grid_edge_projection.hxx"
#include "bioimage_cpp/graph/grid_features.hxx"
#include "bioimage_cpp/graph/grid_graph.hxx"
#include "bioimage_cpp/graph/label_accumulation.hxx"
#include "bioimage_cpp/graph/lifted_from_affinities.hxx"
#include "bioimage_cpp/graph/agglomeration.hxx"
#include "bioimage_cpp/graph/lifted_multicut.hxx"
#include "bioimage_cpp/graph/lifted_multicut/fusion_move.hxx"
#include "bioimage_cpp/graph/lifted_multicut/lifted_from_node_labels.hxx"
#include "bioimage_cpp/graph/multicut.hxx"
#include "bioimage_cpp/graph/mutex_watershed.hxx"
#include "bioimage_cpp/graph/multicut/fusion_move.hxx"
#include "bioimage_cpp/graph/multicut/greedy_additive.hxx"
#include "bioimage_cpp/graph/multicut/greedy_fixation.hxx"
#include "bioimage_cpp/graph/multicut/kernighan_lin.hxx"
#include "bioimage_cpp/graph/node_label_projection.hxx"
#include "bioimage_cpp/graph/proposal_generator.hxx"
#include "bioimage_cpp/graph/proposal_generators/greedy_additive_multicut.hxx"
#include "bioimage_cpp/graph/distributed/block_extraction.hxx"
#include "bioimage_cpp/graph/distributed/merge.hxx"
#include "bioimage_cpp/graph/proposal_generators/watershed.hxx"
#include "bioimage_cpp/graph/rag_coordinates.hxx"
#include "bioimage_cpp/graph/region_adjacency_graph.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <array>
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

using Graph = graph::UndirectedGraph;
using GridGraph2D = graph::GridGraph<2>;
using GridGraph3D = graph::GridGraph<3>;
using Rag = graph::RegionAdjacencyGraph;
using UInt8Array = nb::ndarray<nb::numpy, std::uint8_t, nb::c_contig>;
using UInt64Array = nb::ndarray<nb::numpy, std::uint64_t, nb::c_contig>;
using ConstUInt8Array = nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig>;
using ConstUInt64Array = nb::ndarray<nb::numpy, const std::uint64_t, nb::c_contig>;
using Int64Array = nb::ndarray<nb::numpy, std::int64_t, nb::c_contig>;
using ConstInt64Array = nb::ndarray<nb::numpy, const std::int64_t, nb::c_contig>;
using DoubleArray = nb::ndarray<nb::numpy, double, nb::c_contig>;
using ConstDoubleArray = nb::ndarray<nb::numpy, const double, nb::c_contig>;
using FloatArray = nb::ndarray<nb::numpy, float, nb::c_contig>;
using ConstFloatArray = nb::ndarray<nb::numpy, const float, nb::c_contig>;

template <class T>
using LabelArray = nb::ndarray<nb::numpy, const T, nb::c_contig>;

template <class T>
using FloatingArray = nb::ndarray<nb::numpy, T, nb::c_contig>;
template <class T>
using ConstFloatingArray = nb::ndarray<nb::numpy, const T, nb::c_contig>;

void require_uv_array(const ConstUInt64Array &uvs, const char *argument_name) {
    if (uvs.ndim() != 2 || uvs.shape(1) != 2) {
        throw std::invalid_argument(
            std::string(argument_name) + " must have shape (n_edges, 2)"
        );
    }
}

UInt64Array make_uint64_array(const std::vector<std::size_t> &shape) {
    std::size_t size = 1;
    for (const auto axis_size : shape) {
        size *= axis_size;
    }
    auto *data = new std::uint64_t[size]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<std::uint64_t *>(p); });
    return UInt64Array(data, shape.size(), shape.data(), owner);
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

Int64Array make_int64_array(const std::vector<std::size_t> &shape) {
    std::size_t size = 1;
    for (const auto axis_size : shape) {
        size *= axis_size;
    }
    auto *data = new std::int64_t[size]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<std::int64_t *>(p); });
    return Int64Array(data, shape.size(), shape.data(), owner);
}

DoubleArray make_double_array(const std::vector<std::size_t> &shape) {
    std::size_t size = 1;
    for (const auto axis_size : shape) {
        size *= axis_size;
    }
    auto *data = new double[size]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<double *>(p); });
    return DoubleArray(data, shape.size(), shape.data(), owner);
}

template <class T>
using TypedArray = nb::ndarray<nb::numpy, T, nb::c_contig>;

template <class T>
TypedArray<T> make_typed_array(const std::vector<std::size_t> &shape) {
    std::size_t size = 1;
    for (const auto axis_size : shape) {
        size *= axis_size;
    }
    auto *data = new T[size]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<T *>(p); });
    return TypedArray<T>(data, shape.size(), shape.data(), owner);
}

template <class T>
FloatingArray<T> make_floating_array(const std::vector<std::size_t> &shape) {
    std::size_t size = 1;
    for (const auto axis_size : shape) {
        size *= axis_size;
    }
    auto *data = new T[size]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<T *>(p); });
    return FloatingArray<T>(data, shape.size(), shape.data(), owner);
}

template <class T>
FloatingArray<T> vector_to_floating_array(const std::vector<T> &values) {
    auto result = make_floating_array<T>({values.size()});
    std::copy(values.begin(), values.end(), result.data());
    return result;
}

template <class T>
std::vector<std::ptrdiff_t> const_floating_shape(ConstFloatingArray<T> array) {
    std::vector<std::ptrdiff_t> shape(array.ndim());
    for (std::size_t axis = 0; axis < array.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(array.shape(axis));
    }
    return shape;
}

std::vector<std::ptrdiff_t> const_double_shape(ConstDoubleArray array) {
    std::vector<std::ptrdiff_t> shape(array.ndim());
    for (std::size_t axis = 0; axis < array.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(array.shape(axis));
    }
    return shape;
}

UInt64Array vector_to_uint64_array(const std::vector<std::uint64_t> &values) {
    auto result = make_uint64_array({values.size()});
    std::copy(values.begin(), values.end(), result.data());
    return result;
}

UInt8Array vector_to_uint8_array(const std::vector<std::uint8_t> &values) {
    auto result = make_uint8_array({values.size()});
    std::copy(values.begin(), values.end(), result.data());
    return result;
}

DoubleArray vector_to_double_array(const std::vector<double> &values) {
    auto result = make_double_array({values.size()});
    std::copy(values.begin(), values.end(), result.data());
    return result;
}

UInt64Array edges_to_uv_array(const std::vector<bioimage_cpp::detail::Edge> &edges) {
    auto result = make_uint64_array({edges.size(), 2});
    auto *data = result.data();
    for (std::size_t index = 0; index < edges.size(); ++index) {
        data[2 * index] = edges[index].first;
        data[2 * index + 1] = edges[index].second;
    }
    return result;
}

// Copy a flat per-edge partial-statistics buffer (row-major, 5 columns) into a
// NumPy `(n_edges, 5)` array. Used by the distributed block-extraction bindings.
DoubleArray block_stats_to_array(const std::vector<double> &stats) {
    const auto rows = stats.size() / 5;
    auto result = make_double_array({rows, std::size_t{5}});
    std::copy(stats.begin(), stats.end(), result.data());
    return result;
}

template <std::size_t D>
std::vector<std::uint64_t> coordinate_to_vector(
    const typename graph::GridGraph<D>::Coordinate &coordinate
) {
    return std::vector<std::uint64_t>(coordinate.begin(), coordinate.end());
}

template <std::size_t D>
typename graph::GridGraph<D>::Coordinate coordinate_from_array(
    ConstUInt64Array coordinate,
    const char *argument_name
) {
    if (coordinate.ndim() != 1 || coordinate.shape(0) != D) {
        throw std::invalid_argument(
            std::string(argument_name) + " must be a 1D uint64 array of length " +
            std::to_string(D)
        );
    }
    typename graph::GridGraph<D>::Coordinate result{};
    std::copy(coordinate.data(), coordinate.data() + D, result.begin());
    return result;
}

template <std::size_t D>
std::array<std::int64_t, D> offset_from_array(
    ConstInt64Array offset,
    const char *argument_name
) {
    if (offset.ndim() != 1 || offset.shape(0) != D) {
        throw std::invalid_argument(
            std::string(argument_name) + " must be a 1D int64 array of length " +
            std::to_string(D)
        );
    }
    std::array<std::int64_t, D> result{};
    std::copy(offset.data(), offset.data() + D, result.begin());
    return result;
}

template <std::size_t D>
std::uint64_t grid_node_id(
    const graph::GridGraph<D> &graph,
    ConstUInt64Array coordinate
) {
    return graph.node_id(coordinate_from_array<D>(coordinate, "coordinate"));
}

template <std::size_t D>
UInt64Array grid_coordinates(const graph::GridGraph<D> &graph, const std::uint64_t node) {
    return vector_to_uint64_array(coordinate_to_vector<D>(graph.coordinates(node)));
}

template <std::size_t D>
std::vector<std::uint64_t> grid_shape(const graph::GridGraph<D> &graph) {
    return coordinate_to_vector<D>(graph.shape());
}

template <std::size_t D>
std::vector<std::uint64_t> grid_strides(const graph::GridGraph<D> &graph) {
    return coordinate_to_vector<D>(graph.strides());
}

template <std::size_t D>
UInt64Array grid_edge_coordinates(
    const graph::GridGraph<D> &graph,
    const std::uint64_t edge
) {
    return vector_to_uint64_array(coordinate_to_vector<D>(graph.edge_coordinates(edge).first));
}

template <std::size_t D>
std::int64_t grid_offset_target(
    const graph::GridGraph<D> &graph,
    const std::uint64_t node,
    ConstInt64Array offset
) {
    std::uint64_t target = 0;
    if (!graph.valid_offset_target(node, offset_from_array<D>(offset, "offset"), target)) {
        return -1;
    }
    return static_cast<std::int64_t>(target);
}

template <std::size_t D>
Int64Array grid_project_edge_ids_to_pixels(const graph::GridGraph<D> &graph) {
    const auto &shape = graph.shape();
    std::vector<std::size_t> out_shape(D + 1);
    out_shape[0] = D;
    for (std::size_t d = 0; d < D; ++d) {
        out_shape[d + 1] = shape[d];
    }
    auto result = make_int64_array(out_shape);

    std::size_t total = 1;
    for (const auto axis_size : out_shape) {
        total *= axis_size;
    }
    auto *data = result.data();
    std::fill(data, data + total, static_cast<std::int64_t>(-1));

    std::vector<std::ptrdiff_t> view_shape(out_shape.size());
    for (std::size_t i = 0; i < out_shape.size(); ++i) {
        view_shape[i] = static_cast<std::ptrdiff_t>(out_shape[i]);
    }
    ArrayView<std::int64_t> view{data, view_shape, {}};

    {
        nb::gil_scoped_release release;
        graph::project_edge_ids_to_pixels<D>(graph, view);
    }
    return result;
}

template <std::size_t D>
nb::tuple grid_project_edge_ids_to_pixels_with_offsets(
    const graph::GridGraph<D> &graph,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::optional<std::vector<std::ptrdiff_t>> &strides,
    const std::optional<ConstUInt8Array> &mask
) {
    if (strides.has_value() && mask.has_value()) {
        throw std::invalid_argument("strides and mask cannot be given together");
    }

    std::vector<std::array<std::ptrdiff_t, D>> off_arr(offsets.size());
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        if (offsets[i].size() != D) {
            throw std::invalid_argument(
                "each offset must have length matching graph ndim"
            );
        }
        for (std::size_t d = 0; d < D; ++d) {
            off_arr[i][d] = offsets[i][d];
        }
    }

    const auto &shape = graph.shape();
    std::vector<std::size_t> out_shape(D + 1);
    out_shape[0] = offsets.size();
    for (std::size_t d = 0; d < D; ++d) {
        out_shape[d + 1] = shape[d];
    }
    auto result = make_int64_array(out_shape);

    std::vector<std::ptrdiff_t> view_shape(out_shape.size());
    for (std::size_t i = 0; i < out_shape.size(); ++i) {
        view_shape[i] = static_cast<std::ptrdiff_t>(out_shape[i]);
    }
    ArrayView<std::int64_t> view{result.data(), view_shape, {}};

    std::uint64_t n_valid = 0;
    if (strides.has_value()) {
        if (strides->size() != D) {
            throw std::invalid_argument(
                "strides must have length matching graph ndim"
            );
        }
        std::array<std::ptrdiff_t, D> stride_arr{};
        for (std::size_t d = 0; d < D; ++d) {
            stride_arr[d] = (*strides)[d];
        }
        nb::gil_scoped_release release;
        n_valid = graph::project_edge_ids_to_pixels_with_offsets<D>(
            graph, off_arr, stride_arr, view
        );
    } else if (mask.has_value()) {
        const auto &m = *mask;
        if (m.ndim() != D + 1 || m.shape(0) != offsets.size()) {
            throw std::invalid_argument(
                "mask shape must be (n_offsets, *graph.shape)"
            );
        }
        for (std::size_t d = 0; d < D; ++d) {
            if (m.shape(d + 1) != shape[d]) {
                throw std::invalid_argument(
                    "mask shape must be (n_offsets, *graph.shape)"
                );
            }
        }
        std::vector<std::ptrdiff_t> mask_shape(m.ndim());
        for (std::size_t i = 0; i < m.ndim(); ++i) {
            mask_shape[i] = static_cast<std::ptrdiff_t>(m.shape(i));
        }
        ConstArrayView<std::uint8_t> mview{m.data(), mask_shape, {}};
        nb::gil_scoped_release release;
        n_valid = graph::project_edge_ids_to_pixels_with_offsets<D>(
            graph, off_arr, mview, view
        );
    } else {
        nb::gil_scoped_release release;
        n_valid = graph::project_edge_ids_to_pixels_with_offsets<D>(
            graph, off_arr, view
        );
    }
    return nb::make_tuple(result, n_valid);
}

template <class T, std::size_t D>
FloatingArray<T> grid_boundary_features_t(
    const graph::GridGraph<D> &graph,
    ConstFloatingArray<T> boundary_map
) {
    auto result = make_floating_array<T>({static_cast<std::size_t>(graph.number_of_edges())});
    ConstArrayView<T> boundary_view{
        boundary_map.data(),
        const_floating_shape<T>(boundary_map),
        {},
    };
    ArrayView<T> out_view{
        result.data(),
        {static_cast<std::ptrdiff_t>(graph.number_of_edges())},
        {},
    };

    nb::gil_scoped_release release;
    graph::grid_boundary_features<T, D>(graph, boundary_view, out_view);
    return result;
}

template <class T, std::size_t D>
nb::tuple grid_affinity_features_t(
    const graph::GridGraph<D> &graph,
    ConstFloatingArray<T> affinities,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets
) {
    auto weights = make_floating_array<T>({static_cast<std::size_t>(graph.number_of_edges())});
    auto valid_edges = make_uint8_array({static_cast<std::size_t>(graph.number_of_edges())});
    ConstArrayView<T> affinities_view{
        affinities.data(),
        const_floating_shape<T>(affinities),
        {},
    };
    ArrayView<T> weights_view{
        weights.data(),
        {static_cast<std::ptrdiff_t>(graph.number_of_edges())},
        {},
    };
    ArrayView<std::uint8_t> valid_view{
        valid_edges.data(),
        {static_cast<std::ptrdiff_t>(graph.number_of_edges())},
        {},
    };

    {
        nb::gil_scoped_release release;
        graph::grid_local_affinity_features<T, D>(
            graph, affinities_view, offsets, weights_view, valid_view
        );
    }
    return nb::make_tuple(weights, valid_edges);
}

template <class T, std::size_t D>
nb::tuple grid_affinity_features_with_lifted_t(
    const graph::GridGraph<D> &graph,
    ConstFloatingArray<T> affinities,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets
) {
    ConstArrayView<T> affinities_view{
        affinities.data(),
        const_floating_shape<T>(affinities),
        {},
    };

    graph::GridLiftedAffinityFeatures<T> features;
    {
        nb::gil_scoped_release release;
        features = graph::grid_affinity_features_with_lifted<T, D>(
            graph, affinities_view, offsets
        );
    }

    return nb::make_tuple(
        vector_to_floating_array<T>(features.local_weights),
        vector_to_uint8_array(features.valid_local_edges),
        edges_to_uv_array(features.lifted_uvs),
        vector_to_floating_array<T>(features.lifted_weights),
        vector_to_uint64_array(features.lifted_offset_ids)
    );
}

std::vector<double> double_array_to_vector(
    ConstDoubleArray array,
    const char *argument_name,
    const std::uint64_t expected_size
) {
    if (array.ndim() != 1) {
        throw std::invalid_argument(std::string(argument_name) + " must be a 1D float64 array");
    }
    if (array.shape(0) != static_cast<std::size_t>(expected_size)) {
        throw std::invalid_argument(
            std::string(argument_name) + " length must match expected size"
        );
    }
    const auto *data = array.data();
    return std::vector<double>(data, data + array.shape(0));
}

std::vector<std::uint64_t> uint64_array_to_vector(
    ConstUInt64Array array,
    const char *argument_name,
    const std::uint64_t expected_size
) {
    if (array.ndim() != 1) {
        throw std::invalid_argument(std::string(argument_name) + " must be a 1D uint64 array");
    }
    if (array.shape(0) != static_cast<std::size_t>(expected_size)) {
        throw std::invalid_argument(
            std::string(argument_name) + " length must match expected size"
        );
    }
    const auto *data = array.data();
    return std::vector<std::uint64_t>(data, data + array.shape(0));
}

UInt64Array graph_nodes(const Graph &graph) {
    auto result = make_uint64_array({static_cast<std::size_t>(graph.number_of_nodes())});
    auto *data = result.data();
    for (std::uint64_t node = 0; node < graph.number_of_nodes(); ++node) {
        data[static_cast<std::size_t>(node)] = node;
    }
    return result;
}

UInt64Array graph_edges(const Graph &graph) {
    auto result = make_uint64_array({static_cast<std::size_t>(graph.number_of_edges())});
    auto *data = result.data();
    for (std::uint64_t edge = 0; edge < graph.number_of_edges(); ++edge) {
        data[static_cast<std::size_t>(edge)] = edge;
    }
    return result;
}

UInt64Array graph_uv_ids(const Graph &graph) {
    auto result = make_uint64_array({static_cast<std::size_t>(graph.number_of_edges()), 2});
    auto *data = result.data();
    for (std::uint64_t edge = 0; edge < graph.number_of_edges(); ++edge) {
        const auto uv = graph.uv(edge);
        const auto offset = static_cast<std::size_t>(2 * edge);
        data[offset] = uv.first;
        data[offset + 1] = uv.second;
    }
    return result;
}

UInt64Array graph_insert_edges(Graph &graph, ConstUInt64Array uvs) {
    require_uv_array(uvs, "uvs");
    const auto n_edges = uvs.shape(0);
    auto result = make_uint64_array({n_edges});
    auto *out = result.data();
    const auto *in = uvs.data();

    for (std::size_t index = 0; index < n_edges; ++index) {
        out[index] = graph.insert_edge(in[2 * index], in[2 * index + 1]);
    }
    return result;
}

Int64Array graph_find_edges(const Graph &graph, ConstUInt64Array uvs) {
    require_uv_array(uvs, "uvs");
    const auto n_edges = uvs.shape(0);
    auto result = make_int64_array({n_edges});
    auto *out = result.data();
    const auto *in = uvs.data();

    for (std::size_t index = 0; index < n_edges; ++index) {
        out[index] = graph.find_edge(in[2 * index], in[2 * index + 1]);
    }
    return result;
}

UInt64Array graph_node_adjacency(const Graph &graph, const std::uint64_t node) {
    const auto &adjacency = graph.node_adjacency(node);
    auto result = make_uint64_array({adjacency.size(), 2});
    auto *data = result.data();
    for (std::size_t index = 0; index < adjacency.size(); ++index) {
        data[2 * index] = adjacency[index].node;
        data[2 * index + 1] = adjacency[index].edge;
    }
    return result;
}

UInt64Array graph_serialize(const Graph &graph) {
    auto result = make_uint64_array({static_cast<std::size_t>(graph.serialization_size())});
    auto *data = result.data();
    data[0] = graph.number_of_nodes();
    data[1] = graph.number_of_edges();
    for (std::uint64_t edge = 0; edge < graph.number_of_edges(); ++edge) {
        const auto uv = graph.uv(edge);
        const auto offset = static_cast<std::size_t>(2 + 2 * edge);
        data[offset] = uv.first;
        data[offset + 1] = uv.second;
    }
    return result;
}

Graph graph_from_edges(const std::uint64_t number_of_nodes, ConstUInt64Array uvs) {
    require_uv_array(uvs, "uvs");
    Graph graph(number_of_nodes);
    const auto *in = uvs.data();
    for (std::size_t index = 0; index < uvs.shape(0); ++index) {
        graph.insert_edge(in[2 * index], in[2 * index + 1]);
    }
    return graph;
}

// Bulk-construct a graph from a pre-deduplicated (u, v) array, bypassing the
// per-edge hash dedup that ``insert_edge`` performs. The caller asserts that
// no (u, v) pair appears twice in ``uvs`` and that ``u != v`` in every row.
// Edges receive ids matching their position in ``uvs``. This is the fast path
// for copying an existing graph (its ``uv_ids()`` are unique by construction).
Graph graph_from_unique_edges(const std::uint64_t number_of_nodes, ConstUInt64Array uvs) {
    require_uv_array(uvs, "uvs");
    std::vector<Graph::Edge> edges;
    edges.reserve(static_cast<std::size_t>(uvs.shape(0)));
    const auto *in = uvs.data();
    for (std::size_t index = 0; index < uvs.shape(0); ++index) {
        const auto u = in[2 * index];
        const auto v = in[2 * index + 1];
        if (u >= v) {
            throw std::invalid_argument(
                "uvs must contain canonical edges with u < v"
            );
        }
        if (u >= number_of_nodes || v >= number_of_nodes) {
            throw std::out_of_range("edge endpoint exceeds number_of_nodes");
        }
        if (!edges.empty() && !(edges.back() < Graph::Edge{u, v})) {
            throw std::invalid_argument(
                "uvs must be strictly lexicographically sorted and duplicate-free"
            );
        }
        edges.emplace_back(u, v);
    }
    return Graph::from_sorted_unique_edges(number_of_nodes, std::move(edges));
}

Graph graph_deserialize(ConstUInt64Array serialization) {
    if (serialization.ndim() != 1) {
        throw std::invalid_argument("serialization must be a 1D uint64 array");
    }
    if (serialization.shape(0) < 2) {
        throw std::invalid_argument("serialization must have at least two entries");
    }
    const auto *data = serialization.data();
    const auto number_of_nodes = data[0];
    const auto number_of_edges = data[1];
    const auto expected_size = static_cast<std::size_t>(2 + 2 * number_of_edges);
    if (serialization.shape(0) != expected_size) {
        throw std::invalid_argument(
            "serialization size must be 2 + 2 * number_of_edges"
        );
    }

    Graph graph(number_of_nodes, number_of_edges);
    for (std::uint64_t edge = 0; edge < number_of_edges; ++edge) {
        const auto offset = static_cast<std::size_t>(2 + 2 * edge);
        graph.insert_edge(data[offset], data[offset + 1]);
    }
    return graph;
}

std::vector<std::uint64_t> nodes_to_vector(ConstUInt64Array nodes) {
    if (nodes.ndim() != 1) {
        throw std::invalid_argument("nodes must be a 1D uint64 array");
    }
    std::vector<std::uint64_t> result(nodes.shape(0));
    const auto *data = nodes.data();
    for (std::size_t index = 0; index < nodes.shape(0); ++index) {
        result[index] = data[index];
    }
    return result;
}

std::pair<UInt64Array, UInt64Array> graph_extract_subgraph_from_nodes(
    const Graph &graph,
    ConstUInt64Array nodes
) {
    const auto node_vector = nodes_to_vector(nodes);
    const auto extracted = graph.extract_subgraph_from_nodes(node_vector);
    auto inner = make_uint64_array({extracted.first.size()});
    auto outer = make_uint64_array({extracted.second.size()});
    std::copy(extracted.first.begin(), extracted.first.end(), inner.data());
    std::copy(extracted.second.begin(), extracted.second.end(), outer.data());
    return {inner, outer};
}

UInt64Array graph_edges_from_node_list(const Graph &graph, ConstUInt64Array nodes) {
    const auto extracted = graph_extract_subgraph_from_nodes(graph, nodes);
    return extracted.first;
}

UInt64Array graph_connected_components(const Graph &graph) {
    std::vector<std::uint64_t> labels;
    {
        nb::gil_scoped_release release;
        labels = graph::connected_components(graph);
    }
    return vector_to_uint64_array(labels);
}

UInt64Array graph_connected_components_masked(const Graph &graph, ConstUInt8Array edge_mask) {
    if (edge_mask.ndim() != 1) {
        throw std::invalid_argument("edge_mask must be a 1D uint8 array");
    }
    if (edge_mask.shape(0) != static_cast<std::size_t>(graph.number_of_edges())) {
        throw std::invalid_argument("edge_mask length must match graph number_of_edges");
    }
    std::vector<std::uint64_t> labels;
    {
        nb::gil_scoped_release release;
        labels = graph::connected_components(graph, edge_mask.data());
    }
    return vector_to_uint64_array(labels);
}

template <class T>
using ConstArray1D = nb::ndarray<nb::numpy, const T, nb::c_contig>;

template <class T>
std::vector<T> array_1d_to_vector(
    ConstArray1D<T> array,
    const char *argument_name,
    const std::uint64_t expected_size
) {
    if (array.ndim() != 1) {
        throw std::invalid_argument(std::string(argument_name) + " must be a 1D array");
    }
    if (array.shape(0) != static_cast<std::size_t>(expected_size)) {
        throw std::invalid_argument(
            std::string(argument_name) + " length must match expected size"
        );
    }
    const auto *data = array.data();
    return std::vector<T>(data, data + array.shape(0));
}

template <class T>
nb::ndarray<nb::numpy, T, nb::c_contig> vector_to_array_1d(const std::vector<T> &values) {
    const std::size_t size = values.size();
    auto *data = new T[size];
    std::copy(values.begin(), values.end(), data);
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<T *>(p); });
    const std::array<std::size_t, 1> shape{size};
    return nb::ndarray<nb::numpy, T, nb::c_contig>(data, 1, shape.data(), owner);
}

template <class WeightT, class SeedT>
nb::ndarray<nb::numpy, SeedT, nb::c_contig> graph_edge_weighted_watershed_t(
    const Graph &graph,
    ConstArray1D<WeightT> edge_weights,
    ConstArray1D<SeedT> seeds
) {
    const auto weight_vector =
        array_1d_to_vector<WeightT>(edge_weights, "edge_weights", graph.number_of_edges());
    const auto seed_vector =
        array_1d_to_vector<SeedT>(seeds, "seeds", graph.number_of_nodes());
    std::vector<SeedT> labels;
    {
        nb::gil_scoped_release release;
        labels = graph::edge_weighted_watershed<WeightT, SeedT>(graph, weight_vector, seed_vector);
    }
    return vector_to_array_1d<SeedT>(labels);
}

double multicut_energy(const Graph &graph, ConstDoubleArray costs, ConstUInt64Array labels) {
    const auto cost_vector = double_array_to_vector(costs, "edge_costs", graph.number_of_edges());
    const auto label_vector = uint64_array_to_vector(labels, "labels", graph.number_of_nodes());
    nb::gil_scoped_release release;
    return graph::multicut::energy(graph, cost_vector, label_vector);
}

UInt64Array multicut_greedy_additive(
    const Graph &graph,
    ConstDoubleArray costs,
    const double weight_stop,
    const double node_num_stop,
    const bool add_noise,
    const int seed,
    const double sigma
) {
    const auto cost_vector = double_array_to_vector(costs, "edge_costs", graph.number_of_edges());
    std::vector<std::uint64_t> labels;
    {
        nb::gil_scoped_release release;
        labels = graph::multicut::greedy_additive(
            graph,
            cost_vector,
            weight_stop,
            node_num_stop,
            add_noise,
            seed,
            sigma
        );
    }
    return vector_to_uint64_array(labels);
}

UInt64Array multicut_greedy_fixation(
    const Graph &graph,
    ConstDoubleArray costs,
    const double weight_stop,
    const double node_num_stop
) {
    const auto cost_vector = double_array_to_vector(costs, "edge_costs", graph.number_of_edges());
    std::vector<std::uint64_t> labels;
    {
        nb::gil_scoped_release release;
        labels = graph::multicut::greedy_fixation(graph, cost_vector, weight_stop, node_num_stop);
    }
    return vector_to_uint64_array(labels);
}

UInt64Array multicut_kernighan_lin(
    const Graph &graph,
    ConstDoubleArray costs,
    ConstUInt64Array initial_labels,
    const std::uint64_t number_of_outer_iterations,
    const double epsilon
) {
    const auto cost_vector = double_array_to_vector(costs, "edge_costs", graph.number_of_edges());
    auto label_vector = uint64_array_to_vector(initial_labels, "initial_labels", graph.number_of_nodes());
    {
        nb::gil_scoped_release release;
        label_vector = graph::multicut::kernighan_lin(
            graph,
            cost_vector,
            std::move(label_vector),
            number_of_outer_iterations,
            epsilon
        );
    }
    return vector_to_uint64_array(label_vector);
}

std::pair<UInt64Array, UInt64Array> graph_breadth_first_search(
    const Graph &graph,
    const std::uint64_t source,
    const std::uint64_t max_distance,
    const bool include_source
) {
    std::vector<graph::BfsEntry> entries;
    {
        nb::gil_scoped_release release;
        entries = graph::breadth_first_search(graph, source, max_distance, include_source);
    }
    auto nodes = make_uint64_array({entries.size()});
    auto distances = make_uint64_array({entries.size()});
    auto *node_data = nodes.data();
    auto *distance_data = distances.data();
    for (std::size_t index = 0; index < entries.size(); ++index) {
        node_data[index] = entries[index].node;
        distance_data[index] = entries[index].distance;
    }
    return {nodes, distances};
}

double lifted_multicut_energy(
    const Graph &lifted_graph,
    ConstDoubleArray weights,
    ConstUInt64Array labels
) {
    const auto weight_vector =
        double_array_to_vector(weights, "edge_weights", lifted_graph.number_of_edges());
    const auto label_vector =
        uint64_array_to_vector(labels, "labels", lifted_graph.number_of_nodes());
    nb::gil_scoped_release release;
    return graph::lifted_multicut::energy(lifted_graph, weight_vector, label_vector);
}

UInt64Array lifted_multicut_greedy_additive(
    const Graph &lifted_graph,
    ConstDoubleArray weights,
    const std::uint64_t n_base_edges,
    const double weight_stop,
    const double node_num_stop,
    const bool add_noise,
    const int seed,
    const double sigma
) {
    const auto weight_vector =
        double_array_to_vector(weights, "edge_weights", lifted_graph.number_of_edges());
    if (n_base_edges > lifted_graph.number_of_edges()) {
        throw std::invalid_argument(
            "n_base_edges must be <= lifted graph number_of_edges"
        );
    }
    std::vector<std::uint64_t> labels;
    {
        nb::gil_scoped_release release;
        labels = graph::lifted_multicut::greedy_additive(
            lifted_graph,
            weight_vector,
            n_base_edges,
            weight_stop,
            node_num_stop,
            add_noise,
            seed,
            sigma
        );
    }
    return vector_to_uint64_array(labels);
}

UInt64Array lifted_multicut_kernighan_lin(
    const Graph &base_graph,
    const Graph &lifted_graph,
    ConstDoubleArray weights,
    const std::uint64_t n_base_edges,
    ConstUInt64Array initial_labels,
    const std::uint64_t number_of_outer_iterations,
    const double epsilon
) {
    const auto weight_vector =
        double_array_to_vector(weights, "edge_weights", lifted_graph.number_of_edges());
    auto label_vector =
        uint64_array_to_vector(initial_labels, "initial_labels", base_graph.number_of_nodes());
    if (n_base_edges > lifted_graph.number_of_edges()) {
        throw std::invalid_argument(
            "n_base_edges must be <= lifted graph number_of_edges"
        );
    }
    {
        nb::gil_scoped_release release;
        label_vector = graph::lifted_multicut::kernighan_lin(
            base_graph,
            lifted_graph,
            weight_vector,
            n_base_edges,
            std::move(label_vector),
            number_of_outer_iterations,
            epsilon
        );
    }
    return vector_to_uint64_array(label_vector);
}

template <class WeightT>
UInt64Array mutex_watershed_clustering_t(
    const Graph &graph,
    ConstArray1D<WeightT> edge_costs,
    ConstUInt64Array mutex_uvs,
    ConstArray1D<WeightT> mutex_costs
) {
    const auto edge_cost_vector =
        array_1d_to_vector<WeightT>(edge_costs, "edge_costs", graph.number_of_edges());
    require_uv_array(mutex_uvs, "mutex_uvs");
    const auto n_mutex = mutex_uvs.shape(0);
    const auto mutex_cost_vector =
        array_1d_to_vector<WeightT>(mutex_costs, "mutex_costs", static_cast<std::uint64_t>(n_mutex));

    std::vector<std::array<std::uint64_t, 2>> mutex_uv_vector(n_mutex);
    const auto *uv_data = mutex_uvs.data();
    for (std::size_t index = 0; index < n_mutex; ++index) {
        mutex_uv_vector[index][0] = uv_data[2 * index];
        mutex_uv_vector[index][1] = uv_data[2 * index + 1];
    }

    std::vector<std::uint64_t> labels;
    {
        nb::gil_scoped_release release;
        labels = graph::mutex_watershed_clustering<WeightT>(
            graph, edge_cost_vector, mutex_uv_vector, mutex_cost_vector
        );
    }
    return vector_to_uint64_array(labels);
}

template <class WeightT>
std::pair<UInt64Array, Int64Array> semantic_mutex_watershed_clustering_t(
    const Graph &graph,
    ConstArray1D<WeightT> edge_costs,
    ConstUInt64Array mutex_uvs,
    ConstArray1D<WeightT> mutex_costs,
    ConstUInt64Array semantic_node_classes,
    ConstArray1D<WeightT> semantic_costs
) {
    const auto edge_cost_vector =
        array_1d_to_vector<WeightT>(edge_costs, "edge_costs", graph.number_of_edges());
    require_uv_array(mutex_uvs, "mutex_uvs");
    const auto n_mutex = mutex_uvs.shape(0);
    const auto mutex_cost_vector =
        array_1d_to_vector<WeightT>(mutex_costs, "mutex_costs", static_cast<std::uint64_t>(n_mutex));
    require_uv_array(semantic_node_classes, "semantic_node_classes");
    const auto n_semantic = semantic_node_classes.shape(0);
    const auto semantic_cost_vector = array_1d_to_vector<WeightT>(
        semantic_costs,
        "semantic_costs",
        static_cast<std::uint64_t>(n_semantic)
    );

    std::vector<std::array<std::uint64_t, 2>> mutex_uv_vector(n_mutex);
    {
        const auto *uv_data = mutex_uvs.data();
        for (std::size_t index = 0; index < n_mutex; ++index) {
            mutex_uv_vector[index][0] = uv_data[2 * index];
            mutex_uv_vector[index][1] = uv_data[2 * index + 1];
        }
    }
    std::vector<std::array<std::uint64_t, 2>> semantic_uv_vector(n_semantic);
    {
        const auto *uv_data = semantic_node_classes.data();
        for (std::size_t index = 0; index < n_semantic; ++index) {
            semantic_uv_vector[index][0] = uv_data[2 * index];
            semantic_uv_vector[index][1] = uv_data[2 * index + 1];
        }
    }

    graph::SemanticMutexWatershedResult result;
    {
        nb::gil_scoped_release release;
        result = graph::semantic_mutex_watershed_clustering<WeightT>(
            graph,
            edge_cost_vector,
            mutex_uv_vector,
            mutex_cost_vector,
            semantic_uv_vector,
            semantic_cost_vector
        );
    }
    return std::make_pair(
        vector_to_uint64_array(result.node_labels),
        vector_to_array_1d<std::int64_t>(result.semantic_labels)
    );
}

template <class WeightT>
std::vector<double> array_1d_to_double_vector(
    ConstArray1D<WeightT> array,
    const char *argument_name,
    const std::uint64_t expected_size
) {
    if (array.ndim() != 1) {
        throw std::invalid_argument(std::string(argument_name) + " must be a 1D array");
    }
    if (array.shape(0) != static_cast<std::size_t>(expected_size)) {
        throw std::invalid_argument(
            std::string(argument_name) + " length must match expected size"
        );
    }
    const auto *data = array.data();
    std::vector<double> out(array.shape(0));
    for (std::size_t index = 0; index < out.size(); ++index) {
        out[index] = static_cast<double>(data[index]);
    }
    return out;
}

template <class WeightT>
UInt64Array agglo_edge_weighted_t(
    const Graph &graph,
    ConstArray1D<WeightT> edge_indicators,
    ConstArray1D<WeightT> edge_sizes,
    ConstArray1D<WeightT> node_sizes,
    const std::uint64_t num_clusters_stop,
    const double size_regularizer
) {
    auto indicator_vector = array_1d_to_double_vector<WeightT>(
        edge_indicators, "edge_indicators", graph.number_of_edges()
    );
    auto edge_size_vector = array_1d_to_double_vector<WeightT>(
        edge_sizes, "edge_sizes", graph.number_of_edges()
    );
    auto node_size_vector = array_1d_to_double_vector<WeightT>(
        node_sizes, "node_sizes", graph.number_of_nodes()
    );

    std::vector<std::uint64_t> labels;
    {
        nb::gil_scoped_release release;
        graph::agglomeration::EdgeWeightedClusterPolicy policy(
            std::move(indicator_vector),
            std::move(edge_size_vector),
            std::move(node_size_vector),
            static_cast<std::size_t>(num_clusters_stop),
            size_regularizer
        );
        labels = graph::agglomeration::agglomerative_clustering(graph, policy);
    }
    return vector_to_uint64_array(labels);
}

template <class WeightT>
UInt64Array agglo_node_and_edge_weighted_t(
    const Graph &graph,
    ConstArray1D<WeightT> edge_indicators,
    ConstArray1D<WeightT> edge_sizes,
    ConstArray1D<WeightT> node_sizes,
    ConstFloatingArray<WeightT> node_features,
    const std::uint64_t num_clusters_stop,
    const double size_regularizer,
    const double beta
) {
    auto indicator_vector = array_1d_to_double_vector<WeightT>(
        edge_indicators, "edge_indicators", graph.number_of_edges()
    );
    auto edge_size_vector = array_1d_to_double_vector<WeightT>(
        edge_sizes, "edge_sizes", graph.number_of_edges()
    );
    auto node_size_vector = array_1d_to_double_vector<WeightT>(
        node_sizes, "node_sizes", graph.number_of_nodes()
    );

    if (node_features.ndim() != 2) {
        throw std::invalid_argument(
            "node_features must be a 2D array of shape (n_nodes, n_channels)"
        );
    }
    if (node_features.shape(0) != static_cast<std::size_t>(graph.number_of_nodes())) {
        throw std::invalid_argument(
            "node_features first dimension must equal graph.number_of_nodes"
        );
    }
    const auto n_nodes = static_cast<std::size_t>(node_features.shape(0));
    const auto n_channels = static_cast<std::size_t>(node_features.shape(1));
    std::vector<std::vector<double>> features(n_nodes, std::vector<double>(n_channels));
    const auto *feature_data = node_features.data();
    for (std::size_t node = 0; node < n_nodes; ++node) {
        for (std::size_t channel = 0; channel < n_channels; ++channel) {
            features[node][channel] =
                static_cast<double>(feature_data[node * n_channels + channel]);
        }
    }

    std::vector<std::uint64_t> labels;
    {
        nb::gil_scoped_release release;
        graph::agglomeration::NodeAndEdgeWeightedClusterPolicy policy(
            std::move(indicator_vector),
            std::move(edge_size_vector),
            std::move(node_size_vector),
            std::move(features),
            static_cast<std::size_t>(num_clusters_stop),
            size_regularizer,
            beta
        );
        labels = graph::agglomeration::agglomerative_clustering(graph, policy);
    }
    return vector_to_uint64_array(labels);
}

template <class WeightT>
UInt64Array agglo_mala_t(
    const Graph &graph,
    ConstArray1D<WeightT> edge_indicators,
    const std::uint64_t num_bins,
    const double bin_min,
    const double bin_max,
    const std::uint64_t num_clusters_stop,
    const std::uint64_t num_edges_stop,
    const double threshold
) {
    auto indicator_vector = array_1d_to_double_vector<WeightT>(
        edge_indicators, "edge_indicators", graph.number_of_edges()
    );

    std::vector<std::uint64_t> labels;
    {
        nb::gil_scoped_release release;
        graph::agglomeration::MalaClusterPolicy policy(
            std::move(indicator_vector),
            static_cast<std::size_t>(num_bins),
            bin_min,
            bin_max,
            static_cast<std::size_t>(num_clusters_stop),
            static_cast<std::size_t>(num_edges_stop),
            threshold
        );
        labels = graph::agglomeration::agglomerative_clustering(graph, policy);
    }
    return vector_to_uint64_array(labels);
}

template <class WeightT>
UInt64Array agglo_gasp_t(
    const Graph &graph,
    ConstArray1D<WeightT> edge_weights,
    ConstArray1D<WeightT> edge_sizes,
    ConstUInt8Array is_mergeable,
    const std::uint64_t num_clusters_stop,
    const int linkage
) {
    auto weight_vector = array_1d_to_double_vector<WeightT>(
        edge_weights, "edge_weights", graph.number_of_edges()
    );
    auto edge_size_vector = array_1d_to_double_vector<WeightT>(
        edge_sizes, "edge_sizes", graph.number_of_edges()
    );

    std::vector<std::uint8_t> mergeable_vector;
    if (is_mergeable.ndim() == 1 && is_mergeable.shape(0) > 0) {
        if (is_mergeable.shape(0) != graph.number_of_edges()) {
            throw std::invalid_argument(
                "is_mergeable length must match graph.number_of_edges"
            );
        }
        const auto *data = is_mergeable.data();
        mergeable_vector.assign(data, data + is_mergeable.shape(0));
    }

    if (linkage < 0 || linkage > 5) {
        throw std::invalid_argument(
            "linkage must be in [0, 5] (sum, mean, max, min, abs_max, mutex_watershed)"
        );
    }

    std::vector<std::uint64_t> labels;
    {
        nb::gil_scoped_release release;
        graph::agglomeration::GaspClusterPolicy policy(
            std::move(weight_vector),
            std::move(edge_size_vector),
            std::move(mergeable_vector),
            static_cast<std::size_t>(num_clusters_stop),
            static_cast<graph::agglomeration::GaspLinkage>(linkage)
        );
        labels = graph::agglomeration::agglomerative_clustering(graph, policy);
    }
    return vector_to_uint64_array(labels);
}

UInt64Array multicut_fusion_move(
    const Graph &graph,
    ConstDoubleArray costs,
    ConstUInt64Array initial_labels,
    std::vector<graph::ProposalGeneratorBase *> proposal_generators,
    const graph::multicut::SolverBase *sub_solver,
    const std::size_t number_of_iterations,
    const std::size_t stop_if_no_improvement,
    const std::size_t number_of_threads,
    const std::size_t number_of_parallel_proposals
) {
    if (proposal_generators.empty()) {
        throw std::invalid_argument("proposal_generators must not be empty");
    }
    auto cost_vector = double_array_to_vector(costs, "edge_costs", graph.number_of_edges());
    auto label_vector =
        uint64_array_to_vector(initial_labels, "initial_labels", graph.number_of_nodes());

    graph::multicut::FusionMoveSolver solver(
        std::move(proposal_generators),
        sub_solver,
        number_of_iterations,
        stop_if_no_improvement,
        number_of_threads,
        number_of_parallel_proposals
    );

    std::vector<std::uint64_t> result;
    {
        nb::gil_scoped_release release;
        graph::multicut::Objective objective(graph, std::move(cost_vector), std::move(label_vector));
        result = solver.optimize(objective);
    }
    return vector_to_uint64_array(result);
}

UInt64Array lifted_multicut_fusion_move(
    const Graph &base_graph,
    const Graph &lifted_graph,
    ConstDoubleArray weights,
    const std::uint64_t n_base_edges,
    ConstUInt64Array initial_labels,
    std::vector<graph::ProposalGeneratorBase *> proposal_generators,
    const graph::lifted_multicut::SolverBase *sub_solver,
    const std::size_t number_of_iterations,
    const std::size_t stop_if_no_improvement,
    const std::size_t number_of_threads,
    const std::size_t number_of_parallel_proposals
) {
    if (proposal_generators.empty()) {
        throw std::invalid_argument("proposal_generators must not be empty");
    }
    if (n_base_edges > lifted_graph.number_of_edges()) {
        throw std::invalid_argument(
            "n_base_edges must be <= lifted graph number_of_edges"
        );
    }
    auto weight_vector =
        double_array_to_vector(weights, "edge_weights", lifted_graph.number_of_edges());
    auto label_vector =
        uint64_array_to_vector(initial_labels, "initial_labels", base_graph.number_of_nodes());

    // Decompose the user's lifted graph into base + lifted parts for the
    // Objective constructor (which rebuilds the lifted graph internally). The
    // overhead is one O(E) walk; the C++ Objective consumes only base costs
    // and per-lifted-edge (u, v, weight) triples.
    std::vector<double> base_weights(
        weight_vector.begin(), weight_vector.begin() + static_cast<std::ptrdiff_t>(n_base_edges)
    );
    const auto n_lifted_edges = lifted_graph.number_of_edges() - n_base_edges;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> lifted_uvs;
    lifted_uvs.reserve(static_cast<std::size_t>(n_lifted_edges));
    std::vector<double> lifted_weights;
    lifted_weights.reserve(static_cast<std::size_t>(n_lifted_edges));
    for (std::uint64_t edge = n_base_edges; edge < lifted_graph.number_of_edges(); ++edge) {
        const auto uv = lifted_graph.uv(edge);
        lifted_uvs.emplace_back(uv.first, uv.second);
        lifted_weights.push_back(weight_vector[static_cast<std::size_t>(edge)]);
    }

    graph::lifted_multicut::FusionMoveSolver solver(
        std::move(proposal_generators),
        sub_solver,
        number_of_iterations,
        stop_if_no_improvement,
        number_of_threads,
        number_of_parallel_proposals
    );

    std::vector<std::uint64_t> result;
    {
        nb::gil_scoped_release release;
        graph::lifted_multicut::Objective objective(
            base_graph,
            std::move(base_weights),
            lifted_uvs,
            lifted_weights,
            false
        );
        objective.set_labels(std::move(label_vector));
        result = solver.optimize(objective);
    }
    return vector_to_uint64_array(result);
}

template <class T>
Rag region_adjacency_graph_t(
    LabelArray<T> labels,
    const std::size_t number_of_threads
) {
    std::vector<std::ptrdiff_t> shape(labels.ndim());
    for (std::size_t axis = 0; axis < labels.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(labels.shape(axis));
    }

    ConstArrayView<T> labels_view{
        labels.data(),
        shape,
        {},
    };

    nb::gil_scoped_release release;
    return graph::region_adjacency_graph<T>(labels_view, number_of_threads);
}

template <class T>
graph::RagCoordinates rag_coordinates_t(
    const Rag &rag,
    LabelArray<T> labels,
    const std::size_t number_of_threads
) {
    std::vector<std::ptrdiff_t> shape(labels.ndim());
    for (std::size_t axis = 0; axis < labels.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(labels.shape(axis));
    }

    ConstArrayView<T> labels_view{
        labels.data(),
        shape,
        {},
    };

    nb::gil_scoped_release release;
    return graph::RagCoordinates(rag, labels_view, number_of_threads);
}

UInt64Array rag_coordinates_storage_lengths(const graph::RagCoordinates &coords) {
    return vector_to_uint64_array(coords.storage_lengths());
}

UInt64Array rag_coordinates_edge_coordinates(
    const graph::RagCoordinates &coords,
    const std::uint64_t edge,
    const int edge_direction
) {
    const auto flat = coords.edge_coordinates(edge, edge_direction);
    const auto ndim = coords.ndim();
    const std::size_t n_points = ndim == 0 ? 0 : flat.size() / ndim;
    auto result = make_uint64_array({n_points, ndim});
    std::copy(flat.begin(), flat.end(), result.data());
    return result;
}

template <class V>
TypedArray<V> rag_coordinates_edges_to_volume(
    const graph::RagCoordinates &coords,
    ConstFloatingArray<V> edge_values,
    const int edge_direction,
    const V ignore_value
) {
    if (edge_values.ndim() != 1) {
        throw std::invalid_argument("edge_values must be a 1D array");
    }
    std::vector<V> values(
        edge_values.data(), edge_values.data() + edge_values.shape(0)
    );

    const auto &shape = coords.shape();
    std::vector<std::size_t> out_shape(shape.size());
    std::vector<std::ptrdiff_t> view_shape(shape.size());
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        out_shape[axis] = static_cast<std::size_t>(shape[axis]);
        view_shape[axis] = static_cast<std::ptrdiff_t>(shape[axis]);
    }
    auto result = make_typed_array<V>(out_shape);
    ArrayView<V> view{result.data(), view_shape, {}};

    {
        nb::gil_scoped_release release;
        coords.edges_to_volume(values, view, edge_direction, ignore_value);
    }
    return result;
}

template <class T>
std::vector<std::ptrdiff_t> ndarray_shape(LabelArray<T> array) {
    std::vector<std::ptrdiff_t> shape(array.ndim());
    for (std::size_t axis = 0; axis < array.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(array.shape(axis));
    }
    return shape;
}

std::vector<std::ptrdiff_t> ndarray_shape(ConstDoubleArray array) {
    std::vector<std::ptrdiff_t> shape(array.ndim());
    for (std::size_t axis = 0; axis < array.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(array.shape(axis));
    }
    return shape;
}

std::vector<std::ptrdiff_t> ndarray_shape(DoubleArray array) {
    std::vector<std::ptrdiff_t> shape(array.ndim());
    for (std::size_t axis = 0; axis < array.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(array.shape(axis));
    }
    return shape;
}

// ---- Distributed region-adjacency-graph + edge-feature primitives ----

template <class T>
UInt64Array block_region_adjacency_edges_t(
    LabelArray<T> labels,
    std::vector<std::int64_t> own_begin,
    std::vector<std::int64_t> own_shape,
    const std::size_t number_of_threads
) {
    ConstArrayView<T> labels_view{labels.data(), ndarray_shape(labels), {}};

    std::vector<bioimage_cpp::detail::Edge> edges;
    {
        nb::gil_scoped_release release;
        edges = graph::distributed::block_region_adjacency_edges<T>(
            labels_view, own_begin, own_shape, number_of_threads
        );
    }
    return edges_to_uv_array(edges);
}

template <class T>
nb::tuple block_edge_map_stats_t(
    LabelArray<T> labels,
    ConstDoubleArray edge_map,
    std::vector<std::int64_t> own_begin,
    std::vector<std::int64_t> own_shape,
    const std::size_t number_of_threads
) {
    ConstArrayView<T> labels_view{labels.data(), ndarray_shape(labels), {}};
    ConstArrayView<double> edge_map_view{edge_map.data(), ndarray_shape(edge_map), {}};

    graph::distributed::BlockEdgeStats result;
    {
        nb::gil_scoped_release release;
        result = graph::distributed::block_edge_map_stats<T>(
            labels_view, edge_map_view, own_begin, own_shape, number_of_threads
        );
    }
    return nb::make_tuple(
        edges_to_uv_array(result.edges), block_stats_to_array(result.stats)
    );
}

template <class T>
nb::tuple block_affinity_stats_t(
    LabelArray<T> labels,
    ConstDoubleArray affinities,
    std::vector<std::vector<std::ptrdiff_t>> offsets,
    std::vector<std::int64_t> own_begin,
    std::vector<std::int64_t> own_shape,
    const std::size_t number_of_threads
) {
    ConstArrayView<T> labels_view{labels.data(), ndarray_shape(labels), {}};
    ConstArrayView<double> affinities_view{affinities.data(), ndarray_shape(affinities), {}};

    graph::distributed::BlockEdgeStats result;
    {
        nb::gil_scoped_release release;
        result = graph::distributed::block_affinity_stats<T>(
            labels_view, affinities_view, offsets, own_begin, own_shape, number_of_threads
        );
    }
    return nb::make_tuple(
        edges_to_uv_array(result.edges), block_stats_to_array(result.stats)
    );
}

UInt64Array distributed_merge_edges(ConstUInt64Array edges) {
    ConstArrayView<std::uint64_t> edges_view{edges.data(), ndarray_shape(edges), {}};

    std::vector<bioimage_cpp::detail::Edge> merged;
    {
        nb::gil_scoped_release release;
        merged = graph::distributed::merge_edges(edges_view);
    }
    return edges_to_uv_array(merged);
}

// Mutates `current_stats` in place (the Python wrapper hands the caller's
// accumulator back), so one merge stays O(block edges) instead of copying the
// whole global accumulator per block.
void distributed_merge_block_edge_stats(
    const Graph &global_graph,
    DoubleArray current_stats,
    ConstUInt64Array block_edges,
    ConstDoubleArray block_stats
) {
    ArrayView<double> current_view{current_stats.data(), ndarray_shape(current_stats), {}};
    ConstArrayView<std::uint64_t> block_edges_view{block_edges.data(), ndarray_shape(block_edges), {}};
    ConstArrayView<double> block_stats_view{block_stats.data(), ndarray_shape(block_stats), {}};

    {
        nb::gil_scoped_release release;
        graph::distributed::merge_block_edge_stats(
            global_graph, current_view, block_edges_view, block_stats_view
        );
    }
}

DoubleArray distributed_finalize_edge_features(
    ConstDoubleArray stats,
    const bool compute_complex_features
) {
    if (stats.ndim() != 2 || stats.shape(1) != 5) {
        throw std::invalid_argument("stats must have shape (number_of_edges, 5)");
    }
    const auto rows = static_cast<std::size_t>(stats.shape(0));
    const std::size_t number_of_features = compute_complex_features ? 5 : 2;
    auto result = make_double_array({rows, number_of_features});

    ConstArrayView<double> stats_view{stats.data(), ndarray_shape(stats), {}};
    ArrayView<double> out_view{
        result.data(),
        {
            static_cast<std::ptrdiff_t>(rows),
            static_cast<std::ptrdiff_t>(number_of_features),
        },
        {},
    };
    {
        nb::gil_scoped_release release;
        graph::distributed::finalize_edge_features(
            stats_view, compute_complex_features, out_view
        );
    }
    return result;
}

template <class LabelT>
DoubleArray accumulate_edge_map_features_t(
    const Rag &rag,
    LabelArray<LabelT> labels,
    ConstDoubleArray edge_map,
    const bool compute_complex_features,
    const std::size_t number_of_threads
) {
    const std::size_t number_of_features =
        compute_complex_features ? std::size_t{12} : std::size_t{2};
    auto result = make_double_array({
        static_cast<std::size_t>(rag.number_of_edges()),
        number_of_features
    });

    ConstArrayView<LabelT> labels_view{
        labels.data(),
        ndarray_shape(labels),
        {},
    };
    ConstArrayView<double> edge_map_view{
        edge_map.data(),
        ndarray_shape(edge_map),
        {},
    };
    ArrayView<double> out_view{
        result.data(),
        {
            static_cast<std::ptrdiff_t>(rag.number_of_edges()),
            static_cast<std::ptrdiff_t>(number_of_features),
        },
        {},
    };

    nb::gil_scoped_release release;
    graph::accumulate_edge_map_features<LabelT, double>(
        rag,
        labels_view,
        edge_map_view,
        compute_complex_features,
        number_of_threads,
        out_view
    );
    return result;
}

template <class LabelT>
DoubleArray accumulate_affinity_features_t(
    const Rag &rag,
    LabelArray<LabelT> labels,
    ConstDoubleArray affinities,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const bool compute_complex_features,
    const std::size_t number_of_threads
) {
    const std::size_t number_of_features =
        compute_complex_features ? std::size_t{12} : std::size_t{2};
    auto result = make_double_array({
        static_cast<std::size_t>(rag.number_of_edges()),
        number_of_features
    });

    ConstArrayView<LabelT> labels_view{
        labels.data(),
        ndarray_shape(labels),
        {},
    };
    ConstArrayView<double> affinities_view{
        affinities.data(),
        ndarray_shape(affinities),
        {},
    };
    ArrayView<double> out_view{
        result.data(),
        {
            static_cast<std::ptrdiff_t>(rag.number_of_edges()),
            static_cast<std::ptrdiff_t>(number_of_features),
        },
        {},
    };

    nb::gil_scoped_release release;
    graph::accumulate_affinity_features<LabelT, double>(
        rag,
        labels_view,
        affinities_view,
        offsets,
        compute_complex_features,
        number_of_threads,
        out_view
    );
    return result;
}

template <class LabelT>
UInt64Array lifted_edges_from_affinities_t(
    const Rag &rag,
    LabelArray<LabelT> labels,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::size_t number_of_threads
) {
    ConstArrayView<LabelT> labels_view{
        labels.data(),
        ndarray_shape(labels),
        {},
    };

    std::vector<bioimage_cpp::detail::Edge> lifted_edges;
    {
        nb::gil_scoped_release release;
        lifted_edges = graph::lifted_edges_from_offsets<LabelT>(
            rag, labels_view, offsets, number_of_threads
        );
    }
    auto result = make_uint64_array({lifted_edges.size(), 2});
    auto *data = result.data();
    for (std::size_t index = 0; index < lifted_edges.size(); ++index) {
        data[2 * index] = lifted_edges[index].first;
        data[2 * index + 1] = lifted_edges[index].second;
    }
    return result;
}

template <class LabelT>
UInt64Array lifted_edges_from_node_labels_t(
    const Graph &graph,
    LabelArray<LabelT> node_labels,
    const std::uint64_t graph_depth,
    const std::string &mode,
    std::optional<LabelT> ignore_label,
    const std::size_t number_of_threads
) {
    if (node_labels.ndim() != 1) {
        throw std::invalid_argument("node_labels must be a 1D array");
    }
    if (node_labels.shape(0) != graph.number_of_nodes()) {
        throw std::invalid_argument(
            "node_labels length must match graph number_of_nodes"
        );
    }
    graph::lifted_multicut::LiftedNodeLabelMode mode_enum;
    if (mode == "all") {
        mode_enum = graph::lifted_multicut::LiftedNodeLabelMode::all;
    } else if (mode == "same") {
        mode_enum = graph::lifted_multicut::LiftedNodeLabelMode::same;
    } else if (mode == "different") {
        mode_enum = graph::lifted_multicut::LiftedNodeLabelMode::different;
    } else {
        throw std::invalid_argument(
            "mode must be one of 'all', 'same', 'different', got '" + mode + "'"
        );
    }

    ConstArrayView<LabelT> labels_view{
        node_labels.data(),
        {static_cast<std::ptrdiff_t>(node_labels.shape(0))},
        {},
    };

    std::vector<bioimage_cpp::detail::Edge> lifted_edges;
    {
        nb::gil_scoped_release release;
        lifted_edges = graph::lifted_multicut::lifted_edges_from_node_labels<LabelT>(
            graph, labels_view, graph_depth, mode_enum, ignore_label, number_of_threads
        );
    }
    auto result = make_uint64_array({lifted_edges.size(), 2});
    auto *data = result.data();
    for (std::size_t index = 0; index < lifted_edges.size(); ++index) {
        data[2 * index] = lifted_edges[index].first;
        data[2 * index + 1] = lifted_edges[index].second;
    }
    return result;
}

template <class LabelT>
DoubleArray accumulate_lifted_affinity_features_t(
    LabelArray<LabelT> labels,
    ConstDoubleArray affinities,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    ConstUInt64Array lifted_uvs,
    const bool compute_complex_features,
    const std::size_t number_of_threads
) {
    if (lifted_uvs.ndim() != 2 || lifted_uvs.shape(1) != 2) {
        throw std::invalid_argument("lifted_uvs must have shape (n_lifted, 2)");
    }
    const auto n_lifted = lifted_uvs.shape(0);
    const std::size_t number_of_features =
        compute_complex_features ? std::size_t{12} : std::size_t{2};
    auto result = make_double_array({n_lifted, number_of_features});

    std::vector<bioimage_cpp::detail::Edge> lifted_edges(n_lifted);
    const auto *uv_data = lifted_uvs.data();
    for (std::size_t index = 0; index < n_lifted; ++index) {
        lifted_edges[index] = {uv_data[2 * index], uv_data[2 * index + 1]};
    }

    ConstArrayView<LabelT> labels_view{
        labels.data(),
        ndarray_shape(labels),
        {},
    };
    ConstArrayView<double> affinities_view{
        affinities.data(),
        ndarray_shape(affinities),
        {},
    };
    ArrayView<double> out_view{
        result.data(),
        {
            static_cast<std::ptrdiff_t>(n_lifted),
            static_cast<std::ptrdiff_t>(number_of_features),
        },
        {},
    };

    nb::gil_scoped_release release;
    graph::accumulate_lifted_affinity_features<LabelT, double>(
        labels_view,
        affinities_view,
        offsets,
        lifted_edges,
        compute_complex_features,
        number_of_threads,
        out_view
    );
    return result;
}

template <class LabelT>
UInt64Array project_node_labels_to_pixels_t(
    const Rag &rag,
    LabelArray<LabelT> labels,
    ConstUInt64Array node_labels,
    const std::size_t number_of_threads
) {
    if (node_labels.ndim() != 1) {
        throw std::invalid_argument("node_labels must be a 1D uint64 array");
    }

    std::vector<std::size_t> output_shape(labels.ndim());
    for (std::size_t axis = 0; axis < labels.ndim(); ++axis) {
        output_shape[axis] = labels.shape(axis);
    }
    auto result = make_uint64_array(output_shape);

    ConstArrayView<LabelT> labels_view{
        labels.data(),
        ndarray_shape(labels),
        {},
    };
    ConstArrayView<std::uint64_t> node_labels_view{
        node_labels.data(),
        {static_cast<std::ptrdiff_t>(node_labels.shape(0))},
        {},
    };
    ArrayView<std::uint64_t> out_view{
        result.data(),
        ndarray_shape(labels),
        {},
    };

    nb::gil_scoped_release release;
    graph::project_node_labels_to_pixels<LabelT>(
        rag,
        labels_view,
        node_labels_view,
        number_of_threads,
        out_view
    );
    return result;
}

template <class LabelT, class OtherT>
TypedArray<OtherT> accumulate_labels_t(
    const Rag &rag,
    LabelArray<LabelT> labels,
    LabelArray<OtherT> other_labels,
    const bool has_ignore_value,
    const OtherT ignore_value,
    const std::size_t number_of_threads
) {
    if (labels.ndim() != other_labels.ndim()) {
        throw std::invalid_argument("other_labels shape must match labels shape");
    }
    for (std::size_t axis = 0; axis < labels.ndim(); ++axis) {
        if (labels.shape(axis) != other_labels.shape(axis)) {
            throw std::invalid_argument("other_labels shape must match labels shape");
        }
    }

    auto result = make_typed_array<OtherT>({static_cast<std::size_t>(rag.number_of_nodes())});

    ConstArrayView<LabelT> labels_view{
        labels.data(),
        ndarray_shape(labels),
        {},
    };
    ConstArrayView<OtherT> other_labels_view{
        other_labels.data(),
        ndarray_shape(other_labels),
        {},
    };
    ArrayView<OtherT> out_view{
        result.data(),
        {static_cast<std::ptrdiff_t>(rag.number_of_nodes())},
        {},
    };

    nb::gil_scoped_release release;
    graph::accumulate_labels<LabelT, OtherT>(
        rag,
        labels_view,
        other_labels_view,
        has_ignore_value,
        ignore_value,
        number_of_threads,
        out_view
    );
    return result;
}

} // namespace

void bind_graph(nb::module_ &m) {
    nb::class_<Graph>(m, "UndirectedGraph")
        .def(
            nb::init<std::uint64_t, std::uint64_t>(),
            nb::arg("number_of_nodes") = 0,
            nb::arg("reserve_number_of_edges") = 0
        )
        // Constructor equivalent of the from_edges / from_unique_edges
        // statics. Unlike a def_static, an __init__ overload constructs an
        // instance of the *derived* Python class, so the Python subclass in
        // bioimage_cpp.graph can build itself from an edge array.
        .def(
            "__init__",
            [](Graph *self, const std::uint64_t number_of_nodes, ConstUInt64Array uvs,
               const bool unique) {
                new (self) Graph(
                    unique ? graph_from_unique_edges(number_of_nodes, uvs)
                           : graph_from_edges(number_of_nodes, uvs)
                );
            },
            nb::arg("number_of_nodes"),
            nb::arg("uvs"),
            nb::arg("unique")
        )
        .def(
            "assign",
            &Graph::assign,
            nb::arg("number_of_nodes") = 0,
            nb::arg("reserve_number_of_edges") = 0
        )
        .def_prop_ro("number_of_nodes", &Graph::number_of_nodes)
        .def_prop_ro("number_of_edges", &Graph::number_of_edges)
        .def_prop_ro("node_id_upper_bound", &Graph::node_id_upper_bound)
        .def_prop_ro("edge_id_upper_bound", &Graph::edge_id_upper_bound)
        .def("insert_edge", &Graph::insert_edge, nb::arg("u"), nb::arg("v"))
        .def("find_edge", &Graph::find_edge, nb::arg("u"), nb::arg("v"))
        .def("u", &Graph::u, nb::arg("edge"))
        .def("v", &Graph::v, nb::arg("edge"))
        .def("uv", &Graph::uv, nb::arg("edge"))
        .def("nodes", &graph_nodes)
        .def("edges", &graph_edges)
        .def("uv_ids", &graph_uv_ids)
        .def("insert_edges", &graph_insert_edges, nb::arg("uvs"))
        .def("find_edges", &graph_find_edges, nb::arg("uvs"))
        .def("node_adjacency", &graph_node_adjacency, nb::arg("node"))
        .def_prop_ro("serialization_size", &Graph::serialization_size)
        .def("serialize", &graph_serialize)
        .def(
            "extract_subgraph_from_nodes",
            &graph_extract_subgraph_from_nodes,
            nb::arg("nodes")
        )
        .def("edges_from_node_list", &graph_edges_from_node_list, nb::arg("nodes"))
        .def(
            "freeze",
            &Graph::freeze,
            "Build the internal adjacency representation now (it is otherwise "
            "built lazily on first use). Call this on the construction thread "
            "before sharing the graph with concurrent reader threads: the lazy "
            "build is not thread-safe. No-op if already built; safe to call "
            "repeatedly."
        )
        .def("clone", &Graph::clone)
        .def_static(
            "from_edges",
            &graph_from_edges,
            nb::arg("number_of_nodes"),
            nb::arg("uvs")
        )
        .def_static(
            "from_unique_edges",
            &graph_from_unique_edges,
            nb::arg("number_of_nodes"),
            nb::arg("uvs")
        )
        .def_static(
            "deserialize",
            &graph_deserialize,
            nb::arg("serialization")
        )
        .def_prop_ro("numberOfNodes", &Graph::number_of_nodes)
        .def_prop_ro("numberOfEdges", &Graph::number_of_edges)
        .def_prop_ro("nodeIdUpperBound", &Graph::node_id_upper_bound)
        .def_prop_ro("edgeIdUpperBound", &Graph::edge_id_upper_bound)
        .def_prop_ro("serializationSize", &Graph::serialization_size)
        .def("insertEdge", &Graph::insert_edge, nb::arg("u"), nb::arg("v"))
        .def("findEdge", &Graph::find_edge, nb::arg("u"), nb::arg("v"))
        .def("uvIds", &graph_uv_ids)
        .def("insertEdges", &graph_insert_edges, nb::arg("uvs"))
        .def("findEdges", &graph_find_edges, nb::arg("uvs"))
        .def("nodeAdjacency", &graph_node_adjacency, nb::arg("node"))
        .def(
            "extractSubgraphFromNodes",
            &graph_extract_subgraph_from_nodes,
            nb::arg("nodes")
        )
        .def("edgesFromNodeList", &graph_edges_from_node_list, nb::arg("nodes"));

    nb::class_<GridGraph2D, Graph>(m, "GridGraph2D")
        .def(nb::init<const std::vector<std::uint64_t> &>(), nb::arg("shape"))
        .def_prop_ro("shape", &grid_shape<2>)
        .def_prop_ro("strides", &grid_strides<2>)
        .def_prop_ro("ndim", &GridGraph2D::ndim)
        .def("node_id", &grid_node_id<2>, nb::arg("coordinate"))
        .def("coordinates", &grid_coordinates<2>, nb::arg("node"))
        .def("edge_axis", &GridGraph2D::edge_axis, nb::arg("edge"))
        .def("edge_coordinates", &grid_edge_coordinates<2>, nb::arg("edge"))
        .def("offset_target", &grid_offset_target<2>, nb::arg("node"), nb::arg("offset"))
        .def("project_edge_ids_to_pixels", &grid_project_edge_ids_to_pixels<2>)
        .def(
            "project_edge_ids_to_pixels_with_offsets",
            &grid_project_edge_ids_to_pixels_with_offsets<2>,
            nb::arg("offsets"),
            nb::arg("strides") = std::nullopt,
            nb::arg("mask") = std::nullopt
        )
        .def_prop_ro("numberOfDimensions", &GridGraph2D::ndim)
        .def("nodeId", &grid_node_id<2>, nb::arg("coordinate"))
        .def("edgeAxis", &GridGraph2D::edge_axis, nb::arg("edge"))
        .def("edgeCoordinates", &grid_edge_coordinates<2>, nb::arg("edge"))
        .def("offsetTarget", &grid_offset_target<2>, nb::arg("node"), nb::arg("offset"));

    nb::class_<GridGraph3D, Graph>(m, "GridGraph3D")
        .def(nb::init<const std::vector<std::uint64_t> &>(), nb::arg("shape"))
        .def_prop_ro("shape", &grid_shape<3>)
        .def_prop_ro("strides", &grid_strides<3>)
        .def_prop_ro("ndim", &GridGraph3D::ndim)
        .def("node_id", &grid_node_id<3>, nb::arg("coordinate"))
        .def("coordinates", &grid_coordinates<3>, nb::arg("node"))
        .def("edge_axis", &GridGraph3D::edge_axis, nb::arg("edge"))
        .def("edge_coordinates", &grid_edge_coordinates<3>, nb::arg("edge"))
        .def("offset_target", &grid_offset_target<3>, nb::arg("node"), nb::arg("offset"))
        .def("project_edge_ids_to_pixels", &grid_project_edge_ids_to_pixels<3>)
        .def(
            "project_edge_ids_to_pixels_with_offsets",
            &grid_project_edge_ids_to_pixels_with_offsets<3>,
            nb::arg("offsets"),
            nb::arg("strides") = std::nullopt,
            nb::arg("mask") = std::nullopt
        )
        .def_prop_ro("numberOfDimensions", &GridGraph3D::ndim)
        .def("nodeId", &grid_node_id<3>, nb::arg("coordinate"))
        .def("edgeAxis", &GridGraph3D::edge_axis, nb::arg("edge"))
        .def("edgeCoordinates", &grid_edge_coordinates<3>, nb::arg("edge"))
        .def("offsetTarget", &grid_offset_target<3>, nb::arg("node"), nb::arg("offset"));

    nb::class_<Rag, Graph>(m, "RegionAdjacencyGraph")
        .def_prop_ro("shape", &Rag::shape);

    nb::class_<graph::RagCoordinates>(m, "RagCoordinates")
        .def_prop_ro("ndim", &graph::RagCoordinates::ndim)
        .def_prop_ro("shape", &graph::RagCoordinates::shape)
        .def_prop_ro("number_of_edges", &graph::RagCoordinates::number_of_edges)
        .def("storage_lengths", &rag_coordinates_storage_lengths)
        .def(
            "edge_coordinates",
            &rag_coordinates_edge_coordinates,
            nb::arg("edge"),
            nb::arg("edge_direction")
        )
        .def(
            "_edges_to_volume_float32",
            &rag_coordinates_edges_to_volume<float>,
            nb::arg("edge_values"),
            nb::arg("edge_direction"),
            nb::arg("ignore_value")
        )
        .def(
            "_edges_to_volume_float64",
            &rag_coordinates_edges_to_volume<double>,
            nb::arg("edge_values"),
            nb::arg("edge_direction"),
            nb::arg("ignore_value")
        )
        .def(
            "_edges_to_volume_uint32",
            &rag_coordinates_edges_to_volume<std::uint32_t>,
            nb::arg("edge_values"),
            nb::arg("edge_direction"),
            nb::arg("ignore_value")
        )
        .def(
            "_edges_to_volume_uint64",
            &rag_coordinates_edges_to_volume<std::uint64_t>,
            nb::arg("edge_values"),
            nb::arg("edge_direction"),
            nb::arg("ignore_value")
        );

    const auto register_grid_boundary = [&m]<class T, std::size_t D>(const char *name) {
        m.def(
            name,
            &grid_boundary_features_t<T, D>,
            nb::arg("graph"),
            nb::arg("boundary_map")
        );
    };
    register_grid_boundary.operator()<float, 2>("_grid_boundary_features_2d_float32");
    register_grid_boundary.operator()<double, 2>("_grid_boundary_features_2d_float64");
    register_grid_boundary.operator()<float, 3>("_grid_boundary_features_3d_float32");
    register_grid_boundary.operator()<double, 3>("_grid_boundary_features_3d_float64");

    const auto register_grid_affinity = [&m]<class T, std::size_t D>(const char *name) {
        m.def(
            name,
            &grid_affinity_features_t<T, D>,
            nb::arg("graph"),
            nb::arg("affinities"),
            nb::arg("offsets")
        );
    };
    register_grid_affinity.operator()<float, 2>("_grid_affinity_features_2d_float32");
    register_grid_affinity.operator()<double, 2>("_grid_affinity_features_2d_float64");
    register_grid_affinity.operator()<float, 3>("_grid_affinity_features_3d_float32");
    register_grid_affinity.operator()<double, 3>("_grid_affinity_features_3d_float64");

    const auto register_grid_affinity_lifted = [&m]<class T, std::size_t D>(
        const char *name
    ) {
        m.def(
            name,
            &grid_affinity_features_with_lifted_t<T, D>,
            nb::arg("graph"),
            nb::arg("affinities"),
            nb::arg("offsets")
        );
    };
    register_grid_affinity_lifted.operator()<float, 2>(
        "_grid_affinity_features_with_lifted_2d_float32"
    );
    register_grid_affinity_lifted.operator()<double, 2>(
        "_grid_affinity_features_with_lifted_2d_float64"
    );
    register_grid_affinity_lifted.operator()<float, 3>(
        "_grid_affinity_features_with_lifted_3d_float32"
    );
    register_grid_affinity_lifted.operator()<double, 3>(
        "_grid_affinity_features_with_lifted_3d_float64"
    );

    m.def(
        "_breadth_first_search",
        &graph_breadth_first_search,
        nb::arg("graph"),
        nb::arg("source"),
        nb::arg("max_distance"),
        nb::arg("include_source")
    );
    m.def("_connected_components", &graph_connected_components, nb::arg("graph"));
    m.def(
        "_connected_components_masked",
        &graph_connected_components_masked,
        nb::arg("graph"),
        nb::arg("edge_mask")
    );
    const auto register_watershed = [&m]<class WeightT, class SeedT>(
        const char *name
    ) {
        m.def(
            name,
            &graph_edge_weighted_watershed_t<WeightT, SeedT>,
            nb::arg("graph"),
            nb::arg("edge_weights"),
            nb::arg("seeds")
        );
    };
    register_watershed.operator()<float, std::uint32_t>("_edge_weighted_watershed_float32_uint32");
    register_watershed.operator()<float, std::uint64_t>("_edge_weighted_watershed_float32_uint64");
    register_watershed.operator()<float, std::int32_t>("_edge_weighted_watershed_float32_int32");
    register_watershed.operator()<float, std::int64_t>("_edge_weighted_watershed_float32_int64");
    register_watershed.operator()<double, std::uint32_t>("_edge_weighted_watershed_float64_uint32");
    register_watershed.operator()<double, std::uint64_t>("_edge_weighted_watershed_float64_uint64");
    register_watershed.operator()<double, std::int32_t>("_edge_weighted_watershed_float64_int32");
    register_watershed.operator()<double, std::int64_t>("_edge_weighted_watershed_float64_int64");
    m.def(
        "_multicut_energy",
        &multicut_energy,
        nb::arg("graph"),
        nb::arg("edge_costs"),
        nb::arg("labels")
    );
    m.def(
        "_multicut_greedy_additive",
        &multicut_greedy_additive,
        nb::arg("graph"),
        nb::arg("edge_costs"),
        nb::arg("weight_stop"),
        nb::arg("node_num_stop"),
        nb::arg("add_noise"),
        nb::arg("seed"),
        nb::arg("sigma")
    );
    m.def(
        "_multicut_greedy_fixation",
        &multicut_greedy_fixation,
        nb::arg("graph"),
        nb::arg("edge_costs"),
        nb::arg("weight_stop"),
        nb::arg("node_num_stop")
    );
    m.def(
        "_multicut_kernighan_lin",
        &multicut_kernighan_lin,
        nb::arg("graph"),
        nb::arg("edge_costs"),
        nb::arg("initial_labels"),
        nb::arg("number_of_outer_iterations"),
        nb::arg("epsilon")
    );

    // Multicut sub-solver hierarchy used by fusion moves. The classes are
    // opaque to Python; constructors carry per-solver settings.
    nb::class_<graph::multicut::SolverBase>(m, "_MulticutSolverBase");
    nb::class_<graph::multicut::GreedyAdditiveSolver, graph::multicut::SolverBase>(
        m, "_GreedyAdditiveMulticutSubSolver"
    )
        .def(
            nb::init<double, double, bool, int, double>(),
            nb::arg("weight_stop") = 0.0,
            nb::arg("node_num_stop") = -1.0,
            nb::arg("add_noise") = false,
            nb::arg("seed") = 42,
            nb::arg("sigma") = 1.0
        );
    nb::class_<graph::multicut::GreedyFixationSolver, graph::multicut::SolverBase>(
        m, "_GreedyFixationMulticutSubSolver"
    )
        .def(
            nb::init<double, double>(),
            nb::arg("weight_stop") = 0.0,
            nb::arg("node_num_stop") = -1.0
        );
    nb::class_<graph::multicut::KernighanLinSolver, graph::multicut::SolverBase>(
        m, "_KernighanLinMulticutSubSolver"
    )
        .def(
            nb::init<std::uint64_t, double>(),
            nb::arg("number_of_outer_iterations") = 100,
            nb::arg("epsilon") = 1.0e-6
        );

    // Proposal generators used by fusion moves.
    nb::class_<graph::ProposalGeneratorBase>(m, "_ProposalGeneratorBase");
    nb::class_<graph::WatershedProposalGenerator, graph::ProposalGeneratorBase>(
        m, "_WatershedProposalGenerator"
    )
        .def(
            "__init__",
            [](graph::WatershedProposalGenerator *self,
               const Graph &graph,
               ConstDoubleArray edge_costs,
               double sigma,
               double n_seeds_fraction,
               int seed) {
                auto costs = double_array_to_vector(
                    edge_costs, "edge_costs", graph.number_of_edges()
                );
                new (self) graph::WatershedProposalGenerator(
                    graph, std::move(costs), sigma, n_seeds_fraction, seed
                );
            },
            nb::arg("graph"),
            nb::arg("edge_costs"),
            nb::arg("sigma") = 1.0,
            nb::arg("n_seeds_fraction") = 0.1,
            nb::arg("seed") = 0
        );
    nb::class_<
        graph::GreedyAdditiveMulticutProposalGenerator,
        graph::ProposalGeneratorBase
    >(m, "_GreedyAdditiveMulticutProposalGenerator")
        .def(
            "__init__",
            [](graph::GreedyAdditiveMulticutProposalGenerator *self,
               const Graph &graph,
               ConstDoubleArray edge_costs,
               double sigma,
               double weight_stop,
               double node_num_stop,
               int seed) {
                auto costs = double_array_to_vector(
                    edge_costs, "edge_costs", graph.number_of_edges()
                );
                new (self) graph::GreedyAdditiveMulticutProposalGenerator(
                    graph, std::move(costs), sigma, weight_stop, node_num_stop, seed
                );
            },
            nb::arg("graph"),
            nb::arg("edge_costs"),
            nb::arg("sigma") = 1.0,
            nb::arg("weight_stop") = 0.0,
            nb::arg("node_num_stop") = -1.0,
            nb::arg("seed") = 0
        );

    m.def(
        "_lifted_multicut_energy",
        &lifted_multicut_energy,
        nb::arg("lifted_graph"),
        nb::arg("edge_weights"),
        nb::arg("labels")
    );
    m.def(
        "_lifted_multicut_greedy_additive",
        &lifted_multicut_greedy_additive,
        nb::arg("lifted_graph"),
        nb::arg("edge_weights"),
        nb::arg("n_base_edges"),
        nb::arg("weight_stop"),
        nb::arg("node_num_stop"),
        nb::arg("add_noise"),
        nb::arg("seed"),
        nb::arg("sigma")
    );
    m.def(
        "_lifted_multicut_kernighan_lin",
        &lifted_multicut_kernighan_lin,
        nb::arg("base_graph"),
        nb::arg("lifted_graph"),
        nb::arg("edge_weights"),
        nb::arg("n_base_edges"),
        nb::arg("initial_labels"),
        nb::arg("number_of_outer_iterations"),
        nb::arg("epsilon")
    );

    const auto register_mutex_watershed_clustering = [&m]<class WeightT>(const char *name) {
        m.def(
            name,
            &mutex_watershed_clustering_t<WeightT>,
            nb::arg("graph"),
            nb::arg("edge_costs"),
            nb::arg("mutex_uvs"),
            nb::arg("mutex_costs")
        );
    };
    register_mutex_watershed_clustering.operator()<float>("_mutex_watershed_clustering_float32");
    register_mutex_watershed_clustering.operator()<double>("_mutex_watershed_clustering_float64");

    const auto register_semantic_mutex_watershed_clustering =
        [&m]<class WeightT>(const char *name) {
            m.def(
                name,
                &semantic_mutex_watershed_clustering_t<WeightT>,
                nb::arg("graph"),
                nb::arg("edge_costs"),
                nb::arg("mutex_uvs"),
                nb::arg("mutex_costs"),
                nb::arg("semantic_node_classes"),
                nb::arg("semantic_costs")
            );
        };
    register_semantic_mutex_watershed_clustering
        .operator()<float>("_semantic_mutex_watershed_clustering_float32");
    register_semantic_mutex_watershed_clustering
        .operator()<double>("_semantic_mutex_watershed_clustering_float64");

    const auto register_agglo_edge_weighted = [&m]<class WeightT>(const char *name) {
        m.def(
            name,
            &agglo_edge_weighted_t<WeightT>,
            nb::arg("graph"),
            nb::arg("edge_indicators"),
            nb::arg("edge_sizes"),
            nb::arg("node_sizes"),
            nb::arg("num_clusters_stop"),
            nb::arg("size_regularizer")
        );
    };
    register_agglo_edge_weighted.operator()<float>("_agglo_edge_weighted_float32");
    register_agglo_edge_weighted.operator()<double>("_agglo_edge_weighted_float64");

    const auto register_agglo_node_and_edge_weighted =
        [&m]<class WeightT>(const char *name) {
            m.def(
                name,
                &agglo_node_and_edge_weighted_t<WeightT>,
                nb::arg("graph"),
                nb::arg("edge_indicators"),
                nb::arg("edge_sizes"),
                nb::arg("node_sizes"),
                nb::arg("node_features"),
                nb::arg("num_clusters_stop"),
                nb::arg("size_regularizer"),
                nb::arg("beta")
            );
        };
    register_agglo_node_and_edge_weighted
        .operator()<float>("_agglo_node_and_edge_weighted_float32");
    register_agglo_node_and_edge_weighted
        .operator()<double>("_agglo_node_and_edge_weighted_float64");

    const auto register_agglo_mala = [&m]<class WeightT>(const char *name) {
        m.def(
            name,
            &agglo_mala_t<WeightT>,
            nb::arg("graph"),
            nb::arg("edge_indicators"),
            nb::arg("num_bins"),
            nb::arg("bin_min"),
            nb::arg("bin_max"),
            nb::arg("num_clusters_stop"),
            nb::arg("num_edges_stop"),
            nb::arg("threshold")
        );
    };
    register_agglo_mala.operator()<float>("_agglo_mala_float32");
    register_agglo_mala.operator()<double>("_agglo_mala_float64");

    const auto register_agglo_gasp = [&m]<class WeightT>(const char *name) {
        m.def(
            name,
            &agglo_gasp_t<WeightT>,
            nb::arg("graph"),
            nb::arg("edge_weights"),
            nb::arg("edge_sizes"),
            nb::arg("is_mergeable"),
            nb::arg("num_clusters_stop"),
            nb::arg("linkage")
        );
    };
    register_agglo_gasp.operator()<float>("_agglo_gasp_float32");
    register_agglo_gasp.operator()<double>("_agglo_gasp_float64");

    // Lifted multicut sub-solver hierarchy. Same shape as the multicut sub-
    // solver bindings — opaque to Python, used by future fusion-move drivers.
    nb::class_<graph::lifted_multicut::SolverBase>(m, "_LiftedMulticutSolverBase");
    nb::class_<
        graph::lifted_multicut::GreedyAdditiveSolver,
        graph::lifted_multicut::SolverBase
    >(m, "_GreedyAdditiveLiftedMulticutSubSolver")
        .def(
            nb::init<double, double, bool, int, double>(),
            nb::arg("weight_stop") = 0.0,
            nb::arg("node_num_stop") = -1.0,
            nb::arg("add_noise") = false,
            nb::arg("seed") = 42,
            nb::arg("sigma") = 1.0
        );
    nb::class_<
        graph::lifted_multicut::KernighanLinSolver,
        graph::lifted_multicut::SolverBase
    >(m, "_KernighanLinLiftedMulticutSubSolver")
        .def(
            nb::init<std::uint64_t, double>(),
            nb::arg("number_of_outer_iterations") = 100,
            nb::arg("epsilon") = 1.0e-6
        );

    m.def(
        "_multicut_fusion_move",
        &multicut_fusion_move,
        nb::arg("graph"),
        nb::arg("edge_costs"),
        nb::arg("initial_labels"),
        nb::arg("proposal_generators"),
        nb::arg("sub_solver").none(),
        nb::arg("number_of_iterations"),
        nb::arg("stop_if_no_improvement"),
        nb::arg("number_of_threads"),
        nb::arg("number_of_parallel_proposals")
    );

    m.def(
        "_lifted_multicut_fusion_move",
        &lifted_multicut_fusion_move,
        nb::arg("base_graph"),
        nb::arg("lifted_graph"),
        nb::arg("edge_weights"),
        nb::arg("n_base_edges"),
        nb::arg("initial_labels"),
        nb::arg("proposal_generators"),
        nb::arg("sub_solver").none(),
        nb::arg("number_of_iterations"),
        nb::arg("stop_if_no_improvement"),
        nb::arg("number_of_threads"),
        nb::arg("number_of_parallel_proposals")
    );

    m.def(
        "_region_adjacency_graph_uint32",
        &region_adjacency_graph_t<std::uint32_t>,
        nb::arg("labels"),
        nb::arg("number_of_threads"),
        "Build a region adjacency graph for uint32 labels."
    );
    m.def(
        "_region_adjacency_graph_uint64",
        &region_adjacency_graph_t<std::uint64_t>,
        nb::arg("labels"),
        nb::arg("number_of_threads"),
        "Build a region adjacency graph for uint64 labels."
    );
    m.def(
        "_region_adjacency_graph_int32",
        &region_adjacency_graph_t<std::int32_t>,
        nb::arg("labels"),
        nb::arg("number_of_threads"),
        "Build a region adjacency graph for int32 labels."
    );
    m.def(
        "_region_adjacency_graph_int64",
        &region_adjacency_graph_t<std::int64_t>,
        nb::arg("labels"),
        nb::arg("number_of_threads"),
        "Build a region adjacency graph for int64 labels."
    );

    m.def(
        "_rag_coordinates_uint32",
        &rag_coordinates_t<std::uint32_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("number_of_threads"),
        "Build RAG edge coordinates for uint32 labels."
    );
    m.def(
        "_rag_coordinates_uint64",
        &rag_coordinates_t<std::uint64_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("number_of_threads"),
        "Build RAG edge coordinates for uint64 labels."
    );
    m.def(
        "_rag_coordinates_int32",
        &rag_coordinates_t<std::int32_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("number_of_threads"),
        "Build RAG edge coordinates for int32 labels."
    );
    m.def(
        "_rag_coordinates_int64",
        &rag_coordinates_t<std::int64_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("number_of_threads"),
        "Build RAG edge coordinates for int64 labels."
    );

    m.def(
        "_accumulate_edge_map_features_uint32",
        &accumulate_edge_map_features_t<std::uint32_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("edge_map"),
        nb::arg("compute_complex_features"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_accumulate_edge_map_features_uint64",
        &accumulate_edge_map_features_t<std::uint64_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("edge_map"),
        nb::arg("compute_complex_features"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_accumulate_edge_map_features_int32",
        &accumulate_edge_map_features_t<std::int32_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("edge_map"),
        nb::arg("compute_complex_features"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_accumulate_edge_map_features_int64",
        &accumulate_edge_map_features_t<std::int64_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("edge_map"),
        nb::arg("compute_complex_features"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_accumulate_affinity_features_uint32",
        &accumulate_affinity_features_t<std::uint32_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("compute_complex_features"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_accumulate_affinity_features_uint64",
        &accumulate_affinity_features_t<std::uint64_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("compute_complex_features"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_accumulate_affinity_features_int32",
        &accumulate_affinity_features_t<std::int32_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("compute_complex_features"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_accumulate_affinity_features_int64",
        &accumulate_affinity_features_t<std::int64_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("compute_complex_features"),
        nb::arg("number_of_threads")
    );

    // Distributed region-adjacency-graph + edge-feature primitives.
    m.def(
        "_block_region_adjacency_edges_uint32",
        &block_region_adjacency_edges_t<std::uint32_t>,
        nb::arg("labels"), nb::arg("own_begin"), nb::arg("own_shape"),
        nb::arg("number_of_threads"),
        "Extract owned region-adjacency edges from a uint32 label block."
    );
    m.def(
        "_block_region_adjacency_edges_uint64",
        &block_region_adjacency_edges_t<std::uint64_t>,
        nb::arg("labels"), nb::arg("own_begin"), nb::arg("own_shape"),
        nb::arg("number_of_threads"),
        "Extract owned region-adjacency edges from a uint64 label block."
    );
    m.def(
        "_block_region_adjacency_edges_int32",
        &block_region_adjacency_edges_t<std::int32_t>,
        nb::arg("labels"), nb::arg("own_begin"), nb::arg("own_shape"),
        nb::arg("number_of_threads"),
        "Extract owned region-adjacency edges from an int32 label block."
    );
    m.def(
        "_block_region_adjacency_edges_int64",
        &block_region_adjacency_edges_t<std::int64_t>,
        nb::arg("labels"), nb::arg("own_begin"), nb::arg("own_shape"),
        nb::arg("number_of_threads"),
        "Extract owned region-adjacency edges from an int64 label block."
    );

    m.def(
        "_block_edge_map_stats_uint32",
        &block_edge_map_stats_t<std::uint32_t>,
        nb::arg("labels"), nb::arg("edge_map"), nb::arg("own_begin"),
        nb::arg("own_shape"), nb::arg("number_of_threads")
    );
    m.def(
        "_block_edge_map_stats_uint64",
        &block_edge_map_stats_t<std::uint64_t>,
        nb::arg("labels"), nb::arg("edge_map"), nb::arg("own_begin"),
        nb::arg("own_shape"), nb::arg("number_of_threads")
    );
    m.def(
        "_block_edge_map_stats_int32",
        &block_edge_map_stats_t<std::int32_t>,
        nb::arg("labels"), nb::arg("edge_map"), nb::arg("own_begin"),
        nb::arg("own_shape"), nb::arg("number_of_threads")
    );
    m.def(
        "_block_edge_map_stats_int64",
        &block_edge_map_stats_t<std::int64_t>,
        nb::arg("labels"), nb::arg("edge_map"), nb::arg("own_begin"),
        nb::arg("own_shape"), nb::arg("number_of_threads")
    );

    m.def(
        "_block_affinity_stats_uint32",
        &block_affinity_stats_t<std::uint32_t>,
        nb::arg("labels"), nb::arg("affinities"), nb::arg("offsets"),
        nb::arg("own_begin"), nb::arg("own_shape"), nb::arg("number_of_threads")
    );
    m.def(
        "_block_affinity_stats_uint64",
        &block_affinity_stats_t<std::uint64_t>,
        nb::arg("labels"), nb::arg("affinities"), nb::arg("offsets"),
        nb::arg("own_begin"), nb::arg("own_shape"), nb::arg("number_of_threads")
    );
    m.def(
        "_block_affinity_stats_int32",
        &block_affinity_stats_t<std::int32_t>,
        nb::arg("labels"), nb::arg("affinities"), nb::arg("offsets"),
        nb::arg("own_begin"), nb::arg("own_shape"), nb::arg("number_of_threads")
    );
    m.def(
        "_block_affinity_stats_int64",
        &block_affinity_stats_t<std::int64_t>,
        nb::arg("labels"), nb::arg("affinities"), nb::arg("offsets"),
        nb::arg("own_begin"), nb::arg("own_shape"), nb::arg("number_of_threads")
    );

    m.def("_merge_edges", &distributed_merge_edges, nb::arg("edges"));
    m.def(
        "_merge_block_edge_stats",
        &distributed_merge_block_edge_stats,
        nb::arg("global_graph"), nb::arg("current_stats"),
        nb::arg("block_edges"), nb::arg("block_stats")
    );
    m.def(
        "_finalize_edge_features",
        &distributed_finalize_edge_features,
        nb::arg("stats"), nb::arg("compute_complex_features")
    );
    m.def(
        "_lifted_edges_from_affinities_uint32",
        &lifted_edges_from_affinities_t<std::uint32_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("offsets"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_lifted_edges_from_affinities_uint64",
        &lifted_edges_from_affinities_t<std::uint64_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("offsets"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_lifted_edges_from_affinities_int32",
        &lifted_edges_from_affinities_t<std::int32_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("offsets"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_lifted_edges_from_affinities_int64",
        &lifted_edges_from_affinities_t<std::int64_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("offsets"),
        nb::arg("number_of_threads")
    );

    m.def(
        "_lifted_edges_from_node_labels_uint32",
        &lifted_edges_from_node_labels_t<std::uint32_t>,
        nb::arg("graph"),
        nb::arg("node_labels"),
        nb::arg("graph_depth"),
        nb::arg("mode"),
        nb::arg("ignore_label"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_lifted_edges_from_node_labels_uint64",
        &lifted_edges_from_node_labels_t<std::uint64_t>,
        nb::arg("graph"),
        nb::arg("node_labels"),
        nb::arg("graph_depth"),
        nb::arg("mode"),
        nb::arg("ignore_label"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_lifted_edges_from_node_labels_int32",
        &lifted_edges_from_node_labels_t<std::int32_t>,
        nb::arg("graph"),
        nb::arg("node_labels"),
        nb::arg("graph_depth"),
        nb::arg("mode"),
        nb::arg("ignore_label"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_lifted_edges_from_node_labels_int64",
        &lifted_edges_from_node_labels_t<std::int64_t>,
        nb::arg("graph"),
        nb::arg("node_labels"),
        nb::arg("graph_depth"),
        nb::arg("mode"),
        nb::arg("ignore_label"),
        nb::arg("number_of_threads")
    );

    m.def(
        "_accumulate_lifted_affinity_features_uint32",
        &accumulate_lifted_affinity_features_t<std::uint32_t>,
        nb::arg("labels"),
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("lifted_uvs"),
        nb::arg("compute_complex_features"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_accumulate_lifted_affinity_features_uint64",
        &accumulate_lifted_affinity_features_t<std::uint64_t>,
        nb::arg("labels"),
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("lifted_uvs"),
        nb::arg("compute_complex_features"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_accumulate_lifted_affinity_features_int32",
        &accumulate_lifted_affinity_features_t<std::int32_t>,
        nb::arg("labels"),
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("lifted_uvs"),
        nb::arg("compute_complex_features"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_accumulate_lifted_affinity_features_int64",
        &accumulate_lifted_affinity_features_t<std::int64_t>,
        nb::arg("labels"),
        nb::arg("affinities"),
        nb::arg("offsets"),
        nb::arg("lifted_uvs"),
        nb::arg("compute_complex_features"),
        nb::arg("number_of_threads")
    );

    m.def(
        "_project_node_labels_to_pixels_uint32",
        &project_node_labels_to_pixels_t<std::uint32_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("node_labels"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_project_node_labels_to_pixels_uint64",
        &project_node_labels_to_pixels_t<std::uint64_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("node_labels"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_project_node_labels_to_pixels_int32",
        &project_node_labels_to_pixels_t<std::int32_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("node_labels"),
        nb::arg("number_of_threads")
    );
    m.def(
        "_project_node_labels_to_pixels_int64",
        &project_node_labels_to_pixels_t<std::int64_t>,
        nb::arg("rag"),
        nb::arg("labels"),
        nb::arg("node_labels"),
        nb::arg("number_of_threads")
    );

#define BIC_BIND_ACCUMULATE_LABELS(LSUF, LT, OSUF, OT)              \
    m.def(                                                          \
        "_accumulate_labels_" #LSUF "_" #OSUF,                      \
        &accumulate_labels_t<LT, OT>,                               \
        nb::arg("rag"),                                             \
        nb::arg("labels"),                                          \
        nb::arg("other_labels"),                                    \
        nb::arg("has_ignore_value"),                                \
        nb::arg("ignore_value"),                                    \
        nb::arg("number_of_threads")                                \
    )

    BIC_BIND_ACCUMULATE_LABELS(uint32, std::uint32_t, uint32, std::uint32_t);
    BIC_BIND_ACCUMULATE_LABELS(uint32, std::uint32_t, uint64, std::uint64_t);
    BIC_BIND_ACCUMULATE_LABELS(uint32, std::uint32_t, int32,  std::int32_t);
    BIC_BIND_ACCUMULATE_LABELS(uint32, std::uint32_t, int64,  std::int64_t);
    BIC_BIND_ACCUMULATE_LABELS(uint64, std::uint64_t, uint32, std::uint32_t);
    BIC_BIND_ACCUMULATE_LABELS(uint64, std::uint64_t, uint64, std::uint64_t);
    BIC_BIND_ACCUMULATE_LABELS(uint64, std::uint64_t, int32,  std::int32_t);
    BIC_BIND_ACCUMULATE_LABELS(uint64, std::uint64_t, int64,  std::int64_t);
    BIC_BIND_ACCUMULATE_LABELS(int32,  std::int32_t,  uint32, std::uint32_t);
    BIC_BIND_ACCUMULATE_LABELS(int32,  std::int32_t,  uint64, std::uint64_t);
    BIC_BIND_ACCUMULATE_LABELS(int32,  std::int32_t,  int32,  std::int32_t);
    BIC_BIND_ACCUMULATE_LABELS(int32,  std::int32_t,  int64,  std::int64_t);
    BIC_BIND_ACCUMULATE_LABELS(int64,  std::int64_t,  uint32, std::uint32_t);
    BIC_BIND_ACCUMULATE_LABELS(int64,  std::int64_t,  uint64, std::uint64_t);
    BIC_BIND_ACCUMULATE_LABELS(int64,  std::int64_t,  int32,  std::int32_t);
    BIC_BIND_ACCUMULATE_LABELS(int64,  std::int64_t,  int64,  std::int64_t);

#undef BIC_BIND_ACCUMULATE_LABELS
}

} // namespace bioimage_cpp::bindings
