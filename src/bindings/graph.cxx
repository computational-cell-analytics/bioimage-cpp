#include "graph.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/graph/connected_components.hxx"
#include "bioimage_cpp/graph/feature_accumulation.hxx"
#include "bioimage_cpp/graph/multicut.hxx"
#include "bioimage_cpp/graph/node_label_projection.hxx"
#include "bioimage_cpp/graph/region_adjacency_graph.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

using Graph = graph::UndirectedGraph;
using Rag = graph::RegionAdjacencyGraph;
using UInt64Array = nb::ndarray<nb::numpy, std::uint64_t, nb::c_contig>;
using ConstUInt8Array = nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig>;
using ConstUInt64Array = nb::ndarray<nb::numpy, const std::uint64_t, nb::c_contig>;
using Int64Array = nb::ndarray<nb::numpy, std::int64_t, nb::c_contig>;
using DoubleArray = nb::ndarray<nb::numpy, double, nb::c_contig>;
using ConstDoubleArray = nb::ndarray<nb::numpy, const double, nb::c_contig>;

template <class T>
using LabelArray = nb::ndarray<nb::numpy, const T, nb::c_contig>;

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

UInt64Array vector_to_uint64_array(const std::vector<std::uint64_t> &values) {
    auto result = make_uint64_array({values.size()});
    std::copy(values.begin(), values.end(), result.data());
    return result;
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

template <class LabelT>
DoubleArray accumulate_edge_map_features_t(
    const Rag &rag,
    LabelArray<LabelT> labels,
    ConstDoubleArray edge_map,
    const bool compute_complex_features,
    const std::size_t number_of_threads
) {
    const auto number_of_features = compute_complex_features ? 12 : 2;
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
    const auto number_of_features = compute_complex_features ? 12 : 2;
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

} // namespace

void bind_graph(nb::module_ &m) {
    nb::class_<Graph>(m, "UndirectedGraph")
        .def(
            nb::init<std::uint64_t, std::uint64_t>(),
            nb::arg("number_of_nodes") = 0,
            nb::arg("reserve_number_of_edges") = 0
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
        .def_static(
            "from_edges",
            &graph_from_edges,
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

    nb::class_<Rag, Graph>(m, "RegionAdjacencyGraph")
        .def_prop_ro("shape", &Rag::shape);

    m.def("_connected_components", &graph_connected_components, nb::arg("graph"));
    m.def(
        "_connected_components_masked",
        &graph_connected_components_masked,
        nb::arg("graph"),
        nb::arg("edge_mask")
    );
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
}

} // namespace bioimage_cpp::bindings
