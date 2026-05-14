#include "graph.hxx"

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
using UInt64Array = nb::ndarray<nb::numpy, std::uint64_t, nb::c_contig>;
using ConstUInt64Array = nb::ndarray<nb::numpy, const std::uint64_t, nb::c_contig>;
using Int64Array = nb::ndarray<nb::numpy, std::int64_t, nb::c_contig>;

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
}

} // namespace bioimage_cpp::bindings
