#include "graph.hxx"
#include "ndarray.hxx"

#include "bioimage_cpp/graph/contraction_graph.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

using Graph = graph::UndirectedGraph;
using ContractionGraph = graph::ContractionGraph;
using UInt64Array = nb::ndarray<nb::numpy, std::uint64_t, nb::c_contig>;

template <class T>
using ConstFloatingArray = nb::ndarray<nb::numpy, const T, nb::c_contig>;

graph::ParallelEdgePolicy parallel_edge_policy_from_code(const int code) {
    switch (code) {
    case 0:
        return graph::ParallelEdgePolicy::Merge;
    case 1:
        return graph::ParallelEdgePolicy::Keep;
    default:
        throw std::invalid_argument(
            "unknown parallel edge policy code: " + std::to_string(code)
        );
    }
}

graph::Reduction reduction_from_code(const int code) {
    switch (code) {
    case 0:
        return graph::Reduction::Sum;
    case 1:
        return graph::Reduction::Mean;
    case 2:
        return graph::Reduction::Minimum;
    case 3:
        return graph::Reduction::Maximum;
    default:
        throw std::invalid_argument("unknown reduction code: " + std::to_string(code));
    }
}

template <class T>
void add_values(
    ContractionGraph &contraction,
    const std::string &name,
    const ConstFloatingArray<T> values,
    const int reduction_code,
    const bool node_values
) {
    const auto expected = node_values
        ? contraction.nodes().size()
        : contraction.edges().size();
    if (values.ndim() == 0) {
        throw std::invalid_argument("values must have at least one dimension");
    }
    if (values.shape(0) != expected) {
        throw std::invalid_argument(
            "values.shape[0] must match the original number of " +
            std::string(node_values ? "nodes" : "edges") +
            ", got " + std::to_string(values.shape(0)) +
            " for " + std::to_string(expected)
        );
    }

    std::vector<std::size_t> component_shape;
    component_shape.reserve(values.ndim() - 1);
    std::size_t size = 1;
    for (std::size_t axis = 0; axis < values.ndim(); ++axis) {
        size *= values.shape(axis);
        if (axis > 0) {
            component_shape.push_back(values.shape(axis));
        }
    }
    const auto input = std::span<const T>(values.data(), size);
    if (node_values) {
        contraction.add_node_values<T>(
            name,
            input,
            std::move(component_shape),
            reduction_from_code(reduction_code)
        );
    } else {
        contraction.add_edge_values<T>(
            name,
            input,
            std::move(component_shape),
            reduction_from_code(reduction_code)
        );
    }
}

UInt64Array ids_to_array(const std::vector<std::uint64_t> &ids) {
    return detail::copy_vector_to_array(ids);
}

UInt64Array adjacency_to_array(
    const std::vector<graph::detail::ContractionAdjacency> &adjacency
) {
    auto output = detail::make_array_for_overwrite<std::uint64_t>(
        {adjacency.size(), std::size_t{2}}
    );
    for (std::size_t index = 0; index < adjacency.size(); ++index) {
        output.data()[2 * index] = adjacency[index].node;
        output.data()[2 * index + 1] = adjacency[index].edge;
    }
    return output;
}

template <class T>
nb::object values_to_array(const graph::MaterializedValues<T> &values) {
    return nb::cast(detail::copy_vector_to_array(values.values, values.shape));
}

nb::object values_to_array(const graph::MaterializedValueMap &values) {
    return std::visit(
        [](const auto &typed) {
            return values_to_array(typed);
        },
        values
    );
}

nb::tuple active_node_values(
    const ContractionGraph &contraction,
    const std::string &name
) {
    return nb::make_tuple(
        ids_to_array(contraction.nodes()),
        values_to_array(contraction.active_node_values(name))
    );
}

nb::tuple active_edge_values(
    const ContractionGraph &contraction,
    const std::string &name
) {
    return nb::make_tuple(
        ids_to_array(contraction.edges()),
        values_to_array(contraction.active_edge_values(name))
    );
}

UInt64Array graph_uv_ids(const Graph &graph) {
    auto output = detail::make_array_for_overwrite<std::uint64_t>(
        {static_cast<std::size_t>(graph.number_of_edges()), std::size_t{2}}
    );
    for (std::uint64_t edge = 0; edge < graph.number_of_edges(); ++edge) {
        const auto uv = graph.uv(edge);
        output.data()[2 * edge] = uv.first;
        output.data()[2 * edge + 1] = uv.second;
    }
    return output;
}

nb::dict materialized_values_to_dict(
    const std::vector<graph::MaterializedValueMap> &values
) {
    nb::dict output;
    for (const auto &value : values) {
        std::visit(
            [&output](const auto &typed) {
                output[nb::str(typed.name.c_str())] = values_to_array(typed);
            },
            value
        );
    }
    return output;
}

nb::tuple materialize(ContractionGraph &contraction) {
    auto result = contraction.materialize();
    const auto number_of_nodes = result.graph.number_of_nodes();
    auto uvs = graph_uv_ids(result.graph);
    auto node_values = materialized_values_to_dict(result.node_values);
    auto edge_values = materialized_values_to_dict(result.edge_values);
    auto node_mapping = detail::copy_vector_to_array(result.node_mapping);
    auto edge_mapping = detail::copy_vector_to_array(result.edge_mapping);
    return nb::make_tuple(
        number_of_nodes,
        std::move(uvs),
        std::move(node_values),
        std::move(edge_values),
        std::move(node_mapping),
        std::move(edge_mapping)
    );
}

} // namespace

void bind_graph_contraction(nb::module_ &m) {
    nb::class_<ContractionGraph>(m, "_ContractionGraph")
        .def(
            "__init__",
            [](ContractionGraph *self, const Graph &graph, const int parallel_edges) {
                new (self) ContractionGraph(
                    graph,
                    parallel_edge_policy_from_code(parallel_edges)
                );
            },
            nb::arg("graph"),
            nb::arg("parallel_edges") = 0
        )
        .def_prop_ro("number_of_nodes", &ContractionGraph::number_of_nodes)
        .def_prop_ro("number_of_edges", &ContractionGraph::number_of_edges)
        .def("nodes", &ContractionGraph::nodes)
        .def("edges", &ContractionGraph::edges)
        .def("uv", &ContractionGraph::uv, nb::arg("edge"))
        .def("find_edge", &ContractionGraph::find_edge, nb::arg("u"), nb::arg("v"))
        .def(
            "find_edges",
            [](const ContractionGraph &self, const std::uint64_t u,
               const std::uint64_t v) {
                return ids_to_array(self.find_edges(u, v));
            },
            nb::arg("u"),
            nb::arg("v")
        )
        .def(
            "node_adjacency",
            [](const ContractionGraph &self, const std::uint64_t node) {
                return adjacency_to_array(self.node_adjacency(node));
            },
            nb::arg("node")
        )
        .def("degree", &ContractionGraph::degree, nb::arg("node"))
        .def("is_node_active", &ContractionGraph::node_active, nb::arg("node"))
        .def("is_edge_active", &ContractionGraph::edge_active, nb::arg("edge"))
        .def(
            "representative",
            &ContractionGraph::representative,
            nb::arg("original_node")
        )
        .def(
            "_add_node_values_float32",
            [](ContractionGraph &self, const std::string &name,
               const ConstFloatingArray<float> values, const int reduction) {
                add_values(self, name, values, reduction, true);
            },
            nb::arg("name"),
            nb::arg("values"),
            nb::arg("reduction")
        )
        .def(
            "_add_node_values_float64",
            [](ContractionGraph &self, const std::string &name,
               const ConstFloatingArray<double> values, const int reduction) {
                add_values(self, name, values, reduction, true);
            },
            nb::arg("name"),
            nb::arg("values"),
            nb::arg("reduction")
        )
        .def(
            "_add_edge_values_float32",
            [](ContractionGraph &self, const std::string &name,
               const ConstFloatingArray<float> values, const int reduction) {
                add_values(self, name, values, reduction, false);
            },
            nb::arg("name"),
            nb::arg("values"),
            nb::arg("reduction")
        )
        .def(
            "_add_edge_values_float64",
            [](ContractionGraph &self, const std::string &name,
               const ConstFloatingArray<double> values, const int reduction) {
                add_values(self, name, values, reduction, false);
            },
            nb::arg("name"),
            nb::arg("values"),
            nb::arg("reduction")
        )
        .def(
            "contract_edge",
            &ContractionGraph::contract_edge,
            nb::arg("edge"),
            nb::arg("keep_node") = std::nullopt
        )
        .def("erase_edge", &ContractionGraph::erase_edge, nb::arg("edge"))
        .def("suppress_node", &ContractionGraph::suppress_node, nb::arg("node"))
        .def(
            "node_value",
            [](const ContractionGraph &self, const std::string &name,
               const std::uint64_t node) {
                return values_to_array(self.node_value(name, node));
            },
            nb::arg("name"),
            nb::arg("node")
        )
        .def(
            "edge_value",
            [](const ContractionGraph &self, const std::string &name,
               const std::uint64_t edge) {
                return values_to_array(self.edge_value(name, edge));
            },
            nb::arg("name"),
            nb::arg("edge")
        )
        .def(
            "active_node_values",
            &active_node_values,
            nb::arg("name")
        )
        .def(
            "active_edge_values",
            &active_edge_values,
            nb::arg("name")
        )
        .def("materialize", &materialize);
}

} // namespace bioimage_cpp::bindings
