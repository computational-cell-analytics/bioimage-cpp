#pragma once

#include "bioimage_cpp/util/union_find.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace bioimage_cpp::graph {

inline std::vector<std::uint64_t> dense_labels_from_union_find(
    bioimage_cpp::util::UnionFind &sets,
    const std::uint64_t number_of_nodes
) {
    std::unordered_map<std::uint64_t, std::uint64_t> relabeling;
    std::vector<std::uint64_t> labels(static_cast<std::size_t>(number_of_nodes));
    for (std::uint64_t node = 0; node < number_of_nodes; ++node) {
        const auto root = sets.find(node);
        auto found = relabeling.find(root);
        if (found == relabeling.end()) {
            found = relabeling.emplace(root, static_cast<std::uint64_t>(relabeling.size())).first;
        }
        labels[static_cast<std::size_t>(node)] = found->second;
    }
    return labels;
}

inline std::vector<std::uint64_t> connected_components(
    const UndirectedGraph &graph,
    const std::uint8_t *edge_mask = nullptr
) {
    bioimage_cpp::util::UnionFind sets(static_cast<std::size_t>(graph.number_of_nodes()));
    for (std::uint64_t edge = 0; edge < graph.number_of_edges(); ++edge) {
        if (edge_mask != nullptr && edge_mask[static_cast<std::size_t>(edge)] == 0) {
            continue;
        }
        const auto uv = graph.uv(edge);
        sets.merge(uv.first, uv.second);
    }
    return dense_labels_from_union_find(sets, graph.number_of_nodes());
}

} // namespace bioimage_cpp::graph
