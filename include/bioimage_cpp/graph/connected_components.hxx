#pragma once

#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph {

class UnionFind {
public:
    explicit UnionFind(const std::size_t size) : parents_(size), ranks_(size, 0) {
        std::iota(parents_.begin(), parents_.end(), std::uint64_t{0});
    }

    std::uint64_t find(const std::uint64_t node) {
        if (parents_[static_cast<std::size_t>(node)] != node) {
            parents_[static_cast<std::size_t>(node)] = find(parents_[static_cast<std::size_t>(node)]);
        }
        return parents_[static_cast<std::size_t>(node)];
    }

    void merge(std::uint64_t first, std::uint64_t second) {
        first = find(first);
        second = find(second);
        if (first == second) {
            return;
        }
        if (ranks_[static_cast<std::size_t>(first)] < ranks_[static_cast<std::size_t>(second)]) {
            std::swap(first, second);
        }
        parents_[static_cast<std::size_t>(second)] = first;
        if (ranks_[static_cast<std::size_t>(first)] == ranks_[static_cast<std::size_t>(second)]) {
            ++ranks_[static_cast<std::size_t>(first)];
        }
    }

    void merge_to(std::uint64_t stable, std::uint64_t removed) {
        stable = find(stable);
        removed = find(removed);
        if (stable == removed) {
            return;
        }
        parents_[static_cast<std::size_t>(removed)] = stable;
        if (ranks_[static_cast<std::size_t>(stable)] <= ranks_[static_cast<std::size_t>(removed)]) {
            ranks_[static_cast<std::size_t>(stable)] = ranks_[static_cast<std::size_t>(removed)] + 1;
        }
    }

private:
    std::vector<std::uint64_t> parents_;
    std::vector<std::uint64_t> ranks_;
};

inline std::vector<std::uint64_t> dense_labels_from_union_find(
    UnionFind &sets,
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
    UnionFind sets(static_cast<std::size_t>(graph.number_of_nodes()));
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
