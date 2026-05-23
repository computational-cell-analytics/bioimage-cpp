#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/edge_hash.hxx"
#include "bioimage_cpp/detail/threading.hxx"
#include "bioimage_cpp/graph/breadth_first_search.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace bioimage_cpp::graph::lifted_multicut {

enum class LiftedNodeLabelMode { all, same, different };

// Discover lifted edges from per-node labels by BFS-neighborhood expansion.
//
// For every source node `u` the BFS reports each reachable node `v` together
// with the hop distance. A pair `(u, v)` with `u < v` becomes a lifted edge
// iff:
//   - distance is in [2, graph_depth] (distance 1 corresponds to base edges
//     and is excluded);
//   - neither labels[u] nor labels[v] equals `ignore_label` (when set);
//   - the `mode` predicate matches: `all` keeps every pair, `same` keeps
//     pairs with labels[u] == labels[v], `different` keeps the complement.
//
// Returns the deduplicated set sorted lexicographically with `u < v`.
template <class LabelT>
std::vector<bioimage_cpp::detail::Edge> lifted_edges_from_node_labels(
    const UndirectedGraph &graph,
    const ConstArrayView<LabelT> &node_labels,
    const std::uint64_t graph_depth,
    const LiftedNodeLabelMode mode,
    const std::optional<LabelT> ignore_label,
    const std::size_t number_of_threads
) {
    if (node_labels.ndim() != 1) {
        throw std::invalid_argument(
            "node_labels must be a 1D array"
        );
    }
    if (static_cast<std::uint64_t>(node_labels.shape[0]) != graph.number_of_nodes()) {
        throw std::invalid_argument(
            "node_labels length must match graph number_of_nodes"
        );
    }
    if (graph_depth < 1) {
        throw std::invalid_argument(
            "graph_depth must be >= 1"
        );
    }

    const auto n_nodes = static_cast<std::size_t>(graph.number_of_nodes());
    if (n_nodes == 0) {
        return {};
    }

    const auto n_threads = bioimage_cpp::detail::normalize_thread_count(
        number_of_threads, n_nodes
    );

    const auto *labels = node_labels.data;

    const auto label_pair_passes =
        [&](const LabelT label_u, const LabelT label_v) -> bool {
            if (ignore_label.has_value()) {
                if (label_u == *ignore_label || label_v == *ignore_label) {
                    return false;
                }
            }
            switch (mode) {
                case LiftedNodeLabelMode::all:
                    return true;
                case LiftedNodeLabelMode::same:
                    return label_u == label_v;
                case LiftedNodeLabelMode::different:
                    return label_u != label_v;
            }
            return false;
        };

    using EdgeSet = std::unordered_set<
        bioimage_cpp::detail::Edge, bioimage_cpp::detail::EdgeHash
    >;
    std::vector<EdgeSet> per_thread(n_threads);

    bioimage_cpp::detail::parallel_for_chunks(
        n_threads,
        n_nodes,
        [&](const std::size_t thread_id, const std::size_t begin, const std::size_t end) {
            auto &out = per_thread[thread_id];
            BfsWorkspace workspace;
            for (std::size_t source = begin; source < end; ++source) {
                const auto label_u = labels[source];
                if (ignore_label.has_value() && label_u == *ignore_label) {
                    continue;
                }
                const auto entries = breadth_first_search(
                    graph,
                    static_cast<std::uint64_t>(source),
                    graph_depth,
                    /*include_source=*/false,
                    workspace
                );
                for (const auto &entry : entries) {
                    if (entry.distance < 2) {
                        continue;
                    }
                    if (entry.node <= source) {
                        continue;
                    }
                    const auto label_v = labels[static_cast<std::size_t>(entry.node)];
                    if (!label_pair_passes(label_u, label_v)) {
                        continue;
                    }
                    out.insert(bioimage_cpp::detail::edge_key(
                        static_cast<std::uint64_t>(source), entry.node
                    ));
                }
            }
        }
    );

    EdgeSet merged;
    std::size_t total = 0;
    for (const auto &set : per_thread) {
        total += set.size();
    }
    merged.reserve(total);
    for (auto &set : per_thread) {
        merged.insert(set.begin(), set.end());
    }

    std::vector<bioimage_cpp::detail::Edge> result(merged.begin(), merged.end());
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace bioimage_cpp::graph::lifted_multicut
