#pragma once

#include "bioimage_cpp/detail/mutex_storage.hxx"
#include "bioimage_cpp/detail/relabel.hxx"
#include "bioimage_cpp/detail/semantic_labels.hxx"
#include "bioimage_cpp/detail/union_find.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace bioimage_cpp::graph {

// Result of a semantic mutex watershed clustering: a dense node labeling and a
// per-node semantic class label. Unassigned nodes report ``-1``.
struct SemanticMutexWatershedResult {
    std::vector<std::uint64_t> node_labels;
    std::vector<std::int64_t> semantic_labels;
};

// Semantic mutex watershed on an arbitrary undirected graph.
//
// `graph` defines the attractive edges with one cost per edge in `edge_costs`.
// `mutex_uvs` defines the long-range repulsive (mutex) edges with one cost per
// row in `mutex_costs`. `semantic_node_classes` is an (S, 2) table whose first
// column is a node id and whose second column is a non-negative semantic class
// id, scored by `semantic_costs`. All three edge groups are sorted jointly by
// descending weight and processed in one pass:
//   - attractive: merge the two endpoints' clusters unless a mutex constraint
//     already separates them OR the clusters already carry different semantic
//     classes.
//   - mutex: insert a hard separation between the two endpoints' clusters.
//   - semantic: tag the endpoint's current cluster with the given class id if
//     the cluster is still unassigned. Semantic labels propagate when clusters
//     merge via attractive edges.
//
// Ties in weight are broken by edge group (attractive < mutex < semantic) and
// then by within-group index, for determinism. The returned `node_labels` are
// dense in [0, k) in first-occurrence order; `semantic_labels` is one
// `std::int64_t` per node, equal to the cluster's class id or ``-1`` if the
// cluster received no semantic assignment.
//
// Ported from `compute_semantic_mws_clustering` in the affogato library.
template <class WeightT>
SemanticMutexWatershedResult semantic_mutex_watershed_clustering(
    const UndirectedGraph &graph,
    const std::vector<WeightT> &edge_costs,
    const std::vector<std::array<std::uint64_t, 2>> &mutex_uvs,
    const std::vector<WeightT> &mutex_costs,
    const std::vector<std::array<std::uint64_t, 2>> &semantic_node_classes,
    const std::vector<WeightT> &semantic_costs
) {
    const auto number_of_edges = static_cast<std::size_t>(graph.number_of_edges());
    if (edge_costs.size() != number_of_edges) {
        throw std::invalid_argument(
            "edge_costs size must match graph.number_of_edges(), got edge_costs size=" +
            std::to_string(edge_costs.size()) +
            ", number_of_edges=" + std::to_string(number_of_edges)
        );
    }
    if (mutex_costs.size() != mutex_uvs.size()) {
        throw std::invalid_argument(
            "mutex_costs size must match mutex_uvs size, got mutex_costs size=" +
            std::to_string(mutex_costs.size()) +
            ", mutex_uvs size=" + std::to_string(mutex_uvs.size())
        );
    }
    if (semantic_costs.size() != semantic_node_classes.size()) {
        throw std::invalid_argument(
            "semantic_costs size must match semantic_node_classes size, got semantic_costs size=" +
            std::to_string(semantic_costs.size()) +
            ", semantic_node_classes size=" + std::to_string(semantic_node_classes.size())
        );
    }

    const auto number_of_nodes = static_cast<std::size_t>(graph.number_of_nodes());
    const auto number_of_mutex = mutex_uvs.size();
    const auto number_of_semantic = semantic_node_classes.size();

    for (std::size_t index = 0; index < number_of_mutex; ++index) {
        const auto u = mutex_uvs[index][0];
        const auto v = mutex_uvs[index][1];
        if (u >= number_of_nodes || v >= number_of_nodes) {
            throw std::invalid_argument(
                "mutex_uvs endpoints must be < number_of_nodes, got u=" +
                std::to_string(u) + ", v=" + std::to_string(v) +
                ", number_of_nodes=" + std::to_string(number_of_nodes)
            );
        }
    }
    for (std::size_t index = 0; index < number_of_semantic; ++index) {
        const auto node = semantic_node_classes[index][0];
        if (node >= number_of_nodes) {
            throw std::invalid_argument(
                "semantic_node_classes node ids must be < number_of_nodes, got node=" +
                std::to_string(node) +
                ", number_of_nodes=" + std::to_string(number_of_nodes)
            );
        }
    }

    enum class EdgeKind : std::uint8_t {
        Attractive = 0,
        Mutex = 1,
        Semantic = 2,
    };

    struct WeightedEdge {
        WeightT weight;
        std::uint64_t index;
        EdgeKind kind;
    };

    std::vector<WeightedEdge> edge_order;
    edge_order.reserve(number_of_edges + number_of_mutex + number_of_semantic);
    for (std::size_t index = 0; index < number_of_edges; ++index) {
        edge_order.push_back(
            WeightedEdge{edge_costs[index], static_cast<std::uint64_t>(index), EdgeKind::Attractive}
        );
    }
    for (std::size_t index = 0; index < number_of_mutex; ++index) {
        edge_order.push_back(
            WeightedEdge{mutex_costs[index], static_cast<std::uint64_t>(index), EdgeKind::Mutex}
        );
    }
    for (std::size_t index = 0; index < number_of_semantic; ++index) {
        edge_order.push_back(
            WeightedEdge{semantic_costs[index], static_cast<std::uint64_t>(index), EdgeKind::Semantic}
        );
    }

    std::sort(edge_order.begin(), edge_order.end(), [](const auto &first, const auto &second) {
        if (first.weight != second.weight) {
            return first.weight > second.weight;
        }
        if (first.kind != second.kind) {
            return static_cast<std::uint8_t>(first.kind) < static_cast<std::uint8_t>(second.kind);
        }
        return first.index < second.index;
    });

    bioimage_cpp::detail::UnionFind sets(number_of_nodes);
    MutexStorage mutexes(number_of_nodes);
    SemanticLabeling semantic_labels(number_of_nodes, -1);

    for (const auto &edge : edge_order) {
        const auto edge_index = static_cast<std::size_t>(edge.index);
        if (edge.kind == EdgeKind::Semantic) {
            const auto node = semantic_node_classes[edge_index][0];
            const auto class_id = static_cast<std::int64_t>(semantic_node_classes[edge_index][1]);
            const auto root = sets.find(node);
            assign_semantic_label(root, class_id, semantic_labels);
            continue;
        }

        std::uint64_t u;
        std::uint64_t v;
        if (edge.kind == EdgeKind::Mutex) {
            const auto &pair = mutex_uvs[edge_index];
            u = pair[0];
            v = pair[1];
        } else {
            const auto uv = graph.uv(edge.index);
            u = uv.first;
            v = uv.second;
        }
        if (u == v) {
            continue;
        }

        const auto root_u = sets.find(u);
        const auto root_v = sets.find(v);
        if (root_u == root_v) {
            continue;
        }
        if (check_semantic_constraint(root_u, root_v, semantic_labels)) {
            continue;
        }
        if (check_mutex(root_u, root_v, mutexes)) {
            continue;
        }

        if (edge.kind == EdgeKind::Mutex) {
            insert_mutex(root_u, root_v, mutexes);
        } else {
            const auto new_root = sets.unite_roots(root_u, root_v);
            const auto old_root = (new_root == root_u) ? root_v : root_u;
            merge_mutexes(old_root, new_root, mutexes);
            merge_semantic_labels(new_root, old_root, semantic_labels);
        }
    }

    std::vector<std::uint64_t> roots(number_of_nodes);
    std::vector<std::int64_t> semantic_out(number_of_nodes);
    for (std::size_t node = 0; node < number_of_nodes; ++node) {
        const auto root = sets.find(static_cast<std::uint64_t>(node));
        roots[node] = root;
        semantic_out[node] = semantic_labels[root];
    }

    SemanticMutexWatershedResult result;
    result.node_labels = bioimage_cpp::detail::dense_relabel(roots);
    result.semantic_labels = std::move(semantic_out);
    return result;
}

} // namespace bioimage_cpp::graph
