#pragma once

#include "bioimage_cpp/graph/agglomeration/cluster_policy_base.hxx"
#include "bioimage_cpp/graph/agglomeration/detail.hxx"
#include "bioimage_cpp/graph/multicut/detail.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"
#include "bioimage_cpp/util/union_find.hxx"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bioimage_cpp::graph::agglomeration {

// Hierarchical agglomerative clustering driven by a `ClusterPolicyBase`.
//
// The driver owns the dynamic graph, union-find and heap. The policy carries
// its own per-edge / per-node state (sizes, histograms, features, cannot-link
// constraints, ...) and decides per iteration whether to merge, skip, or
// stop. Returns dense node labels in `[0, k)` via the union-find roots.
inline std::vector<std::uint64_t> agglomerative_clustering(
    const UndirectedGraph &graph,
    ClusterPolicyBase &policy
) {
    multicut::detail::DynamicGraph dynamic_graph(graph);
    util::UnionFind sets(static_cast<std::size_t>(graph.number_of_nodes()));
    ClusterPolicyBase::EdgeHeap heap;
    heap.reset_capacity(static_cast<std::size_t>(graph.number_of_edges()));
    policy.initialize(graph, dynamic_graph, heap);

    while (!heap.empty() && dynamic_graph.alive_count > 1) {
        if (policy.is_done(dynamic_graph)) {
            break;
        }
        const auto top = heap.top();
        const auto action = policy.next_action(top.key, top.priority, dynamic_graph);
        if (action == ClusterPolicyBase::Action::kStop) {
            break;
        }
        if (action == ClusterPolicyBase::Action::kRejectEdge) {
            heap.pop();
            continue;
        }
        const auto &edge = dynamic_graph.edges[top.key];
        detail::agglo_merge_dynamic_nodes(
            dynamic_graph, sets, heap, edge.u, edge.v, policy
        );
    }
    return multicut::detail::labels_from_sets(sets, graph);
}

} // namespace bioimage_cpp::graph::agglomeration
