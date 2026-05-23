#pragma once

#include "bioimage_cpp/graph/agglomeration/cluster_policy_base.hxx"
#include "bioimage_cpp/graph/multicut/detail.hxx"
#include "bioimage_cpp/util/union_find.hxx"

#include <cstddef>
#include <utility>

namespace bioimage_cpp::graph::agglomeration::detail {

// Contract the edge between super-nodes `u` and `v`. Structurally a clone of
// `multicut::detail::merge_dynamic_nodes`, but delegates the per-fold weight
// update (and the rekey-priority recomputation) to a policy.
//
// On each fold of two edges that connect the same pair of super-nodes,
// `policy.merge_edges(existing_id, fold_id)` returns the new heap priority
// for the surviving edge; the surviving edge's cached `weight` and heap
// entry are updated to that value. On the no-fold rekey branch the priority
// is recomputed via `policy.rekeyed_priority(...)` from the current heap
// value — policies whose priority depends on node-level state (e.g. the
// harmonic size factor) recompute; policies that don't keep the priority.
//
// `policy.merge_nodes(stable, removed)` is invoked once, before the per-fold
// loop, so policies that update node-level state (sizes, features, mutex
// storage) operate on the *pre-fold* roots — matching the order
// `existing_id` itself was created.
template <class Policy>
inline std::size_t agglo_merge_dynamic_nodes(
    multicut::detail::DynamicGraph &dynamic_graph,
    util::UnionFind &sets,
    ClusterPolicyBase::EdgeHeap &heap,
    std::size_t u,
    std::size_t v,
    Policy &policy
) {
    u = static_cast<std::size_t>(sets.find(u));
    v = static_cast<std::size_t>(sets.find(v));
    if (u == v) {
        return u;
    }

    auto stable = u;
    auto removed = v;
    if (dynamic_graph.adjacency[stable].size() < dynamic_graph.adjacency[removed].size()) {
        std::swap(stable, removed);
    }
    sets.merge_to(stable, removed);
    policy.merge_nodes(stable, removed);

    // Stamp stable's neighbors so each removed-neighbor lookup is O(1).
    for (const auto &entry : dynamic_graph.adjacency[stable]) {
        dynamic_graph.scratch_edge_id[entry.neighbor] = entry.edge_id;
    }

    // Erase the contracted edge from heap + stable adjacency.
    const auto contracted_edge_id = dynamic_graph.scratch_edge_id[removed];
    heap.erase(contracted_edge_id);
    dynamic_graph.scratch_edge_id[removed] = multicut::detail::no_edge;
    multicut::detail::internal::erase_by_neighbor(
        dynamic_graph.adjacency[stable], removed
    );

    // Snapshot removed's neighbors before mutating its adjacency.
    const auto removed_neighbors = dynamic_graph.adjacency[removed];

    for (const auto &entry : removed_neighbors) {
        const auto neighbor = entry.neighbor;
        const auto removed_edge_id = entry.edge_id;
        if (neighbor == stable) {
            continue;
        }

        const auto existing_id = dynamic_graph.scratch_edge_id[neighbor];
        if (existing_id == multicut::detail::no_edge) {
            // Rekey: removed-side edge inherits its endpoint rename. The
            // policy decides whether the priority changes (default: keep).
            dynamic_graph.adjacency[stable].push_back({neighbor, removed_edge_id});
            dynamic_graph.scratch_edge_id[neighbor] = removed_edge_id;
            multicut::detail::internal::rename_neighbor(
                dynamic_graph.adjacency[neighbor], removed, stable
            );
            auto &edge = dynamic_graph.edges[removed_edge_id];
            if (edge.u == removed) {
                edge.u = stable;
            } else {
                edge.v = stable;
            }
            const auto current_priority = edge.weight;
            const auto new_priority = policy.rekeyed_priority(
                removed_edge_id, stable, neighbor, current_priority
            );
            if (new_priority != current_priority) {
                edge.weight = new_priority;
                // The edge may have been previously popped via kRejectEdge
                // (GASP cannot-link), in which case it is no longer in the
                // heap. Use push_or_change to handle both cases.
                if (heap.contains(removed_edge_id)) {
                    heap.change(removed_edge_id, new_priority);
                }
            }
        } else {
            // Fold: both stable and removed had an edge to `neighbor`. Let the
            // policy merge the per-edge state into `existing_id` and tell us
            // the new heap priority; then drop the removed-side edge.
            const auto new_priority = policy.merge_edges(
                existing_id, removed_edge_id, stable, neighbor
            );
            dynamic_graph.edges[existing_id].weight = new_priority;
            heap.erase(removed_edge_id);
            multicut::detail::internal::erase_by_neighbor(
                dynamic_graph.adjacency[neighbor], removed
            );
            if (heap.contains(existing_id)) {
                heap.change(existing_id, new_priority);
            }
        }
    }

    // Clear scratch via the updated stable adjacency.
    for (const auto &entry : dynamic_graph.adjacency[stable]) {
        dynamic_graph.scratch_edge_id[entry.neighbor] = multicut::detail::no_edge;
    }

    dynamic_graph.adjacency[removed].clear();
    dynamic_graph.alive[removed] = false;
    --dynamic_graph.alive_count;
    policy.contract_edge_done(stable, dynamic_graph, heap);
    return stable;
}

} // namespace bioimage_cpp::graph::agglomeration::detail
