#pragma once

#include "bioimage_cpp/detail/indexed_heap.hxx"
#include "bioimage_cpp/graph/detail/contraction_topology.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <cstddef>
#include <functional>

namespace bioimage_cpp::graph::agglomeration {

// Strategy interface for hierarchical agglomerative clustering.
//
// A cluster policy carries all per-edge / per-node auxiliary state required
// to compute heap priorities (edge sizes, node sizes, histograms, features,
// signed weights, and cannot-link masks). The driver owns the contraction
// topology, union-find, and heap.
//
// The agglo heap is a min-heap (smallest priority pops first), matching
// nifty's convention: edge indicators in the edge-weighted / node+edge-
// weighted / MALA policies are interpreted as boundary strengths, so the
// weakest boundary is the strongest merge candidate. The GASP policy
// stores ``-|weight|`` to recover max-heap-on-absolute-value semantics on
// top of the same min-heap container.
class ClusterPolicyBase {
public:
    using Topology = graph::detail::ContractionTopology;
    using EdgeHeap =
        bioimage_cpp::detail::DenseIndexedHeap<double, std::greater<double>>;

    // Decision returned by `next_action`. The driver acts as follows:
    //   kMerge      → contract the heap-top edge between its current endpoints
    //   kRejectEdge → pop the heap-top edge and continue (no contraction)
    //   kStop       → terminate the agglomeration loop
    enum class Action { kMerge, kRejectEdge, kStop };

    virtual ~ClusterPolicyBase() = default;

    // Seed the heap and initialize policy state.
    virtual void initialize(
        const UndirectedGraph &graph,
        EdgeHeap &heap
    ) = 0;

    // Check termination independently of the heap top.
    virtual bool is_done(const Topology &topology) const = 0;

    // Heap-top-dependent action. Called after `is_done` returns false and
    // before any contraction is attempted. `edge_id` is the heap top key
    // and `priority` is its cached priority.
    virtual Action next_action(
        std::size_t edge_id,
        double priority,
        const Topology &topology
    ) = 0;

    // Called once per contraction, before the per-fold loop, to let the
    // policy update node-level state (node sizes, features, mutex storage).
    // Roots `stable` and `removed` are super-node ids; `stable` survives.
    virtual void merge_nodes(std::size_t stable, std::size_t removed) = 0;

    // Called per fold when two edges between the same pair of super-nodes
    // collapse into one. Updates the policy's per-edge state for the
    // surviving edge `existing_id` and returns the new heap priority for
    // that edge. `u_new` and `v_new` are the current super-node endpoints
    // of the surviving edge (both already reflect any node-level updates
    // applied by `merge_nodes`).
    virtual double merge_edges(
        std::size_t existing_id,
        std::size_t fold_id,
        std::size_t u_new,
        std::size_t v_new
    ) = 0;

    // Priority for a rekeyed (no-fold) edge whose endpoint has just been
    // renamed from `removed` to `stable`. The default keeps the current
    // priority — appropriate for policies whose priority does not depend on
    // node-level state (Mala, GASP). Policies whose priority does depend on
    // node sizes / features (edge-weighted, node+edge-weighted) override.
    virtual double rekeyed_priority(
        std::size_t edge_id,
        std::size_t u_new,
        std::size_t v_new,
        double current_priority
    ) {
        (void)edge_id;
        (void)u_new;
        (void)v_new;
        return current_priority;
    }

    // Final hook called after the per-fold loop with the (now finalised)
    // adjacency of `stable`. Policies whose priority depends on node-level
    // state (e.g. the harmonic size factor) use this to recompute the
    // priority of every edge incident to `stable` whose endpoint sizes have
    // changed. Default: no-op.
    virtual void contract_edge_done(
        std::size_t stable,
        const Topology &topology,
        EdgeHeap &heap
    ) {
        (void)stable;
        (void)topology;
        (void)heap;
    }
};

} // namespace bioimage_cpp::graph::agglomeration
