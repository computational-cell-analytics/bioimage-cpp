#pragma once

#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/graph/agglomeration/cluster_policy_base.hxx"
#include "bioimage_cpp/graph/agglomeration/detail.hxx"
#include "bioimage_cpp/graph/connected_components.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"
#include "bioimage_cpp/util/union_find.hxx"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bioimage_cpp::graph::agglomeration {

// Hierarchical agglomerative clustering driven by a `ClusterPolicyBase`.
//
// The policy owns objective-specific state and controls merge decisions.
inline std::vector<std::uint64_t> agglomerative_clustering(
    const UndirectedGraph &graph,
    ClusterPolicyBase &policy
) {
    BIOIMAGE_PROFILE_INIT(profile);
    ClusterPolicyBase::Topology topology;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "topology_reset");
        topology.reset(graph);
    }
    util::UnionFind sets(static_cast<std::size_t>(graph.number_of_nodes()));
    ClusterPolicyBase::EdgeHeap heap;
    heap.reset_capacity(static_cast<std::size_t>(graph.number_of_edges()));
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "initialize");
        policy.initialize(graph, heap);
    }
    while (!heap.empty() && topology.number_of_nodes() > 1) {
        if (policy.is_done(topology)) {
            break;
        }
        const auto top = heap.top();
        const auto action = policy.next_action(
            top.key,
            top.priority,
            topology
        );
        if (action == ClusterPolicyBase::Action::kStop) {
            break;
        }
        if (action == ClusterPolicyBase::Action::kRejectEdge) {
            heap.pop();
            continue;
        }
        detail::contract_edge(
            topology,
            sets,
            heap,
            top.key,
            policy,
            profile
        );
    }
    std::vector<std::uint64_t> labels;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "labels_from_sets");
        labels = dense_labels_from_union_find(
            sets,
            graph.number_of_nodes()
        );
    }
    BIOIMAGE_PROFILE_REPORT(profile);
    return labels;
}

} // namespace bioimage_cpp::graph::agglomeration
