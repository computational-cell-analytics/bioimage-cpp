#pragma once

#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/graph/agglomeration/cluster_policy_base.hxx"
#include "bioimage_cpp/util/union_find.hxx"

#include <cstddef>
#include <cstdint>

namespace bioimage_cpp::graph::agglomeration::detail {

template <class Policy>
class ContractionObserver {
public:
    ContractionObserver(
        ClusterPolicyBase::Topology &topology,
        util::UnionFind &sets,
        ClusterPolicyBase::EdgeHeap &heap,
        Policy &policy
    )
        : topology_(topology),
          sets_(sets),
          heap_(heap),
          policy_(policy) {
    }

    void begin_contraction(
        const std::uint64_t kept_node,
        const std::uint64_t removed_node,
        const std::uint64_t removed_edge
    ) {
        (void)removed_edge;
        sets_.merge_to(kept_node, removed_node);
        policy_.merge_nodes(
            static_cast<std::size_t>(kept_node),
            static_cast<std::size_t>(removed_node)
        );
    }

    void edge_deactivated(const std::uint64_t edge) {
        heap_.erase(static_cast<std::size_t>(edge));
    }

    void edge_rekeyed(const graph::detail::EdgeRekey &rekey) {
        const auto edge = static_cast<std::size_t>(rekey.edge);
        if (!heap_.contains(edge)) {
            return;
        }
        const auto [u, v] = topology_.uv(rekey.edge);
        const auto current_priority = heap_.priority_of(edge);
        const auto new_priority = policy_.rekeyed_priority(
            edge,
            static_cast<std::size_t>(u),
            static_cast<std::size_t>(v),
            current_priority
        );
        if (new_priority != current_priority) {
            heap_.change(edge, new_priority);
        }
    }

    void edges_folded(
        const std::uint64_t kept_edge_id,
        const std::uint64_t removed_edge_id
    ) {
        const auto kept_edge = static_cast<std::size_t>(kept_edge_id);
        const auto [u, v] = topology_.uv(kept_edge_id);
        const auto new_priority = policy_.merge_edges(
            kept_edge,
            static_cast<std::size_t>(removed_edge_id),
            static_cast<std::size_t>(u),
            static_cast<std::size_t>(v)
        );
        if (heap_.contains(kept_edge)) {
            heap_.change(kept_edge, new_priority);
        }
    }

    void end_contraction(const std::uint64_t kept_node) {
        policy_.contract_edge_done(
            static_cast<std::size_t>(kept_node),
            topology_,
            heap_
        );
    }

private:
    ClusterPolicyBase::Topology &topology_;
    util::UnionFind &sets_;
    ClusterPolicyBase::EdgeHeap &heap_;
    Policy &policy_;
};

template <class Policy, class Profiler>
inline std::uint64_t contract_edge(
    ClusterPolicyBase::Topology &topology,
    util::UnionFind &sets,
    ClusterPolicyBase::EdgeHeap &heap,
    const std::size_t edge,
    Policy &policy,
    Profiler &profile
) {
    const auto edge_id = static_cast<std::uint64_t>(edge);
    const auto [kept, removed] =
        topology.preferred_contraction_nodes(edge_id);
    ContractionObserver observer(topology, sets, heap, policy);
    std::uint64_t kept_node;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "contract");
        kept_node =
            topology.contract_edge_observed_known_nodes<true, true>(
                edge_id,
                kept,
                removed,
                observer
            );
    }
    return kept_node;
}

} // namespace bioimage_cpp::graph::agglomeration::detail
