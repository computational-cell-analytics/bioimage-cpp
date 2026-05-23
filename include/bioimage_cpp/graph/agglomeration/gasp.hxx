#pragma once

#include "bioimage_cpp/detail/mutex_storage.hxx"
#include "bioimage_cpp/graph/agglomeration/cluster_policy_base.hxx"
#include "bioimage_cpp/graph/multicut/detail.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::agglomeration {

// Linkage criterion for GASP. The criterion determines how parallel edges
// fold when two clusters merge and how the agglomeration terminates.
//
// For all linkages except ``kMutexWatershed`` the heap priority is
// ``-edge_weight`` (signed): the most attractive edge pops first, and the
// agglomeration stops as soon as no positive-weight edges remain. This
// matches nifty's "stop when no merge candidates are left" behaviour and
// avoids doing unbounded work after all attractive evidence has been
// consumed (otherwise critical for ``kSum`` whose combined weight can grow
// without bound).
//
// ``kMutexWatershed`` instead uses priority ``-|edge_weight|`` so the
// largest-magnitude edge pops first; a negative-weight pop installs a
// permanent cannot-link constraint between the two clusters and is
// rejected, exactly matching the mutex-watershed algorithm.
enum class GaspLinkage {
    kSum = 0,
    kMean = 1,
    kMax = 2,
    kMin = 3,
    kAbsMax = 4,
    kMutexWatershed = 5,
};

// Generalized Algorithm for Signed graph Partitioning (Bailoni et al.).
//
// Edge weights are signed (positive = attractive, negative = repulsive). The
// optional `is_mergeable` mask marks edges that may never trigger a merge —
// they are processed in priority order to install permanent cannot-link
// constraints between the clusters they connect. Cannot-link constraints
// propagate as clusters grow via the standard `MutexStorage` helper.
class GaspClusterPolicy final : public ClusterPolicyBase {
public:
    GaspClusterPolicy(
        std::vector<double> edge_weights,
        std::vector<double> edge_sizes,
        std::vector<std::uint8_t> is_mergeable,
        const std::size_t num_clusters_stop,
        const GaspLinkage linkage
    )
        : edge_weight_(std::move(edge_weights)),
          edge_size_(std::move(edge_sizes)),
          is_mergeable_(std::move(is_mergeable)),
          num_clusters_stop_(num_clusters_stop),
          linkage_(linkage) {
        if (edge_weight_.size() != edge_size_.size()) {
            throw std::invalid_argument(
                "edge_weights and edge_sizes must have the same length, got "
                "edge_weights.size=" + std::to_string(edge_weight_.size()) +
                ", edge_sizes.size=" + std::to_string(edge_size_.size())
            );
        }
        if (!is_mergeable_.empty() && is_mergeable_.size() != edge_weight_.size()) {
            throw std::invalid_argument(
                "is_mergeable must be empty or have the same length as "
                "edge_weights, got is_mergeable.size=" +
                std::to_string(is_mergeable_.size()) +
                ", edge_weights.size=" + std::to_string(edge_weight_.size())
            );
        }
    }

    void initialize(
        const UndirectedGraph &graph,
        DynamicGraph &dynamic_graph,
        EdgeHeap &heap
    ) override {
        const auto n_edges = static_cast<std::size_t>(graph.number_of_edges());
        const auto n_nodes = static_cast<std::size_t>(graph.number_of_nodes());
        if (edge_weight_.size() != n_edges) {
            throw std::invalid_argument(
                "edge_weights length must match graph.number_of_edges, got "
                "length=" + std::to_string(edge_weight_.size()) +
                ", number_of_edges=" + std::to_string(n_edges)
            );
        }
        if (is_mergeable_.empty()) {
            is_mergeable_.assign(n_edges, 1);
        }
        cannot_link_.assign(n_nodes, {});

        std::vector<EdgeHeap::Entry> entries;
        entries.reserve(n_edges);
        for (std::uint64_t edge_id = 0; edge_id < graph.number_of_edges(); ++edge_id) {
            const auto uv = graph.uv(edge_id);
            const auto u = static_cast<std::size_t>(uv.first);
            const auto v = static_cast<std::size_t>(uv.second);
            const auto edge_index = static_cast<std::size_t>(edge_id);
            const double priority = priority_of(edge_weight_[edge_index]);
            auto &edge = dynamic_graph.edges[edge_index];
            edge.u = u;
            edge.v = v;
            edge.weight = priority;
            edge.is_constraint = 0;
            dynamic_graph.adjacency[u].push_back({v, edge_index});
            dynamic_graph.adjacency[v].push_back({u, edge_index});
            entries.push_back({edge_index, priority});
        }
        heap.build_heap(std::move(entries));
    }

    bool is_done(const DynamicGraph &dynamic_graph) const override {
        return dynamic_graph.alive_count <= num_clusters_stop_;
    }

    Action next_action(
        std::size_t edge_id,
        double priority,
        const DynamicGraph &dynamic_graph
    ) override {
        (void)priority;
        const auto &edge = dynamic_graph.edges[edge_id];
        const auto u = static_cast<std::uint64_t>(edge.u);
        const auto v = static_cast<std::uint64_t>(edge.v);
        if (check_mutex(u, v, cannot_link_)) {
            return Action::kRejectEdge;
        }
        // Non-mutex-watershed linkages use signed-weight priority and stop
        // as soon as the top of the heap is non-positive (no attractive
        // edges remain). Matches `nifty.graph.agglo`'s `isDone` behaviour
        // and prevents `kSum` / `kAbsMax` from running away when negative
        // weights flood the queue.
        if (linkage_ != GaspLinkage::kMutexWatershed) {
            if (edge_weight_[edge_id] <= 0.0) {
                return Action::kStop;
            }
        }
        if (!is_mergeable_[edge_id]) {
            insert_mutex(u, v, cannot_link_);
            return Action::kRejectEdge;
        }
        if (linkage_ == GaspLinkage::kMutexWatershed) {
            if (edge_weight_[edge_id] <= 0.0) {
                insert_mutex(u, v, cannot_link_);
                return Action::kRejectEdge;
            }
        }
        return Action::kMerge;
    }

    void merge_nodes(std::size_t stable, std::size_t removed) override {
        merge_mutexes(
            static_cast<std::uint64_t>(removed),
            static_cast<std::uint64_t>(stable),
            cannot_link_
        );
    }

    double merge_edges(
        std::size_t existing_id,
        std::size_t fold_id,
        std::size_t u_new,
        std::size_t v_new
    ) override {
        (void)u_new;
        (void)v_new;
        const double wa = edge_weight_[existing_id];
        const double wb = edge_weight_[fold_id];
        const double sa = edge_size_[existing_id];
        const double sb = edge_size_[fold_id];
        double combined = wa;
        switch (linkage_) {
            case GaspLinkage::kSum:
                combined = wa + wb;
                break;
            case GaspLinkage::kMean: {
                const double total = sa + sb;
                combined = total > 0.0 ? (sa * wa + sb * wb) / total : wa;
                break;
            }
            case GaspLinkage::kMax:
                combined = std::max(wa, wb);
                break;
            case GaspLinkage::kMin:
                combined = std::min(wa, wb);
                break;
            case GaspLinkage::kAbsMax:
                combined = std::abs(wa) >= std::abs(wb) ? wa : wb;
                break;
            case GaspLinkage::kMutexWatershed:
                // Behaves like absolute max with sign preserved. Once two
                // super-nodes are merged on a positive edge, any folded
                // repulsive evidence is absorbed via abs-max; subsequent
                // negative-sign heap tops trigger cannot-link rejection in
                // `next_action`.
                combined = std::abs(wa) >= std::abs(wb) ? wa : wb;
                break;
        }
        edge_weight_[existing_id] = combined;
        edge_size_[existing_id] = sa + sb;
        // Mergeable iff both sides were mergeable: a single non-mergeable
        // contribution makes the surviving edge a cannot-link candidate.
        is_mergeable_[existing_id] =
            (is_mergeable_[existing_id] != 0 && is_mergeable_[fold_id] != 0) ? 1 : 0;
        return priority_of(combined);
    }

private:
    // For non-mutex-watershed linkages the priority is the negated signed
    // weight (min-heap pops the most-positive weight first; the loop
    // terminates when the top is non-positive). The mutex-watershed
    // linkage uses ``-|weight|`` so the largest-magnitude edge pops first
    // regardless of sign.
    double priority_of(const double weight) const {
        if (linkage_ == GaspLinkage::kMutexWatershed) {
            return -std::abs(weight);
        }
        return -weight;
    }

    std::vector<double> edge_weight_;
    std::vector<double> edge_size_;
    std::vector<std::uint8_t> is_mergeable_;
    std::size_t num_clusters_stop_;
    GaspLinkage linkage_;
    MutexStorage cannot_link_;
};

} // namespace bioimage_cpp::graph::agglomeration
