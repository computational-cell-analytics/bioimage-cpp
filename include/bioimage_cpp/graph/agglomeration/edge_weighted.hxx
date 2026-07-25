#pragma once

#include "bioimage_cpp/graph/agglomeration/cluster_policy_base.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::agglomeration {

// Hierarchical edge-weighted agglomerative clustering.
//
// Equivalent of `nifty.graph.agglo.edgeWeightedClusterPolicy`. Each iteration
// contracts the heap-top edge; priorities are `edge_indicator * sFac` where
// `sFac = 2 / (1/sizeU^sr + 1/sizeV^sr)` is a harmonic-mean size regulariser
// (sr = `size_regularizer`). Folded edges combine their indicators via a
// size-weighted average; node sizes add.
class EdgeWeightedClusterPolicy final : public ClusterPolicyBase {
public:
    EdgeWeightedClusterPolicy(
        std::vector<double> edge_indicators,
        std::vector<double> edge_sizes,
        std::vector<double> node_sizes,
        const std::size_t num_clusters_stop,
        const double size_regularizer
    )
        : edge_indicator_(std::move(edge_indicators)),
          edge_size_(std::move(edge_sizes)),
          node_size_(std::move(node_sizes)),
          num_clusters_stop_(num_clusters_stop),
          size_regularizer_(size_regularizer) {
        if (edge_indicator_.size() != edge_size_.size()) {
            throw std::invalid_argument(
                "edge_indicators and edge_sizes must have the same length, got "
                "edge_indicators.size=" + std::to_string(edge_indicator_.size()) +
                ", edge_sizes.size=" + std::to_string(edge_size_.size())
            );
        }
    }

    void initialize(
        const UndirectedGraph &graph,
        EdgeHeap &heap
    ) override {
        const auto n_edges = static_cast<std::size_t>(graph.number_of_edges());
        if (edge_indicator_.size() != n_edges) {
            throw std::invalid_argument(
                "edge_indicators length must match graph.number_of_edges, got "
                "length=" + std::to_string(edge_indicator_.size()) +
                ", number_of_edges=" + std::to_string(n_edges)
            );
        }
        if (node_size_.size() != static_cast<std::size_t>(graph.number_of_nodes())) {
            throw std::invalid_argument(
                "node_sizes length must match graph.number_of_nodes, got "
                "length=" + std::to_string(node_size_.size()) +
                ", number_of_nodes=" + std::to_string(graph.number_of_nodes())
            );
        }

        std::vector<EdgeHeap::Entry> entries;
        entries.reserve(n_edges);
        for (std::uint64_t edge_id = 0; edge_id < graph.number_of_edges(); ++edge_id) {
            const auto uv = graph.uv(edge_id);
            const auto u = static_cast<std::size_t>(uv.first);
            const auto v = static_cast<std::size_t>(uv.second);
            const auto edge_index = static_cast<std::size_t>(edge_id);
            const auto priority = priority_of(edge_index, u, v);
            entries.push_back({edge_index, priority});
        }
        heap.build_heap(std::move(entries));
    }

    bool is_done(const Topology &topology) const override {
        return topology.number_of_nodes() <= num_clusters_stop_;
    }

    Action next_action(
        std::size_t edge_id,
        double priority,
        const Topology &topology
    ) override {
        (void)edge_id;
        (void)priority;
        (void)topology;
        return Action::kMerge;
    }

    void merge_nodes(std::size_t stable, std::size_t removed) override {
        node_size_[stable] += node_size_[removed];
    }

    double merge_edges(
        std::size_t existing_id,
        std::size_t fold_id,
        std::size_t u_new,
        std::size_t v_new
    ) override {
        const double size_a = edge_size_[existing_id];
        const double size_d = edge_size_[fold_id];
        const double total = size_a + size_d;
        if (total > 0.0) {
            edge_indicator_[existing_id] =
                (size_a * edge_indicator_[existing_id]
                 + size_d * edge_indicator_[fold_id]) / total;
        }
        edge_size_[existing_id] = total;
        return priority_of(existing_id, u_new, v_new);
    }

    double rekeyed_priority(
        std::size_t edge_id,
        std::size_t u_new,
        std::size_t v_new,
        double current_priority
    ) override {
        (void)current_priority;
        return priority_of(edge_id, u_new, v_new);
    }

    void contract_edge_done(
        std::size_t stable,
        const Topology &topology,
        EdgeHeap &heap
    ) override {
        for (const auto &entry : topology.node_adjacency(stable)) {
            const auto edge_id = static_cast<std::size_t>(entry.edge);
            const auto neighbor = static_cast<std::size_t>(entry.node);
            const auto new_priority = priority_of(edge_id, stable, neighbor);
            if (heap.contains(edge_id)
                && heap.priority_of(edge_id) != new_priority) {
                heap.change(edge_id, new_priority);
            }
        }
    }

private:
    double priority_of(std::size_t edge_id, std::size_t u, std::size_t v) const {
        return edge_indicator_[edge_id] * size_factor(u, v);
    }

    double size_factor(std::size_t u, std::size_t v) const {
        if (size_regularizer_ == 0.0) {
            return 1.0;
        }
        const double su = std::pow(node_size_[u], size_regularizer_);
        const double sv = std::pow(node_size_[v], size_regularizer_);
        if (su == 0.0 || sv == 0.0) {
            return 0.0;
        }
        return 2.0 / (1.0 / su + 1.0 / sv);
    }

    std::vector<double> edge_indicator_;
    std::vector<double> edge_size_;
    std::vector<double> node_size_;
    std::size_t num_clusters_stop_;
    double size_regularizer_;
};

} // namespace bioimage_cpp::graph::agglomeration
