#pragma once

#include "bioimage_cpp/graph/agglomeration/cluster_policy_base.hxx"
#include "bioimage_cpp/graph/multicut/detail.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::agglomeration {

// Hierarchical agglomeration that blends per-edge indicators with a node-
// feature distance. Equivalent of
// `nifty.graph.agglo.nodeAndEdgeWeightedClusterPolicy`.
//
// Priority for edge (u, v):
//   fromNodes = sqrt(sum_c (feat[u][c] - feat[v][c])^2)
//   fromEdge  = edge_indicator
//   priority  = (beta * fromNodes + (1 - beta) * fromEdge) * sFac
// with the same harmonic-mean `sFac` as `EdgeWeightedClusterPolicy`.
//
// Node features and node sizes aggregate via size-weighted means on each
// contraction; edge state aggregates as in the edge-weighted policy.
class NodeAndEdgeWeightedClusterPolicy final : public ClusterPolicyBase {
public:
    NodeAndEdgeWeightedClusterPolicy(
        std::vector<double> edge_indicators,
        std::vector<double> edge_sizes,
        std::vector<double> node_sizes,
        std::vector<std::vector<double>> node_features,
        const std::size_t num_clusters_stop,
        const double size_regularizer,
        const double beta
    )
        : edge_indicator_(std::move(edge_indicators)),
          edge_size_(std::move(edge_sizes)),
          node_size_(std::move(node_sizes)),
          node_features_(std::move(node_features)),
          num_clusters_stop_(num_clusters_stop),
          size_regularizer_(size_regularizer),
          beta_(beta) {
        if (edge_indicator_.size() != edge_size_.size()) {
            throw std::invalid_argument(
                "edge_indicators and edge_sizes must have the same length, got "
                "edge_indicators.size=" + std::to_string(edge_indicator_.size()) +
                ", edge_sizes.size=" + std::to_string(edge_size_.size())
            );
        }
        if (node_size_.size() != node_features_.size()) {
            throw std::invalid_argument(
                "node_sizes and node_features must have the same length, got "
                "node_sizes.size=" + std::to_string(node_size_.size()) +
                ", node_features.size=" + std::to_string(node_features_.size())
            );
        }
        if (!node_features_.empty()) {
            num_channels_ = node_features_.front().size();
            for (const auto &row : node_features_) {
                if (row.size() != num_channels_) {
                    throw std::invalid_argument(
                        "node_features rows must all have the same length"
                    );
                }
            }
        }
    }

    void initialize(
        const UndirectedGraph &graph,
        DynamicGraph &dynamic_graph,
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
            auto &edge = dynamic_graph.edges[edge_index];
            edge.u = u;
            edge.v = v;
            const auto priority = priority_of(edge_index, u, v);
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
        (void)edge_id;
        (void)priority;
        (void)dynamic_graph;
        return Action::kMerge;
    }

    void merge_nodes(std::size_t stable, std::size_t removed) override {
        const double size_a = node_size_[stable];
        const double size_d = node_size_[removed];
        const double total = size_a + size_d;
        if (total > 0.0 && num_channels_ > 0) {
            auto &feat_a = node_features_[stable];
            const auto &feat_d = node_features_[removed];
            for (std::size_t channel = 0; channel < num_channels_; ++channel) {
                feat_a[channel] =
                    (size_a * feat_a[channel] + size_d * feat_d[channel]) / total;
            }
        }
        node_size_[stable] = total;
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
        DynamicGraph &dynamic_graph,
        EdgeHeap &heap
    ) override {
        for (const auto &entry : dynamic_graph.adjacency[stable]) {
            const auto edge_id = entry.edge_id;
            const auto neighbor = entry.neighbor;
            const auto new_priority = priority_of(edge_id, stable, neighbor);
            auto &edge = dynamic_graph.edges[edge_id];
            if (edge.weight != new_priority) {
                edge.weight = new_priority;
                if (heap.contains(edge_id)) {
                    heap.change(edge_id, new_priority);
                }
            }
        }
    }

private:
    double feature_distance(std::size_t u, std::size_t v) const {
        if (num_channels_ == 0) {
            return 0.0;
        }
        const auto &fa = node_features_[u];
        const auto &fb = node_features_[v];
        double sum = 0.0;
        for (std::size_t channel = 0; channel < num_channels_; ++channel) {
            const double delta = fa[channel] - fb[channel];
            sum += delta * delta;
        }
        return std::sqrt(sum);
    }

    double priority_of(std::size_t edge_id, std::size_t u, std::size_t v) const {
        const double from_nodes = feature_distance(u, v);
        const double from_edge = edge_indicator_[edge_id];
        const double base = beta_ * from_nodes + (1.0 - beta_) * from_edge;
        return base * size_factor(u, v);
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
    std::vector<std::vector<double>> node_features_;
    std::size_t num_clusters_stop_;
    double size_regularizer_;
    double beta_;
    std::size_t num_channels_ = 0;
};

} // namespace bioimage_cpp::graph::agglomeration
