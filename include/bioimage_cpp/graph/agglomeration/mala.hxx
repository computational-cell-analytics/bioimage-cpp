#pragma once

#include "bioimage_cpp/graph/agglomeration/cluster_policy_base.hxx"
#include "bioimage_cpp/graph/multicut/detail.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::agglomeration {

// Histogram-based MALA cluster policy (Funke et al.). Each edge carries a
// histogram of indicators seen across contractions; the priority is the
// histogram's running median. Histograms add element-wise on merge. The
// agglomeration stops when the heap top crosses `threshold`, or when the
// cluster / edge count drops to the configured stop.
//
// Set `num_clusters_stop = 0` or `num_edges_stop = 0` to disable the
// respective count-based stop. The threshold stop is always active.
class MalaClusterPolicy final : public ClusterPolicyBase {
public:
    using BinCount = std::uint32_t;

    MalaClusterPolicy(
        std::vector<double> edge_indicators,
        const std::size_t num_bins,
        const double bin_min,
        const double bin_max,
        const std::size_t num_clusters_stop,
        const std::size_t num_edges_stop,
        const double threshold
    )
        : initial_indicators_(std::move(edge_indicators)),
          num_bins_(num_bins),
          bin_min_(bin_min),
          bin_max_(bin_max),
          num_clusters_stop_(num_clusters_stop),
          num_edges_stop_(num_edges_stop),
          threshold_(threshold) {
        if (num_bins_ == 0) {
            throw std::invalid_argument("num_bins must be >= 1");
        }
        if (!(bin_max_ > bin_min_)) {
            throw std::invalid_argument(
                "bin_max must be > bin_min, got bin_min=" +
                std::to_string(bin_min_) + ", bin_max=" + std::to_string(bin_max_)
            );
        }
    }

    void initialize(
        const UndirectedGraph &graph,
        DynamicGraph &dynamic_graph,
        EdgeHeap &heap
    ) override {
        const auto n_edges = static_cast<std::size_t>(graph.number_of_edges());
        if (initial_indicators_.size() != n_edges) {
            throw std::invalid_argument(
                "edge_indicators length must match graph.number_of_edges, got "
                "length=" + std::to_string(initial_indicators_.size()) +
                ", number_of_edges=" + std::to_string(n_edges)
            );
        }
        histograms_.assign(n_edges, std::vector<BinCount>(num_bins_, 0));
        active_edges_ = n_edges;

        std::vector<EdgeHeap::Entry> entries;
        entries.reserve(n_edges);
        for (std::uint64_t edge_id = 0; edge_id < graph.number_of_edges(); ++edge_id) {
            const auto uv = graph.uv(edge_id);
            const auto u = static_cast<std::size_t>(uv.first);
            const auto v = static_cast<std::size_t>(uv.second);
            const auto edge_index = static_cast<std::size_t>(edge_id);
            const double indicator = initial_indicators_[edge_index];
            histograms_[edge_index][bin_index(indicator)] = 1;
            const double priority = indicator;
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
        if (num_clusters_stop_ > 0 && dynamic_graph.alive_count <= num_clusters_stop_) {
            return true;
        }
        if (num_edges_stop_ > 0 && active_edges_ <= num_edges_stop_) {
            return true;
        }
        return false;
    }

    Action next_action(
        std::size_t edge_id,
        double priority,
        const DynamicGraph &dynamic_graph
    ) override {
        (void)edge_id;
        (void)dynamic_graph;
        if (priority >= threshold_) {
            return Action::kStop;
        }
        return Action::kMerge;
    }

    void merge_nodes(std::size_t stable, std::size_t removed) override {
        (void)stable;
        (void)removed;
    }

    double merge_edges(
        std::size_t existing_id,
        std::size_t fold_id,
        std::size_t u_new,
        std::size_t v_new
    ) override {
        (void)u_new;
        (void)v_new;
        auto &target = histograms_[existing_id];
        const auto &source = histograms_[fold_id];
        for (std::size_t bin = 0; bin < num_bins_; ++bin) {
            target[bin] += source[bin];
        }
        --active_edges_;
        return median_of(target);
    }

    // No `contract_edge_done` override: Mala priorities depend only on the
    // surviving histogram, not on node sizes, so on-stable edges retain
    // their priorities.

private:
    std::size_t bin_index(double value) const {
        if (value <= bin_min_) {
            return 0;
        }
        if (value >= bin_max_) {
            return num_bins_ - 1;
        }
        const double position = (value - bin_min_) / (bin_max_ - bin_min_);
        auto index = static_cast<std::size_t>(position * static_cast<double>(num_bins_));
        if (index >= num_bins_) {
            index = num_bins_ - 1;
        }
        return index;
    }

    double bin_center(std::size_t bin) const {
        const double step = (bin_max_ - bin_min_) / static_cast<double>(num_bins_);
        return bin_min_ + step * (static_cast<double>(bin) + 0.5);
    }

    double median_of(const std::vector<BinCount> &histogram) const {
        std::uint64_t total = 0;
        for (const auto count : histogram) {
            total += count;
        }
        if (total == 0) {
            return bin_min_;
        }
        const std::uint64_t half = (total + 1) / 2;
        std::uint64_t cumulative = 0;
        for (std::size_t bin = 0; bin < histogram.size(); ++bin) {
            cumulative += histogram[bin];
            if (cumulative >= half) {
                return bin_center(bin);
            }
        }
        return bin_center(histogram.size() - 1);
    }

    std::vector<double> initial_indicators_;
    std::size_t num_bins_;
    double bin_min_;
    double bin_max_;
    std::size_t num_clusters_stop_;
    std::size_t num_edges_stop_;
    double threshold_;
    std::vector<std::vector<BinCount>> histograms_;
    std::size_t active_edges_ = 0;
};

} // namespace bioimage_cpp::graph::agglomeration
