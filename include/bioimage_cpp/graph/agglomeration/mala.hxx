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
//
// Binning matches ``nifty::histogram::Histogram``:
//   fbin(v) = (v - min) / (max - min) * (N - 1)
// is the fractional bin index in ``[0, N - 1]``. Inserts split their weight
// linearly between ``floor(fbin)`` and ``ceil(fbin)`` and the bin index
// maps back to a value via ``b -> min + b / (N - 1) * (max - min)``.
// Median computation reproduces nifty's quantile loop (see ``median_of``
// below).
class MalaClusterPolicy final : public ClusterPolicyBase {
public:
    using BinCount = double;

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
        histograms_.assign(n_edges, std::vector<BinCount>(num_bins_, 0.0));
        active_edges_ = n_edges;

        std::vector<EdgeHeap::Entry> entries;
        entries.reserve(n_edges);
        for (std::uint64_t edge_id = 0; edge_id < graph.number_of_edges(); ++edge_id) {
            const auto uv = graph.uv(edge_id);
            const auto u = static_cast<std::size_t>(uv.first);
            const auto v = static_cast<std::size_t>(uv.second);
            const auto edge_index = static_cast<std::size_t>(edge_id);
            const double indicator = initial_indicators_[edge_index];
            insert_into(histograms_[edge_index], indicator, 1.0);
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
    // Fractional bin index for ``value`` in ``[bin_min_, bin_max_]``,
    // returned in ``[0, num_bins_ - 1]``. Matches nifty's
    // ``Histogram::fbin``: a value at ``bin_max_`` lands exactly on bin
    // index ``num_bins_ - 1`` rather than past the last bin.
    double fbin(double value) const {
        if (value <= bin_min_) {
            return 0.0;
        }
        if (value >= bin_max_) {
            return static_cast<double>(num_bins_ - 1);
        }
        const double normalized =
            (value - bin_min_) / (bin_max_ - bin_min_);
        return normalized * static_cast<double>(num_bins_ - 1);
    }

    // Map a fractional bin index back to a value in
    // ``[bin_min_, bin_max_]``. Matches nifty's
    // ``Histogram::fbinToValue``.
    double bin_to_value(double fbin_value) const {
        const double t = fbin_value / static_cast<double>(num_bins_ - 1);
        return (1.0 - t) * bin_min_ + t * bin_max_;
    }

    // Insert ``weight`` mass at ``value`` into ``histogram``, splitting
    // linearly between the two surrounding integer bins (nifty's
    // ``Histogram::insert``).
    void insert_into(
        std::vector<BinCount> &histogram, double value, double weight
    ) const {
        const double b = fbin(value);
        const double low = std::floor(b);
        const double high = std::ceil(b);
        if (low + 0.5 >= high) {
            histogram[static_cast<std::size_t>(low)] += weight;
        } else {
            const double w_low = high - b;
            const double w_high = b - low;
            histogram[static_cast<std::size_t>(low)] += weight * w_low;
            histogram[static_cast<std::size_t>(high)] += weight * w_high;
        }
    }

    // 0.5 quantile of the running histogram, reproducing the formula in
    // ``nifty::histogram::quantiles`` byte-for-byte (note that nifty's
    // ``binWidth`` is ``(bin_max - bin_min) / num_bins`` — *not*
    // ``/(num_bins - 1)`` — and the formula mixes that into the bin-index
    // axis; we follow it exactly for parity with nifty's MALA output).
    double median_of(const std::vector<BinCount> &histogram) const {
        double total = 0.0;
        for (const auto count : histogram) {
            total += count;
        }
        if (total == 0.0) {
            return bin_min_;
        }
        const double bin_width =
            (bin_max_ - bin_min_) / static_cast<double>(num_bins_);
        const double quant = 0.5 * total;
        double csum = 0.0;
        for (std::size_t bin = 0; bin < histogram.size(); ++bin) {
            const double new_csum = csum + histogram[bin];
            if (csum <= quant && new_csum >= quant) {
                if (bin == 0) {
                    return bin_to_value(0.0);
                }
                const double lbin =
                    static_cast<double>(static_cast<long>(bin) - 1)
                    + bin_width / 2.0;
                const double m = histogram[bin];
                const double c = csum - lbin * m;
                return bin_to_value((quant - c) / m);
            }
            csum = new_csum;
        }
        return bin_to_value(static_cast<double>(num_bins_ - 1));
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
