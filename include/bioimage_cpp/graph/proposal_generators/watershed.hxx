#pragma once

#include "bioimage_cpp/graph/edge_weighted_watershed.hxx"
#include "bioimage_cpp/graph/proposal_generator.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph {

// Watershed proposal generator: noisy edge weights + random seeds at endpoints
// of negative-cost edges, then `edge_weighted_watershed`. Mirrors the workhorse
// generator used by nifty's multicut fusion moves.
//
// `n_seeds_fraction` is interpreted as a fraction of `number_of_nodes` when
// <= 1.0 and as an absolute seed-pair count otherwise.
class WatershedProposalGenerator final : public ProposalGeneratorBase {
public:
    WatershedProposalGenerator(
        const UndirectedGraph &graph,
        std::vector<double> edge_costs,
        const double sigma = 1.0,
        const double n_seeds_fraction = 0.1,
        const int seed = 0
    )
        : graph_(graph),
          edge_costs_(std::move(edge_costs)),
          sigma_(sigma),
          n_seeds_fraction_(n_seeds_fraction),
          seed_(seed),
          generator_(static_cast<std::mt19937::result_type>(seed)),
          noise_(0.0, sigma) {
        if (edge_costs_.size() != static_cast<std::size_t>(graph.number_of_edges())) {
            throw std::invalid_argument(
                "edge_costs length must equal number_of_edges"
            );
        }
        for (std::uint64_t edge = 0; edge < graph.number_of_edges(); ++edge) {
            if (edge_costs_[static_cast<std::size_t>(edge)] < 0.0) {
                negative_edges_.push_back(edge);
            }
        }
    }

    void generate(
        const std::vector<std::uint64_t> &current_labels,
        std::vector<std::uint64_t> &proposal
    ) override {
        (void)current_labels;
        const auto number_of_nodes = graph_.number_of_nodes();
        proposal.assign(static_cast<std::size_t>(number_of_nodes), 0);

        if (negative_edges_.empty()) {
            return;
        }

        std::vector<float> noisy_costs(edge_costs_.size());
        for (std::size_t edge = 0; edge < edge_costs_.size(); ++edge) {
            noisy_costs[edge] = static_cast<float>(edge_costs_[edge] + noise_(generator_));
        }

        std::size_t n_seed_pairs;
        if (n_seeds_fraction_ <= 1.0) {
            n_seed_pairs = static_cast<std::size_t>(
                static_cast<double>(number_of_nodes) * n_seeds_fraction_ + 0.5
            );
        } else {
            n_seed_pairs = static_cast<std::size_t>(n_seeds_fraction_ + 0.5);
        }
        n_seed_pairs = std::max(std::size_t{1}, n_seed_pairs);
        n_seed_pairs = std::min(negative_edges_.size(), n_seed_pairs);

        std::vector<std::uint64_t> seeds(static_cast<std::size_t>(number_of_nodes), 0);
        std::uniform_int_distribution<std::size_t> edge_dist(0, negative_edges_.size() - 1);
        std::uint64_t next_label = 1;
        for (std::size_t i = 0; i < n_seed_pairs; ++i) {
            const auto edge = negative_edges_[edge_dist(generator_)];
            const auto uv = graph_.uv(edge);
            seeds[static_cast<std::size_t>(uv.first)] = next_label++;
            seeds[static_cast<std::size_t>(uv.second)] = next_label++;
        }

        proposal = edge_weighted_watershed<float, std::uint64_t>(graph_, noisy_costs, seeds);
    }

    void reset() override {
        generator_.seed(static_cast<std::mt19937::result_type>(seed_));
    }

private:
    const UndirectedGraph &graph_;
    std::vector<double> edge_costs_;
    double sigma_;
    double n_seeds_fraction_;
    int seed_;
    std::vector<std::uint64_t> negative_edges_;
    std::mt19937 generator_;
    std::normal_distribution<double> noise_;
};

} // namespace bioimage_cpp::graph
