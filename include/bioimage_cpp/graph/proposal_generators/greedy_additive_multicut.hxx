#pragma once

#include "bioimage_cpp/graph/multicut/greedy_additive.hxx"
#include "bioimage_cpp/graph/proposal_generator.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph {

// Greedy-additive multicut proposal generator. Repeatedly invokes the greedy
// additive multicut solver with additive Gaussian noise on the edge weights;
// the seed is incremented per call so consecutive proposals differ.
//
// Multicut-specific in the sense that it solves a multicut to produce a
// proposal, but the *output* is just a node labeling and is reusable from any
// fusion-move driver (e.g. lifted multicut).
class GreedyAdditiveMulticutProposalGenerator final : public ProposalGeneratorBase {
public:
    GreedyAdditiveMulticutProposalGenerator(
        const UndirectedGraph &graph,
        std::vector<double> edge_costs,
        const double sigma = 1.0,
        const double weight_stop = 0.0,
        const double node_num_stop = -1.0,
        const int seed = 0
    )
        : graph_(graph),
          edge_costs_(std::move(edge_costs)),
          sigma_(sigma),
          weight_stop_(weight_stop),
          node_num_stop_(node_num_stop),
          seed_(seed),
          call_count_(0) {
        if (edge_costs_.size() != static_cast<std::size_t>(graph.number_of_edges())) {
            throw std::invalid_argument(
                "edge_costs length must equal number_of_edges"
            );
        }
    }

    void generate(
        const std::vector<std::uint64_t> &current_labels,
        std::vector<std::uint64_t> &proposal
    ) override {
        (void)current_labels;
        proposal = multicut::greedy_additive(
            graph_,
            edge_costs_,
            weight_stop_,
            node_num_stop_,
            true,
            mixed_seed(seed_, call_count_),
            sigma_
        );
        ++call_count_;
    }

    void reset() override {
        call_count_ = 0;
    }

private:
    // Mix the base seed and the call counter so proposals from different
    // (slot, iteration) cells never realign. A plain `seed_ + call_count_`
    // collided with the per-slot `seed + slot` offset applied by the fusion-move
    // driver (slot 0 at iteration i+1 reused the seed of slot 1 at iteration i),
    // recomputing identical proposals and shrinking effective diversity. A
    // SplitMix64 finalizer over the packed (seed, counter) pair removes the
    // linearity.
    static int mixed_seed(const int base, const std::size_t counter) {
        std::uint64_t z =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(base)) << 32) |
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(counter));
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return static_cast<int>(static_cast<std::uint32_t>(z));
    }

    const UndirectedGraph &graph_;
    std::vector<double> edge_costs_;
    double sigma_;
    double weight_stop_;
    double node_num_stop_;
    int seed_;
    std::size_t call_count_;
};

} // namespace bioimage_cpp::graph
