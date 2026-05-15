#pragma once

#include <cstdint>
#include <vector>

namespace bioimage_cpp::graph {

// Base class for proposal generators consumed by fusion-move solvers.
//
// A proposal generator is constructed once per problem with whatever data it
// needs (graph, edge costs, hyperparameters, RNG seed). The driver then calls
// `generate(current, proposal)` once per fusion-move iteration. Each
// implementation may use `current` (e.g. for warm-start) or ignore it.
//
// Output proposals must:
//   - Have the same length as the graph's node count.
//   - Use label `0` reserved for "background" only if the implementation
//     explicitly documents this; the fusion-move driver does not interpret
//     label values, only their equality classes.
//
// Implementations should be deterministic given their constructor seed and
// the number of times `generate` has been called.
class ProposalGeneratorBase {
public:
    virtual ~ProposalGeneratorBase() = default;

    virtual void generate(
        const std::vector<std::uint64_t> &current_labels,
        std::vector<std::uint64_t> &proposal
    ) = 0;

    virtual void reset() {}
};

} // namespace bioimage_cpp::graph
