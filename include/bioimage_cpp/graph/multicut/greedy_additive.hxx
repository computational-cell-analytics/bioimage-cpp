#pragma once

#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/graph/multicut/detail.hxx"
#include "bioimage_cpp/graph/multicut/objective.hxx"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace bioimage_cpp::graph::multicut {

// Reusable scratch state for `greedy_additive`.
struct GreedyAdditiveWorkspace {
    detail::ContractionState state;

    void reset(const UndirectedGraph &graph) {
        state.reset(graph);
    }
};

inline std::vector<std::uint64_t> greedy_additive(
    const UndirectedGraph &graph,
    const std::vector<double> &costs,
    const double weight_stop,
    const double node_num_stop,
    const bool add_noise,
    const int seed,
    const double sigma,
    GreedyAdditiveWorkspace &workspace
) {
    BIOIMAGE_PROFILE_INIT(profile);
    validate_costs(graph, costs);
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "workspace_reset");
        workspace.reset(graph);
    }
    auto &state = workspace.state;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "initialize");
        detail::initialize_contraction_state(
            graph, costs, state, false, add_noise, seed, sigma
        );
    }

    while (!state.heap.empty() && state.topology.number_of_nodes() > 1) {
        const auto top = state.heap.top();
        if (top.priority <= weight_stop) {
            break;
        }
        if (node_num_stop > 0.0
            && state.topology.number_of_nodes()
                <= detail::stop_node_count(graph, node_num_stop)) {
            break;
        }
        detail::contract_edge(state, top.key, false, profile);
    }
    std::vector<std::uint64_t> labels;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "labels_from_sets");
        labels = detail::labels_from_sets(state.union_find, graph);
    }
    BIOIMAGE_PROFILE_REPORT(profile);
    return labels;
}

inline std::vector<std::uint64_t> greedy_additive(
    const UndirectedGraph &graph,
    const std::vector<double> &costs,
    const double weight_stop,
    const double node_num_stop,
    const bool add_noise,
    const int seed,
    const double sigma
) {
    GreedyAdditiveWorkspace workspace;
    return greedy_additive(
        graph, costs, weight_stop, node_num_stop, add_noise, seed, sigma, workspace
    );
}

class GreedyAdditiveSolver final : public CloneableSolverBase {
public:
    GreedyAdditiveSolver(
        const double weight_stop = 0.0,
        const double node_num_stop = -1.0,
        const bool add_noise = false,
        const int seed = 42,
        const double sigma = 1.0
    )
        : weight_stop_(weight_stop),
          node_num_stop_(node_num_stop),
          add_noise_(add_noise),
          seed_(seed),
          sigma_(sigma) {
    }

    std::vector<std::uint64_t> optimize(Objective &objective) const override {
        auto labels = greedy_additive(
            objective.graph(),
            objective.costs(),
            weight_stop_,
            node_num_stop_,
            add_noise_,
            seed_,
            sigma_
        );
        objective.set_labels(labels);
        return labels;
    }

    std::unique_ptr<CloneableSolverBase> clone() const override {
        return std::make_unique<GreedyAdditiveSolver>(*this);
    }

private:
    double weight_stop_;
    double node_num_stop_;
    bool add_noise_;
    int seed_;
    double sigma_;
};

} // namespace bioimage_cpp::graph::multicut
