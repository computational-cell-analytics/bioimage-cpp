#pragma once

#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/graph/lifted_multicut/detail.hxx"
#include "bioimage_cpp/graph/lifted_multicut/objective.hxx"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bioimage_cpp::graph::lifted_multicut {

// Reusable scratch state for `lifted_greedy_additive`.
struct GreedyAdditiveWorkspace {
    detail::ContractionState state;

    void reset(const UndirectedGraph &lifted_graph) {
        state.reset(lifted_graph);
    }
};

// Greedy-additive multicut on the lifted graph. Identical contraction flow to
// multicut::greedy_additive, with one twist: lifted edges are never contracted
// directly. They influence the energy via weight accumulation during merges —
// a lifted edge between super-nodes contributes its weight to the contracted
// edge whenever the two super-nodes also share a non-lifted edge, and the
// combined edge stops being lifted as soon as a base-graph path connects its
// endpoints.
inline std::vector<std::uint64_t> greedy_additive(
    const UndirectedGraph &lifted_graph,
    const std::vector<double> &weights,
    const std::uint64_t n_base_edges,
    const double weight_stop,
    const double node_num_stop,
    const bool add_noise,
    const int seed,
    const double sigma,
    GreedyAdditiveWorkspace &workspace
) {
    BIOIMAGE_PROFILE_INIT(profile);
    validate_weights(lifted_graph, weights);
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "workspace_reset");
        workspace.reset(lifted_graph);
    }
    auto &state = workspace.state;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "initialize");
        detail::initialize_contraction_state(
            lifted_graph,
            weights,
            n_base_edges,
            state,
            add_noise,
            seed,
            sigma
        );
    }

    while (!state.heap.empty() && state.topology.number_of_nodes() > 1) {
        const auto top = state.heap.top();
        if (top.priority <= weight_stop) {
            break;
        }
        if (node_num_stop > 0.0
            && state.topology.number_of_nodes()
                <= detail::stop_node_count(
                    lifted_graph,
                    node_num_stop
                )) {
            break;
        }
        detail::contract_edge(state, top.key, profile);
    }
    std::vector<std::uint64_t> labels;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "labels_from_sets");
        labels = detail::labels_from_sets(state.union_find, lifted_graph);
    }
    BIOIMAGE_PROFILE_REPORT(profile);
    return labels;
}

inline std::vector<std::uint64_t> greedy_additive(
    const UndirectedGraph &lifted_graph,
    const std::vector<double> &weights,
    const std::uint64_t n_base_edges,
    const double weight_stop,
    const double node_num_stop,
    const bool add_noise,
    const int seed,
    const double sigma
) {
    GreedyAdditiveWorkspace workspace;
    return greedy_additive(
        lifted_graph,
        weights,
        n_base_edges,
        weight_stop,
        node_num_stop,
        add_noise,
        seed,
        sigma,
        workspace
    );
}

class GreedyAdditiveSolver final : public SolverBase {
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
          sigma_(sigma) {}

    std::vector<std::uint64_t> optimize(Objective &objective) const override {
        auto labels = greedy_additive(
            objective.lifted_graph(),
            objective.weights(),
            objective.number_of_base_edges(),
            weight_stop_,
            node_num_stop_,
            add_noise_,
            seed_,
            sigma_
        );
        objective.set_labels(labels);
        return labels;
    }

private:
    double weight_stop_;
    double node_num_stop_;
    bool add_noise_;
    int seed_;
    double sigma_;
};

} // namespace bioimage_cpp::graph::lifted_multicut
