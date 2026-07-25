#pragma once

#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/graph/multicut/detail.hxx"
#include "bioimage_cpp/graph/multicut/objective.hxx"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bioimage_cpp::graph::multicut {

inline std::vector<std::uint64_t> greedy_fixation(
    const UndirectedGraph &graph,
    const std::vector<double> &costs,
    const double weight_stop,
    const double node_num_stop
) {
    BIOIMAGE_PROFILE_INIT(profile);
    validate_costs(graph, costs);
    detail::ContractionState state;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "workspace_reset");
        state.reset(graph);
    }
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "initialize");
        detail::initialize_contraction_state(graph, costs, state, true);
    }

    while (!state.heap.empty() && state.topology.number_of_nodes() > 1) {
        const auto top = state.heap.top();
        // Priority is |weight|, so this also handles weight == 0 (stop).
        if (top.priority <= weight_stop) {
            break;
        }
        if (node_num_stop > 0.0
            && state.topology.number_of_nodes()
                <= detail::stop_node_count(graph, node_num_stop)) {
            break;
        }
        const auto edge_id = top.key;
        auto &objective_edge = state.topology.edge_payload(edge_id);
        if (objective_edge.weight > 0.0) {
            detail::contract_edge(state, edge_id, true, profile);
        } else {
            // A negative edge installs a persistent constraint.
            state.heap.pop();
            objective_edge.is_constraint = 1;
        }
    }
    std::vector<std::uint64_t> labels;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "labels_from_sets");
        labels = detail::labels_from_sets(state.union_find, graph);
    }
    BIOIMAGE_PROFILE_REPORT(profile);
    return labels;
}

class GreedyFixationSolver final : public SolverBase {
public:
    GreedyFixationSolver(const double weight_stop = 0.0, const double node_num_stop = -1.0)
        : weight_stop_(weight_stop),
          node_num_stop_(node_num_stop) {
    }

    std::vector<std::uint64_t> optimize(Objective &objective) const override {
        auto labels = greedy_fixation(objective.graph(), objective.costs(), weight_stop_, node_num_stop_);
        objective.set_labels(labels);
        return labels;
    }

private:
    double weight_stop_;
    double node_num_stop_;
};

} // namespace bioimage_cpp::graph::multicut
