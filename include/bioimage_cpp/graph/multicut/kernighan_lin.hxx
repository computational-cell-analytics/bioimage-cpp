#pragma once

#include "bioimage_cpp/graph/multicut/objective.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace bioimage_cpp::graph::multicut {

inline std::vector<std::uint64_t> kernighan_lin(
    const UndirectedGraph &graph,
    const std::vector<double> &costs,
    std::vector<std::uint64_t> labels,
    const std::uint64_t number_of_outer_iterations,
    const double epsilon
) {
    validate_costs(graph, costs);
    validate_labels(graph, labels);
    labels = dense_relabel(labels);
    double current_energy = energy(graph, costs, labels);

    for (std::uint64_t iteration = 0; iteration < number_of_outer_iterations; ++iteration) {
        bool changed = false;
        auto next_label = static_cast<std::uint64_t>(*std::max_element(labels.begin(), labels.end()) + 1);
        for (std::uint64_t node = 0; node < graph.number_of_nodes(); ++node) {
            const auto old_label = labels[static_cast<std::size_t>(node)];
            std::unordered_set<std::uint64_t> candidate_labels;
            for (const auto adjacency : graph.node_adjacency(node)) {
                candidate_labels.insert(labels[static_cast<std::size_t>(adjacency.node)]);
            }
            candidate_labels.insert(next_label);

            auto best_label = old_label;
            auto best_energy = current_energy;
            for (const auto candidate : candidate_labels) {
                if (candidate == old_label) {
                    continue;
                }
                double delta = 0.0;
                for (const auto adjacency : graph.node_adjacency(node)) {
                    const auto neighbor_label = labels[static_cast<std::size_t>(adjacency.node)];
                    const auto cost = costs[static_cast<std::size_t>(adjacency.edge)];
                    const auto was_cut = old_label != neighbor_label;
                    const auto is_cut = candidate != neighbor_label;
                    if (was_cut != is_cut) {
                        delta += is_cut ? cost : -cost;
                    }
                }
                const auto proposed_energy = current_energy + delta;
                if (proposed_energy + epsilon < best_energy) {
                    best_energy = proposed_energy;
                    best_label = candidate;
                }
            }
            labels[static_cast<std::size_t>(node)] = best_label;
            if (best_label != old_label) {
                current_energy = best_energy;
                changed = true;
                if (best_label == next_label) {
                    ++next_label;
                }
            }
        }
        labels = dense_relabel(labels);
        current_energy = energy(graph, costs, labels);
        if (!changed) {
            break;
        }
    }
    return dense_relabel(labels);
}

class KernighanLinSolver final : public SolverBase {
public:
    KernighanLinSolver(
        const std::uint64_t number_of_outer_iterations = 100,
        const double epsilon = 1.0e-6
    )
        : number_of_outer_iterations_(number_of_outer_iterations),
          epsilon_(epsilon) {
    }

    std::vector<std::uint64_t> optimize(Objective &objective) const override {
        auto labels = kernighan_lin(
            objective.graph(),
            objective.costs(),
            objective.labels(),
            number_of_outer_iterations_,
            epsilon_
        );
        objective.set_labels(labels);
        return labels;
    }

private:
    std::uint64_t number_of_outer_iterations_;
    double epsilon_;
};

} // namespace bioimage_cpp::graph::multicut
