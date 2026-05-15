#pragma once

#include "bioimage_cpp/graph/detail/fusion_contract.hxx"
#include "bioimage_cpp/graph/multicut/greedy_additive.hxx"
#include "bioimage_cpp/graph/multicut/objective.hxx"
#include "bioimage_cpp/graph/proposal_generator.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::multicut {

class FusionMoveSolver final : public SolverBase {
public:
    // The proposal generator is borrowed; the caller owns its lifetime.
    // The sub-solver pointer is optional: when null, the driver uses a default
    // greedy-additive sub-solver internally.
    //
    // `number_of_threads` and `number_of_parallel_proposals` are reserved for
    // future use; v1 only supports the single-threaded pairwise
    // (proposal, current) fuse and rejects other values.
    FusionMoveSolver(
        ProposalGeneratorBase &proposal_generator,
        const SolverBase *sub_solver = nullptr,
        const std::size_t number_of_iterations = 10,
        const std::size_t stop_if_no_improvement = 4,
        const std::size_t number_of_threads = 1,
        const std::size_t number_of_parallel_proposals = 2
    )
        : proposal_generator_(proposal_generator),
          sub_solver_(sub_solver),
          number_of_iterations_(number_of_iterations),
          stop_if_no_improvement_(stop_if_no_improvement) {
        if (number_of_threads != 1) {
            throw std::invalid_argument(
                "FusionMoveSolver currently supports number_of_threads=1 only"
            );
        }
        if (number_of_parallel_proposals != 2) {
            throw std::invalid_argument(
                "FusionMoveSolver currently supports number_of_parallel_proposals=2 only"
            );
        }
    }

    std::vector<std::uint64_t> optimize(Objective &objective) const override {
        const auto &graph = objective.graph();
        const auto &costs = objective.costs();
        const auto number_of_nodes = graph.number_of_nodes();

        std::vector<std::uint64_t> current = objective.labels();
        if (number_of_nodes == 0 || graph.number_of_edges() == 0) {
            objective.set_labels(current);
            return objective.labels();
        }

        const auto default_sub_solver = GreedyAdditiveSolver();
        const auto &sub_solver = sub_solver_ != nullptr
            ? *sub_solver_
            : static_cast<const SolverBase &>(default_sub_solver);

        // Warm start from greedy-additive if the caller passed the trivial
        // singleton labeling.
        if (is_singleton_labeling(current)) {
            Objective warm_objective(graph, costs);
            current = default_sub_solver.optimize(warm_objective);
        }

        double current_energy = energy(graph, costs, current);

        std::vector<std::uint64_t> proposal(static_cast<std::size_t>(number_of_nodes));
        std::size_t iterations_without_improvement = 0;

        for (std::size_t iteration = 0; iteration < number_of_iterations_; ++iteration) {
            proposal_generator_.generate(current, proposal);

            const auto fused = fuse_pair(graph, costs, current, proposal, sub_solver);
            const auto fused_energy = energy(graph, costs, fused);
            const auto proposal_energy = energy(graph, costs, proposal);

            // Best-of safety net across the three candidates so an iteration
            // can never raise the running energy.
            double best_energy = current_energy;
            const std::vector<std::uint64_t> *best = &current;
            if (fused_energy < best_energy) {
                best_energy = fused_energy;
                best = &fused;
            }
            if (proposal_energy < best_energy) {
                best_energy = proposal_energy;
                best = &proposal;
            }

            if (best_energy < current_energy) {
                current = *best;
                current_energy = best_energy;
                iterations_without_improvement = 0;
            } else {
                ++iterations_without_improvement;
                if (iterations_without_improvement >= stop_if_no_improvement_) {
                    break;
                }
            }
        }

        objective.set_labels(current);
        return objective.labels();
    }

private:
    static bool is_singleton_labeling(const std::vector<std::uint64_t> &labels) {
        for (std::size_t index = 0; index < labels.size(); ++index) {
            if (labels[index] != static_cast<std::uint64_t>(index)) {
                return false;
            }
        }
        return true;
    }

    static std::vector<std::uint64_t> fuse_pair(
        const UndirectedGraph &graph,
        const std::vector<double> &costs,
        const std::vector<std::uint64_t> &current,
        const std::vector<std::uint64_t> &proposal,
        const SolverBase &sub_solver
    ) {
        const auto number_of_nodes = static_cast<std::size_t>(graph.number_of_nodes());
        std::vector<std::uint64_t> stacked(2 * number_of_nodes);
        std::copy(current.begin(), current.end(), stacked.begin());
        std::copy(proposal.begin(), proposal.end(), stacked.begin() + number_of_nodes);

        auto contraction = ::bioimage_cpp::graph::detail::contract_by_agreement(
            graph, stacked.data(), 2, number_of_nodes
        );

        const auto &contracted_graph = contraction.contracted_graph;
        const auto number_of_contracted_edges = contracted_graph.number_of_edges();

        std::vector<double> contracted_costs(
            static_cast<std::size_t>(number_of_contracted_edges), 0.0
        );
        for (std::uint64_t edge = 0; edge < graph.number_of_edges(); ++edge) {
            const auto target = contraction.contracted_edge_of_original[
                static_cast<std::size_t>(edge)
            ];
            if (target < 0) {
                continue;
            }
            contracted_costs[static_cast<std::size_t>(target)] +=
                costs[static_cast<std::size_t>(edge)];
        }

        if (number_of_contracted_edges == 0) {
            std::vector<std::uint64_t> result(number_of_nodes);
            for (std::uint64_t node = 0; node < graph.number_of_nodes(); ++node) {
                result[static_cast<std::size_t>(node)] = contraction.root_of_node[
                    static_cast<std::size_t>(node)
                ];
            }
            return result;
        }

        Objective sub_objective(contracted_graph, std::move(contracted_costs));
        const auto sub_labels = sub_solver.optimize(sub_objective);

        std::vector<std::uint64_t> result(number_of_nodes);
        for (std::uint64_t node = 0; node < graph.number_of_nodes(); ++node) {
            const auto root = contraction.root_of_node[static_cast<std::size_t>(node)];
            result[static_cast<std::size_t>(node)] = sub_labels[
                static_cast<std::size_t>(root)
            ];
        }
        return result;
    }

    ProposalGeneratorBase &proposal_generator_;
    const SolverBase *sub_solver_;
    std::size_t number_of_iterations_;
    std::size_t stop_if_no_improvement_;
};

} // namespace bioimage_cpp::graph::multicut
