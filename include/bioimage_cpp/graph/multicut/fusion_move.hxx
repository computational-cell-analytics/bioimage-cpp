#pragma once

#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/detail/threading.hxx"
#include "bioimage_cpp/graph/detail/fusion_contract.hxx"
#include "bioimage_cpp/graph/multicut/greedy_additive.hxx"
#include "bioimage_cpp/graph/multicut/objective.hxx"
#include "bioimage_cpp/graph/proposal_generator.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::multicut {

class FusionMoveSolver final : public SolverBase {
public:
    // Each entry in `proposal_generators` is one parallel-proposal source.
    // The container must have exactly `number_of_parallel_proposals` entries.
    // When `number_of_threads > 1` the workers index the generators by
    // proposal slot; each generator must have independent state (own RNG,
    // own scratch). The pointers are borrowed; the caller owns lifetimes.
    //
    // `sub_solver` is optional: nullptr uses an internal greedy-additive
    // sub-solver with a shared workspace per worker.
    FusionMoveSolver(
        std::vector<ProposalGeneratorBase *> proposal_generators,
        const SolverBase *sub_solver = nullptr,
        const std::size_t number_of_iterations = 10,
        const std::size_t stop_if_no_improvement = 4,
        const std::size_t number_of_threads = 1,
        const std::size_t number_of_parallel_proposals = 2
    )
        : proposal_generators_(std::move(proposal_generators)),
          sub_solver_(sub_solver),
          number_of_iterations_(number_of_iterations),
          stop_if_no_improvement_(stop_if_no_improvement),
          number_of_threads_(number_of_threads),
          number_of_parallel_proposals_(number_of_parallel_proposals) {
        if (number_of_parallel_proposals < 1) {
            throw std::invalid_argument(
                "number_of_parallel_proposals must be >= 1"
            );
        }
        if (number_of_threads < 1) {
            throw std::invalid_argument("number_of_threads must be >= 1");
        }
        if (proposal_generators_.size() != number_of_parallel_proposals) {
            throw std::invalid_argument(
                "proposal_generators length must equal number_of_parallel_proposals"
            );
        }
        for (const auto *pgen : proposal_generators_) {
            if (pgen == nullptr) {
                throw std::invalid_argument("proposal_generators must not contain null");
            }
        }
    }

    std::vector<std::uint64_t> optimize(Objective &objective) const override {
        BIOIMAGE_PROFILE_INIT(profile);
        const auto &graph = objective.graph();
        const auto &costs = objective.costs();
        const auto number_of_nodes = graph.number_of_nodes();

        std::vector<std::uint64_t> current = objective.labels();
        if (number_of_nodes == 0 || graph.number_of_edges() == 0) {
            objective.set_labels(current);
            return objective.labels();
        }

        // Proposal generators may read graph.node_adjacency() concurrently in the
        // stage-1 parallel region (the greedy-additive generator does, via
        // DynamicGraph::reset). The lazy CSR rebuild is not thread-safe, and the
        // warm-start below only freezes the graph for a singleton initial labeling,
        // so freeze on this thread before fan-out. See UndirectedGraph thread-safety.
        graph.freeze();

        // One workspace per worker thread; reused across the warm-start, every
        // pairwise fuse, and the stage-2 joint fuse.
        const auto effective_threads = ::bioimage_cpp::detail::normalize_thread_count(
            number_of_threads_, number_of_parallel_proposals_
        );
        std::vector<GreedyAdditiveWorkspace> workspaces(effective_threads);

        // Warm-start from greedy-additive if the caller passed the trivial
        // singleton labeling.
        if (is_singleton_labeling(current)) {
            BIOIMAGE_PROFILE_SCOPE(profile, "warm_start");
            current = greedy_additive(
                graph, costs, 0.0, -1.0, false, 42, 1.0, workspaces[0]
            );
        }

        double current_energy;
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "energy_eval");
            current_energy = energy(graph, costs, current);
        }

        // Per-proposal-slot buffers. The proposal generator writes into
        // `proposal_buffers[p]`; the pairwise-fuse writes into
        // `fused_buffers[p]`. Both are reused across iterations.
        const std::size_t P = number_of_parallel_proposals_;
        std::vector<std::vector<std::uint64_t>> proposal_buffers(P);
        std::vector<std::vector<std::uint64_t>> fused_buffers(P);
        std::vector<double> proposal_energies(P);
        std::vector<double> fused_energies(P);
        std::vector<unsigned char> is_leftover(P);

        constexpr double kEnergyEps = 1e-7;

        std::size_t iterations_without_improvement = 0;

        for (std::size_t iteration = 0; iteration < number_of_iterations_; ++iteration) {
            // === Stage 1: parallel proposal generation + parallel pairwise fuse ===

            // Snapshot current under no mutation (only the calling thread writes
            // to `current` between iterations, so workers can read it freely).
            const auto &current_snapshot = current;

            std::fill(is_leftover.begin(), is_leftover.end(), 0);

            {
                BIOIMAGE_PROFILE_SCOPE(profile, "proposal_and_pairwise_fuse");
                ::bioimage_cpp::detail::parallel_for_chunks(
                    effective_threads,
                    P,
                    [&](const std::size_t thread_id, const std::size_t begin, const std::size_t end) {
                        auto &workspace = workspaces[thread_id];
                        for (std::size_t p = begin; p < end; ++p) {
                            proposal_generators_[p]->generate(
                                current_snapshot, proposal_buffers[p]
                            );
                            proposal_energies[p] = energy(graph, costs, proposal_buffers[p]);

                            fuse_pair_into(
                                graph,
                                costs,
                                current_snapshot,
                                proposal_buffers[p],
                                sub_solver_,
                                workspace,
                                fused_buffers[p]
                            );
                            fused_energies[p] = energy(graph, costs, fused_buffers[p]);
                        }
                    }
                );
            }

            // === Aggregate stage-1 results sequentially (small loop) ===
            // Track the best candidate across {current, proposals, fused}.
            // Collect "leftovers" — fuse results that did not improve on
            // `current_energy` and were not effectively equal — for stage 2.
            double best_energy = current_energy;
            const std::vector<std::uint64_t> *best = &current;
            std::size_t leftover_count = 0;
            for (std::size_t p = 0; p < P; ++p) {
                if (proposal_energies[p] < best_energy) {
                    best_energy = proposal_energies[p];
                    best = &proposal_buffers[p];
                }
                if (fused_energies[p] < best_energy) {
                    best_energy = fused_energies[p];
                    best = &fused_buffers[p];
                }
                if (fused_energies[p] > current_energy + kEnergyEps) {
                    is_leftover[p] = 1;
                    ++leftover_count;
                }
            }

            // === Stage 2: joint multi-proposal fuse on leftovers ===
            std::vector<std::uint64_t> joint_result;
            double joint_energy = std::numeric_limits<double>::infinity();
            if (leftover_count >= 2) {
                BIOIMAGE_PROFILE_SCOPE(profile, "joint_fuse");
                std::vector<const std::vector<std::uint64_t> *> leftovers;
                leftovers.reserve(leftover_count);
                for (std::size_t p = 0; p < P; ++p) {
                    if (is_leftover[p]) {
                        leftovers.push_back(&fused_buffers[p]);
                    }
                }
                joint_result = fuse_multi(
                    graph, costs, leftovers, sub_solver_, workspaces[0], profile
                );
                {
                    BIOIMAGE_PROFILE_SCOPE(profile, "energy_eval");
                    joint_energy = energy(graph, costs, joint_result);
                }
                if (joint_energy < best_energy) {
                    best_energy = joint_energy;
                    best = &joint_result;
                }
            }

            // === Update current under the best-of safety net ===
            if (best_energy + kEnergyEps < current_energy) {
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
        BIOIMAGE_PROFILE_REPORT(profile);
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

    // Pairwise fuse that writes into a caller-provided output buffer (used by
    // the parallel stage-1 loop so workers don't allocate).
    static void fuse_pair_into(
        const UndirectedGraph &graph,
        const std::vector<double> &costs,
        const std::vector<std::uint64_t> &current,
        const std::vector<std::uint64_t> &proposal,
        const SolverBase *sub_solver,
        GreedyAdditiveWorkspace &workspace,
        std::vector<std::uint64_t> &output
    ) {
        const std::array<const std::vector<std::uint64_t> *, 2> proposals{
            &current, &proposal
        };
        std::vector<const std::vector<std::uint64_t> *> proposal_list(
            proposals.begin(), proposals.end()
        );
        ::bioimage_cpp::detail::NullProfiler null_profile;
        output = fuse_multi(graph, costs, proposal_list, sub_solver, workspace, null_profile);
    }

    // Multi-input fuse: contract by agreement over all N proposals, sum costs
    // onto the contracted edges, sub-solve, lift labels back. N=2 is the
    // pairwise case; N>2 is the stage-2 joint fuse on leftovers.
    template <class ProfilerT>
    static std::vector<std::uint64_t> fuse_multi(
        const UndirectedGraph &graph,
        const std::vector<double> &costs,
        const std::vector<const std::vector<std::uint64_t> *> &proposals,
        const SolverBase *sub_solver,
        GreedyAdditiveWorkspace &greedy_workspace,
        [[maybe_unused]] ProfilerT &profile
    ) {
        const auto number_of_nodes = static_cast<std::size_t>(graph.number_of_nodes());
        const auto n_proposals = proposals.size();

        std::vector<std::uint64_t> stacked(n_proposals * number_of_nodes);
        for (std::size_t p = 0; p < n_proposals; ++p) {
            std::copy(
                proposals[p]->begin(),
                proposals[p]->end(),
                stacked.begin() + static_cast<std::ptrdiff_t>(p * number_of_nodes)
            );
        }

        ::bioimage_cpp::graph::detail::AgreementContraction contraction;
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "agreement_contract");
            contraction = ::bioimage_cpp::graph::detail::contract_by_agreement(
                graph, stacked.data(), n_proposals, number_of_nodes
            );
        }

        const auto &contracted_graph = contraction.contracted_graph;
        const auto number_of_contracted_edges = contracted_graph.number_of_edges();

        std::vector<double> contracted_costs(
            static_cast<std::size_t>(number_of_contracted_edges), 0.0
        );
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "cost_aggregate");
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

        std::vector<std::uint64_t> sub_labels;
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "sub_solve");
            if (sub_solver == nullptr) {
                sub_labels = greedy_additive(
                    contracted_graph,
                    contracted_costs,
                    0.0,
                    -1.0,
                    false,
                    42,
                    1.0,
                    greedy_workspace
                );
            } else {
                Objective sub_objective(contracted_graph, std::move(contracted_costs));
                sub_labels = sub_solver->optimize(sub_objective);
            }
        }

        std::vector<std::uint64_t> result(number_of_nodes);
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "lift");
            for (std::uint64_t node = 0; node < graph.number_of_nodes(); ++node) {
                const auto root = contraction.root_of_node[static_cast<std::size_t>(node)];
                result[static_cast<std::size_t>(node)] = sub_labels[
                    static_cast<std::size_t>(root)
                ];
            }
        }
        return result;
    }

    std::vector<ProposalGeneratorBase *> proposal_generators_;
    const SolverBase *sub_solver_;
    std::size_t number_of_iterations_;
    std::size_t stop_if_no_improvement_;
    std::size_t number_of_threads_;
    std::size_t number_of_parallel_proposals_;
};

} // namespace bioimage_cpp::graph::multicut
