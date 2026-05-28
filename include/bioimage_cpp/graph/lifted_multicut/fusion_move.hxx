#pragma once

#include "bioimage_cpp/detail/edge_hash.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/detail/threading.hxx"
#include "bioimage_cpp/graph/detail/fusion_contract.hxx"
#include "bioimage_cpp/graph/lifted_multicut/greedy_additive.hxx"
#include "bioimage_cpp/graph/lifted_multicut/objective.hxx"
#include "bioimage_cpp/graph/proposal_generator.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::lifted_multicut {

// Fusion-move solver for lifted multicut. Mirrors the multicut driver:
// stage-1 parallel proposal generation + parallel pairwise fuses, stage-2
// sequential joint multi-fuse on leftover candidates, best-of safety net,
// optional warm-start from the trivial singleton labeling.
//
// The proposal generators emit node labelings over the *base* graph. Agreement
// contraction is also computed on the base graph only (lifted edges are not
// candidate contraction edges in lifted multicut). Both base and lifted
// weights are then aggregated onto the contracted subproblem:
//   - A surviving base edge contributes its weight to the contracted base
//     edge it maps onto.
//   - A surviving lifted edge whose endpoints map to roots (ru, rv) that
//     already share a contracted base edge contributes its weight to that
//     base edge.
//   - Otherwise the lifted edge becomes a new contracted lifted edge in the
//     subproblem, with duplicates accumulated.
//
// The sub-solver is then a lifted-multicut solver over (contracted_base,
// contracted_lifted, contracted_weights). Default: lifted greedy-additive
// with a per-thread workspace reused across fuses.
class FusionMoveSolver final : public SolverBase {
public:
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
        const auto &base_graph = objective.graph();
        const auto &lifted_graph = objective.lifted_graph();
        const auto &weights = objective.weights();
        const auto n_base_edges = objective.number_of_base_edges();
        const auto number_of_nodes = base_graph.number_of_nodes();

        std::vector<std::uint64_t> current = objective.labels();
        if (number_of_nodes == 0 || lifted_graph.number_of_edges() == 0) {
            objective.set_labels(current);
            return objective.labels();
        }

        // Proposal generators read base_graph.node_adjacency() concurrently in the
        // stage-1 parallel region (the greedy-additive generator does, via
        // DynamicGraph::reset). The lazy CSR rebuild is not thread-safe, and the
        // warm-start below freezes the *lifted* graph, not the base graph, so freeze
        // the base graph on this thread before fan-out. See UndirectedGraph
        // thread-safety. (The lifted graph is only read by edge iteration here.)
        base_graph.freeze();

        const auto effective_threads = ::bioimage_cpp::detail::normalize_thread_count(
            number_of_threads_, number_of_parallel_proposals_
        );
        std::vector<GreedyAdditiveWorkspace> workspaces(effective_threads);

        if (is_singleton_labeling(current)) {
            BIOIMAGE_PROFILE_SCOPE(profile, "warm_start");
            current = greedy_additive(
                lifted_graph, weights, n_base_edges,
                0.0, -1.0, false, 42, 1.0, workspaces[0]
            );
        }

        double current_energy;
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "energy_eval");
            current_energy = energy(lifted_graph, weights, current);
        }

        const std::size_t P = number_of_parallel_proposals_;
        std::vector<std::vector<std::uint64_t>> proposal_buffers(P);
        std::vector<std::vector<std::uint64_t>> fused_buffers(P);
        std::vector<double> proposal_energies(P);
        std::vector<double> fused_energies(P);
        std::vector<unsigned char> is_leftover(P);

        constexpr double kEnergyEps = 1e-7;

        std::size_t iterations_without_improvement = 0;

        for (std::size_t iteration = 0; iteration < number_of_iterations_; ++iteration) {
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
                            proposal_energies[p] = energy(
                                lifted_graph, weights, proposal_buffers[p]
                            );

                            fuse_pair_into(
                                base_graph,
                                lifted_graph,
                                weights,
                                n_base_edges,
                                current_snapshot,
                                proposal_buffers[p],
                                sub_solver_,
                                workspace,
                                fused_buffers[p]
                            );
                            fused_energies[p] = energy(
                                lifted_graph, weights, fused_buffers[p]
                            );
                        }
                    }
                );
            }

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
                    base_graph,
                    lifted_graph,
                    weights,
                    n_base_edges,
                    leftovers,
                    sub_solver_,
                    workspaces[0],
                    profile
                );
                {
                    BIOIMAGE_PROFILE_SCOPE(profile, "energy_eval");
                    joint_energy = energy(lifted_graph, weights, joint_result);
                }
                if (joint_energy < best_energy) {
                    best_energy = joint_energy;
                    best = &joint_result;
                }
            }

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

    static void fuse_pair_into(
        const UndirectedGraph &base_graph,
        const UndirectedGraph &lifted_graph,
        const std::vector<double> &weights,
        const std::uint64_t n_base_edges,
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
        output = fuse_multi(
            base_graph, lifted_graph, weights, n_base_edges,
            proposal_list, sub_solver, workspace, null_profile
        );
    }

    // Build the contracted lifted-multicut subproblem from N proposals,
    // solve it, lift labels back. N=2 is the pairwise case; N>2 is the
    // stage-2 joint fuse on leftovers.
    template <class ProfilerT>
    static std::vector<std::uint64_t> fuse_multi(
        const UndirectedGraph &base_graph,
        const UndirectedGraph &lifted_graph,
        const std::vector<double> &weights,
        const std::uint64_t n_base_edges,
        const std::vector<const std::vector<std::uint64_t> *> &proposals,
        const SolverBase *sub_solver,
        GreedyAdditiveWorkspace &greedy_workspace,
        [[maybe_unused]] ProfilerT &profile
    ) {
        const auto number_of_nodes = static_cast<std::size_t>(base_graph.number_of_nodes());
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
                base_graph, stacked.data(), n_proposals, number_of_nodes
            );
        }

        const auto &contracted_base = contraction.contracted_graph;
        const auto n_contracted_base =
            static_cast<std::size_t>(contracted_base.number_of_edges());

        // Aggregate base costs onto contracted base edges.
        std::vector<double> contracted_base_weights(n_contracted_base, 0.0);
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "cost_aggregate_base");
            for (std::uint64_t edge = 0; edge < n_base_edges; ++edge) {
                const auto target = contraction.contracted_edge_of_original[
                    static_cast<std::size_t>(edge)
                ];
                if (target < 0) {
                    continue;
                }
                contracted_base_weights[static_cast<std::size_t>(target)] +=
                    weights[static_cast<std::size_t>(edge)];
            }
        }

        using ::bioimage_cpp::detail::Edge;
        using ::bioimage_cpp::detail::EdgeHash;
        using ::bioimage_cpp::detail::edge_key;

        // Look up which (ru, rv) pairs already exist as contracted base edges
        // (so lifted edges between those roots fold into that base edge rather
        // than introducing a new contracted lifted edge). The contracted base
        // graph was built without `edge_lookup_`, so build our own map.
        std::unordered_map<Edge, std::size_t, EdgeHash> base_lookup;
        base_lookup.reserve(n_contracted_base);
        for (std::uint64_t e = 0; e < contracted_base.number_of_edges(); ++e) {
            const auto uv = contracted_base.uv(e);
            base_lookup.emplace(uv, static_cast<std::size_t>(e));
        }

        // Walk lifted edges. Three outcomes per edge:
        //   - endpoints in same root: dropped (never cut).
        //   - endpoints share a contracted base edge: accumulate onto base.
        //   - otherwise: emit/accumulate a new contracted lifted edge.
        std::vector<std::pair<std::uint64_t, std::uint64_t>> new_lifted_uvs;
        std::vector<double> new_lifted_weights;
        std::unordered_map<Edge, std::size_t, EdgeHash> lifted_lookup;
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "cost_aggregate_lifted");
            const auto n_total_edges = lifted_graph.number_of_edges();
            for (std::uint64_t edge = n_base_edges; edge < n_total_edges; ++edge) {
                const auto uv = lifted_graph.uv(edge);
                const auto ru = contraction.root_of_node[static_cast<std::size_t>(uv.first)];
                const auto rv = contraction.root_of_node[static_cast<std::size_t>(uv.second)];
                if (ru == rv) {
                    continue;
                }
                const auto key = edge_key(ru, rv);
                const auto base_it = base_lookup.find(key);
                if (base_it != base_lookup.end()) {
                    contracted_base_weights[base_it->second] +=
                        weights[static_cast<std::size_t>(edge)];
                    continue;
                }
                const auto lifted_it = lifted_lookup.find(key);
                if (lifted_it != lifted_lookup.end()) {
                    new_lifted_weights[lifted_it->second] +=
                        weights[static_cast<std::size_t>(edge)];
                } else {
                    const auto index = new_lifted_uvs.size();
                    new_lifted_uvs.emplace_back(key.first, key.second);
                    new_lifted_weights.push_back(
                        weights[static_cast<std::size_t>(edge)]
                    );
                    lifted_lookup.emplace(key, index);
                }
            }
        }

        // Degenerate case: nothing left to optimize — all proposals already
        // agreed everywhere that mattered. The agreement labeling itself is
        // the answer.
        if (n_contracted_base == 0 && new_lifted_uvs.empty()) {
            std::vector<std::uint64_t> result(number_of_nodes);
            for (std::uint64_t node = 0; node < base_graph.number_of_nodes(); ++node) {
                result[static_cast<std::size_t>(node)] = contraction.root_of_node[
                    static_cast<std::size_t>(node)
                ];
            }
            return result;
        }

        // Build the contracted Objective. The Objective constructor
        // re-inserts base edges into its internal lifted graph and appends
        // the new lifted edges; weights are stored as
        // [base_costs..., lifted_costs...] in the same order.
        std::vector<std::uint64_t> sub_labels;
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "sub_solve");
            Objective sub_objective(
                contracted_base,
                std::move(contracted_base_weights),
                new_lifted_uvs,
                new_lifted_weights,
                false
            );
            if (sub_solver == nullptr) {
                sub_labels = greedy_additive(
                    sub_objective.lifted_graph(),
                    sub_objective.weights(),
                    sub_objective.number_of_base_edges(),
                    0.0,
                    -1.0,
                    false,
                    42,
                    1.0,
                    greedy_workspace
                );
            } else {
                sub_labels = sub_solver->optimize(sub_objective);
            }
        }

        std::vector<std::uint64_t> result(number_of_nodes);
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "lift");
            for (std::uint64_t node = 0; node < base_graph.number_of_nodes(); ++node) {
                const auto root = contraction.root_of_node[
                    static_cast<std::size_t>(node)
                ];
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

} // namespace bioimage_cpp::graph::lifted_multicut
