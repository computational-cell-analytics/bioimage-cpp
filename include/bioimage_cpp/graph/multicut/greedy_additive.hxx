#pragma once

#include "bioimage_cpp/graph/multicut/detail.hxx"
#include "bioimage_cpp/graph/multicut/objective.hxx"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bioimage_cpp::graph::multicut {

// Reusable scratch state for `greedy_additive`. Construct once and call
// `greedy_additive(..., workspace)` repeatedly to avoid per-call allocation
// of the DynamicGraph, UnionFind, and EdgeHeap. Capacities only grow; the
// internal vectors are reset (not freed) between calls.
struct GreedyAdditiveWorkspace {
    detail::DynamicGraph dynamic_graph;
    bioimage_cpp::detail::UnionFind union_find{0};
    detail::EdgeHeap heap;

    void reset(const UndirectedGraph &graph) {
        dynamic_graph.reset(graph);
        union_find.reset(static_cast<std::size_t>(graph.number_of_nodes()));
        heap.reset_capacity(static_cast<std::size_t>(graph.number_of_edges()));
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
    validate_costs(graph, costs);
    workspace.reset(graph);
    auto &dynamic_graph = workspace.dynamic_graph;
    auto &sets = workspace.union_find;
    auto &heap = workspace.heap;
    detail::initialize_dynamic_graph(graph, costs, dynamic_graph, heap, false, add_noise, seed, sigma);

    while (!heap.empty() && dynamic_graph.alive_count > 1) {
        const auto top = heap.top();
        if (top.priority <= weight_stop) {
            break;
        }
        if (node_num_stop > 0.0
            && dynamic_graph.alive_count <= detail::stop_node_count(graph, node_num_stop)) {
            break;
        }
        const auto &edge = dynamic_graph.edges[top.key];
        detail::merge_dynamic_nodes(dynamic_graph, sets, heap, edge.u, edge.v, false);
    }
    return detail::labels_from_sets(sets, graph);
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

private:
    double weight_stop_;
    double node_num_stop_;
    bool add_noise_;
    int seed_;
    double sigma_;
};

} // namespace bioimage_cpp::graph::multicut
