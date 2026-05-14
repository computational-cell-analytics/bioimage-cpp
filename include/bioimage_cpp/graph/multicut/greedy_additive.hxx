#pragma once

#include "bioimage_cpp/graph/multicut/detail.hxx"
#include "bioimage_cpp/graph/multicut/objective.hxx"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bioimage_cpp::graph::multicut {

inline std::vector<std::uint64_t> greedy_additive(
    const UndirectedGraph &graph,
    const std::vector<double> &costs,
    const double weight_stop,
    const double node_num_stop,
    const bool add_noise,
    const int seed,
    const double sigma
) {
    validate_costs(graph, costs);
    detail::DynamicGraph dynamic_graph(graph);
    bioimage_cpp::detail::UnionFind sets(static_cast<std::size_t>(graph.number_of_nodes()));
    detail::EdgeHeap heap;
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
