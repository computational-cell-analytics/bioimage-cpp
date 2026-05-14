#pragma once

#include "bioimage_cpp/graph/multicut/detail.hxx"
#include "bioimage_cpp/graph/multicut/objective.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <unordered_map>
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
    UnionFind sets(static_cast<std::size_t>(graph.number_of_nodes()));
    std::priority_queue<detail::QueueEdge> queue;
    detail::initialize_dynamic_graph(graph, costs, dynamic_graph, queue, false, add_noise, seed, sigma);

    while (!queue.empty() && dynamic_graph.alive_count > 1) {
        auto edge = queue.top();
        queue.pop();
        const auto u = std::min(edge.u, edge.v);
        const auto v = std::max(edge.u, edge.v);
        const auto *dynamic_edge = dynamic_graph.edge(u, v);
        if (dynamic_edge == nullptr || edge.edition < dynamic_edge->edition) {
            continue;
        }
        const auto weight = dynamic_edge->weight;
        if (weight <= weight_stop) {
            break;
        }
        if (node_num_stop > 0.0 && dynamic_graph.alive_count <= detail::stop_node_count(graph, node_num_stop)) {
            break;
        }
        detail::merge_dynamic_nodes(dynamic_graph, sets, queue, u, v, false);
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
