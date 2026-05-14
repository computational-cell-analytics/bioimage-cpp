#pragma once

#include "bioimage_cpp/graph/multicut/detail.hxx"
#include "bioimage_cpp/graph/multicut/objective.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <vector>

namespace bioimage_cpp::graph::multicut {

inline std::vector<std::uint64_t> greedy_fixation(
    const UndirectedGraph &graph,
    const std::vector<double> &costs,
    const double weight_stop,
    const double node_num_stop
) {
    validate_costs(graph, costs);
    detail::DynamicGraph dynamic_graph(graph);
    bioimage_cpp::detail::UnionFind sets(static_cast<std::size_t>(graph.number_of_nodes()));
    std::priority_queue<detail::QueueEdge> queue;
    detail::initialize_dynamic_graph(graph, costs, dynamic_graph, queue, true);

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
        if (std::abs(weight) <= weight_stop) {
            break;
        }
        if (node_num_stop > 0.0 && dynamic_graph.alive_count <= detail::stop_node_count(graph, node_num_stop)) {
            break;
        }
        if (dynamic_graph.has_constraint(u, v)) {
            continue;
        }
        if (weight > 0.0) {
            detail::merge_dynamic_nodes(dynamic_graph, sets, queue, u, v, true);
        } else if (weight < 0.0) {
            dynamic_graph.add_constraint(u, v);
        }
    }
    return detail::labels_from_sets(sets, graph);
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
