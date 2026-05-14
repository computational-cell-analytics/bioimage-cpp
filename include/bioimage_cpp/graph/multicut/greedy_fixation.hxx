#pragma once

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
    validate_costs(graph, costs);
    detail::DynamicGraph dynamic_graph(graph);
    bioimage_cpp::detail::UnionFind sets(static_cast<std::size_t>(graph.number_of_nodes()));
    detail::EdgeHeap heap;
    detail::initialize_dynamic_graph(graph, costs, dynamic_graph, heap, true);

    while (!heap.empty() && dynamic_graph.alive_count > 1) {
        const auto top = heap.top();
        // Priority is |weight|, so this also handles weight == 0 (stop).
        if (top.priority <= weight_stop) {
            break;
        }
        if (node_num_stop > 0.0
            && dynamic_graph.alive_count <= detail::stop_node_count(graph, node_num_stop)) {
            break;
        }
        const auto edge_id = top.key;
        const auto &edge = dynamic_graph.edges[edge_id];
        if (edge.weight > 0.0) {
            detail::merge_dynamic_nodes(dynamic_graph, sets, heap, edge.u, edge.v, true);
        } else {
            // weight < 0: forbid merging through this edge. merge_dynamic_nodes
            // propagates the flag onto any merged successor edges.
            heap.pop();
            dynamic_graph.edges[edge_id].is_constraint = 1;
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
