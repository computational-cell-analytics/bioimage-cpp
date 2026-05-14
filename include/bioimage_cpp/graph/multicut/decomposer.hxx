#pragma once

#include "bioimage_cpp/graph/connected_components.hxx"
#include "bioimage_cpp/graph/multicut/objective.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace bioimage_cpp::graph::multicut {

class DecomposerSolver final : public SolverBase {
public:
    explicit DecomposerSolver(const SolverBase &sub_solver, const SolverBase *fallthrough_solver = nullptr)
        : sub_solver_(sub_solver),
          fallthrough_solver_(fallthrough_solver) {
    }

    std::vector<std::uint64_t> optimize(Objective &objective) const override {
        std::vector<std::uint8_t> positive_edges(static_cast<std::size_t>(objective.graph().number_of_edges()), 0);
        for (std::size_t edge = 0; edge < objective.costs().size(); ++edge) {
            positive_edges[edge] = objective.costs()[edge] > 0.0 ? 1 : 0;
        }

        const auto components = connected_components(objective.graph(), positive_edges.data());
        const auto number_of_components = components.empty()
            ? std::uint64_t{0}
            : (*std::max_element(components.begin(), components.end()) + 1);
        if (number_of_components <= 1) {
            const auto &solver = fallthrough_solver_ == nullptr ? sub_solver_ : *fallthrough_solver_;
            auto labels = solver.optimize(objective);
            objective.set_labels(labels);
            return labels;
        }

        std::vector<std::uint64_t> labels(static_cast<std::size_t>(objective.graph().number_of_nodes()));
        std::uint64_t label_offset = 0;
        for (std::uint64_t component = 0; component < number_of_components; ++component) {
            std::vector<std::uint64_t> nodes;
            for (std::uint64_t node = 0; node < objective.graph().number_of_nodes(); ++node) {
                if (components[static_cast<std::size_t>(node)] == component) {
                    nodes.push_back(node);
                }
            }
            if (nodes.size() == 1) {
                labels[static_cast<std::size_t>(nodes[0])] = label_offset++;
                continue;
            }

            std::vector<std::uint64_t> local_ids(static_cast<std::size_t>(objective.graph().number_of_nodes()), std::numeric_limits<std::uint64_t>::max());
            for (std::uint64_t local_node = 0; local_node < nodes.size(); ++local_node) {
                local_ids[static_cast<std::size_t>(nodes[static_cast<std::size_t>(local_node)])] = local_node;
            }

            const auto extracted = objective.graph().extract_subgraph_from_nodes(nodes);
            UndirectedGraph subgraph(static_cast<std::uint64_t>(nodes.size()), static_cast<std::uint64_t>(extracted.first.size()));
            std::vector<double> sub_costs;
            sub_costs.reserve(extracted.first.size());
            for (const auto edge_id : extracted.first) {
                const auto uv = objective.graph().uv(edge_id);
                subgraph.insert_edge(
                    local_ids[static_cast<std::size_t>(uv.first)],
                    local_ids[static_cast<std::size_t>(uv.second)]
                );
                sub_costs.push_back(objective.costs()[static_cast<std::size_t>(edge_id)]);
            }

            Objective sub_objective(subgraph, std::move(sub_costs));
            auto sub_labels = dense_relabel(sub_solver_.optimize(sub_objective));
            for (std::size_t local_node = 0; local_node < nodes.size(); ++local_node) {
                labels[static_cast<std::size_t>(nodes[local_node])] = sub_labels[local_node] + label_offset;
            }
            label_offset += *std::max_element(sub_labels.begin(), sub_labels.end()) + 1;
        }

        objective.set_labels(labels);
        return objective.labels();
    }

private:
    const SolverBase &sub_solver_;
    const SolverBase *fallthrough_solver_;
};

} // namespace bioimage_cpp::graph::multicut
