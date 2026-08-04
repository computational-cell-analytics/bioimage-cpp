#pragma once

#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/detail/threading.hxx"
#include "bioimage_cpp/graph/connected_components.hxx"
#include "bioimage_cpp/graph/multicut/greedy_additive.hxx"
#include "bioimage_cpp/graph/multicut/objective.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::multicut {

class DecomposerSolver final : public SolverBase {
public:
    DecomposerSolver(
        const CloneableSolverBase &sub_solver,
        const CloneableSolverBase *fallthrough_solver = nullptr,
        const std::size_t number_of_threads = 0
    )
        : sub_solver_(clone_solver(sub_solver)),
          fallthrough_solver_(
              fallthrough_solver == nullptr
                  ? nullptr
                  : clone_solver(*fallthrough_solver)
          ),
          number_of_threads_(number_of_threads) {
    }

    std::vector<std::uint64_t> optimize(Objective &objective) const override {
        BIOIMAGE_PROFILE_INIT(profile);

        if (
            fallthrough_solver_ == nullptr
            && dynamic_cast<const GreedyAdditiveSolver *>(
                sub_solver_.get()
            ) != nullptr
        ) {
            auto labels = sub_solver_->optimize(objective);
            BIOIMAGE_PROFILE_REPORT(profile);
            return labels;
        }

        const auto &graph = objective.graph();
        const auto &costs = objective.costs();
        const auto number_of_nodes =
            static_cast<std::size_t>(graph.number_of_nodes());
        const auto number_of_edges =
            static_cast<std::size_t>(graph.number_of_edges());

        std::vector<std::uint64_t> component_of_node;
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "component_find");
            std::vector<std::uint8_t> positive_edge(number_of_edges);
            for (std::size_t edge = 0; edge < number_of_edges; ++edge) {
                positive_edge[edge] = costs[edge] > 0.0 ? 1 : 0;
            }
            component_of_node = connected_components(
                graph,
                positive_edge.empty() ? nullptr : positive_edge.data()
            );
        }

        const auto number_of_components = component_of_node.empty()
            ? std::size_t{0}
            : static_cast<std::size_t>(
                *std::max_element(component_of_node.begin(), component_of_node.end())
            ) + 1;
        if (number_of_components <= 1) {
            const auto &solver = fallthrough_solver_ == nullptr
                ? sub_solver_
                : fallthrough_solver_;
            auto labels = solver->optimize(objective);
            BIOIMAGE_PROFILE_REPORT(profile);
            return labels;
        }

        std::vector<std::size_t> node_offsets(number_of_components + 1, 0);
        std::vector<std::size_t> edge_offsets(number_of_components + 1, 0);
        std::vector<std::uint64_t> grouped_nodes(number_of_nodes);
        std::vector<std::uint64_t> global_to_local(number_of_nodes);
        std::vector<std::uint64_t> grouped_edge_ids;

        {
            BIOIMAGE_PROFILE_SCOPE(profile, "component_group");
            for (const auto component : component_of_node) {
                ++node_offsets[static_cast<std::size_t>(component) + 1];
            }
            for (std::size_t component = 0;
                 component < number_of_components;
                 ++component) {
                node_offsets[component + 1] += node_offsets[component];
            }

            auto node_cursor = node_offsets;
            for (std::size_t node = 0; node < number_of_nodes; ++node) {
                const auto component =
                    static_cast<std::size_t>(component_of_node[node]);
                const auto position = node_cursor[component]++;
                grouped_nodes[position] = static_cast<std::uint64_t>(node);
                global_to_local[node] =
                    static_cast<std::uint64_t>(position - node_offsets[component]);
            }

            for (std::size_t edge = 0; edge < number_of_edges; ++edge) {
                const auto uv = graph.uv(static_cast<std::uint64_t>(edge));
                const auto u_component =
                    component_of_node[static_cast<std::size_t>(uv.first)];
                const auto v_component =
                    component_of_node[static_cast<std::size_t>(uv.second)];
                if (u_component == v_component) {
                    ++edge_offsets[static_cast<std::size_t>(u_component) + 1];
                }
            }
            for (std::size_t component = 0;
                 component < number_of_components;
                 ++component) {
                edge_offsets[component + 1] += edge_offsets[component];
            }

            grouped_edge_ids.resize(edge_offsets.back());
            auto edge_cursor = edge_offsets;
            for (std::size_t edge = 0; edge < number_of_edges; ++edge) {
                const auto uv = graph.uv(static_cast<std::uint64_t>(edge));
                const auto u_component =
                    component_of_node[static_cast<std::size_t>(uv.first)];
                const auto v_component =
                    component_of_node[static_cast<std::size_t>(uv.second)];
                if (u_component != v_component) {
                    continue;
                }
                grouped_edge_ids[
                    edge_cursor[static_cast<std::size_t>(u_component)]++
                ] = static_cast<std::uint64_t>(edge);
            }
        }

        struct Task {
            std::size_t component;
            std::size_t node_begin;
            std::size_t node_end;
            std::size_t edge_begin;
            std::size_t edge_end;
            std::unique_ptr<CloneableSolverBase> solver;
        };

        std::vector<Task> tasks;
        tasks.reserve(number_of_components);
        for (std::size_t component = 0;
             component < number_of_components;
             ++component) {
            const auto node_begin = node_offsets[component];
            const auto node_end = node_offsets[component + 1];
            if (node_end - node_begin <= 1) {
                continue;
            }
            tasks.push_back(
                Task{
                    component,
                    node_begin,
                    node_end,
                    edge_offsets[component],
                    edge_offsets[component + 1],
                    clone_solver(*sub_solver_)
                }
            );
        }
        std::stable_sort(
            tasks.begin(),
            tasks.end(),
            [](const Task &left, const Task &right) {
                const auto left_edges = left.edge_end - left.edge_begin;
                const auto right_edges = right.edge_end - right.edge_begin;
                if (left_edges != right_edges) {
                    return left_edges > right_edges;
                }
                return left.component < right.component;
            }
        );

        std::vector<std::vector<std::uint64_t>> component_labels(
            number_of_components
        );
        const auto solve_task = [&](Task &task) {
            std::vector<UndirectedGraph::Edge> local_edges;
            std::vector<double> local_costs;
            const auto task_edge_count = task.edge_end - task.edge_begin;
            local_edges.reserve(task_edge_count);
            local_costs.reserve(task_edge_count);

            for (std::size_t position = task.edge_begin;
                 position < task.edge_end;
                 ++position) {
                const auto edge = grouped_edge_ids[position];
                const auto uv = graph.uv(edge);
                auto local_u =
                    global_to_local[static_cast<std::size_t>(uv.first)];
                auto local_v =
                    global_to_local[static_cast<std::size_t>(uv.second)];
                if (local_v < local_u) {
                    std::swap(local_u, local_v);
                }
                local_edges.emplace_back(local_u, local_v);
                local_costs.push_back(costs[static_cast<std::size_t>(edge)]);
            }

            auto subgraph = UndirectedGraph::from_unique_edges(
                static_cast<std::uint64_t>(task.node_end - task.node_begin),
                std::move(local_edges),
                false
            );
            Objective sub_objective(subgraph, std::move(local_costs));
            component_labels[task.component] = dense_relabel(
                task.solver->optimize(sub_objective)
            );
        };

        {
            BIOIMAGE_PROFILE_SCOPE(profile, "parallel_solve");
            const auto effective_threads =
                ::bioimage_cpp::detail::normalize_thread_count(
                    number_of_threads_, tasks.size()
                );
            std::atomic<std::size_t> next_task{0};
            ::bioimage_cpp::detail::parallel_for_chunks(
                effective_threads,
                effective_threads,
                [&](
                    const std::size_t,
                    const std::size_t,
                    const std::size_t
                ) {
                    while (true) {
                        const auto task_index = next_task.fetch_add(
                            1, std::memory_order_relaxed
                        );
                        if (task_index >= tasks.size()) {
                            break;
                        }
                        solve_task(tasks[task_index]);
                    }
                }
            );
        }

        std::vector<std::uint64_t> global_labels(number_of_nodes);
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "label_assembly");
            std::uint64_t label_offset = 0;
            for (std::size_t component = 0;
                 component < number_of_components;
                 ++component) {
                const auto begin = node_offsets[component];
                const auto end = node_offsets[component + 1];
                if (end - begin == 1) {
                    global_labels[
                        static_cast<std::size_t>(grouped_nodes[begin])
                    ] = label_offset++;
                    continue;
                }

                const auto &labels = component_labels[component];
                if (labels.size() != end - begin || labels.empty()) {
                    throw std::runtime_error(
                        "component solver returned an invalid label count"
                    );
                }
                for (std::size_t local_node = 0;
                     local_node < labels.size();
                     ++local_node) {
                    const auto global_node = grouped_nodes[begin + local_node];
                    global_labels[static_cast<std::size_t>(global_node)] =
                        labels[local_node] + label_offset;
                }
                label_offset +=
                    *std::max_element(labels.begin(), labels.end()) + 1;
            }
        }

        objective.set_labels(std::move(global_labels));
        BIOIMAGE_PROFILE_REPORT(profile);
        return objective.labels();
    }

private:
    std::unique_ptr<CloneableSolverBase> sub_solver_;
    std::unique_ptr<CloneableSolverBase> fallthrough_solver_;
    std::size_t number_of_threads_;
};

} // namespace bioimage_cpp::graph::multicut
