#pragma once

#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::multicut {

inline void validate_costs(const UndirectedGraph &graph, const std::vector<double> &costs) {
    if (costs.size() != static_cast<std::size_t>(graph.number_of_edges())) {
        throw std::invalid_argument("edge costs length must match graph number_of_edges");
    }
}

inline void validate_labels(const UndirectedGraph &graph, const std::vector<std::uint64_t> &labels) {
    if (labels.size() != static_cast<std::size_t>(graph.number_of_nodes())) {
        throw std::invalid_argument("labels length must match graph number_of_nodes");
    }
}

inline std::vector<std::uint64_t> singleton_labels(const UndirectedGraph &graph) {
    std::vector<std::uint64_t> labels(static_cast<std::size_t>(graph.number_of_nodes()));
    std::iota(labels.begin(), labels.end(), std::uint64_t{0});
    return labels;
}

inline std::vector<std::uint64_t> dense_relabel(const std::vector<std::uint64_t> &labels) {
    std::unordered_map<std::uint64_t, std::uint64_t> relabeling;
    std::vector<std::uint64_t> result(labels.size());
    for (std::size_t index = 0; index < labels.size(); ++index) {
        auto found = relabeling.find(labels[index]);
        if (found == relabeling.end()) {
            found = relabeling.emplace(labels[index], static_cast<std::uint64_t>(relabeling.size())).first;
        }
        result[index] = found->second;
    }
    return result;
}

inline double energy(
    const UndirectedGraph &graph,
    const std::vector<double> &costs,
    const std::vector<std::uint64_t> &labels
) {
    validate_costs(graph, costs);
    validate_labels(graph, labels);
    double result = 0.0;
    for (std::uint64_t edge = 0; edge < graph.number_of_edges(); ++edge) {
        const auto uv = graph.uv(edge);
        if (labels[static_cast<std::size_t>(uv.first)] != labels[static_cast<std::size_t>(uv.second)]) {
            result += costs[static_cast<std::size_t>(edge)];
        }
    }
    return result;
}

class Objective {
public:
    Objective(const UndirectedGraph &graph, std::vector<double> costs)
        : graph_(graph),
          costs_(std::move(costs)),
          labels_(singleton_labels(graph)) {
        validate_costs(graph_, costs_);
    }

    Objective(
        const UndirectedGraph &graph,
        std::vector<double> costs,
        std::vector<std::uint64_t> labels
    )
        : graph_(graph),
          costs_(std::move(costs)),
          labels_(std::move(labels)) {
        validate_costs(graph_, costs_);
        validate_labels(graph_, labels_);
    }

    [[nodiscard]] const UndirectedGraph &graph() const {
        return graph_;
    }

    [[nodiscard]] const std::vector<double> &costs() const {
        return costs_;
    }

    [[nodiscard]] const std::vector<std::uint64_t> &labels() const {
        return labels_;
    }

    void set_labels(std::vector<std::uint64_t> labels) {
        validate_labels(graph_, labels);
        labels_ = dense_relabel(labels);
    }

    void reset_labels() {
        labels_ = singleton_labels(graph_);
    }

    [[nodiscard]] double eval() const {
        return energy(graph_, costs_, labels_);
    }

    [[nodiscard]] double eval(const std::vector<std::uint64_t> &labels) const {
        return energy(graph_, costs_, labels);
    }

private:
    const UndirectedGraph &graph_;
    std::vector<double> costs_;
    std::vector<std::uint64_t> labels_;
};

class SolverBase {
public:
    virtual ~SolverBase() = default;
    virtual std::vector<std::uint64_t> optimize(Objective &objective) const = 0;
};

} // namespace bioimage_cpp::graph::multicut
