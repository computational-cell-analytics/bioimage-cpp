#pragma once

#include "bioimage_cpp/detail/relabel.hxx"
#include "bioimage_cpp/graph/breadth_first_search.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::lifted_multicut {

inline void validate_weights(
    const UndirectedGraph &lifted_graph,
    const std::vector<double> &weights
) {
    if (weights.size() != static_cast<std::size_t>(lifted_graph.number_of_edges())) {
        throw std::invalid_argument(
            "edge weights length must match lifted graph number_of_edges"
        );
    }
}

inline void validate_labels(
    const UndirectedGraph &graph,
    const std::vector<std::uint64_t> &labels
) {
    if (labels.size() != static_cast<std::size_t>(graph.number_of_nodes())) {
        throw std::invalid_argument("labels length must match graph number_of_nodes");
    }
}

inline std::vector<std::uint64_t> singleton_labels(const UndirectedGraph &graph) {
    std::vector<std::uint64_t> labels(static_cast<std::size_t>(graph.number_of_nodes()));
    std::iota(labels.begin(), labels.end(), std::uint64_t{0});
    return labels;
}

using bioimage_cpp::detail::dense_relabel;

// Energy of a node labeling on a lifted graph: sum of weights across cut
// edges. The lifted graph's edge set is base ∪ lifted, so both contribute.
inline double energy(
    const UndirectedGraph &lifted_graph,
    const std::vector<double> &weights,
    const std::vector<std::uint64_t> &labels
) {
    validate_weights(lifted_graph, weights);
    validate_labels(lifted_graph, labels);
    double result = 0.0;
    for (std::uint64_t edge = 0; edge < lifted_graph.number_of_edges(); ++edge) {
        const auto uv = lifted_graph.uv(edge);
        if (labels[static_cast<std::size_t>(uv.first)]
            != labels[static_cast<std::size_t>(uv.second)]) {
            result += weights[static_cast<std::size_t>(edge)];
        }
    }
    return result;
}

// Build a lifted graph as a superset of the base graph. The first
// `base_graph.number_of_edges()` edges of the returned graph are the base
// edges in the same order; subsequent edges are the lifted edges. Duplicate
// (u, v) pairs across base and lifted are not inserted twice — the lifted
// "edge" simply contributes its weight to the corresponding base edge.
//
// Returns:
//   - lifted_graph: the constructed UndirectedGraph.
//   - weights: vector of length `lifted_graph.number_of_edges()` with the base
//     weights followed by lifted contributions. Weights for (u, v) that
//     coincide with a base edge are added to the base weight when
//     `overwrite_existing == false`, or replace it when `true`.
struct LiftedGraphBuild {
    UndirectedGraph lifted_graph;
    std::vector<double> weights;
};

inline LiftedGraphBuild build_lifted_graph(
    const UndirectedGraph &base_graph,
    const std::vector<double> &base_weights,
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> &lifted_uvs,
    const std::vector<double> &lifted_weights,
    const bool overwrite_existing = false
) {
    validate_weights(base_graph, base_weights);
    if (lifted_uvs.size() != lifted_weights.size()) {
        throw std::invalid_argument(
            "lifted_uvs and lifted_weights must have the same length"
        );
    }

    UndirectedGraph lifted_graph(
        base_graph.number_of_nodes(),
        base_graph.number_of_edges() + static_cast<std::uint64_t>(lifted_uvs.size())
    );
    for (std::uint64_t edge = 0; edge < base_graph.number_of_edges(); ++edge) {
        const auto uv = base_graph.uv(edge);
        lifted_graph.insert_edge(uv.first, uv.second);
    }
    std::vector<double> weights = base_weights;
    weights.resize(static_cast<std::size_t>(lifted_graph.number_of_edges()), 0.0);

    for (std::size_t i = 0; i < lifted_uvs.size(); ++i) {
        const auto u = lifted_uvs[i].first;
        const auto v = lifted_uvs[i].second;
        const auto pre = lifted_graph.number_of_edges();
        const auto edge = lifted_graph.insert_edge(u, v);
        const auto weight = lifted_weights[i];
        if (lifted_graph.number_of_edges() > pre) {
            weights.push_back(weight);
        } else {
            if (overwrite_existing) {
                weights[static_cast<std::size_t>(edge)] = weight;
            } else {
                weights[static_cast<std::size_t>(edge)] += weight;
            }
        }
    }
    return LiftedGraphBuild{std::move(lifted_graph), std::move(weights)};
}

// Lifted multicut objective. Stores a reference to the base graph and owns the
// lifted graph (= base ∪ lifted edges) and the per-lifted-edge weights. The
// node-labeling lives over the base graph's node set.
//
// Invariants:
//   - lifted_graph().uv(e) == base_graph.uv(e) for every base edge id e.
//   - weights().size() == lifted_graph().number_of_edges().
class Objective {
public:
    Objective(
        const UndirectedGraph &base_graph,
        std::vector<double> base_weights
    )
        : base_graph_(base_graph),
          lifted_graph_(build_base_only_(base_graph, base_weights)),
          weights_(std::move(base_weights)),
          labels_(singleton_labels(base_graph)),
          n_base_edges_(base_graph.number_of_edges()) {
        validate_weights(lifted_graph_, weights_);
    }

    Objective(
        const UndirectedGraph &base_graph,
        std::vector<double> base_weights,
        const std::vector<std::pair<std::uint64_t, std::uint64_t>> &lifted_uvs,
        const std::vector<double> &lifted_weights,
        const bool overwrite_existing = false
    )
        : base_graph_(base_graph),
          lifted_graph_(),
          weights_(),
          labels_(singleton_labels(base_graph)),
          n_base_edges_(base_graph.number_of_edges()) {
        auto build = build_lifted_graph(
            base_graph, base_weights, lifted_uvs, lifted_weights, overwrite_existing
        );
        lifted_graph_ = std::move(build.lifted_graph);
        weights_ = std::move(build.weights);
    }

    [[nodiscard]] const UndirectedGraph &graph() const { return base_graph_; }
    [[nodiscard]] const UndirectedGraph &lifted_graph() const { return lifted_graph_; }
    [[nodiscard]] const std::vector<double> &weights() const { return weights_; }
    [[nodiscard]] std::uint64_t number_of_base_edges() const { return n_base_edges_; }
    [[nodiscard]] std::uint64_t number_of_lifted_edges() const {
        return lifted_graph_.number_of_edges() - n_base_edges_;
    }
    [[nodiscard]] const std::vector<std::uint64_t> &labels() const { return labels_; }

    // Insert or update a lifted edge. Returns the edge id in the lifted graph
    // and whether the edge is newly inserted. If the (u, v) pair already
    // exists (as a base edge or as a previously inserted lifted edge), the
    // weight is accumulated unless `overwrite` is true. Inserting a brand-new
    // edge appends it after all existing edges, so base edges remain at the
    // head of the lifted graph.
    std::pair<std::uint64_t, bool> set_cost(
        const std::uint64_t u,
        const std::uint64_t v,
        const double weight,
        const bool overwrite = false
    ) {
        const auto pre = lifted_graph_.number_of_edges();
        const auto edge = lifted_graph_.insert_edge(u, v);
        if (lifted_graph_.number_of_edges() > pre) {
            weights_.push_back(weight);
            return {edge, true};
        }
        if (overwrite) {
            weights_[static_cast<std::size_t>(edge)] = weight;
        } else {
            weights_[static_cast<std::size_t>(edge)] += weight;
        }
        return {edge, false};
    }

    // Insert lifted edges between every pair of base-graph nodes within
    // `max_distance` hops (excluding the source). Each new edge is created
    // with weight 0; callers are expected to call `set_cost` afterwards or
    // pre-populate weights another way. Existing edges are left untouched.
    void insert_lifted_edges_bfs(const std::uint64_t max_distance) {
        BfsWorkspace workspace;
        for (std::uint64_t source = 0; source < base_graph_.number_of_nodes(); ++source) {
            const auto reached = breadth_first_search(
                base_graph_, source, max_distance, false, workspace
            );
            for (const auto entry : reached) {
                if (entry.node > source) {
                    set_cost(source, entry.node, 0.0, false);
                }
            }
        }
    }

    void set_labels(std::vector<std::uint64_t> labels) {
        validate_labels(base_graph_, labels);
        labels_ = dense_relabel(labels);
    }

    void reset_labels() {
        labels_ = singleton_labels(base_graph_);
    }

    [[nodiscard]] double eval() const {
        return energy(lifted_graph_, weights_, labels_);
    }

    [[nodiscard]] double eval(const std::vector<std::uint64_t> &labels) const {
        return energy(lifted_graph_, weights_, labels);
    }

private:
    static UndirectedGraph build_base_only_(
        const UndirectedGraph &base_graph,
        const std::vector<double> &base_weights
    ) {
        validate_weights(base_graph, base_weights);
        UndirectedGraph lifted_graph(
            base_graph.number_of_nodes(), base_graph.number_of_edges()
        );
        for (std::uint64_t edge = 0; edge < base_graph.number_of_edges(); ++edge) {
            const auto uv = base_graph.uv(edge);
            lifted_graph.insert_edge(uv.first, uv.second);
        }
        return lifted_graph;
    }

    const UndirectedGraph &base_graph_;
    UndirectedGraph lifted_graph_;
    std::vector<double> weights_;
    std::vector<std::uint64_t> labels_;
    std::uint64_t n_base_edges_;
};

class SolverBase {
public:
    virtual ~SolverBase() = default;
    virtual std::vector<std::uint64_t> optimize(Objective &objective) const = 0;
};

} // namespace bioimage_cpp::graph::lifted_multicut
