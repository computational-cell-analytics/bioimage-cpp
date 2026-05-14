#pragma once

#include "bioimage_cpp/detail/edge_hash.hxx"
#include "bioimage_cpp/detail/union_find.hxx"
#include "bioimage_cpp/graph/multicut/objective.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::multicut {

namespace detail_kl {

struct ClusterPair {
    std::uint64_t a;
    std::uint64_t b;
    double cut_weight;
};

inline std::vector<ClusterPair> compute_cluster_pairs(
    const UndirectedGraph &graph,
    const std::vector<double> &costs,
    const std::vector<std::uint64_t> &labels
) {
    using bioimage_cpp::detail::Edge;
    using bioimage_cpp::detail::EdgeHash;
    using bioimage_cpp::detail::edge_key;

    std::unordered_map<Edge, double, EdgeHash> table;
    table.reserve(static_cast<std::size_t>(graph.number_of_edges()));
    for (std::uint64_t edge = 0; edge < graph.number_of_edges(); ++edge) {
        const auto uv = graph.uv(edge);
        const auto la = labels[static_cast<std::size_t>(uv.first)];
        const auto lb = labels[static_cast<std::size_t>(uv.second)];
        if (la == lb) {
            continue;
        }
        table[edge_key(la, lb)] += costs[static_cast<std::size_t>(edge)];
    }

    std::vector<ClusterPair> result;
    result.reserve(table.size());
    for (const auto &entry : table) {
        result.push_back({entry.first.first, entry.first.second, entry.second});
    }
    std::sort(result.begin(), result.end(), [](const ClusterPair &lhs, const ClusterPair &rhs) {
        return std::tie(lhs.a, lhs.b) < std::tie(rhs.a, rhs.b);
    });
    return result;
}

inline std::vector<std::vector<std::uint64_t>> build_cluster_to_nodes(
    const std::vector<std::uint64_t> &labels,
    const std::uint64_t number_of_clusters
) {
    std::vector<std::vector<std::uint64_t>> result(static_cast<std::size_t>(number_of_clusters));
    for (std::uint64_t node = 0; node < labels.size(); ++node) {
        result[static_cast<std::size_t>(labels[static_cast<std::size_t>(node)])].push_back(node);
    }
    return result;
}

// Scratch buffers reused across every chain in one outer iteration.
// All vectors are sized to the number of graph nodes once. Each chain only
// reads and writes entries for nodes it touches, so it must reset its touched
// entries (via the queue node list) at the end of the chain.
struct ChainBuffers {
    std::vector<char> in_pair;            // 1 while the node belongs to the chain's queue
    std::vector<double> gain;             // current gain of moving the node to the other side
    std::vector<std::uint64_t> version;   // monotonic gain version; heap entries pin a version
    std::vector<char> locked;             // 1 once the node has been popped during the chain
    std::vector<std::uint64_t> tentative; // tentative cluster id as the chain progresses

    explicit ChainBuffers(const std::size_t n_nodes)
        : in_pair(n_nodes, 0),
          gain(n_nodes, 0.0),
          version(n_nodes, 0),
          locked(n_nodes, 0),
          tentative(n_nodes, 0) {}
};

struct HeapEntry {
    double gain;
    std::uint64_t node;
    std::uint64_t version;

    bool operator<(const HeapEntry &other) const {
        return gain < other.gain;
    }
};

struct ChainScratch {
    std::vector<std::uint64_t> queue_nodes;
    std::vector<HeapEntry> heap;

    void clear() {
        queue_nodes.clear();
        heap.clear();
    }
};

// Run a Kernighan-Lin move-chain on the bipartition (cluster_a, cluster_b).
//
// Mutates `labels` and `cluster_to_nodes` if the chain commits any moves.
// `cluster_to_nodes[c]` is treated as an append-only list of "nodes ever in
// cluster c during this outer iteration"; the filter on `labels[v] == c`
// removes stale entries on the fly. Returns the committed cumulative gain
// (>= 0).
inline double run_chain(
    const UndirectedGraph &graph,
    const std::vector<double> &costs,
    std::vector<std::uint64_t> &labels,
    std::vector<std::vector<std::uint64_t>> &cluster_to_nodes,
    ChainBuffers &bufs,
    ChainScratch &scratch,
    const std::uint64_t cluster_a,
    const std::uint64_t cluster_b,
    const double epsilon
) {
    if (cluster_a == cluster_b) {
        return 0.0;
    }

    scratch.clear();
    auto &queue_nodes = scratch.queue_nodes;
    const auto &stale_a = cluster_to_nodes[static_cast<std::size_t>(cluster_a)];
    const auto &stale_b = cluster_to_nodes[static_cast<std::size_t>(cluster_b)];
    queue_nodes.reserve(stale_a.size() + stale_b.size());

    std::size_t live_a = 0;
    std::size_t live_b = 0;
    for (const auto v : stale_a) {
        if (labels[static_cast<std::size_t>(v)] == cluster_a) {
            queue_nodes.push_back(v);
            bufs.in_pair[static_cast<std::size_t>(v)] = 1;
            ++live_a;
        }
    }
    for (const auto v : stale_b) {
        if (labels[static_cast<std::size_t>(v)] == cluster_b) {
            queue_nodes.push_back(v);
            bufs.in_pair[static_cast<std::size_t>(v)] = 1;
            ++live_b;
        }
    }
    // No two-side swap is possible if either side is empty or both sides have
    // a single node (the only non-trivial outcome is a join, handled by
    // apply_joins).
    if (live_a == 0 || live_b == 0 || (live_a == 1 && live_b == 1)) {
        for (const auto v : queue_nodes) {
            bufs.in_pair[static_cast<std::size_t>(v)] = 0;
        }
        return 0.0;
    }

    for (const auto v : queue_nodes) {
        const auto v_label = labels[static_cast<std::size_t>(v)];
        bufs.version[static_cast<std::size_t>(v)] = 0;
        bufs.locked[static_cast<std::size_t>(v)] = 0;
        bufs.tentative[static_cast<std::size_t>(v)] = v_label;

        double w_to_a = 0.0;
        double w_to_b = 0.0;
        for (const auto adj : graph.node_adjacency(v)) {
            if (!bufs.in_pair[static_cast<std::size_t>(adj.node)]) {
                continue;
            }
            const auto c = costs[static_cast<std::size_t>(adj.edge)];
            if (labels[static_cast<std::size_t>(adj.node)] == cluster_a) {
                w_to_a += c;
            } else {
                w_to_b += c;
            }
        }
        bufs.gain[static_cast<std::size_t>(v)] =
            (v_label == cluster_a) ? (w_to_b - w_to_a) : (w_to_a - w_to_b);
    }

    auto &heap = scratch.heap;
    heap.reserve(queue_nodes.size() * 4);
    for (const auto v : queue_nodes) {
        heap.push_back({bufs.gain[static_cast<std::size_t>(v)], v, 0});
    }
    std::make_heap(heap.begin(), heap.end());

    struct Move {
        std::uint64_t node;
        std::uint64_t new_label;
    };
    std::vector<Move> chain;
    chain.reserve(queue_nodes.size());

    double cumulative = 0.0;
    double best_cumulative = 0.0;
    std::size_t best_prefix = 0;
    // The chain keeps running through negative moves because a later prefix
    // can still recover. In practice deep negative runs almost never improve
    // best_cumulative, so cap the lookahead at a small constant after each new
    // best — this matches the heuristic nifty uses and is the dominant runtime
    // saving for large clusters.
    constexpr std::size_t max_steps_without_improvement = 32;
    std::size_t steps_since_best = 0;

    while (!heap.empty()) {
        const auto top = heap.front();
        std::pop_heap(heap.begin(), heap.end());
        heap.pop_back();
        if (bufs.locked[static_cast<std::size_t>(top.node)]) {
            continue;
        }
        if (top.version != bufs.version[static_cast<std::size_t>(top.node)]) {
            continue;
        }

        const auto v = top.node;
        const auto old_label = bufs.tentative[static_cast<std::size_t>(v)];
        const auto new_label = (old_label == cluster_a) ? cluster_b : cluster_a;

        bufs.tentative[static_cast<std::size_t>(v)] = new_label;
        cumulative += bufs.gain[static_cast<std::size_t>(v)];
        chain.push_back({v, new_label});
        bufs.locked[static_cast<std::size_t>(v)] = 1;

        if (cumulative > best_cumulative + epsilon) {
            best_cumulative = cumulative;
            best_prefix = chain.size();
            steps_since_best = 0;
        } else {
            ++steps_since_best;
            if (steps_since_best > max_steps_without_improvement) {
                break;
            }
        }

        for (const auto adj : graph.node_adjacency(v)) {
            const auto u = adj.node;
            if (!bufs.in_pair[static_cast<std::size_t>(u)] || bufs.locked[static_cast<std::size_t>(u)]) {
                continue;
            }
            const auto c = costs[static_cast<std::size_t>(adj.edge)];
            const auto u_label = bufs.tentative[static_cast<std::size_t>(u)];
            // v moved from old_label to new_label. From u's perspective the
            // edge v-u flips between "same side" and "other side":
            //   * u on v's old side  → gain(u) += 2c
            //   * u on v's new side  → gain(u) -= 2c
            if (u_label == old_label) {
                bufs.gain[static_cast<std::size_t>(u)] += 2.0 * c;
            } else if (u_label == new_label) {
                bufs.gain[static_cast<std::size_t>(u)] -= 2.0 * c;
            }
            ++bufs.version[static_cast<std::size_t>(u)];
            heap.push_back({
                bufs.gain[static_cast<std::size_t>(u)],
                u,
                bufs.version[static_cast<std::size_t>(u)],
            });
            std::push_heap(heap.begin(), heap.end());
        }
    }

    for (const auto v : queue_nodes) {
        bufs.in_pair[static_cast<std::size_t>(v)] = 0;
    }

    if (best_cumulative > epsilon) {
        for (std::size_t i = 0; i < best_prefix; ++i) {
            const auto v = chain[i].node;
            const auto new_label = chain[i].new_label;
            labels[static_cast<std::size_t>(v)] = new_label;
            cluster_to_nodes[static_cast<std::size_t>(new_label)].push_back(v);
        }
        return best_cumulative;
    }
    return 0.0;
}

// Single-node polish pass. For each node, find the adjacent cluster (if any)
// for which the unilateral move-out gain is positive, then move. The chain
// already considers these moves implicitly, but its greedy ordering can lock
// a node into a suboptimal commit prefix; this cheap O(|V| + |E|) sweep
// recovers those decisions without altering converged optima.
inline bool single_node_polish(
    const UndirectedGraph &graph,
    const std::vector<double> &costs,
    std::vector<std::uint64_t> &labels,
    const double epsilon
) {
    bool improved = false;
    // Hoist scratch across nodes; clear via the touched-keys list per node.
    std::unordered_map<std::uint64_t, double> sums;
    std::vector<std::uint64_t> touched;
    for (std::uint64_t v = 0; v < graph.number_of_nodes(); ++v) {
        sums.clear();
        touched.clear();
        for (const auto adj : graph.node_adjacency(v)) {
            const auto label = labels[static_cast<std::size_t>(adj.node)];
            auto it = sums.find(label);
            if (it == sums.end()) {
                sums.emplace(label, costs[static_cast<std::size_t>(adj.edge)]);
                touched.push_back(label);
            } else {
                it->second += costs[static_cast<std::size_t>(adj.edge)];
            }
        }
        const auto cur = labels[static_cast<std::size_t>(v)];
        const auto cur_it = sums.find(cur);
        const auto cur_sum = (cur_it == sums.end()) ? 0.0 : cur_it->second;

        double best_gain = 0.0;
        std::uint64_t best = cur;
        for (const auto candidate : touched) {
            if (candidate == cur) {
                continue;
            }
            const auto gain = sums[candidate] - cur_sum;
            if (gain > best_gain + epsilon) {
                best_gain = gain;
                best = candidate;
            }
        }
        if (best != cur) {
            labels[static_cast<std::size_t>(v)] = best;
            improved = true;
        }
    }
    return improved;
}

// Apply beneficial cluster joins in a single pass using union-find over cluster
// ids. Merging two clusters across cut_weight > 0 edges removes those edges
// from the multicut energy.
inline bool apply_joins(
    std::vector<std::uint64_t> &labels,
    const std::vector<ClusterPair> &pairs,
    const std::uint64_t number_of_clusters,
    const double epsilon
) {
    if (number_of_clusters == 0) {
        return false;
    }
    bioimage_cpp::detail::UnionFind sets(static_cast<std::size_t>(number_of_clusters));
    bool any_join = false;
    for (const auto &pair : pairs) {
        if (pair.cut_weight <= epsilon) {
            continue;
        }
        const auto root_a = sets.find(pair.a);
        const auto root_b = sets.find(pair.b);
        if (root_a == root_b) {
            continue;
        }
        sets.merge(root_a, root_b);
        any_join = true;
    }
    if (any_join) {
        for (auto &label : labels) {
            label = sets.find(label);
        }
    }
    return any_join;
}

} // namespace detail_kl

inline std::vector<std::uint64_t> kernighan_lin(
    const UndirectedGraph &graph,
    const std::vector<double> &costs,
    std::vector<std::uint64_t> labels,
    const std::uint64_t number_of_outer_iterations,
    const double epsilon
) {
    validate_costs(graph, costs);
    validate_labels(graph, labels);
    labels = dense_relabel(labels);

    detail_kl::ChainBuffers bufs(static_cast<std::size_t>(graph.number_of_nodes()));
    detail_kl::ChainScratch scratch;

    for (std::uint64_t iteration = 0; iteration < number_of_outer_iterations; ++iteration) {
        bool improved = false;

        const auto pairs_for_chain = detail_kl::compute_cluster_pairs(graph, costs, labels);
        const auto number_of_clusters = labels.empty()
            ? std::uint64_t{0}
            : (*std::max_element(labels.begin(), labels.end()) + 1);
        auto cluster_to_nodes = detail_kl::build_cluster_to_nodes(labels, number_of_clusters);

        for (const auto &pair : pairs_for_chain) {
            const auto delta = detail_kl::run_chain(
                graph, costs, labels, cluster_to_nodes, bufs, scratch, pair.a, pair.b, epsilon
            );
            if (delta > epsilon) {
                improved = true;
            }
        }

        const auto pairs_for_join = detail_kl::compute_cluster_pairs(graph, costs, labels);
        const auto current_number_of_clusters = labels.empty()
            ? std::uint64_t{0}
            : (*std::max_element(labels.begin(), labels.end()) + 1);
        if (detail_kl::apply_joins(labels, pairs_for_join, current_number_of_clusters, epsilon)) {
            improved = true;
        }

        if (detail_kl::single_node_polish(graph, costs, labels, epsilon)) {
            improved = true;
        }

        labels = dense_relabel(labels);
        if (!improved) {
            break;
        }
    }
    return labels;
}

class KernighanLinSolver final : public SolverBase {
public:
    KernighanLinSolver(
        const std::uint64_t number_of_outer_iterations = 100,
        const double epsilon = 1.0e-6
    )
        : number_of_outer_iterations_(number_of_outer_iterations),
          epsilon_(epsilon) {
    }

    std::vector<std::uint64_t> optimize(Objective &objective) const override {
        auto labels = kernighan_lin(
            objective.graph(),
            objective.costs(),
            objective.labels(),
            number_of_outer_iterations_,
            epsilon_
        );
        objective.set_labels(labels);
        return labels;
    }

private:
    std::uint64_t number_of_outer_iterations_;
    double epsilon_;
};

} // namespace bioimage_cpp::graph::multicut
