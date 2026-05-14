#pragma once

#include "bioimage_cpp/detail/edge_hash.hxx"
#include "bioimage_cpp/detail/indexed_heap.hxx"
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

// Per-node scratch reused across chains.
//
// - `in_pair`     : 1 while the node sits in (A ∪ B) for the current chain.
// - `moved`       : 1 once the node has been popped (tentatively moved this
//                   chain). Pairs with `in_pair` to distinguish "moved" from
//                   "non-bordered, never pushed".
// - `cross_count` : number of cross-side bipartition neighbors. Matches
//                   nifty's `referenced_by`: a node is only allowed in the
//                   heap once this is positive, which restricts the chain to
//                   border moves and avoids committing "orphan" single-node
//                   migrations into an arbitrary neighbor cluster. Without
//                   this restriction the chain commits negative-internal-weight
//                   nodes into B even when their true best home is a new
//                   cluster, locking them out of the later split phase.
// - `stash_gain`  : the node's current gain estimate, kept in sync regardless
//                   of whether the node is currently in the heap. Used to seed
//                   `heap.push` when a non-bordered node first joins the
//                   border.
struct ChainBuffers {
    std::vector<char> in_pair;
    std::vector<char> moved;
    std::vector<std::uint32_t> cross_count;
    std::vector<double> stash_gain;

    explicit ChainBuffers(const std::size_t n_nodes)
        : in_pair(n_nodes, 0),
          moved(n_nodes, 0),
          cross_count(n_nodes, 0),
          stash_gain(n_nodes, 0.0) {}
};

struct ChainScratch {
    std::vector<std::uint64_t> queue_nodes;
    bioimage_cpp::detail::DenseIndexedHeap<double> heap;

    explicit ChainScratch(const std::size_t n_nodes) : heap(n_nodes) {}
};

// Run a Kernighan-Lin move-chain on the bipartition (cluster_a, cluster_b).
//
// Mutates `labels` and `cluster_to_nodes` if the chain commits any moves.
// `cluster_to_nodes[c]` is treated as an append-only list of "nodes ever in
// cluster c during this outer iteration"; the filter on `labels[v] == c`
// removes stale entries on the fly. Returns the committed cumulative gain
// (>= 0).
//
// Also handles single-cluster splits: pass `cluster_b` as a fresh label
// (no live members) and the chain will try to peel off a subset of `cluster_a`
// into the new label.
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

    auto &queue_nodes = scratch.queue_nodes;
    auto &heap = scratch.heap;
    queue_nodes.clear();
    heap.clear();

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
    // Skip if no non-trivial move exists: the cluster_b == fresh-label split
    // case is allowed when cluster_a has at least two live nodes.
    if (live_a + live_b < 2 || (live_a == 1 && live_b == 1)) {
        for (const auto v : queue_nodes) {
            bufs.in_pair[static_cast<std::size_t>(v)] = 0;
        }
        return 0.0;
    }

    // For pair-chains (both sides non-empty) the chain is restricted to nodes
    // with at least one cross-side neighbor. For splits (B empty) every live
    // A node is eligible because there is no border yet — the first move has
    // to peel off the weakest-attached interior node.
    const bool is_split = (live_b == 0);
    for (const auto v : queue_nodes) {
        double w_to_a = 0.0;
        double w_to_b = 0.0;
        std::uint32_t cross = 0;
        const auto v_label = labels[static_cast<std::size_t>(v)];
        for (const auto adj : graph.node_adjacency(v)) {
            const auto u_key = static_cast<std::size_t>(adj.node);
            if (!bufs.in_pair[u_key]) {
                continue;
            }
            const auto c = costs[static_cast<std::size_t>(adj.edge)];
            const auto u_label = labels[u_key];
            if (u_label == cluster_a) {
                w_to_a += c;
            } else {
                w_to_b += c;
            }
            if (u_label != v_label) {
                ++cross;
            }
        }
        const double gain_v = (v_label == cluster_a) ? (w_to_b - w_to_a) : (w_to_a - w_to_b);
        const auto v_key = static_cast<std::size_t>(v);
        bufs.stash_gain[v_key] = gain_v;
        bufs.cross_count[v_key] = cross;
        if (is_split || cross > 0) {
            heap.push(v_key, gain_v);
        }
    }

    struct Move {
        std::uint64_t node;
        std::uint64_t new_label;
    };
    std::vector<Move> chain;
    chain.reserve(queue_nodes.size());

    double cumulative = 0.0;
    double best_cumulative = 0.0;
    std::size_t best_prefix = 0;

    while (!heap.empty()) {
        const auto top = heap.pop();
        const auto v = static_cast<std::uint64_t>(top.key);
        const auto gain_v = top.priority;
        const auto v_key = static_cast<std::size_t>(v);
        const auto old_label = labels[v_key];
        const auto new_label = (old_label == cluster_a) ? cluster_b : cluster_a;

        bufs.moved[v_key] = 1;
        cumulative += gain_v;
        chain.push_back({v, new_label});

        if (cumulative > best_cumulative + epsilon) {
            best_cumulative = cumulative;
            best_prefix = chain.size();
        }

        for (const auto adj : graph.node_adjacency(v)) {
            const auto u_key = static_cast<std::size_t>(adj.node);
            if (!bufs.in_pair[u_key] || bufs.moved[u_key]) {
                continue;
            }
            const auto c = costs[static_cast<std::size_t>(adj.edge)];
            const auto u_label = labels[u_key];
            const double delta = (u_label == old_label) ? 2.0 * c : -2.0 * c;
            bufs.stash_gain[u_key] += delta;
            if (heap.contains(u_key)) {
                heap.change(u_key, bufs.stash_gain[u_key]);
            }
            // Border maintenance. For pair-chains, only nodes that are
            // currently bordered may be popped. A node becomes bordered when
            // it gains its first cross-side neighbor (cross_count 0 -> 1) and
            // un-borders when it loses its last (cross_count -> 0).
            if (u_label == old_label) {
                ++bufs.cross_count[u_key];
                if (!is_split && !heap.contains(u_key)) {
                    heap.push(u_key, bufs.stash_gain[u_key]);
                }
            } else {
                if (bufs.cross_count[u_key] > 0) {
                    --bufs.cross_count[u_key];
                }
                if (!is_split && bufs.cross_count[u_key] == 0
                    && heap.contains(u_key)) {
                    heap.erase(u_key);
                }
            }
        }
    }

    for (const auto v : queue_nodes) {
        const auto v_key = static_cast<std::size_t>(v);
        bufs.in_pair[v_key] = 0;
        bufs.moved[v_key] = 0;
        bufs.cross_count[v_key] = 0;
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

    const auto n_nodes = static_cast<std::size_t>(graph.number_of_nodes());
    detail_kl::ChainBuffers bufs(n_nodes);
    detail_kl::ChainScratch scratch(n_nodes);

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

        // Try splitting each existing cluster off a fresh label. Pair-chains
        // can only swap members between existing clusters, so without this
        // pass the algorithm can never *increase* the partition count — any
        // local minimum that requires breaking up a cluster is unreachable.
        // Whether a given problem actually benefits depends on whether the
        // pair-chain phase leaves any cluster with internally-negative-weight
        // nodes.
        std::uint64_t next_label = number_of_clusters;
        for (std::uint64_t cluster = 0; cluster < number_of_clusters; ++cluster) {
            while (true) {
                if (next_label >= cluster_to_nodes.size()) {
                    cluster_to_nodes.resize(static_cast<std::size_t>(next_label) + 1);
                }
                const auto delta = detail_kl::run_chain(
                    graph, costs, labels, cluster_to_nodes, bufs, scratch, cluster, next_label, epsilon
                );
                if (delta <= epsilon) {
                    break;
                }
                improved = true;
                ++next_label;
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
