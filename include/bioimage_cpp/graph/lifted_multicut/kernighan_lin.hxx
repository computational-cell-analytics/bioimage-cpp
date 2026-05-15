#pragma once

#include "bioimage_cpp/detail/edge_hash.hxx"
#include "bioimage_cpp/detail/indexed_heap.hxx"
#include "bioimage_cpp/detail/union_find.hxx"
#include "bioimage_cpp/graph/connected_components.hxx"
#include "bioimage_cpp/graph/lifted_multicut/objective.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::lifted_multicut {

namespace detail_kl {

struct ClusterPair {
    std::uint64_t a;
    std::uint64_t b;
};

// Build the set of cluster pairs that share at least one base-graph edge.
// These are the only pairs eligible for the two-cut chain in lifted KL —
// merging non-base-adjacent clusters is undone by the post-iteration
// connected-component repartition.
inline std::vector<ClusterPair> compute_base_cluster_pairs(
    const UndirectedGraph &base_graph,
    const std::vector<std::uint64_t> &labels
) {
    using bioimage_cpp::detail::Edge;
    using bioimage_cpp::detail::EdgeHash;
    using bioimage_cpp::detail::edge_key;

    std::unordered_map<Edge, std::uint8_t, EdgeHash> table;
    table.reserve(static_cast<std::size_t>(base_graph.number_of_edges()));
    for (std::uint64_t edge = 0; edge < base_graph.number_of_edges(); ++edge) {
        const auto uv = base_graph.uv(edge);
        const auto la = labels[static_cast<std::size_t>(uv.first)];
        const auto lb = labels[static_cast<std::size_t>(uv.second)];
        if (la == lb) {
            continue;
        }
        table.emplace(edge_key(la, lb), std::uint8_t{1});
    }

    std::vector<ClusterPair> result;
    result.reserve(table.size());
    for (const auto &entry : table) {
        result.push_back({entry.first.first, entry.first.second});
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

// Same per-node scratch shape as multicut::detail_kl::ChainBuffers but
// duplicated to keep the two KL files independent (the helpers are small).
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

// Run a Kernighan-Lin move-chain on (cluster_a, cluster_b), restricted by the
// base graph for connectivity (cross_count) and driven by the lifted graph
// for energy gains.
//
// As a parallel branch, also consider fully merging cluster_b into cluster_a
// — that is the limit of a chain that moves every B-node into A. The merge
// alternative is committed if its lifted-cut gain beats the best chain prefix
// gain. This matches nifty's two-cut behaviour.
//
// Returns the committed cumulative gain (>= 0).
inline double run_chain(
    const UndirectedGraph &base_graph,
    const UndirectedGraph &lifted_graph,
    const std::vector<double> &lifted_weights,
    const std::uint64_t n_base_edges,
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
    if (live_a + live_b < 2 || (live_a == 1 && live_b == 1)) {
        for (const auto v : queue_nodes) {
            bufs.in_pair[static_cast<std::size_t>(v)] = 0;
        }
        return 0.0;
    }

    const bool is_split = (live_b == 0);

    // First pass: compute initial gains (over lifted graph) and cross_count
    // (over base graph), and accumulate the cross-side lifted-weight sum used
    // by the merge alternative.
    double gain_from_merging_double = 0.0;
    for (const auto v : queue_nodes) {
        double w_to_a = 0.0;
        double w_to_b = 0.0;
        std::uint32_t cross = 0;
        const auto v_label = labels[static_cast<std::size_t>(v)];
        for (const auto adj : lifted_graph.node_adjacency(v)) {
            const auto u_key = static_cast<std::size_t>(adj.node);
            if (!bufs.in_pair[u_key]) {
                continue;
            }
            const auto w = lifted_weights[static_cast<std::size_t>(adj.edge)];
            const auto u_label = labels[u_key];
            if (u_label == cluster_a) {
                w_to_a += w;
            } else {
                w_to_b += w;
            }
            const bool is_base = adj.edge < n_base_edges;
            if (u_label != v_label) {
                if (is_base) {
                    ++cross;
                }
                gain_from_merging_double += w;
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
    const double gain_from_merging = is_split ? 0.0 : 0.5 * gain_from_merging_double;

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

        for (const auto adj : lifted_graph.node_adjacency(v)) {
            const auto u_key = static_cast<std::size_t>(adj.node);
            if (!bufs.in_pair[u_key] || bufs.moved[u_key]) {
                continue;
            }
            const auto w = lifted_weights[static_cast<std::size_t>(adj.edge)];
            const auto u_label = labels[u_key];
            const double delta = (u_label == old_label) ? 2.0 * w : -2.0 * w;
            bufs.stash_gain[u_key] += delta;
            if (heap.contains(u_key)) {
                heap.change(u_key, bufs.stash_gain[u_key]);
            }
            // Border maintenance is base-graph-only. Lifted-only edges affect
            // the gain but cannot make a node bordered.
            if (adj.edge < n_base_edges) {
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
    }

    for (const auto v : queue_nodes) {
        const auto v_key = static_cast<std::size_t>(v);
        bufs.in_pair[v_key] = 0;
        bufs.moved[v_key] = 0;
        bufs.cross_count[v_key] = 0;
    }

    // Decide between best chain prefix and full-merge alternative.
    if (gain_from_merging > best_cumulative + epsilon && gain_from_merging > epsilon) {
        // Move all live B-nodes into A.
        for (const auto v : queue_nodes) {
            if (labels[static_cast<std::size_t>(v)] == cluster_b) {
                labels[static_cast<std::size_t>(v)] = cluster_a;
                cluster_to_nodes[static_cast<std::size_t>(cluster_a)].push_back(v);
            }
        }
        return gain_from_merging;
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

// Re-split labels so that every cluster is connected in the base graph.
// Returns the new labeling (dense relabeled).
inline std::vector<std::uint64_t> enforce_base_connectivity(
    const UndirectedGraph &base_graph,
    const std::vector<std::uint64_t> &labels
) {
    const auto n_edges = static_cast<std::size_t>(base_graph.number_of_edges());
    std::vector<std::uint8_t> mask(n_edges);
    for (std::uint64_t edge = 0; edge < base_graph.number_of_edges(); ++edge) {
        const auto uv = base_graph.uv(edge);
        mask[static_cast<std::size_t>(edge)] =
            labels[static_cast<std::size_t>(uv.first)]
                == labels[static_cast<std::size_t>(uv.second)]
            ? 1 : 0;
    }
    return connected_components(base_graph, mask.empty() ? nullptr : mask.data());
}

} // namespace detail_kl

inline std::vector<std::uint64_t> kernighan_lin(
    const UndirectedGraph &base_graph,
    const UndirectedGraph &lifted_graph,
    const std::vector<double> &lifted_weights,
    const std::uint64_t n_base_edges,
    std::vector<std::uint64_t> labels,
    const std::uint64_t number_of_outer_iterations,
    const double epsilon
) {
    validate_weights(lifted_graph, lifted_weights);
    validate_labels(base_graph, labels);

    // Make sure every cluster is base-graph connected before we start; the
    // chain assumes this invariant.
    labels = detail_kl::enforce_base_connectivity(base_graph, labels);
    labels = dense_relabel(labels);

    double current_energy = energy(lifted_graph, lifted_weights, labels);
    auto last_good = labels;
    double last_good_energy = current_energy;

    const auto n_nodes = static_cast<std::size_t>(base_graph.number_of_nodes());
    detail_kl::ChainBuffers bufs(n_nodes);
    detail_kl::ChainScratch scratch(n_nodes);

    for (std::uint64_t iteration = 0; iteration < number_of_outer_iterations; ++iteration) {
        bool improved = false;

        const auto pairs = detail_kl::compute_base_cluster_pairs(base_graph, labels);
        const auto number_of_clusters = labels.empty()
            ? std::uint64_t{0}
            : (*std::max_element(labels.begin(), labels.end()) + 1);
        auto cluster_to_nodes = detail_kl::build_cluster_to_nodes(labels, number_of_clusters);

        for (const auto &pair : pairs) {
            const auto delta = detail_kl::run_chain(
                base_graph,
                lifted_graph,
                lifted_weights,
                n_base_edges,
                labels,
                cluster_to_nodes,
                bufs,
                scratch,
                pair.a,
                pair.b,
                epsilon
            );
            if (delta > epsilon) {
                improved = true;
            }
        }

        // Try splitting each existing cluster off a fresh label.
        std::uint64_t next_label = number_of_clusters;
        for (std::uint64_t cluster = 0; cluster < number_of_clusters; ++cluster) {
            while (true) {
                if (next_label >= cluster_to_nodes.size()) {
                    cluster_to_nodes.resize(static_cast<std::size_t>(next_label) + 1);
                }
                const auto delta = detail_kl::run_chain(
                    base_graph,
                    lifted_graph,
                    lifted_weights,
                    n_base_edges,
                    labels,
                    cluster_to_nodes,
                    bufs,
                    scratch,
                    cluster,
                    next_label,
                    epsilon
                );
                if (delta <= epsilon) {
                    break;
                }
                improved = true;
                ++next_label;
            }
        }

        // Re-split any cluster that became disconnected in the base graph.
        labels = detail_kl::enforce_base_connectivity(base_graph, labels);
        labels = dense_relabel(labels);

        const auto new_energy = energy(lifted_graph, lifted_weights, labels);
        if (new_energy + epsilon < last_good_energy) {
            last_good = labels;
            last_good_energy = new_energy;
            current_energy = new_energy;
        } else {
            // No improvement — revert to last good labeling and stop.
            labels = last_good;
            current_energy = last_good_energy;
            break;
        }

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
          epsilon_(epsilon) {}

    std::vector<std::uint64_t> optimize(Objective &objective) const override {
        auto labels = kernighan_lin(
            objective.graph(),
            objective.lifted_graph(),
            objective.weights(),
            objective.number_of_base_edges(),
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

} // namespace bioimage_cpp::graph::lifted_multicut
