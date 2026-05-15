#pragma once

#include "bioimage_cpp/detail/edge_hash.hxx"
#include "bioimage_cpp/detail/indexed_heap.hxx"
#include "bioimage_cpp/detail/profile.hxx"
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

// Per-edge entry of the per-node filtered adjacency built once during
// `chain_gain_init` and consumed by `chain_loop`. Lifted-graph adjacency is
// walked exactly once per chain (during init) and the few entries that
// survive the in-pair filter are appended here, with their weight and base-
// vs-lifted classification cached. The chain loop then iterates only these
// surviving entries — for typical small pairs in a large graph that's a
// >10× reduction in inner-loop iterations versus re-walking
// ``lifted_graph.node_adjacency(v)`` from scratch on every move.
struct FilteredAdj {
    std::uint64_t node;
    double weight;
    bool is_base;
};

struct ChainScratch {
    std::vector<std::uint64_t> queue_nodes;
    bioimage_cpp::detail::DenseIndexedHeap<double> heap;

    // Per-node CSR-style range into ``filtered_entries``. ``filtered_count[v]``
    // is set whenever v is part of the current chain's pair; readers must
    // therefore index only into nodes known to be in-pair (i.e. nodes popped
    // from the heap, all of which were pushed during init).
    std::vector<std::uint64_t> filtered_offset;
    std::vector<std::uint32_t> filtered_count;
    std::vector<FilteredAdj> filtered_entries;

    explicit ChainScratch(const std::size_t n_nodes)
        : heap(n_nodes),
          filtered_offset(n_nodes, 0),
          filtered_count(n_nodes, 0) {
        filtered_entries.reserve(n_nodes * 4);
    }
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
template <class ProfilerT>
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
    const double epsilon,
    [[maybe_unused]] ProfilerT &profile
) {
    if (cluster_a == cluster_b) {
        return 0.0;
    }

    auto &queue_nodes = scratch.queue_nodes;
    auto &heap = scratch.heap;
    auto &filtered_entries = scratch.filtered_entries;
    queue_nodes.clear();
    heap.clear();
    filtered_entries.clear();

    std::size_t live_a = 0;
    std::size_t live_b = 0;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "chain_init");
        const auto &stale_a = cluster_to_nodes[static_cast<std::size_t>(cluster_a)];
        const auto &stale_b = cluster_to_nodes[static_cast<std::size_t>(cluster_b)];
        queue_nodes.reserve(stale_a.size() + stale_b.size());

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
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "chain_gain_init");
    for (const auto v : queue_nodes) {
        double w_to_a = 0.0;
        double w_to_b = 0.0;
        std::uint32_t cross = 0;
        const auto v_label = labels[static_cast<std::size_t>(v)];
        const auto v_key = static_cast<std::size_t>(v);
        const auto filter_start = filtered_entries.size();
        for (const auto adj : lifted_graph.node_adjacency(v)) {
            const auto u_key = static_cast<std::size_t>(adj.node);
            if (!bufs.in_pair[u_key]) {
                continue;
            }
            const auto w = lifted_weights[static_cast<std::size_t>(adj.edge)];
            const auto u_label = labels[u_key];
            const bool is_base = adj.edge < n_base_edges;
            filtered_entries.push_back({adj.node, w, is_base});
            if (u_label == cluster_a) {
                w_to_a += w;
            } else {
                w_to_b += w;
            }
            if (u_label != v_label) {
                if (is_base) {
                    ++cross;
                }
                gain_from_merging_double += w;
            }
        }
        scratch.filtered_offset[v_key] = filter_start;
        scratch.filtered_count[v_key] =
            static_cast<std::uint32_t>(filtered_entries.size() - filter_start);
        const double gain_v = (v_label == cluster_a) ? (w_to_b - w_to_a) : (w_to_a - w_to_b);
        bufs.stash_gain[v_key] = gain_v;
        bufs.cross_count[v_key] = cross;
        if (is_split || cross > 0) {
            heap.push(v_key, gain_v);
        }
    }
    } // end chain_gain_init scope
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

    {
    BIOIMAGE_PROFILE_SCOPE(profile, "chain_loop");
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

        // Walk the pre-built in-pair adjacency; each entry is guaranteed
        // ``bufs.in_pair[u_key] == 1`` so we only need to filter on
        // ``bufs.moved``. ``was_in_heap`` is cached so the three subsequent
        // heap operations don't each re-query the locator.
        const auto offset = scratch.filtered_offset[v_key];
        const auto count = scratch.filtered_count[v_key];
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto &fa = scratch.filtered_entries[offset + i];
            const auto u_key = static_cast<std::size_t>(fa.node);
            if (bufs.moved[u_key]) {
                continue;
            }
            const auto u_label = labels[u_key];
            const double delta = (u_label == old_label) ? 2.0 * fa.weight : -2.0 * fa.weight;
            bufs.stash_gain[u_key] += delta;
            const bool was_in_heap = heap.contains(u_key);
            if (was_in_heap) {
                heap.change(u_key, bufs.stash_gain[u_key]);
            }
            // Border maintenance is base-graph-only. Lifted-only edges affect
            // the gain but cannot make a node bordered.
            if (fa.is_base) {
                if (u_label == old_label) {
                    ++bufs.cross_count[u_key];
                    if (!is_split && !was_in_heap) {
                        heap.push(u_key, bufs.stash_gain[u_key]);
                    }
                } else {
                    if (bufs.cross_count[u_key] > 0) {
                        --bufs.cross_count[u_key];
                    }
                    if (!is_split && bufs.cross_count[u_key] == 0 && was_in_heap) {
                        heap.erase(u_key);
                    }
                }
            }
        }
    }

    } // end chain_loop scope

    {
    BIOIMAGE_PROFILE_SCOPE(profile, "chain_cleanup");
    for (const auto v : queue_nodes) {
        const auto v_key = static_cast<std::size_t>(v);
        bufs.in_pair[v_key] = 0;
        bufs.moved[v_key] = 0;
        bufs.cross_count[v_key] = 0;
    }
    } // end chain_cleanup scope

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
    BIOIMAGE_PROFILE_INIT(profile);
    validate_weights(lifted_graph, lifted_weights);
    validate_labels(base_graph, labels);

    // Make sure every cluster is base-graph connected before we start; the
    // chain assumes this invariant.
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "cc_repartition");
        labels = detail_kl::enforce_base_connectivity(base_graph, labels);
        labels = dense_relabel(labels);
    }

    double current_energy;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "energy_eval");
        current_energy = energy(lifted_graph, lifted_weights, labels);
    }
    auto last_good = labels;
    double last_good_energy = current_energy;

    const auto n_nodes = static_cast<std::size_t>(base_graph.number_of_nodes());
    detail_kl::ChainBuffers bufs(n_nodes);
    detail_kl::ChainScratch scratch(n_nodes);

    for (std::uint64_t iteration = 0; iteration < number_of_outer_iterations; ++iteration) {
        bool improved = false;

        std::vector<detail_kl::ClusterPair> pairs;
        std::uint64_t number_of_clusters = 0;
        std::vector<std::vector<std::uint64_t>> cluster_to_nodes;
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "compute_pairs");
            pairs = detail_kl::compute_base_cluster_pairs(base_graph, labels);
            number_of_clusters = labels.empty()
                ? std::uint64_t{0}
                : (*std::max_element(labels.begin(), labels.end()) + 1);
            cluster_to_nodes = detail_kl::build_cluster_to_nodes(labels, number_of_clusters);
        }

        {
            BIOIMAGE_PROFILE_SCOPE(profile, "pair_chains");
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
                    epsilon,
                    profile
                );
                if (delta > epsilon) {
                    improved = true;
                }
            }
        }

        {
            BIOIMAGE_PROFILE_SCOPE(profile, "cluster_splits");
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
                        epsilon,
                        profile
                    );
                    if (delta <= epsilon) {
                        break;
                    }
                    improved = true;
                    ++next_label;
                }
            }
        }

        {
            BIOIMAGE_PROFILE_SCOPE(profile, "cc_repartition");
            labels = detail_kl::enforce_base_connectivity(base_graph, labels);
            labels = dense_relabel(labels);
        }

        double new_energy;
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "energy_eval");
            new_energy = energy(lifted_graph, lifted_weights, labels);
        }
        if (new_energy + epsilon < last_good_energy) {
            last_good = labels;
            last_good_energy = new_energy;
            current_energy = new_energy;
        } else {
            labels = last_good;
            current_energy = last_good_energy;
            break;
        }

        if (!improved) {
            break;
        }
    }
    BIOIMAGE_PROFILE_REPORT(profile);
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
