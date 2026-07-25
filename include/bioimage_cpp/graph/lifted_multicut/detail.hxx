#pragma once

#include "bioimage_cpp/detail/indexed_heap.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/graph/connected_components.hxx"
#include "bioimage_cpp/graph/detail/contraction_topology.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"
#include "bioimage_cpp/util/union_find.hxx"

#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::lifted_multicut::detail {

using EdgeHeap = bioimage_cpp::detail::DenseIndexedHeap<double>;

struct ObjectiveEdge {
    double weight = 0.0;
    unsigned char is_lifted = 0;
};

struct ContractionState {
    graph::detail::BasicContractionTopology<ObjectiveEdge> topology;
    bioimage_cpp::util::UnionFind union_find{0};
    EdgeHeap heap;

    void reset(const UndirectedGraph &lifted_graph) {
        topology.reset(lifted_graph, ParallelEdgePolicy::Merge);
        union_find.reset(
            static_cast<std::size_t>(lifted_graph.number_of_nodes())
        );
        const auto number_of_edges =
            static_cast<std::size_t>(lifted_graph.number_of_edges());
        heap.reset_capacity(number_of_edges);
    }
};

inline void initialize_contraction_state(
    const UndirectedGraph &lifted_graph,
    const std::vector<double> &weights,
    const std::uint64_t number_of_base_edges,
    ContractionState &state,
    const bool add_noise = false,
    const int seed = 42,
    const double sigma = 1.0
) {
    std::vector<EdgeHeap::Entry> heap_entries;
    heap_entries.reserve(static_cast<std::size_t>(number_of_base_edges));

    std::mt19937 generator(seed);
    std::normal_distribution<double> noise(0.0, sigma);
    for (std::uint64_t edge = 0;
         edge < lifted_graph.number_of_edges();
         ++edge) {
        double weight = weights[static_cast<std::size_t>(edge)];
        if (add_noise) {
            weight += noise(generator);
        }
        const auto edge_id = static_cast<std::size_t>(edge);
        auto &edge_payload = state.topology.edge_payload(edge);
        edge_payload = {
            weight,
            static_cast<unsigned char>(edge < number_of_base_edges ? 0 : 1)
        };
        if (edge_payload.is_lifted == 0) {
            heap_entries.push_back({edge_id, weight});
        }
    }
    state.heap.build_heap(std::move(heap_entries));
}

class ContractionObserver {
public:
    explicit ContractionObserver(ContractionState &state)
        : state_(state) {
    }

    void begin_contraction(
        const std::uint64_t kept_node,
        const std::uint64_t removed_node,
        const std::uint64_t removed_edge
    ) {
        (void)removed_edge;
        state_.union_find.merge_to(kept_node, removed_node);
    }

    void edge_deactivated(const std::uint64_t edge) {
        state_.heap.erase(static_cast<std::size_t>(edge));
    }

    void edge_rekeyed(const graph::detail::EdgeRekey &rekey) {
        (void)rekey;
    }

    void edges_folded(
        const std::uint64_t kept_edge_id,
        const std::uint64_t removed_edge_id
    ) {
        const auto kept_edge = static_cast<std::size_t>(kept_edge_id);
        auto &kept = state_.topology.edge_payload(kept_edge_id);
        const auto &removed =
            state_.topology.edge_payload(removed_edge_id);
        kept.weight += removed.weight;
        kept.is_lifted =
            (kept.is_lifted != 0 && removed.is_lifted != 0)
            ? 1
            : 0;
        if (kept.is_lifted != 0) {
            state_.heap.erase(kept_edge);
        } else {
            state_.heap.push_or_change(
                kept_edge,
                kept.weight
            );
        }
    }

    void end_contraction(const std::uint64_t kept_node) {
        (void)kept_node;
    }

private:
    ContractionState &state_;
};

template <class Profiler>
inline std::uint64_t contract_edge(
    ContractionState &state,
    const std::size_t edge,
    Profiler &profile
) {
    const auto edge_id = static_cast<std::uint64_t>(edge);
    const auto [kept, removed] =
        state.topology.preferred_contraction_nodes(edge_id);
    ContractionObserver observer(state);
    std::uint64_t kept_node;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "contract");
        kept_node =
            state.topology.contract_edge_observed_known_nodes<false, true>(
                edge_id,
                kept,
                removed,
                observer
            );
    }
    return kept_node;
}

inline std::vector<std::uint64_t> labels_from_sets(
    bioimage_cpp::util::UnionFind &sets,
    const UndirectedGraph &graph
) {
    return dense_labels_from_union_find(sets, graph.number_of_nodes());
}

inline std::size_t stop_node_count(
    const UndirectedGraph &graph,
    const double node_num_stop
) {
    return node_num_stop >= 1.0
        ? static_cast<std::size_t>(node_num_stop)
        : static_cast<std::size_t>(
              double(graph.number_of_nodes()) * node_num_stop + 0.5
          );
}

} // namespace bioimage_cpp::graph::lifted_multicut::detail
