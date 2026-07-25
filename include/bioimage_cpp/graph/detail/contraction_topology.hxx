#pragma once

#include "bioimage_cpp/graph/contraction_options.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::detail {

inline constexpr std::uint64_t invalid_contraction_id =
    std::numeric_limits<std::uint64_t>::max();

struct ContractionAdjacency {
    std::uint64_t node;
    std::uint64_t edge;
};

struct EmptyContractionEdgePayload {};

template <class EdgePayload>
struct BasicContractionEdge {
    std::uint64_t u = 0;
    std::uint64_t v = 0;
    [[no_unique_address]] EdgePayload payload;
};

using ContractionEdge =
    BasicContractionEdge<EmptyContractionEdgePayload>;

struct EdgeRekey {
    std::uint64_t edge;
    std::uint64_t old_u;
    std::uint64_t old_v;
    std::uint64_t new_u;
    std::uint64_t new_v;
};

struct EdgeFold {
    std::uint64_t kept_edge;
    std::uint64_t removed_edge;
};

// Callers reuse this record across mutations to avoid one allocation per edge
// contraction. Solver adapters can use the same record to update heaps and
// objective-specific state after the topology changes.
struct ContractionChange {
    std::uint64_t kept_node = invalid_contraction_id;
    std::uint64_t removed_node = invalid_contraction_id;
    std::uint64_t removed_edge = invalid_contraction_id;
    std::uint64_t replacement_edge = invalid_contraction_id;
    std::vector<std::uint64_t> deactivated_edges;
    std::vector<EdgeRekey> rekeyed_edges;
    std::vector<EdgeFold> folded_edges;

    void clear() {
        kept_node = invalid_contraction_id;
        removed_node = invalid_contraction_id;
        removed_edge = invalid_contraction_id;
        replacement_edge = invalid_contraction_id;
        deactivated_edges.clear();
        rekeyed_edges.clear();
        folded_edges.clear();
    }
};

class ContractionChangeObserver {
public:
    explicit ContractionChangeObserver(ContractionChange &change)
        : change_(change) {
        change_.clear();
    }

    void begin_contraction(
        const std::uint64_t kept_node,
        const std::uint64_t removed_node,
        const std::uint64_t removed_edge
    ) {
        change_.kept_node = kept_node;
        change_.removed_node = removed_node;
        change_.removed_edge = removed_edge;
    }

    void edge_deactivated(const std::uint64_t edge) {
        change_.deactivated_edges.push_back(edge);
    }

    void edge_rekeyed(const EdgeRekey &rekey) {
        change_.rekeyed_edges.push_back(rekey);
    }

    void edges_folded(
        const std::uint64_t kept_edge,
        const std::uint64_t removed_edge
    ) {
        change_.folded_edges.push_back({kept_edge, removed_edge});
    }

    void end_contraction(const std::uint64_t kept_node) {
        (void)kept_node;
    }

private:
    ContractionChange &change_;
};

// Mutable topology with stable sparse ids. Parallel-edge merging is optional.
// An edge payload keeps solver state next to its topology when this improves
// locality. The topology does not interpret the payload.
template <class EdgePayload = EmptyContractionEdgePayload>
class BasicContractionTopology {
public:
    using NodeId = std::uint64_t;
    using EdgeId = std::uint64_t;
    using Edge = UndirectedGraph::Edge;

    BasicContractionTopology() = default;

    explicit BasicContractionTopology(
        const UndirectedGraph &graph,
        const ParallelEdgePolicy parallel_policy = ParallelEdgePolicy::Merge
    ) {
        reset(graph, parallel_policy);
    }

    BasicContractionTopology(
        const NodeId number_of_nodes,
        const std::vector<Edge> &edges,
        const ParallelEdgePolicy parallel_policy
    )
        : adjacency_(static_cast<std::size_t>(number_of_nodes)),
          nodes_active_(static_cast<std::size_t>(number_of_nodes), true),
          active_node_count_(static_cast<std::size_t>(number_of_nodes)),
          parallel_policy_(parallel_policy),
          scratch_edge_(
              static_cast<std::size_t>(number_of_nodes),
              invalid_contraction_id
          ) {
        edges_.reserve(edges.size());
        for (const auto &edge : edges) {
            add_initial_edge(edge.first, edge.second);
        }
        active_edge_count_ = edges_.size();
    }

    // Reset from a simple graph while retaining allocated storage. This path
    // reads only the edge list, so it does not trigger a lazy CSR rebuild.
    void reset(
        const UndirectedGraph &graph,
        const ParallelEdgePolicy parallel_policy = ParallelEdgePolicy::Merge
    ) {
        const auto number_of_nodes = graph.number_of_nodes();
        const auto &edges = graph.uv_ids();
        const auto node_count = static_cast<std::size_t>(number_of_nodes);

        for (auto &adjacency : adjacency_) {
            adjacency.clear();
        }
        adjacency_.resize(node_count);
        edges_.resize(edges.size());
        edges_active_.assign(edges.size(), true);
        nodes_active_.assign(node_count, true);
        active_node_count_ = node_count;
        active_edge_count_ = edges.size();
        parallel_policy_ = parallel_policy;

        scratch_edge_.assign(node_count, 0);
        for (const auto &[u, v] : edges) {
            if (u >= number_of_nodes || v >= number_of_nodes) {
                throw std::out_of_range("edge endpoint is out of range");
            }
            if (u == v) {
                throw std::invalid_argument("self edges are not supported");
            }
            ++scratch_edge_[static_cast<std::size_t>(u)];
            ++scratch_edge_[static_cast<std::size_t>(v)];
        }
        for (NodeId node = 0; node < number_of_nodes; ++node) {
            adjacency_[static_cast<std::size_t>(node)].reserve(
                static_cast<std::size_t>(
                    scratch_edge_[static_cast<std::size_t>(node)]
                )
            );
        }
        std::fill(
            scratch_edge_.begin(),
            scratch_edge_.end(),
            invalid_contraction_id
        );

        for (std::size_t edge = 0; edge < edges.size(); ++edge) {
            const auto [u, v] = edges[edge];
            edges_[edge] = {u, v};
            add_adjacency(u, v, static_cast<EdgeId>(edge));
        }
    }

    [[nodiscard]] NodeId node_id_upper_bound() const {
        return adjacency_.empty() ? 0 : static_cast<NodeId>(adjacency_.size() - 1);
    }

    [[nodiscard]] EdgeId edge_id_upper_bound() const {
        return edges_.empty() ? 0 : static_cast<EdgeId>(edges_.size() - 1);
    }

    [[nodiscard]] std::size_t node_capacity() const {
        return adjacency_.size();
    }

    [[nodiscard]] std::size_t edge_capacity() const {
        return edges_.size();
    }

    [[nodiscard]] std::size_t number_of_nodes() const {
        return active_node_count_;
    }

    [[nodiscard]] std::size_t number_of_edges() const {
        return active_edge_count_;
    }

    [[nodiscard]] bool node_active(const NodeId node) const {
        validate_node_id(node);
        return nodes_active_[static_cast<std::size_t>(node)];
    }

    [[nodiscard]] bool edge_active(const EdgeId edge) const {
        validate_edge_id(edge);
        return edges_active_[static_cast<std::size_t>(edge)];
    }

    [[nodiscard]] Edge uv(const EdgeId edge) const {
        validate_active_edge(edge);
        const auto &entry = edges_[static_cast<std::size_t>(edge)];
        return {entry.u, entry.v};
    }

    [[nodiscard]] EdgePayload &edge_payload(const EdgeId edge) {
        return edges_[static_cast<std::size_t>(edge)].payload;
    }

    [[nodiscard]] const EdgePayload &edge_payload(const EdgeId edge) const {
        return edges_[static_cast<std::size_t>(edge)].payload;
    }

    [[nodiscard]] std::size_t degree(const NodeId node) const {
        validate_active_node(node);
        return adjacency_[static_cast<std::size_t>(node)].size();
    }

    // Select the larger adjacency as the survivor. The edge must be active.
    [[nodiscard]] std::pair<NodeId, NodeId> preferred_contraction_nodes(
        const EdgeId edge
    ) const {
        const auto &entry = edges_[static_cast<std::size_t>(edge)];
        const auto degree_u =
            adjacency_[static_cast<std::size_t>(entry.u)].size();
        const auto degree_v =
            adjacency_[static_cast<std::size_t>(entry.v)].size();
        if (degree_u < degree_v) {
            return {entry.v, entry.u};
        }
        return {entry.u, entry.v};
    }

    [[nodiscard]] bool can_suppress_node(const NodeId node) const {
        validate_node_id(node);
        if (!nodes_active_[static_cast<std::size_t>(node)]) {
            return false;
        }
        const auto &incident = adjacency_[static_cast<std::size_t>(node)];
        return incident.size() == 2 &&
            incident[0].edge != incident[1].edge &&
            incident[0].node != incident[1].node;
    }

    [[nodiscard]] const std::vector<ContractionAdjacency> &node_adjacency(
        const NodeId node
    ) const {
        validate_active_node(node);
        return adjacency_[static_cast<std::size_t>(node)];
    }

    [[nodiscard]] std::vector<NodeId> active_nodes() const {
        std::vector<NodeId> result;
        result.reserve(active_node_count_);
        for (NodeId node = 0; node < adjacency_.size(); ++node) {
            if (nodes_active_[static_cast<std::size_t>(node)]) {
                result.push_back(node);
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<EdgeId> active_edges() const {
        std::vector<EdgeId> result;
        result.reserve(active_edge_count_);
        for (EdgeId edge = 0; edge < edges_.size(); ++edge) {
            if (edges_active_[static_cast<std::size_t>(edge)]) {
                result.push_back(edge);
            }
        }
        return result;
    }

    [[nodiscard]] std::int64_t find_edge(const NodeId u, const NodeId v) const {
        validate_active_node(u);
        validate_active_node(v);
        auto result = invalid_contraction_id;
        for (const auto &adjacency : adjacency_[static_cast<std::size_t>(u)]) {
            if (adjacency.node == v) {
                result = std::min(result, adjacency.edge);
            }
        }
        return result == invalid_contraction_id
            ? -1
            : static_cast<std::int64_t>(result);
    }

    [[nodiscard]] std::vector<EdgeId> find_edges(
        const NodeId u,
        const NodeId v
    ) const {
        validate_active_node(u);
        validate_active_node(v);
        std::vector<EdgeId> result;
        for (const auto &adjacency : adjacency_[static_cast<std::size_t>(u)]) {
            if (adjacency.node == v) {
                result.push_back(adjacency.edge);
            }
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    NodeId contract_edge(
        const EdgeId edge,
        const std::optional<NodeId> keep_node,
        ContractionChange &change
    ) {
        ContractionChangeObserver observer(change);
        return contract_edge_observed(edge, keep_node, observer);
    }

    // Contract an edge and report mutation events as they occur. Solvers use
    // this path to keep objective state in cache and avoid event-buffer writes.
    template <
        bool TrackEdgeActivity = true,
        bool AssumeMergePolicy = false,
        class Observer
    >
    NodeId contract_edge_observed(
        const EdgeId edge,
        const std::optional<NodeId> keep_node,
        Observer &observer
    ) {
        validate_active_edge(edge);
        const auto contracted = edges_[static_cast<std::size_t>(edge)];
        if (contracted.u == contracted.v) {
            throw std::invalid_argument("cannot contract a self edge");
        }

        auto kept = contracted.u;
        auto removed = contracted.v;
        if (keep_node.has_value()) {
            if (*keep_node != contracted.u && *keep_node != contracted.v) {
                throw std::invalid_argument(
                    "keep_node must be an endpoint of the contracted edge"
                );
            }
            kept = *keep_node;
            removed = kept == contracted.u ? contracted.v : contracted.u;
        } else {
            const auto degree_u = adjacency_[static_cast<std::size_t>(contracted.u)].size();
            const auto degree_v = adjacency_[static_cast<std::size_t>(contracted.v)].size();
            if (degree_v > degree_u ||
                (degree_v == degree_u && contracted.v < contracted.u)) {
                kept = contracted.v;
                removed = contracted.u;
            }
        }

        return contract_edge_observed_known_nodes<
            TrackEdgeActivity,
            AssumeMergePolicy
        >(edge, kept, removed, observer);
    }

    // The caller has selected two current endpoints of `edge`.
    // `TrackEdgeActivity = false` leaves edge activity and edge counts stale.
    // `AssumeMergePolicy = true` requires a topology with merge policy.
    template <
        bool TrackEdgeActivity = true,
        bool AssumeMergePolicy = false,
        class Observer
    >
    NodeId contract_edge_observed_known_nodes(
        const EdgeId edge,
        const NodeId kept,
        const NodeId removed,
        Observer &observer
    ) {
        observer.begin_contraction(kept, removed, edge);

        if constexpr (AssumeMergePolicy) {
            stamp_neighbors(kept);
        } else if (parallel_policy_ == ParallelEdgePolicy::Merge) {
            stamp_neighbors(kept);
        }

        if constexpr (AssumeMergePolicy) {
            erase_adjacency_neighbor(
                adjacency_[static_cast<std::size_t>(kept)],
                removed
            );
        } else if (parallel_policy_ == ParallelEdgePolicy::Merge) {
            erase_adjacency_neighbor(
                adjacency_[static_cast<std::size_t>(kept)],
                removed
            );
        } else {
            erase_adjacency_edge(
                adjacency_[static_cast<std::size_t>(kept)],
                edge
            );
        }
        if constexpr (TrackEdgeActivity) {
            edges_active_[static_cast<std::size_t>(edge)] = false;
            --active_edge_count_;
        }
        observer.edge_deactivated(edge);
        if constexpr (AssumeMergePolicy) {
            scratch_edge_[static_cast<std::size_t>(removed)] =
                invalid_contraction_id;
        } else if (parallel_policy_ == ParallelEdgePolicy::Merge) {
            scratch_edge_[static_cast<std::size_t>(removed)] =
                invalid_contraction_id;
        }

        const auto &removed_adjacency =
            adjacency_[static_cast<std::size_t>(removed)];
        for (const auto &adjacency : removed_adjacency) {
            const auto current_edge = adjacency.edge;
            const auto neighbor = adjacency.node;
            if (neighbor == kept) {
                if constexpr (AssumeMergePolicy) {
                    continue;
                } else if (current_edge != edge) {
                    if (parallel_policy_ == ParallelEdgePolicy::Merge) {
                        erase_adjacency_neighbor(
                            adjacency_[static_cast<std::size_t>(kept)],
                            removed
                        );
                    } else {
                        erase_adjacency_edge(
                            adjacency_[static_cast<std::size_t>(kept)],
                            current_edge
                        );
                    }
                    if constexpr (TrackEdgeActivity) {
                        edges_active_[
                            static_cast<std::size_t>(current_edge)
                        ] = false;
                        --active_edge_count_;
                    }
                    observer.edge_deactivated(current_edge);
                }
                continue;
            }

            EdgeId existing;
            if constexpr (AssumeMergePolicy) {
                existing = scratch_edge_[static_cast<std::size_t>(neighbor)];
            } else {
                existing =
                    parallel_policy_ == ParallelEdgePolicy::Merge
                        ? scratch_edge_[static_cast<std::size_t>(neighbor)]
                        : invalid_contraction_id;
            }
            if (existing != invalid_contraction_id) {
                erase_adjacency_neighbor(
                    adjacency_[static_cast<std::size_t>(neighbor)],
                    removed
                );
                if constexpr (TrackEdgeActivity) {
                    edges_active_[static_cast<std::size_t>(current_edge)] =
                        false;
                    --active_edge_count_;
                }
                observer.edge_deactivated(current_edge);
                observer.edges_folded(existing, current_edge);
                continue;
            }

            const auto old_edge =
                edges_[static_cast<std::size_t>(current_edge)];
            adjacency_[static_cast<std::size_t>(kept)].push_back(
                {neighbor, current_edge}
            );
            if constexpr (AssumeMergePolicy) {
                scratch_edge_[static_cast<std::size_t>(neighbor)] = current_edge;
            } else if (parallel_policy_ == ParallelEdgePolicy::Merge) {
                scratch_edge_[static_cast<std::size_t>(neighbor)] = current_edge;
            }
            if constexpr (AssumeMergePolicy) {
                rename_adjacency_neighbor_unchecked(
                    adjacency_[static_cast<std::size_t>(neighbor)],
                    removed,
                    kept
                );
            } else if (parallel_policy_ == ParallelEdgePolicy::Merge) {
                rename_adjacency_neighbor(
                    adjacency_[static_cast<std::size_t>(neighbor)],
                    removed,
                    kept
                );
            } else {
                rename_adjacency_edge(
                    adjacency_[static_cast<std::size_t>(neighbor)],
                    current_edge,
                    kept
                );
            }
            auto &new_edge =
                edges_[static_cast<std::size_t>(current_edge)];
            if (new_edge.u == removed) {
                new_edge.u = kept;
            } else if constexpr (AssumeMergePolicy) {
                new_edge.v = kept;
            } else {
                if (new_edge.v == removed) {
                    new_edge.v = kept;
                } else {
                    throw std::runtime_error(
                        "edge is not incident to the rekeyed node"
                    );
                }
            }
            observer.edge_rekeyed(
                {
                    current_edge,
                    old_edge.u,
                    old_edge.v,
                    new_edge.u,
                    new_edge.v,
                }
            );
        }

        if constexpr (AssumeMergePolicy) {
            clear_stamps(kept);
        } else if (parallel_policy_ == ParallelEdgePolicy::Merge) {
            clear_stamps(kept);
        }
        adjacency_[static_cast<std::size_t>(removed)].clear();
        nodes_active_[static_cast<std::size_t>(removed)] = false;
        --active_node_count_;
        observer.end_contraction(kept);
        return kept;
    }

    void erase_edge(const EdgeId edge, ContractionChange &change) {
        change.clear();
        validate_active_edge(edge);
        change.removed_edge = edge;
        deactivate_edge(edge);
        change.deactivated_edges.push_back(edge);
    }

    EdgeId suppress_node(const NodeId node, ContractionChange &change) {
        change.clear();
        validate_active_node(node);
        const auto incident = adjacency_[static_cast<std::size_t>(node)];
        if (incident.size() != 2) {
            throw std::invalid_argument(
                "suppress_node requires degree 2, got degree=" +
                std::to_string(incident.size())
            );
        }
        if (incident[0].edge == incident[1].edge) {
            throw std::invalid_argument("cannot suppress a node incident to a self edge");
        }

        const auto edge_a = incident[0].edge;
        const auto edge_b = incident[1].edge;
        const auto neighbor_a = incident[0].node;
        const auto neighbor_b = incident[1].node;
        if (neighbor_a == neighbor_b) {
            throw std::invalid_argument(
                "suppress_node would create a self edge"
            );
        }

        change.removed_node = node;
        deactivate_edge(edge_a);
        deactivate_edge(edge_b);
        change.deactivated_edges.push_back(edge_a);
        change.deactivated_edges.push_back(edge_b);

        EdgeId replacement = invalid_contraction_id;
        if (parallel_policy_ == ParallelEdgePolicy::Merge &&
            neighbor_a != neighbor_b) {
            const auto found = find_edge(neighbor_a, neighbor_b);
            if (found >= 0) {
                replacement = static_cast<EdgeId>(found);
                change.folded_edges.push_back({replacement, edge_a});
                change.folded_edges.push_back({replacement, edge_b});
            }
        }

        if (replacement == invalid_contraction_id) {
            replacement = std::min(edge_a, edge_b);
            const auto folded = replacement == edge_a ? edge_b : edge_a;
            auto &replacement_entry = edges_[static_cast<std::size_t>(replacement)];
            const auto old_u = replacement_entry.u;
            const auto old_v = replacement_entry.v;
            replacement_entry.u = neighbor_a;
            replacement_entry.v = neighbor_b;
            edges_active_[static_cast<std::size_t>(replacement)] = true;
            ++active_edge_count_;
            add_adjacency(neighbor_a, neighbor_b, replacement);
            if (old_u != node && old_v != node) {
                throw std::runtime_error(
                    "suppressed edge is not incident to the suppressed node"
                );
            }
            change.rekeyed_edges.push_back(
                {
                    replacement,
                    old_u,
                    old_v,
                    neighbor_a,
                    neighbor_b,
                }
            );
            change.folded_edges.push_back({replacement, folded});
        }

        adjacency_[static_cast<std::size_t>(node)].clear();
        nodes_active_[static_cast<std::size_t>(node)] = false;
        --active_node_count_;
        change.replacement_edge = replacement;
        return replacement;
    }

private:
    void validate_node_id(const NodeId node) const {
        if (node >= adjacency_.size()) {
            throw std::out_of_range(
                "node id " + std::to_string(node) + " is out of range"
            );
        }
    }

    void validate_edge_id(const EdgeId edge) const {
        if (edge >= edges_.size()) {
            throw std::out_of_range(
                "edge id " + std::to_string(edge) + " is out of range"
            );
        }
    }

    void validate_active_node(const NodeId node) const {
        validate_node_id(node);
        if (!nodes_active_[static_cast<std::size_t>(node)]) {
            throw std::invalid_argument(
                "node " + std::to_string(node) + " is inactive"
            );
        }
    }

    void validate_active_edge(const EdgeId edge) const {
        validate_edge_id(edge);
        if (!edges_active_[static_cast<std::size_t>(edge)]) {
            throw std::invalid_argument(
                "edge " + std::to_string(edge) + " is inactive"
            );
        }
    }

    void add_initial_edge(const NodeId u, const NodeId v) {
        if (u >= adjacency_.size() || v >= adjacency_.size()) {
            throw std::out_of_range("edge endpoint is out of range");
        }
        if (u == v) {
            throw std::invalid_argument("self edges are not supported");
        }
        if (parallel_policy_ == ParallelEdgePolicy::Merge) {
            for (const auto &adjacency : adjacency_[static_cast<std::size_t>(u)]) {
                if (adjacency.node == v) {
                    throw std::invalid_argument("parallel edges are not supported");
                }
            }
        }
        const auto edge = static_cast<EdgeId>(edges_.size());
        edges_.push_back({u, v});
        edges_active_.push_back(true);
        add_adjacency(u, v, edge);
    }

    void add_adjacency(const NodeId u, const NodeId v, const EdgeId edge) {
        adjacency_[static_cast<std::size_t>(u)].push_back({v, edge});
        adjacency_[static_cast<std::size_t>(v)].push_back({u, edge});
    }

    static void erase_adjacency_edge(
        std::vector<ContractionAdjacency> &adjacency,
        const EdgeId edge
    ) {
        for (std::size_t index = 0; index < adjacency.size(); ++index) {
            if (adjacency[index].edge == edge) {
                adjacency[index] = adjacency.back();
                adjacency.pop_back();
                return;
            }
        }
    }

    static void erase_adjacency_neighbor(
        std::vector<ContractionAdjacency> &adjacency,
        const NodeId node
    ) {
        for (std::size_t index = 0; index < adjacency.size(); ++index) {
            if (adjacency[index].node == node) {
                adjacency[index] = adjacency.back();
                adjacency.pop_back();
                return;
            }
        }
    }

    static void rename_adjacency_neighbor(
        std::vector<ContractionAdjacency> &adjacency,
        const NodeId old_node,
        const NodeId new_node
    ) {
        for (auto &entry : adjacency) {
            if (entry.node == old_node) {
                entry.node = new_node;
                return;
            }
        }
        throw std::runtime_error("edge adjacency is inconsistent");
    }

    static void rename_adjacency_neighbor_unchecked(
        std::vector<ContractionAdjacency> &adjacency,
        const NodeId old_node,
        const NodeId new_node
    ) {
        for (auto &entry : adjacency) {
            if (entry.node == old_node) {
                entry.node = new_node;
                return;
            }
        }
    }

    static void rename_adjacency_edge(
        std::vector<ContractionAdjacency> &adjacency,
        const EdgeId edge,
        const NodeId new_node
    ) {
        for (auto &entry : adjacency) {
            if (entry.edge == edge) {
                entry.node = new_node;
                return;
            }
        }
        throw std::runtime_error("edge adjacency is inconsistent");
    }

    void deactivate_edge(const EdgeId edge) {
        const auto edge_index = static_cast<std::size_t>(edge);
        auto &entry = edges_[edge_index];
        if (!edges_active_[edge_index]) {
            return;
        }
        erase_adjacency_edge(adjacency_[static_cast<std::size_t>(entry.u)], edge);
        erase_adjacency_edge(adjacency_[static_cast<std::size_t>(entry.v)], edge);
        edges_active_[edge_index] = false;
        --active_edge_count_;
    }

    void stamp_neighbors(const NodeId node) {
        for (const auto &adjacency : adjacency_[static_cast<std::size_t>(node)]) {
            if (adjacency.node != node) {
                scratch_edge_[static_cast<std::size_t>(adjacency.node)] =
                    adjacency.edge;
            }
        }
    }

    void clear_stamps(const NodeId node) {
        for (const auto &adjacency : adjacency_[static_cast<std::size_t>(node)]) {
            if (adjacency.node != node) {
                scratch_edge_[static_cast<std::size_t>(adjacency.node)] =
                    invalid_contraction_id;
            }
        }
    }

    std::vector<std::vector<ContractionAdjacency>> adjacency_;
    std::vector<BasicContractionEdge<EdgePayload>> edges_;
    std::vector<bool> edges_active_;
    std::vector<bool> nodes_active_;
    std::size_t active_node_count_ = 0;
    std::size_t active_edge_count_ = 0;
    ParallelEdgePolicy parallel_policy_ = ParallelEdgePolicy::Merge;
    std::vector<EdgeId> scratch_edge_;
};

using ContractionTopology = BasicContractionTopology<>;

} // namespace bioimage_cpp::graph::detail
