#pragma once

#include "bioimage_cpp/detail/edge_hash.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph {

struct Adjacency {
    std::uint64_t node;
    std::uint64_t edge;
};

class UndirectedGraph {
public:
    using NodeId = std::uint64_t;
    using EdgeId = std::uint64_t;
    using Edge = detail::Edge;
    using AdjacencyList = std::vector<Adjacency>;

    explicit UndirectedGraph(
        const NodeId number_of_nodes = 0,
        const EdgeId reserve_number_of_edges = 0
    )
        : UndirectedGraph(number_of_nodes, reserve_number_of_edges, reserve_number_of_edges) {
    }

    virtual ~UndirectedGraph() = default;

    void assign(
        const NodeId number_of_nodes = 0,
        const EdgeId reserve_number_of_edges = 0
    ) {
        number_of_nodes_ = number_of_nodes;
        edges_.clear();
        edge_lookup_.clear();
        adjacency_.clear();
        adjacency_.resize(static_cast<std::size_t>(number_of_nodes));
        edges_.reserve(static_cast<std::size_t>(reserve_number_of_edges));
        edge_lookup_.reserve(static_cast<std::size_t>(reserve_number_of_edges));
    }

    [[nodiscard]] NodeId number_of_nodes() const {
        return number_of_nodes_;
    }

    [[nodiscard]] EdgeId number_of_edges() const {
        return static_cast<EdgeId>(edges_.size());
    }

    [[nodiscard]] NodeId node_id_upper_bound() const {
        return number_of_nodes_ == 0 ? 0 : number_of_nodes_ - 1;
    }

    [[nodiscard]] EdgeId edge_id_upper_bound() const {
        return edges_.empty() ? 0 : static_cast<EdgeId>(edges_.size() - 1);
    }

    [[nodiscard]] std::vector<NodeId> nodes() const {
        std::vector<NodeId> result(static_cast<std::size_t>(number_of_nodes_));
        for (NodeId node = 0; node < number_of_nodes_; ++node) {
            result[static_cast<std::size_t>(node)] = node;
        }
        return result;
    }

    [[nodiscard]] std::vector<EdgeId> edges() const {
        std::vector<EdgeId> result(edges_.size());
        for (EdgeId edge = 0; edge < edges_.size(); ++edge) {
            result[static_cast<std::size_t>(edge)] = edge;
        }
        return result;
    }

    [[nodiscard]] Edge uv(const EdgeId edge) const {
        validate_edge(edge);
        return edges_[static_cast<std::size_t>(edge)];
    }

    [[nodiscard]] NodeId u(const EdgeId edge) const {
        return uv(edge).first;
    }

    [[nodiscard]] NodeId v(const EdgeId edge) const {
        return uv(edge).second;
    }

    [[nodiscard]] const std::vector<Edge> &uv_ids() const {
        return edges_;
    }

    [[nodiscard]] const AdjacencyList &node_adjacency(const NodeId node) const {
        validate_node(node);
        return adjacency_[static_cast<std::size_t>(node)];
    }

    virtual EdgeId insert_edge(const NodeId u, const NodeId v) {
        validate_node(u);
        validate_node(v);
        if (u == v) {
            throw std::invalid_argument("self edges are not supported");
        }

        const auto key = detail::edge_key(u, v);
        const auto found = edge_lookup_.find(key);
        if (found != edge_lookup_.end()) {
            return found->second;
        }
        return insert_new_edge(key.first, key.second);
    }

    [[nodiscard]] virtual std::int64_t find_edge(const NodeId u, const NodeId v) const {
        validate_node(u);
        validate_node(v);
        if (u == v) {
            return -1;
        }

        const auto key = detail::edge_key(u, v);
        const auto found = edge_lookup_.find(key);
        if (found == edge_lookup_.end()) {
            return -1;
        }
        return static_cast<std::int64_t>(found->second);
    }

    [[nodiscard]] std::uint64_t serialization_size() const {
        return 2 + 2 * number_of_edges();
    }

    // Fast construction from a pre-sorted, deduplicated edge list.
    //
    // Precondition: `edges` is sorted ascending by `(u, v)` with `u < v` in
    // every entry, and contains no duplicates. No node id may equal or exceed
    // `number_of_nodes`. The call takes ownership of `edges` and uses it as
    // the graph's edge storage, bypassing the per-edge hash dedup that
    // `insert_edge` performs — useful when bulk-building a contracted graph
    // whose unique edges are already known.
    //
    // When `populate_lookup` is false, the edge-lookup hash map is left empty
    // and `find_edge`/`insert_edge` are not available on the returned graph.
    // Used by the fusion-move contraction primitive, whose sub-solver only
    // walks edges and adjacency lists.
    static UndirectedGraph from_sorted_unique_edges(
        const NodeId number_of_nodes,
        std::vector<Edge> edges,
        const bool populate_lookup = true
    ) {
        const auto n_edges = static_cast<EdgeId>(edges.size());
        UndirectedGraph graph(
            number_of_nodes,
            n_edges,
            populate_lookup ? n_edges : 0
        );
        graph.edges_ = std::move(edges);
        graph.rebuild_adjacency_from_edges();
        if (populate_lookup) {
            for (std::size_t index = 0; index < graph.edges_.size(); ++index) {
                graph.edge_lookup_.emplace(
                    graph.edges_[index], static_cast<EdgeId>(index)
                );
            }
        }
        return graph;
    }

    [[nodiscard]] std::pair<std::vector<EdgeId>, std::vector<EdgeId>>
    extract_subgraph_from_nodes(const std::vector<NodeId> &nodes) const {
        std::unordered_set<NodeId> node_set;
        node_set.reserve(nodes.size());
        for (const auto node : nodes) {
            validate_node(node);
            node_set.insert(node);
        }

        std::vector<EdgeId> inner_edges;
        std::vector<EdgeId> outer_edges;
        for (const auto u : nodes) {
            for (const auto adjacency : node_adjacency(u)) {
                const auto v = adjacency.node;
                const auto edge = adjacency.edge;
                if (node_set.find(v) != node_set.end()) {
                    if (u < v) {
                        inner_edges.push_back(edge);
                    }
                } else {
                    outer_edges.push_back(edge);
                }
            }
        }
        return {inner_edges, outer_edges};
    }

protected:
    // Internal constructor that lets subclasses opt out of pre-reserving the
    // `(u, v) -> edge_id` hash map. Subclasses (e.g. `GridGraph`) that build
    // their edges analytically and never populate `edge_lookup_` for their
    // intrinsic edges should pass `reserve_lookup = 0` so we don't allocate
    // millions of empty hash buckets that are never written to.
    UndirectedGraph(
        const NodeId number_of_nodes,
        const EdgeId reserve_edges,
        const EdgeId reserve_lookup
    )
        : number_of_nodes_(number_of_nodes),
          adjacency_(static_cast<std::size_t>(number_of_nodes)) {
        edges_.reserve(static_cast<std::size_t>(reserve_edges));
        edge_lookup_.reserve(static_cast<std::size_t>(reserve_lookup));
    }

    EdgeId insert_new_edge(const NodeId u, const NodeId v) {
        const auto edge = static_cast<EdgeId>(edges_.size());
        edges_.emplace_back(u, v);
        edge_lookup_.emplace(Edge{u, v}, edge);
        adjacency_[static_cast<std::size_t>(u)].push_back(Adjacency{v, edge});
        adjacency_[static_cast<std::size_t>(v)].push_back(Adjacency{u, edge});
        return edge;
    }

    EdgeId insert_new_edge_without_lookup(const NodeId u, const NodeId v) {
        const auto edge = static_cast<EdgeId>(edges_.size());
        edges_.emplace_back(u, v);
        adjacency_[static_cast<std::size_t>(u)].push_back(Adjacency{v, edge});
        adjacency_[static_cast<std::size_t>(v)].push_back(Adjacency{u, edge});
        return edge;
    }

    // Mutable access to the underlying edge list for subclasses that emit
    // edges in bulk. The caller is responsible for re-establishing the
    // invariants between `edges_` and `adjacency_` (see
    // `rebuild_adjacency_from_edges`) before exposing the graph to clients.
    std::vector<Edge> &access_edges() {
        return edges_;
    }

    // Bulk-populate `adjacency_` from `edges_`. Assumes `edges_` already
    // contains every edge in the order the subclass wants its `EdgeId`s
    // assigned (the index into `edges_` becomes the edge id).
    //
    // Computes the exact degree of each node in one pass, reserves that
    // capacity per `adjacency_[u]`, then fills in a second pass. With
    // exact capacity reserved up front each `adjacency_[u]` gets a single
    // allocation of optimal size — no geometric-growth reallocs, no
    // fragmentation from intermediate buffers. For grid-shaped topologies
    // where one push_back per axis would otherwise hammer many small
    // vectors, this is dramatically faster than `insert_new_edge_*`
    // accumulating both `edges_` and `adjacency_` in lockstep.
    //
    // Existing contents of `adjacency_` are discarded.
    void rebuild_adjacency_from_edges() {
        std::vector<std::size_t> degree(static_cast<std::size_t>(number_of_nodes_), 0);
        for (const auto &edge : edges_) {
            ++degree[static_cast<std::size_t>(edge.first)];
            ++degree[static_cast<std::size_t>(edge.second)];
        }
        for (std::size_t node = 0; node < adjacency_.size(); ++node) {
            adjacency_[node].clear();
            adjacency_[node].reserve(degree[node]);
        }
        for (std::size_t index = 0; index < edges_.size(); ++index) {
            const auto &edge = edges_[index];
            const auto edge_id = static_cast<EdgeId>(index);
            adjacency_[static_cast<std::size_t>(edge.first)].push_back(
                Adjacency{edge.second, edge_id}
            );
            adjacency_[static_cast<std::size_t>(edge.second)].push_back(
                Adjacency{edge.first, edge_id}
            );
        }
    }

    void validate_node(const NodeId node) const {
        if (node >= number_of_nodes_) {
            throw std::out_of_range(
                "node id must be < number_of_nodes, got node id=" +
                std::to_string(node) + ", number_of_nodes=" +
                std::to_string(number_of_nodes_)
            );
        }
    }

    void validate_edge(const EdgeId edge) const {
        if (edge >= edges_.size()) {
            throw std::out_of_range(
                "edge id must be < number_of_edges, got edge id=" +
                std::to_string(edge) + ", number_of_edges=" +
                std::to_string(edges_.size())
            );
        }
    }

private:
    NodeId number_of_nodes_;
    std::vector<Edge> edges_;
    std::vector<AdjacencyList> adjacency_;
    std::unordered_map<Edge, EdgeId, detail::EdgeHash> edge_lookup_;
};

} // namespace bioimage_cpp::graph
