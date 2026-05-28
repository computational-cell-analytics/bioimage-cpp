#pragma once

#include "bioimage_cpp/detail/edge_hash.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph {

struct Adjacency {
    std::uint64_t node;
    std::uint64_t edge;
};

// Undirected graph storing adjacency in CSR-style (compressed sparse row)
// layout: a single contiguous `adjacency_data_` array of `2 * E` `Adjacency`
// entries, plus an `adjacency_offsets_` array of length `N + 1` giving the
// start of each node's adjacency slice. This replaces the previous
// `std::vector<std::vector<Adjacency>>` representation, which paid for one
// allocator call per node and scattered the per-node buffers throughout the
// heap — the dominant cost in `GridGraph` construction on large 3D problems.
//
// CSR is rebuilt lazily: incremental `insert_edge*` only appends to `edges_`
// (and `edge_lookup_`) and marks the adjacency dirty; the first subsequent
// `node_adjacency` read triggers a single bulk rebuild. Bulk construction
// paths (`from_sorted_unique_edges`, subclass `build_edges` overrides) call
// `rebuild_adjacency_from_edges()` explicitly, which keeps reads cheap and
// thread-safe.
//
// Thread safety: the lazy rebuild is not internally synchronized. If two
// threads each take the first `node_adjacency` read on a still-dirty graph
// they race on the rebuild — concurrently reallocating `adjacency_data_` and
// overwriting `adjacency_offsets_` — which corrupts the CSR (garbage neighbor
// ids, out-of-bounds reads) and intermittently segfaults. The rule:
//
//   Any algorithm that reads `node_adjacency` (directly, or via
//   `breadth_first_search`, `extract_subgraph_from_nodes`, or a sub-solver
//   such as `multicut::greedy_additive`'s `DynamicGraph::reset`) from
//   `parallel_for_chunks` or other threads MUST `freeze()` the graph on the
//   calling thread *before* the fan-out.
//
// Once frozen (or built via `from_sorted_unique_edges`, which rebuilds the CSR
// eagerly), the graph has no mutable read path and is safe to share by
// `const&` across reader threads. Graphs built incrementally via `insert_edge*`
// (including the `from_edges` binding and `region_adjacency_graph`) start dirty.
class UndirectedGraph {
public:
    using NodeId = std::uint64_t;
    using EdgeId = std::uint64_t;
    using Edge = detail::Edge;
    // Non-owning view over a node's adjacency slice in the CSR buffer.
    // Backward-compatible with the previous `std::vector<Adjacency>` type
    // for the uses we have today (range-for and `.size()`); not assignable
    // and not extendable.
    using AdjacencyList = std::span<const Adjacency>;

    explicit UndirectedGraph(
        const NodeId number_of_nodes = 0,
        const EdgeId reserve_number_of_edges = 0
    )
        : UndirectedGraph(number_of_nodes, reserve_number_of_edges, reserve_number_of_edges) {
    }

    virtual ~UndirectedGraph() = default;

    // CSR data lives in a `unique_ptr<Adjacency[]>`, so the type is move-only.
    // The user-declared destructor above suppresses implicit move generation,
    // so we re-default the moves explicitly. Copies are deleted on purpose:
    // the previous vector-of-vectors layout was implicitly copyable but every
    // such copy paid for a deep clone of millions of small adjacency vectors,
    // and no caller in this codebase actually needs that. Add an explicit
    // deep-copy method if one becomes necessary.
    UndirectedGraph(const UndirectedGraph &) = delete;
    UndirectedGraph &operator=(const UndirectedGraph &) = delete;
    UndirectedGraph(UndirectedGraph &&) noexcept = default;
    UndirectedGraph &operator=(UndirectedGraph &&) noexcept = default;

    void assign(
        const NodeId number_of_nodes = 0,
        const EdgeId reserve_number_of_edges = 0
    ) {
        number_of_nodes_ = number_of_nodes;
        edges_.clear();
        edges_.reserve(static_cast<std::size_t>(reserve_number_of_edges));
        edge_lookup_.clear();
        edge_lookup_.reserve(static_cast<std::size_t>(reserve_number_of_edges));
        adjacency_offsets_.clear();
        adjacency_data_.reset();
        adjacency_data_size_ = 0;
        adjacency_dirty_ = true;
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

    // Adjacency slice of `node`. The first call on a dirty graph triggers a
    // non-thread-safe lazy CSR rebuild (mutable write through this `const`
    // method); call `freeze()` on the construction thread before sharing the
    // graph with concurrent readers. See the class-level thread-safety note.
    [[nodiscard]] AdjacencyList node_adjacency(const NodeId node) const {
        validate_node(node);
        ensure_adjacency_built();
        const auto begin = adjacency_offsets_[static_cast<std::size_t>(node)];
        const auto end = adjacency_offsets_[static_cast<std::size_t>(node) + 1];
        return AdjacencyList(
            adjacency_data_.get() + begin,
            static_cast<std::size_t>(end - begin)
        );
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

    // Force the CSR adjacency to be rebuilt if it's currently stale. Use this
    // after a batch of `insert_edge*` calls, before handing the graph to
    // multiple reader threads or before holding a span returned from
    // `node_adjacency` across other graph operations. Lazy rebuild from a
    // dirty state happens via `mutable` writes inside a `const` method and is
    // not internally synchronized; once a graph is "frozen" via `freeze`
    // (or via an explicit `rebuild_adjacency_from_edges` from a subclass
    // constructor) it is safe to share by `const&` across threads.
    void freeze() const {
        ensure_adjacency_built();
    }

    // Explicit deep copy. The previous vector-of-vectors layout made
    // `UndirectedGraph` implicitly copyable; switching to CSR with a
    // `unique_ptr` buffer made the class move-only, so callers that need
    // a clone use this method. If the source graph's CSR is already built
    // we copy the buffer so the clone doesn't pay another rebuild on
    // first read.
    [[nodiscard]] UndirectedGraph clone() const {
        UndirectedGraph copy(
            number_of_nodes_,
            static_cast<EdgeId>(edges_.size()),
            edge_lookup_.empty() ? 0 : static_cast<EdgeId>(edges_.size())
        );
        copy.edges_ = edges_;
        copy.edge_lookup_ = edge_lookup_;
        if (adjacency_dirty_) {
            copy.adjacency_dirty_ = true;
        } else {
            copy.adjacency_offsets_ = adjacency_offsets_;
            copy.adjacency_data_size_ = adjacency_data_size_;
            copy.adjacency_data_ =
                std::make_unique_for_overwrite<Adjacency[]>(adjacency_data_size_);
            std::copy_n(
                adjacency_data_.get(), adjacency_data_size_, copy.adjacency_data_.get()
            );
            copy.adjacency_dirty_ = false;
        }
        return copy;
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
        : number_of_nodes_(number_of_nodes) {
        edges_.reserve(static_cast<std::size_t>(reserve_edges));
        edge_lookup_.reserve(static_cast<std::size_t>(reserve_lookup));
    }

    EdgeId insert_new_edge(const NodeId u, const NodeId v) {
        const auto edge = static_cast<EdgeId>(edges_.size());
        edges_.emplace_back(u, v);
        edge_lookup_.emplace(Edge{u, v}, edge);
        adjacency_dirty_ = true;
        return edge;
    }

    EdgeId insert_new_edge_without_lookup(const NodeId u, const NodeId v) {
        const auto edge = static_cast<EdgeId>(edges_.size());
        edges_.emplace_back(u, v);
        adjacency_dirty_ = true;
        return edge;
    }

    // Mutable access to the underlying edge list for subclasses that emit
    // edges in bulk. Marks adjacency dirty; the next `node_adjacency` read
    // (or an explicit `rebuild_adjacency_from_edges` call) will rebuild
    // the CSR buffers.
    std::vector<Edge> &access_edges() {
        adjacency_dirty_ = true;
        return edges_;
    }

    // Build the CSR adjacency from `edges_`. After this call,
    // `adjacency_offsets_` has length `N + 1`, `adjacency_data_` has length
    // `2 * E`, and the slice `adjacency_data_[offsets[u]:offsets[u+1]]` is
    // exactly the adjacency of node `u`.
    //
    // Builds in three sequential passes:
    //  1. Count degree per node (one pass over `edges_`).
    //  2. Prefix-sum into `adjacency_offsets_` (one pass over nodes).
    //  3. Place each edge's two adjacency entries via a fill cursor copied
    //     from the offsets (one pass over `edges_`).
    //
    // All writes in pass (3) land in a single contiguous buffer, so cache
    // behavior is good; there are no per-node allocator calls. This is
    // typically 4-5x faster than the previous vector-of-vectors fill on
    // large grids.
    void rebuild_adjacency_from_edges() {
        const auto n_nodes = static_cast<std::size_t>(number_of_nodes_);
        const auto data_size = 2 * edges_.size();
        // Pass 1: count per-node degree into the offsets buffer (shifted by 1).
        adjacency_offsets_.assign(n_nodes + 1, 0);
        for (const auto &edge : edges_) {
            ++adjacency_offsets_[static_cast<std::size_t>(edge.first) + 1];
            ++adjacency_offsets_[static_cast<std::size_t>(edge.second) + 1];
        }
        // Pass 2: inclusive prefix sum turns degree-counts into slice starts.
        for (std::size_t node = 1; node <= n_nodes; ++node) {
            adjacency_offsets_[node] += adjacency_offsets_[node - 1];
        }
        // Pass 3: allocate the contiguous adjacency buffer.
        // `make_unique_for_overwrite` leaves trivially-default-constructible
        // `Adjacency` elements UNINITIALIZED — every slot is overwritten in
        // the fill pass below, so the zero-init that `std::vector::resize`
        // would do is pure overhead. On a 12M-edge grid this skips ~125 ms
        // of memset, though the underlying page-fault cost shifts to the
        // fill pass; net win is modest but real.
        static_assert(std::is_trivially_default_constructible_v<Adjacency>);
        adjacency_data_ = std::make_unique_for_overwrite<Adjacency[]>(data_size);
        adjacency_data_size_ = data_size;
        // Pass 4: `cursor[u]` is the next free slot in node `u`'s adjacency
        // range. Initialized from the slice starts and incremented per write.
        std::vector<std::uint64_t> cursor(adjacency_offsets_);
        Adjacency *const data = adjacency_data_.get();
        for (std::size_t index = 0; index < edges_.size(); ++index) {
            const auto &edge = edges_[index];
            const auto edge_id = static_cast<EdgeId>(index);
            data[cursor[static_cast<std::size_t>(edge.first)]++] =
                Adjacency{edge.second, edge_id};
            data[cursor[static_cast<std::size_t>(edge.second)]++] =
                Adjacency{edge.first, edge_id};
        }
        adjacency_dirty_ = false;
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
    void ensure_adjacency_built() const {
        if (adjacency_dirty_) {
            const_cast<UndirectedGraph *>(this)->rebuild_adjacency_from_edges();
        }
    }

    NodeId number_of_nodes_;
    std::vector<Edge> edges_;
    // CSR adjacency. `adjacency_offsets_` has length `N + 1`;
    // `adjacency_data_` has length `2 * number_of_edges()` after a rebuild.
    // The `mutable` qualifiers allow the lazy rebuild from a const reader
    // path — see `ensure_adjacency_built`. Writers (insert / bulk-edit
    // helpers) set `adjacency_dirty_ = true` and the rebuild happens on the
    // next read.
    mutable std::vector<std::uint64_t> adjacency_offsets_;
    // Heap-allocated CSR data buffer. Using `unique_ptr<T[]>` instead of
    // `std::vector<T>` lets `make_unique_for_overwrite` skip the
    // zero-initialization that `vector::resize` would force — a substantial
    // saving for 12 M-edge graphs where the buffer is ~384 MB and every
    // slot is overwritten in the fill pass anyway.
    mutable std::unique_ptr<Adjacency[]> adjacency_data_;
    mutable std::size_t adjacency_data_size_ = 0;
    mutable bool adjacency_dirty_ = true;
    std::unordered_map<Edge, EdgeId, detail::EdgeHash> edge_lookup_;
};

} // namespace bioimage_cpp::graph
