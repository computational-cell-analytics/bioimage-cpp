#pragma once

#include "bioimage_cpp/graph/contraction_options.hxx"
#include "bioimage_cpp/graph/detail/contraction_topology.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"
#include "bioimage_cpp/util/union_find.hxx"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace bioimage_cpp::graph {

enum class Reduction {
    Sum,
    Mean,
    Minimum,
    Maximum,
};

template <std::floating_point T>
struct MaterializedValues {
    std::string name;
    std::vector<T> values;
    std::vector<std::size_t> shape;
};

using MaterializedValueMap = std::variant<
    MaterializedValues<float>,
    MaterializedValues<double>
>;

struct MaterializedContraction {
    UndirectedGraph graph;
    std::vector<MaterializedValueMap> node_values;
    std::vector<MaterializedValueMap> edge_values;
    std::vector<std::int64_t> node_mapping;
    std::vector<std::int64_t> edge_mapping;
};

template <std::floating_point T>
class ReducedValueMap {
public:
    ReducedValueMap(
        std::string name,
        const std::span<const T> values,
        std::vector<std::size_t> component_shape,
        const std::size_t item_count,
        const Reduction reduction
    )
        : name_(std::move(name)),
          component_shape_(std::move(component_shape)),
          component_size_(shape_size(component_shape_)),
          values_(values.begin(), values.end()),
          counts_(reduction == Reduction::Mean ? item_count : 0, 1),
          mean_sums_(reduction == Reduction::Mean ? values.size() : 0),
          reduction_(reduction) {
        if (values_.size() != checked_product(item_count, component_size_)) {
            throw std::invalid_argument(
                "values size does not match the item count and component shape"
            );
        }
        if (reduction_ == Reduction::Mean) {
            std::transform(
                values_.begin(),
                values_.end(),
                mean_sums_.begin(),
                [](const T value) {
                    return static_cast<long double>(value);
                }
            );
        }
    }

    [[nodiscard]] const std::string &name() const {
        return name_;
    }

    [[nodiscard]] const std::vector<std::size_t> &component_shape() const {
        return component_shape_;
    }

    [[nodiscard]] std::size_t component_size() const {
        return component_size_;
    }

    [[nodiscard]] Reduction reduction() const {
        return reduction_;
    }

    void reduce(const std::uint64_t kept, const std::uint64_t removed) {
        const auto kept_index = static_cast<std::size_t>(kept);
        const auto removed_index = static_cast<std::size_t>(removed);
        if (kept_index == removed_index) {
            return;
        }
        const auto kept_offset = checked_product(kept_index, component_size_);
        const auto removed_offset = checked_product(removed_index, component_size_);

        if (reduction_ == Reduction::Mean) {
            const auto kept_count = counts_[kept_index];
            const auto removed_count = counts_[removed_index];
            const auto total = kept_count + removed_count;
            const auto total_weight = static_cast<long double>(total);
            for (std::size_t component = 0; component < component_size_; ++component) {
                auto &sum = mean_sums_[kept_offset + component];
                sum += mean_sums_[removed_offset + component];
                values_[kept_offset + component] =
                    static_cast<T>(sum / total_weight);
            }
            counts_[kept_index] = total;
            return;
        }

        for (std::size_t component = 0; component < component_size_; ++component) {
            auto &a = values_[kept_offset + component];
            const auto b = values_[removed_offset + component];
            if (std::isnan(a) || std::isnan(b)) {
                a = std::numeric_limits<T>::quiet_NaN();
            } else if (reduction_ == Reduction::Sum) {
                a += b;
            } else if (reduction_ == Reduction::Minimum) {
                a = std::min(a, b);
            } else {
                a = std::max(a, b);
            }
        }
    }

    [[nodiscard]] MaterializedValues<T> gather(
        const std::span<const std::uint64_t> item_ids
    ) const {
        std::vector<std::size_t> shape;
        shape.reserve(component_shape_.size() + 1);
        shape.push_back(item_ids.size());
        shape.insert(shape.end(), component_shape_.begin(), component_shape_.end());

        std::vector<T> output;
        output.reserve(checked_product(item_ids.size(), component_size_));
        for (const auto item : item_ids) {
            const auto offset =
                checked_product(static_cast<std::size_t>(item), component_size_);
            output.insert(
                output.end(),
                values_.begin() + static_cast<std::ptrdiff_t>(offset),
                values_.begin() + static_cast<std::ptrdiff_t>(offset + component_size_)
            );
        }
        return {name_, std::move(output), std::move(shape)};
    }

    [[nodiscard]] MaterializedValues<T> value(const std::uint64_t item) const {
        std::vector<std::size_t> shape = component_shape_;
        const auto offset =
            checked_product(static_cast<std::size_t>(item), component_size_);
        std::vector<T> output(
            values_.begin() + static_cast<std::ptrdiff_t>(offset),
            values_.begin() + static_cast<std::ptrdiff_t>(offset + component_size_)
        );
        return {name_, std::move(output), std::move(shape)};
    }

private:
    static std::size_t checked_product(
        const std::size_t first,
        const std::size_t second
    ) {
        if (second != 0 &&
            first > std::numeric_limits<std::size_t>::max() / second) {
            throw std::overflow_error("value-map shape overflows size_t");
        }
        return first * second;
    }

    static std::size_t shape_size(const std::vector<std::size_t> &shape) {
        std::size_t size = 1;
        for (const auto extent : shape) {
            size = checked_product(size, extent);
        }
        return size;
    }

    std::string name_;
    std::vector<std::size_t> component_shape_;
    std::size_t component_size_;
    std::vector<T> values_;
    std::vector<std::uint64_t> counts_;
    std::vector<long double> mean_sums_;
    Reduction reduction_;
};

using ReducedValueMapVariant = std::variant<
    ReducedValueMap<float>,
    ReducedValueMap<double>
>;

// Stateful graph contraction with named floating-point value maps. The input
// graph is simple. Parallel edges created by mutations can merge immediately or
// remain separate until materialization.
class ContractionGraph {
public:
    using NodeId = std::uint64_t;
    using EdgeId = std::uint64_t;
    using Edge = UndirectedGraph::Edge;

    explicit ContractionGraph(
        const UndirectedGraph &graph,
        const ParallelEdgePolicy parallel_edge_policy = ParallelEdgePolicy::Merge
    )
        : topology_(graph, parallel_edge_policy),
          node_sets_(static_cast<std::size_t>(graph.number_of_nodes())),
          edge_parent_(static_cast<std::size_t>(graph.number_of_edges())),
          edge_state_(
              static_cast<std::size_t>(graph.number_of_edges()),
              EdgeState::Active
          ),
          parallel_edge_policy_(parallel_edge_policy) {
        initialize_edge_parents();
    }

    [[nodiscard]] std::size_t number_of_nodes() const {
        return topology_.number_of_nodes();
    }

    [[nodiscard]] std::size_t number_of_edges() const {
        return topology_.number_of_edges();
    }

    [[nodiscard]] std::vector<NodeId> nodes() const {
        return topology_.active_nodes();
    }

    [[nodiscard]] std::vector<EdgeId> edges() const {
        return topology_.active_edges();
    }

    [[nodiscard]] Edge uv(const EdgeId edge) const {
        return topology_.uv(edge);
    }

    [[nodiscard]] std::int64_t find_edge(const NodeId u, const NodeId v) const {
        return topology_.find_edge(u, v);
    }

    [[nodiscard]] std::vector<EdgeId> find_edges(
        const NodeId u,
        const NodeId v
    ) const {
        return topology_.find_edges(u, v);
    }

    [[nodiscard]] ParallelEdgePolicy parallel_edge_policy() const {
        return parallel_edge_policy_;
    }

    [[nodiscard]] const std::vector<detail::ContractionAdjacency> &node_adjacency(
        const NodeId node
    ) const {
        return topology_.node_adjacency(node);
    }

    [[nodiscard]] std::size_t degree(const NodeId node) const {
        return topology_.degree(node);
    }

    [[nodiscard]] bool can_suppress_node(const NodeId node) const {
        return topology_.can_suppress_node(node);
    }

    [[nodiscard]] bool node_active(const NodeId node) const {
        return topology_.node_active(node);
    }

    [[nodiscard]] bool edge_active(const EdgeId edge) const {
        return topology_.edge_active(edge);
    }

    NodeId representative(const NodeId original_node) {
        if (original_node >= node_sets_.size()) {
            throw std::out_of_range("original_node is out of range");
        }
        return node_sets_.find(original_node);
    }

    template <std::floating_point T>
    void add_node_values(
        std::string name,
        const std::span<const T> values,
        std::vector<std::size_t> component_shape,
        const Reduction reduction
    ) {
        require_registration_allowed(name, node_values_, "node");
        node_values_.emplace_back(
            ReducedValueMap<T>(
                std::move(name),
                values,
                std::move(component_shape),
                topology_.node_capacity(),
                reduction
            )
        );
    }

    template <std::floating_point T>
    void add_edge_values(
        std::string name,
        const std::span<const T> values,
        std::vector<std::size_t> component_shape,
        const Reduction reduction
    ) {
        require_registration_allowed(name, edge_values_, "edge");
        edge_values_.emplace_back(
            ReducedValueMap<T>(
                std::move(name),
                values,
                std::move(component_shape),
                topology_.edge_capacity(),
                reduction
            )
        );
    }

    NodeId contract_edge(
        const EdgeId edge,
        const std::optional<NodeId> keep_node = std::nullopt
    ) {
        mutation_started_ = true;
        const auto kept = topology_.contract_edge(edge, keep_node, change_);
        apply_node_reduction(change_.kept_node, change_.removed_node);
        node_sets_.merge_to(change_.kept_node, change_.removed_node);
        apply_removed_edge_states(EdgeState::Internal);
        apply_edge_folds();
        return kept;
    }

    void erase_edge(const EdgeId edge) {
        mutation_started_ = true;
        topology_.erase_edge(edge, change_);
        set_edge_state(edge, EdgeState::Deleted);
    }

    EdgeId suppress_node(const NodeId node) {
        mutation_started_ = true;
        const auto replacement = topology_.suppress_node(node, change_);
        apply_edge_folds();
        return replacement;
    }

    [[nodiscard]] const ReducedValueMapVariant &node_value_map(
        const std::string &name
    ) const {
        return find_value_map(node_values_, name, "node");
    }

    [[nodiscard]] const ReducedValueMapVariant &edge_value_map(
        const std::string &name
    ) const {
        return find_value_map(edge_values_, name, "edge");
    }

    [[nodiscard]] MaterializedValueMap node_value(
        const std::string &name,
        const NodeId node
    ) const {
        if (!topology_.node_active(node)) {
            throw std::invalid_argument("node is inactive");
        }
        return std::visit([node](const auto &map) {
            return MaterializedValueMap(map.value(node));
        }, node_value_map(name));
    }

    [[nodiscard]] MaterializedValueMap edge_value(
        const std::string &name,
        const EdgeId edge
    ) const {
        if (!topology_.edge_active(edge)) {
            throw std::invalid_argument("edge is inactive");
        }
        return std::visit([edge](const auto &map) {
            return MaterializedValueMap(map.value(edge));
        }, edge_value_map(name));
    }

    [[nodiscard]] MaterializedValueMap active_node_values(
        const std::string &name
    ) const {
        const auto ids = topology_.active_nodes();
        return std::visit([&ids](const auto &map) {
            return MaterializedValueMap(map.gather(ids));
        }, node_value_map(name));
    }

    [[nodiscard]] MaterializedValueMap active_edge_values(
        const std::string &name
    ) const {
        const auto ids = topology_.active_edges();
        return std::visit([&ids](const auto &map) {
            return MaterializedValueMap(map.gather(ids));
        }, edge_value_map(name));
    }

    // Build an independent simple-graph snapshot. Parallel active edges fold
    // only in the snapshot. Multiple input items can map to one output id.
    MaterializedContraction materialize() {
        const auto active_nodes = topology_.active_nodes();
        std::vector<std::int64_t> dense_node(topology_.node_capacity(), -1);
        for (std::size_t dense = 0; dense < active_nodes.size(); ++dense) {
            dense_node[static_cast<std::size_t>(active_nodes[dense])] =
                static_cast<std::int64_t>(dense);
        }

        struct MaterializedEdge {
            Edge uv;
            EdgeId stable_edge;
        };
        std::vector<MaterializedEdge> materialized_edges;
        materialized_edges.reserve(topology_.number_of_edges());
        for (const auto edge : topology_.active_edges()) {
            const auto uv = topology_.uv(edge);
            const auto dense_u =
                static_cast<NodeId>(dense_node[static_cast<std::size_t>(uv.first)]);
            const auto dense_v =
                static_cast<NodeId>(dense_node[static_cast<std::size_t>(uv.second)]);
            if (dense_u == dense_v) {
                throw std::runtime_error(
                    "cannot materialize a graph with active self edges"
                );
            }
            materialized_edges.push_back(
                {{std::min(dense_u, dense_v), std::max(dense_u, dense_v)}, edge}
            );
        }
        std::sort(
            materialized_edges.begin(),
            materialized_edges.end(),
            [](const auto &a, const auto &b) {
                return a.uv != b.uv
                    ? a.uv < b.uv
                    : a.stable_edge < b.stable_edge;
            }
        );

        std::vector<Edge> output_edges;
        std::vector<EdgeId> active_edges;
        std::vector<std::pair<EdgeId, EdgeId>> snapshot_folds;
        output_edges.reserve(materialized_edges.size());
        active_edges.reserve(materialized_edges.size());
        snapshot_folds.reserve(materialized_edges.size());
        std::vector<std::int64_t> dense_edge(topology_.edge_capacity(), -1);
        for (const auto &edge : materialized_edges) {
            if (output_edges.empty() || output_edges.back() != edge.uv) {
                output_edges.push_back(edge.uv);
                active_edges.push_back(edge.stable_edge);
            } else {
                snapshot_folds.emplace_back(active_edges.back(), edge.stable_edge);
            }
            dense_edge[static_cast<std::size_t>(edge.stable_edge)] =
                static_cast<std::int64_t>(output_edges.size() - 1);
        }

        auto graph = UndirectedGraph::from_sorted_unique_edges(
            static_cast<NodeId>(active_nodes.size()),
            std::move(output_edges),
            true
        );

        std::vector<std::int64_t> node_mapping(topology_.node_capacity(), -1);
        for (NodeId original = 0; original < topology_.node_capacity(); ++original) {
            const auto root = node_sets_.find(original);
            if (topology_.node_active(root)) {
                node_mapping[static_cast<std::size_t>(original)] =
                    dense_node[static_cast<std::size_t>(root)];
            }
        }

        std::vector<std::int64_t> edge_mapping(topology_.edge_capacity(), -1);
        for (EdgeId original = 0; original < topology_.edge_capacity(); ++original) {
            const auto root = edge_root(original);
            if (edge_state_[static_cast<std::size_t>(root)] == EdgeState::Active &&
                topology_.edge_active(root)) {
                edge_mapping[static_cast<std::size_t>(original)] =
                    dense_edge[static_cast<std::size_t>(root)];
            }
        }

        std::vector<MaterializedValueMap> node_values;
        node_values.reserve(node_values_.size());
        for (const auto &map : node_values_) {
            node_values.push_back(std::visit(
                [&active_nodes](const auto &typed) {
                    return MaterializedValueMap(typed.gather(active_nodes));
                },
                map
            ));
        }

        std::vector<MaterializedValueMap> edge_values;
        edge_values.reserve(edge_values_.size());
        for (const auto &map : edge_values_) {
            edge_values.push_back(std::visit(
                [&active_edges, &snapshot_folds](const auto &typed) {
                    auto snapshot = typed;
                    for (const auto &[kept, removed] : snapshot_folds) {
                        snapshot.reduce(kept, removed);
                    }
                    return MaterializedValueMap(snapshot.gather(active_edges));
                },
                map
            ));
        }

        return {
            std::move(graph),
            std::move(node_values),
            std::move(edge_values),
            std::move(node_mapping),
            std::move(edge_mapping),
        };
    }

private:
    enum class EdgeState : unsigned char {
        Active,
        Internal,
        Deleted,
    };

    template <class Function>
    static decltype(auto) visit_value(
        const ReducedValueMapVariant &map,
        Function &&function
    ) {
        return std::visit(std::forward<Function>(function), map);
    }

    static const std::string &map_name(const ReducedValueMapVariant &map) {
        return visit_value(map, [](const auto &typed) -> const std::string & {
            return typed.name();
        });
    }

    static const ReducedValueMapVariant &find_value_map(
        const std::vector<ReducedValueMapVariant> &maps,
        const std::string &name,
        const char *kind
    ) {
        for (const auto &map : maps) {
            if (map_name(map) == name) {
                return map;
            }
        }
        throw std::invalid_argument(
            std::string("unknown ") + kind + " value map: " + name
        );
    }

    void require_registration_allowed(
        const std::string &name,
        const std::vector<ReducedValueMapVariant> &maps,
        const char *kind
    ) const {
        if (mutation_started_) {
            throw std::invalid_argument(
                "value maps must be registered before the first graph mutation"
            );
        }
        if (name.empty()) {
            throw std::invalid_argument("value map name must not be empty");
        }
        for (const auto &map : maps) {
            if (map_name(map) == name) {
                throw std::invalid_argument(
                    std::string(kind) + " value map already exists: " + name
                );
            }
        }
    }

    void initialize_edge_parents() {
        for (EdgeId edge = 0; edge < edge_parent_.size(); ++edge) {
            edge_parent_[static_cast<std::size_t>(edge)] = edge;
        }
    }

    EdgeId edge_root(EdgeId edge) {
        auto current = static_cast<std::size_t>(edge);
        while (edge_parent_[current] != current) {
            edge_parent_[current] =
                edge_parent_[static_cast<std::size_t>(edge_parent_[current])];
            current = static_cast<std::size_t>(edge_parent_[current]);
        }
        return static_cast<EdgeId>(current);
    }

    void set_edge_state(const EdgeId edge, const EdgeState state) {
        const auto root = edge_root(edge);
        edge_state_[static_cast<std::size_t>(root)] = state;
    }

    void apply_removed_edge_states(const EdgeState state) {
        for (const auto edge : change_.deactivated_edges) {
            const auto folded = std::any_of(
                change_.folded_edges.begin(),
                change_.folded_edges.end(),
                [edge](const auto &fold) {
                    return fold.removed_edge == edge;
                }
            );
            if (!folded) {
                set_edge_state(edge, state);
            }
        }
    }

    void apply_node_reduction(const NodeId kept, const NodeId removed) {
        for (auto &map : node_values_) {
            std::visit(
                [kept, removed](auto &typed) {
                    typed.reduce(kept, removed);
                },
                map
            );
        }
    }

    void apply_edge_folds() {
        for (const auto &fold : change_.folded_edges) {
            for (auto &map : edge_values_) {
                std::visit(
                    [&fold](auto &typed) {
                        typed.reduce(fold.kept_edge, fold.removed_edge);
                    },
                    map
                );
            }
            const auto kept_root = edge_root(fold.kept_edge);
            const auto removed_root = edge_root(fold.removed_edge);
            if (kept_root != removed_root) {
                edge_parent_[static_cast<std::size_t>(removed_root)] = kept_root;
            }
        }
    }

    detail::ContractionTopology topology_;
    util::UnionFind node_sets_;
    std::vector<EdgeId> edge_parent_;
    std::vector<EdgeState> edge_state_;
    std::vector<ReducedValueMapVariant> node_values_;
    std::vector<ReducedValueMapVariant> edge_values_;
    detail::ContractionChange change_;
    ParallelEdgePolicy parallel_edge_policy_ = ParallelEdgePolicy::Merge;
    bool mutation_started_ = false;
};

} // namespace bioimage_cpp::graph
