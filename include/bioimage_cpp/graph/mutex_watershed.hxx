#pragma once

#include "bioimage_cpp/detail/mutex_storage.hxx"
#include "bioimage_cpp/detail/relabel.hxx"
#include "bioimage_cpp/detail/union_find.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace bioimage_cpp::graph {

// Mutex watershed clustering on an arbitrary undirected graph.
//
// `graph` defines the attractive edges with one cost per edge in `edge_costs`.
// `mutex_uvs` defines the long-range repulsive (mutex) edges as (u, v) pairs
// with one cost per edge in `mutex_costs`. Higher costs win — they are
// processed first when sorting jointly by descending weight.
//
// Returns dense node labels in [0, k) following first-occurrence order.
//
// This is a port of `compute_mws_clustering` from the affogato library,
// adapted to bioimage-cpp's UndirectedGraph and detail/ primitives.
//
// Templated on the weight type. Concrete instantiations for `float` and
// `double` are provided by the binding layer; other floating types are
// supported but must be instantiated explicitly.
template <class WeightT>
std::vector<std::uint64_t> mutex_watershed_clustering(
    const UndirectedGraph &graph,
    const std::vector<WeightT> &edge_costs,
    const std::vector<std::array<std::uint64_t, 2>> &mutex_uvs,
    const std::vector<WeightT> &mutex_costs
) {
    const auto number_of_edges = static_cast<std::size_t>(graph.number_of_edges());
    if (edge_costs.size() != number_of_edges) {
        throw std::invalid_argument(
            "edge_costs size must match graph.number_of_edges(), got edge_costs size=" +
            std::to_string(edge_costs.size()) +
            ", number_of_edges=" + std::to_string(number_of_edges)
        );
    }
    if (mutex_costs.size() != mutex_uvs.size()) {
        throw std::invalid_argument(
            "mutex_costs size must match mutex_uvs size, got mutex_costs size=" +
            std::to_string(mutex_costs.size()) +
            ", mutex_uvs size=" + std::to_string(mutex_uvs.size())
        );
    }

    const auto number_of_nodes = static_cast<std::size_t>(graph.number_of_nodes());
    const auto number_of_mutex = mutex_uvs.size();

    for (std::size_t index = 0; index < number_of_mutex; ++index) {
        const auto u = mutex_uvs[index][0];
        const auto v = mutex_uvs[index][1];
        if (u >= number_of_nodes || v >= number_of_nodes) {
            throw std::invalid_argument(
                "mutex_uvs endpoints must be < number_of_nodes, got u=" +
                std::to_string(u) + ", v=" + std::to_string(v) +
                ", number_of_nodes=" + std::to_string(number_of_nodes)
            );
        }
    }

    struct WeightedEdge {
        WeightT weight;
        std::uint64_t index;
        bool is_mutex;
    };

    std::vector<WeightedEdge> edge_order;
    edge_order.reserve(number_of_edges + number_of_mutex);
    for (std::size_t index = 0; index < number_of_edges; ++index) {
        edge_order.push_back(
            WeightedEdge{edge_costs[index], static_cast<std::uint64_t>(index), false}
        );
    }
    for (std::size_t index = 0; index < number_of_mutex; ++index) {
        edge_order.push_back(
            WeightedEdge{mutex_costs[index], static_cast<std::uint64_t>(index), true}
        );
    }

    std::sort(edge_order.begin(), edge_order.end(), [](const auto &first, const auto &second) {
        if (first.weight != second.weight) {
            return first.weight > second.weight;
        }
        if (first.is_mutex != second.is_mutex) {
            return !first.is_mutex;
        }
        return first.index < second.index;
    });

    bioimage_cpp::detail::UnionFind sets(number_of_nodes);
    MutexStorage mutexes(number_of_nodes);

    for (const auto &edge : edge_order) {
        std::uint64_t u;
        std::uint64_t v;
        if (edge.is_mutex) {
            const auto &pair = mutex_uvs[static_cast<std::size_t>(edge.index)];
            u = pair[0];
            v = pair[1];
        } else {
            const auto uv = graph.uv(edge.index);
            u = uv.first;
            v = uv.second;
        }
        if (u == v) {
            continue;
        }

        const auto root_u = sets.find(u);
        const auto root_v = sets.find(v);
        if (root_u == root_v) {
            continue;
        }

        if (edge.is_mutex) {
            insert_mutex(root_u, root_v, mutexes);
        } else {
            if (check_mutex(root_u, root_v, mutexes)) {
                continue;
            }
            const auto new_root = sets.unite_roots(root_u, root_v);
            const auto old_root = (new_root == root_u) ? root_v : root_u;
            merge_mutexes(old_root, new_root, mutexes);
        }
    }

    std::vector<std::uint64_t> roots(number_of_nodes);
    for (std::size_t node = 0; node < number_of_nodes; ++node) {
        roots[node] = sets.find(static_cast<std::uint64_t>(node));
    }
    return bioimage_cpp::detail::dense_relabel(roots);
}

} // namespace bioimage_cpp::graph
