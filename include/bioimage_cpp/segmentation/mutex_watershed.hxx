#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/union_find.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bioimage_cpp {

using MutexStorage = std::vector<std::unordered_set<std::uint64_t>>;

inline bool check_mutex(
    const std::uint64_t first,
    const std::uint64_t second,
    const MutexStorage &mutexes
) {
    const auto &first_mutexes = mutexes[first];
    const auto &second_mutexes = mutexes[second];
    if (first_mutexes.size() < second_mutexes.size()) {
        return first_mutexes.find(second) != first_mutexes.end();
    }
    return second_mutexes.find(first) != second_mutexes.end();
}

inline void insert_mutex(
    const std::uint64_t first,
    const std::uint64_t second,
    MutexStorage &mutexes
) {
    mutexes[first].insert(second);
    mutexes[second].insert(first);
}

inline void merge_mutexes(
    const std::uint64_t root_from,
    const std::uint64_t root_to,
    MutexStorage &mutexes
) {
    auto &mutexes_from = mutexes[root_from];
    auto &mutexes_to = mutexes[root_to];

    for (const auto other_root : mutexes_from) {
        auto &other_mutexes = mutexes[other_root];
        other_mutexes.erase(root_from);
        if (other_root != root_to) {
            other_mutexes.insert(root_to);
            mutexes_to.insert(other_root);
        }
    }
    mutexes_to.erase(root_from);
    mutexes_to.erase(root_to);
    mutexes_from.clear();
}

template <class T>
void mutex_watershed_grid(
    const ConstArrayView<T> &affinities,
    const ConstArrayView<std::uint8_t> &valid_edges,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::size_t number_of_attractive_channels,
    const ArrayView<std::uint64_t> &out
) {
    if (affinities.ndim() != 3 && affinities.ndim() != 4) {
        throw std::invalid_argument(
            "affinities must have shape (channels, y, x) or (channels, z, y, x), got ndim=" +
            std::to_string(affinities.ndim())
        );
    }
    if (offsets.empty()) {
        throw std::invalid_argument("offsets must not be empty");
    }
    if (valid_edges.shape != affinities.shape) {
        throw std::invalid_argument("valid_edges shape must match affinities shape");
    }

    const auto number_of_channels = static_cast<std::size_t>(affinities.shape[0]);
    const auto spatial_ndim = static_cast<std::size_t>(affinities.ndim() - 1);
    if (offsets.size() != number_of_channels) {
        throw std::invalid_argument(
            "offsets length must match affinities channel count, got offsets length=" +
            std::to_string(offsets.size()) + ", channels=" + std::to_string(number_of_channels)
        );
    }
    if (number_of_attractive_channels > number_of_channels) {
        throw std::invalid_argument("number_of_attractive_channels must be <= number of channels");
    }
    for (const auto &offset : offsets) {
        if (offset.size() != spatial_ndim) {
            throw std::invalid_argument(
                "each offset must have length matching the spatial ndim, got spatial ndim=" +
                std::to_string(spatial_ndim)
            );
        }
    }

    std::vector<std::ptrdiff_t> spatial_shape(
        affinities.shape.begin() + 1,
        affinities.shape.end()
    );
    if (out.shape != spatial_shape) {
        throw std::invalid_argument("out shape must match affinities spatial shape");
    }

    const auto number_of_nodes = static_cast<std::uint64_t>(std::accumulate(
        spatial_shape.begin(),
        spatial_shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    ));
    const auto spatial_strides = detail::c_order_strides(spatial_shape);

    std::vector<std::ptrdiff_t> offset_strides(number_of_channels, 0);
    for (std::size_t channel = 0; channel < number_of_channels; ++channel) {
        for (std::size_t axis = 0; axis < spatial_ndim; ++axis) {
            offset_strides[channel] += offsets[channel][axis] * spatial_strides[axis];
        }
    }

    const auto number_of_edges = number_of_nodes * number_of_channels;
    std::vector<std::uint64_t> edge_order(static_cast<std::size_t>(number_of_edges));
    std::iota(edge_order.begin(), edge_order.end(), std::uint64_t{0});
    std::sort(edge_order.begin(), edge_order.end(), [&](const std::uint64_t first, const std::uint64_t second) {
        const T first_weight = affinities.data[first];
        const T second_weight = affinities.data[second];
        if (first_weight == second_weight) {
            return first < second;
        }
        return first_weight > second_weight;
    });

    detail::UnionFind sets(static_cast<std::size_t>(number_of_nodes));
    MutexStorage mutexes(static_cast<std::size_t>(number_of_nodes));

    for (const auto edge_id : edge_order) {
        if (valid_edges.data[edge_id] == 0) {
            continue;
        }

        const auto channel = static_cast<std::size_t>(edge_id / number_of_nodes);
        const auto u = edge_id % number_of_nodes;
        if (!detail::is_valid_grid_edge(u, offsets[channel], spatial_shape, spatial_strides)) {
            continue;
        }

        const auto v_signed = static_cast<std::int64_t>(u) + static_cast<std::int64_t>(offset_strides[channel]);
        const auto v = static_cast<std::uint64_t>(v_signed);
        std::uint64_t root_u = sets.find(u);
        std::uint64_t root_v = sets.find(v);
        if (root_u == root_v || check_mutex(root_u, root_v, mutexes)) {
            continue;
        }

        const bool is_mutex_edge = channel >= number_of_attractive_channels;
        if (is_mutex_edge) {
            insert_mutex(root_u, root_v, mutexes);
        } else {
            const auto new_root = sets.unite_roots(root_u, root_v);
            const auto old_root = (new_root == root_u) ? root_v : root_u;
            merge_mutexes(old_root, new_root, mutexes);
        }
    }

    std::unordered_map<std::uint64_t, std::uint64_t> label_map;
    for (std::uint64_t node = 0; node < number_of_nodes; ++node) {
        const auto root = sets.find(node);
        auto found = label_map.find(root);
        if (found == label_map.end()) {
            const auto next_label = static_cast<std::uint64_t>(label_map.size() + 1);
            found = label_map.emplace(root, next_label).first;
        }
        out.data[node] = found->second;
    }
}

} // namespace bioimage_cpp
