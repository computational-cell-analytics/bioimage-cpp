#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph {

class RegionAdjacencyGraph : public UndirectedGraph {
public:
    using UndirectedGraph::UndirectedGraph;

    explicit RegionAdjacencyGraph(
        const std::uint64_t number_of_nodes,
        std::vector<std::uint64_t> shape
    )
        : UndirectedGraph(number_of_nodes),
          shape_(std::move(shape)) {
    }

    [[nodiscard]] const std::vector<std::uint64_t> &shape() const {
        return shape_;
    }

private:
    std::vector<std::uint64_t> shape_;
};

namespace detail {

using Edge = UndirectedGraph::Edge;

struct EdgeHash {
    std::size_t operator()(const Edge &edge) const {
        const auto first = static_cast<std::size_t>(edge.first);
        const auto second = static_cast<std::size_t>(edge.second);
        return first ^ (second + 0x9e3779b97f4a7c15ULL + (first << 6U) + (first >> 2U));
    }
};

inline Edge edge_key(std::uint64_t u, std::uint64_t v) {
    if (v < u) {
        std::swap(u, v);
    }
    return {u, v};
}

template <class T>
std::uint64_t checked_label_to_node(const T value) {
    if constexpr (std::is_signed_v<T>) {
        if (value < 0) {
            throw std::invalid_argument("labels must not contain negative values");
        }
    }
    return static_cast<std::uint64_t>(value);
}

template <class T>
std::uint64_t max_label(const ConstArrayView<T> &labels) {
    const auto number_of_pixels = static_cast<std::size_t>(std::accumulate(
        labels.shape.begin(),
        labels.shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    ));
    std::uint64_t max_value = 0;
    for (std::size_t index = 0; index < number_of_pixels; ++index) {
        max_value = std::max(max_value, checked_label_to_node(labels.data[index]));
    }
    return max_value;
}

inline std::size_t normalize_thread_count(
    const std::size_t requested,
    const std::size_t number_of_work_items
) {
    if (number_of_work_items == 0) {
        return 1;
    }
    std::size_t n_threads = requested;
    if (n_threads == 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) {
            n_threads = 1;
        }
    }
    return std::max<std::size_t>(1, std::min(n_threads, number_of_work_items));
}

template <class T>
void add_edge_if_different(
    const T *data,
    const std::size_t first,
    const std::size_t second,
    std::unordered_set<Edge, EdgeHash> &edges
) {
    const auto u = checked_label_to_node(data[first]);
    const auto v = checked_label_to_node(data[second]);
    if (u != v) {
        edges.insert(edge_key(u, v));
    }
}

template <class T>
void scan_2d_chunk(
    const T *data,
    const std::size_t height,
    const std::size_t width,
    const std::size_t y_begin,
    const std::size_t y_end,
    std::unordered_set<Edge, EdgeHash> &edges
) {
    for (std::size_t y = y_begin; y < y_end; ++y) {
        const auto row_offset = y * width;
        for (std::size_t x = 0; x < width; ++x) {
            const auto pixel = row_offset + x;
            if (x + 1 < width) {
                add_edge_if_different(data, pixel, pixel + 1, edges);
            }
            if (y + 1 < height) {
                add_edge_if_different(data, pixel, pixel + width, edges);
            }
        }
    }
}

template <class T>
void scan_3d_chunk(
    const T *data,
    const std::size_t depth,
    const std::size_t height,
    const std::size_t width,
    const std::size_t z_begin,
    const std::size_t z_end,
    std::unordered_set<Edge, EdgeHash> &edges
) {
    const auto slice_size = height * width;
    for (std::size_t z = z_begin; z < z_end; ++z) {
        const auto slice_offset = z * slice_size;
        for (std::size_t y = 0; y < height; ++y) {
            const auto row_offset = slice_offset + y * width;
            for (std::size_t x = 0; x < width; ++x) {
                const auto pixel = row_offset + x;
                if (x + 1 < width) {
                    add_edge_if_different(data, pixel, pixel + 1, edges);
                }
                if (y + 1 < height) {
                    add_edge_if_different(data, pixel, pixel + width, edges);
                }
                if (z + 1 < depth) {
                    add_edge_if_different(data, pixel, pixel + slice_size, edges);
                }
            }
        }
    }
}

inline std::vector<Edge> merge_edge_sets(
    const std::vector<std::unordered_set<Edge, EdgeHash>> &per_thread_edges
) {
    std::unordered_set<Edge, EdgeHash> merged;
    for (const auto &edges : per_thread_edges) {
        merged.insert(edges.begin(), edges.end());
    }

    std::vector<Edge> sorted_edges(merged.begin(), merged.end());
    std::sort(sorted_edges.begin(), sorted_edges.end());
    return sorted_edges;
}

} // namespace detail

template <class T>
RegionAdjacencyGraph region_adjacency_graph(
    const ConstArrayView<T> &labels,
    const std::size_t number_of_threads
) {
    if (labels.ndim() != 2 && labels.ndim() != 3) {
        throw std::invalid_argument(
            "labels must be a 2D or 3D array, got ndim=" +
            std::to_string(labels.ndim())
        );
    }
    for (const auto axis_size : labels.shape) {
        if (axis_size <= 0) {
            throw std::invalid_argument("labels must not have empty dimensions");
        }
    }

    const auto max_node = detail::max_label(labels);
    if (max_node == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("maximum label is too large");
    }

    std::vector<std::uint64_t> shape;
    shape.reserve(labels.shape.size());
    for (const auto axis_size : labels.shape) {
        shape.push_back(static_cast<std::uint64_t>(axis_size));
    }

    const auto work_items = static_cast<std::size_t>(labels.shape[0]);
    const auto n_threads = detail::normalize_thread_count(number_of_threads, work_items);
    std::vector<std::unordered_set<detail::Edge, detail::EdgeHash>> per_thread_edges(n_threads);
    std::vector<std::thread> threads;
    threads.reserve(n_threads > 0 ? n_threads - 1 : 0);

    const auto run_chunk = [&](const std::size_t thread_id) {
        const auto begin = thread_id * work_items / n_threads;
        const auto end = (thread_id + 1) * work_items / n_threads;
        if (labels.ndim() == 2) {
            detail::scan_2d_chunk(
                labels.data,
                static_cast<std::size_t>(labels.shape[0]),
                static_cast<std::size_t>(labels.shape[1]),
                begin,
                end,
                per_thread_edges[thread_id]
            );
        } else {
            detail::scan_3d_chunk(
                labels.data,
                static_cast<std::size_t>(labels.shape[0]),
                static_cast<std::size_t>(labels.shape[1]),
                static_cast<std::size_t>(labels.shape[2]),
                begin,
                end,
                per_thread_edges[thread_id]
            );
        }
    };

    for (std::size_t thread_id = 1; thread_id < n_threads; ++thread_id) {
        threads.emplace_back(run_chunk, thread_id);
    }
    run_chunk(0);
    for (auto &thread : threads) {
        thread.join();
    }

    auto graph = RegionAdjacencyGraph(max_node + 1, std::move(shape));
    for (const auto edge : detail::merge_edge_sets(per_thread_edges)) {
        graph.insert_edge(edge.first, edge.second);
    }
    return graph;
}

} // namespace bioimage_cpp::graph
