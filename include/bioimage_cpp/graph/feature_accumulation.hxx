#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/graph/region_adjacency_graph.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace bioimage_cpp::graph {

namespace detail_features {

struct SimpleStats {
    double sum = 0.0;
    std::uint64_t count = 0;

    void add(const double value) {
        sum += value;
        ++count;
    }

    void merge(const SimpleStats &other) {
        sum += other.sum;
        count += other.count;
    }
};

struct ComplexStats {
    double sum = 0.0;
    double sum_of_squares = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    std::uint64_t count = 0;
    std::vector<double> values;

    void add(const double value) {
        sum += value;
        sum_of_squares += value * value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        ++count;
        values.push_back(value);
    }

    void merge(ComplexStats &other) {
        sum += other.sum;
        sum_of_squares += other.sum_of_squares;
        minimum = std::min(minimum, other.minimum);
        maximum = std::max(maximum, other.maximum);
        count += other.count;
        values.insert(values.end(), other.values.begin(), other.values.end());
        other.values.clear();
    }
};

inline double percentile(std::vector<double> &values, const double percentile) {
    if (values.empty()) {
        return 0.0;
    }
    if (values.size() == 1) {
        return values.front();
    }

    const double clipped = std::clamp(percentile, 0.0, 100.0);
    const double position = clipped * static_cast<double>(values.size() - 1) / 100.0;
    const auto lower_index = static_cast<std::size_t>(std::floor(position));
    const auto upper_index = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower_index);

    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(lower_index), values.end());
    const double lower = values[lower_index];
    if (upper_index == lower_index) {
        return lower;
    }
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(upper_index), values.end());
    const double upper = values[upper_index];
    return (1.0 - weight) * lower + weight * upper;
}

inline std::size_t number_of_pixels(const std::vector<std::ptrdiff_t> &shape) {
    return static_cast<std::size_t>(std::accumulate(
        shape.begin(),
        shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    ));
}

inline void require_same_spatial_shape(
    const std::vector<std::ptrdiff_t> &labels_shape,
    const std::vector<std::ptrdiff_t> &data_shape,
    const char *argument_name
) {
    if (labels_shape != data_shape) {
        throw std::invalid_argument(
            std::string(argument_name) + " shape must match labels shape"
        );
    }
}

template <class LabelT>
std::uint64_t label_at(const LabelT *labels, const std::size_t index) {
    return detail::checked_label_to_node(labels[index]);
}

inline std::int64_t edge_for_labels(
    const RegionAdjacencyGraph &rag,
    const std::uint64_t u,
    const std::uint64_t v
) {
    if (u == v) {
        return -1;
    }
    return rag.find_edge(u, v);
}

inline std::vector<std::ptrdiff_t> c_order_strides(const std::vector<std::ptrdiff_t> &shape) {
    std::vector<std::ptrdiff_t> strides(shape.size(), 1);
    for (std::ptrdiff_t axis = static_cast<std::ptrdiff_t>(shape.size()) - 2; axis >= 0; --axis) {
        strides[static_cast<std::size_t>(axis)] =
            strides[static_cast<std::size_t>(axis + 1)] * shape[static_cast<std::size_t>(axis + 1)];
    }
    return strides;
}

template <class LabelT, class ValueT, class Stats>
void scan_edge_map_2d_chunk(
    const RegionAdjacencyGraph &rag,
    const LabelT *labels,
    const ValueT *edge_map,
    const std::size_t height,
    const std::size_t width,
    const std::size_t y_begin,
    const std::size_t y_end,
    std::vector<Stats> &stats
) {
    for (std::size_t y = y_begin; y < y_end; ++y) {
        const auto row_offset = y * width;
        for (std::size_t x = 0; x < width; ++x) {
            const auto pixel = row_offset + x;
            const auto u = label_at(labels, pixel);
            if (x + 1 < width) {
                const auto neighbor = pixel + 1;
                const auto edge = edge_for_labels(rag, u, label_at(labels, neighbor));
                if (edge >= 0) {
                    stats[static_cast<std::size_t>(edge)].add(0.5 * (edge_map[pixel] + edge_map[neighbor]));
                }
            }
            if (y + 1 < height) {
                const auto neighbor = pixel + width;
                const auto edge = edge_for_labels(rag, u, label_at(labels, neighbor));
                if (edge >= 0) {
                    stats[static_cast<std::size_t>(edge)].add(0.5 * (edge_map[pixel] + edge_map[neighbor]));
                }
            }
        }
    }
}

template <class LabelT, class ValueT, class Stats>
void scan_edge_map_3d_chunk(
    const RegionAdjacencyGraph &rag,
    const LabelT *labels,
    const ValueT *edge_map,
    const std::size_t depth,
    const std::size_t height,
    const std::size_t width,
    const std::size_t z_begin,
    const std::size_t z_end,
    std::vector<Stats> &stats
) {
    const auto slice_size = height * width;
    for (std::size_t z = z_begin; z < z_end; ++z) {
        const auto slice_offset = z * slice_size;
        for (std::size_t y = 0; y < height; ++y) {
            const auto row_offset = slice_offset + y * width;
            for (std::size_t x = 0; x < width; ++x) {
                const auto pixel = row_offset + x;
                const auto u = label_at(labels, pixel);
                if (x + 1 < width) {
                    const auto neighbor = pixel + 1;
                    const auto edge = edge_for_labels(rag, u, label_at(labels, neighbor));
                    if (edge >= 0) {
                        stats[static_cast<std::size_t>(edge)].add(0.5 * (edge_map[pixel] + edge_map[neighbor]));
                    }
                }
                if (y + 1 < height) {
                    const auto neighbor = pixel + width;
                    const auto edge = edge_for_labels(rag, u, label_at(labels, neighbor));
                    if (edge >= 0) {
                        stats[static_cast<std::size_t>(edge)].add(0.5 * (edge_map[pixel] + edge_map[neighbor]));
                    }
                }
                if (z + 1 < depth) {
                    const auto neighbor = pixel + slice_size;
                    const auto edge = edge_for_labels(rag, u, label_at(labels, neighbor));
                    if (edge >= 0) {
                        stats[static_cast<std::size_t>(edge)].add(0.5 * (edge_map[pixel] + edge_map[neighbor]));
                    }
                }
            }
        }
    }
}

inline bool valid_offset_target(
    const std::uint64_t node,
    const std::vector<std::ptrdiff_t> &offset,
    const std::vector<std::ptrdiff_t> &shape,
    const std::vector<std::ptrdiff_t> &strides,
    std::uint64_t &target
) {
    std::int64_t target_signed = static_cast<std::int64_t>(node);
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        const auto coord = static_cast<std::ptrdiff_t>(node / static_cast<std::uint64_t>(strides[axis])) % shape[axis];
        const auto neighbor = coord + offset[axis];
        if (neighbor < 0 || neighbor >= shape[axis]) {
            return false;
        }
        target_signed += static_cast<std::int64_t>(offset[axis] * strides[axis]);
    }
    target = static_cast<std::uint64_t>(target_signed);
    return true;
}

template <class LabelT, class ValueT, class Stats>
void scan_affinity_chunk(
    const RegionAdjacencyGraph &rag,
    const LabelT *labels,
    const ValueT *affinities,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::vector<std::ptrdiff_t> &shape,
    const std::size_t node_begin,
    const std::size_t node_end,
    std::vector<Stats> &stats
) {
    const auto spatial_strides = c_order_strides(shape);
    const auto number_of_nodes = static_cast<std::uint64_t>(number_of_pixels(shape));
    for (std::size_t channel = 0; channel < offsets.size(); ++channel) {
        const auto channel_offset = static_cast<std::uint64_t>(channel) * number_of_nodes;
        for (std::uint64_t node = node_begin; node < node_end; ++node) {
            std::uint64_t target = 0;
            if (!valid_offset_target(node, offsets[channel], shape, spatial_strides, target)) {
                continue;
            }
            const auto edge = edge_for_labels(rag, label_at(labels, node), label_at(labels, target));
            if (edge >= 0) {
                stats[static_cast<std::size_t>(edge)].add(affinities[channel_offset + node]);
            }
        }
    }
}

template <class Stats>
std::vector<Stats> merge_stats(
    std::vector<std::vector<Stats>> &per_thread_stats,
    const std::size_t number_of_edges
) {
    std::vector<Stats> merged(number_of_edges);
    for (auto &thread_stats : per_thread_stats) {
        for (std::size_t edge = 0; edge < number_of_edges; ++edge) {
            merged[edge].merge(thread_stats[edge]);
        }
    }
    return merged;
}

template <>
inline std::vector<SimpleStats> merge_stats(
    std::vector<std::vector<SimpleStats>> &per_thread_stats,
    const std::size_t number_of_edges
) {
    std::vector<SimpleStats> merged(number_of_edges);
    for (const auto &thread_stats : per_thread_stats) {
        for (std::size_t edge = 0; edge < number_of_edges; ++edge) {
            merged[edge].merge(thread_stats[edge]);
        }
    }
    return merged;
}

inline void write_simple_features(
    const std::vector<SimpleStats> &stats,
    const ArrayView<double> &out
) {
    for (std::size_t edge = 0; edge < stats.size(); ++edge) {
        const auto &edge_stats = stats[edge];
        const auto offset = 2 * edge;
        out.data[offset] = edge_stats.count == 0 ? 0.0 : edge_stats.sum / static_cast<double>(edge_stats.count);
        out.data[offset + 1] = static_cast<double>(edge_stats.count);
    }
}

inline void write_complex_features(
    std::vector<ComplexStats> &stats,
    const ArrayView<double> &out
) {
    for (std::size_t edge = 0; edge < stats.size(); ++edge) {
        auto &edge_stats = stats[edge];
        const auto offset = 12 * edge;
        if (edge_stats.count == 0) {
            for (std::size_t feature = 0; feature < 12; ++feature) {
                out.data[offset + feature] = 0.0;
            }
            continue;
        }

        const auto count = static_cast<double>(edge_stats.count);
        const auto mean = edge_stats.sum / count;
        const auto variance = std::max(0.0, edge_stats.sum_of_squares / count - mean * mean);
        out.data[offset] = mean;
        out.data[offset + 1] = percentile(edge_stats.values, 50.0);
        out.data[offset + 2] = std::sqrt(variance);
        out.data[offset + 3] = edge_stats.minimum;
        out.data[offset + 4] = edge_stats.maximum;
        out.data[offset + 5] = percentile(edge_stats.values, 5.0);
        out.data[offset + 6] = percentile(edge_stats.values, 10.0);
        out.data[offset + 7] = percentile(edge_stats.values, 25.0);
        out.data[offset + 8] = percentile(edge_stats.values, 75.0);
        out.data[offset + 9] = percentile(edge_stats.values, 90.0);
        out.data[offset + 10] = percentile(edge_stats.values, 95.0);
        out.data[offset + 11] = count;
    }
}

} // namespace detail_features

template <class LabelT, class ValueT>
void accumulate_edge_map_features(
    const RegionAdjacencyGraph &rag,
    const ConstArrayView<LabelT> &labels,
    const ConstArrayView<ValueT> &edge_map,
    const bool compute_complex_features,
    const std::size_t number_of_threads,
    const ArrayView<double> &out
) {
    if (labels.ndim() != 2 && labels.ndim() != 3) {
        throw std::invalid_argument("labels must be a 2D or 3D array");
    }
    detail_features::require_same_spatial_shape(labels.shape, edge_map.shape, "edge_map");

    const auto expected_features = compute_complex_features ? 12 : 2;
    if (out.shape != std::vector<std::ptrdiff_t>{static_cast<std::ptrdiff_t>(rag.number_of_edges()), expected_features}) {
        throw std::invalid_argument("out shape must be (number_of_edges, number_of_features)");
    }

    const auto work_items = static_cast<std::size_t>(labels.shape[0]);
    const auto n_threads = detail::normalize_thread_count(number_of_threads, work_items);
    std::vector<std::thread> threads;
    threads.reserve(n_threads > 0 ? n_threads - 1 : 0);

    if (compute_complex_features) {
        std::vector<std::vector<detail_features::ComplexStats>> per_thread_stats(
            n_threads,
            std::vector<detail_features::ComplexStats>(static_cast<std::size_t>(rag.number_of_edges()))
        );
        const auto run_chunk = [&](const std::size_t thread_id) {
            const auto begin = thread_id * work_items / n_threads;
            const auto end = (thread_id + 1) * work_items / n_threads;
            if (labels.ndim() == 2) {
                detail_features::scan_edge_map_2d_chunk(
                    rag, labels.data, edge_map.data,
                    static_cast<std::size_t>(labels.shape[0]),
                    static_cast<std::size_t>(labels.shape[1]),
                    begin, end, per_thread_stats[thread_id]
                );
            } else {
                detail_features::scan_edge_map_3d_chunk(
                    rag, labels.data, edge_map.data,
                    static_cast<std::size_t>(labels.shape[0]),
                    static_cast<std::size_t>(labels.shape[1]),
                    static_cast<std::size_t>(labels.shape[2]),
                    begin, end, per_thread_stats[thread_id]
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
        auto stats = detail_features::merge_stats(per_thread_stats, static_cast<std::size_t>(rag.number_of_edges()));
        detail_features::write_complex_features(stats, out);
    } else {
        std::vector<std::vector<detail_features::SimpleStats>> per_thread_stats(
            n_threads,
            std::vector<detail_features::SimpleStats>(static_cast<std::size_t>(rag.number_of_edges()))
        );
        const auto run_chunk = [&](const std::size_t thread_id) {
            const auto begin = thread_id * work_items / n_threads;
            const auto end = (thread_id + 1) * work_items / n_threads;
            if (labels.ndim() == 2) {
                detail_features::scan_edge_map_2d_chunk(
                    rag, labels.data, edge_map.data,
                    static_cast<std::size_t>(labels.shape[0]),
                    static_cast<std::size_t>(labels.shape[1]),
                    begin, end, per_thread_stats[thread_id]
                );
            } else {
                detail_features::scan_edge_map_3d_chunk(
                    rag, labels.data, edge_map.data,
                    static_cast<std::size_t>(labels.shape[0]),
                    static_cast<std::size_t>(labels.shape[1]),
                    static_cast<std::size_t>(labels.shape[2]),
                    begin, end, per_thread_stats[thread_id]
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
        auto stats = detail_features::merge_stats(per_thread_stats, static_cast<std::size_t>(rag.number_of_edges()));
        detail_features::write_simple_features(stats, out);
    }
}

template <class LabelT, class ValueT>
void accumulate_affinity_features(
    const RegionAdjacencyGraph &rag,
    const ConstArrayView<LabelT> &labels,
    const ConstArrayView<ValueT> &affinities,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const bool compute_complex_features,
    const std::size_t number_of_threads,
    const ArrayView<double> &out
) {
    if (labels.ndim() != 2 && labels.ndim() != 3) {
        throw std::invalid_argument("labels must be a 2D or 3D array");
    }
    if (affinities.ndim() != labels.ndim() + 1) {
        throw std::invalid_argument("affinities must have shape (channels, *labels.shape)");
    }
    if (static_cast<std::size_t>(affinities.shape[0]) != offsets.size()) {
        throw std::invalid_argument("offsets length must match affinities channel count");
    }
    for (std::size_t axis = 0; axis < labels.shape.size(); ++axis) {
        if (affinities.shape[axis + 1] != labels.shape[axis]) {
            throw std::invalid_argument("affinities spatial shape must match labels shape");
        }
    }
    for (const auto &offset : offsets) {
        if (offset.size() != static_cast<std::size_t>(labels.ndim())) {
            throw std::invalid_argument("each offset must match labels ndim");
        }
    }

    const auto expected_features = compute_complex_features ? 12 : 2;
    if (out.shape != std::vector<std::ptrdiff_t>{static_cast<std::ptrdiff_t>(rag.number_of_edges()), expected_features}) {
        throw std::invalid_argument("out shape must be (number_of_edges, number_of_features)");
    }

    const auto number_of_nodes = detail_features::number_of_pixels(labels.shape);
    const auto n_threads = detail::normalize_thread_count(number_of_threads, number_of_nodes);
    std::vector<std::thread> threads;
    threads.reserve(n_threads > 0 ? n_threads - 1 : 0);

    if (compute_complex_features) {
        std::vector<std::vector<detail_features::ComplexStats>> per_thread_stats(
            n_threads,
            std::vector<detail_features::ComplexStats>(static_cast<std::size_t>(rag.number_of_edges()))
        );
        const auto run_chunk = [&](const std::size_t thread_id) {
            const auto begin = thread_id * number_of_nodes / n_threads;
            const auto end = (thread_id + 1) * number_of_nodes / n_threads;
            detail_features::scan_affinity_chunk(
                rag, labels.data, affinities.data, offsets, labels.shape,
                begin, end, per_thread_stats[thread_id]
            );
        };
        for (std::size_t thread_id = 1; thread_id < n_threads; ++thread_id) {
            threads.emplace_back(run_chunk, thread_id);
        }
        run_chunk(0);
        for (auto &thread : threads) {
            thread.join();
        }
        auto stats = detail_features::merge_stats(per_thread_stats, static_cast<std::size_t>(rag.number_of_edges()));
        detail_features::write_complex_features(stats, out);
    } else {
        std::vector<std::vector<detail_features::SimpleStats>> per_thread_stats(
            n_threads,
            std::vector<detail_features::SimpleStats>(static_cast<std::size_t>(rag.number_of_edges()))
        );
        const auto run_chunk = [&](const std::size_t thread_id) {
            const auto begin = thread_id * number_of_nodes / n_threads;
            const auto end = (thread_id + 1) * number_of_nodes / n_threads;
            detail_features::scan_affinity_chunk(
                rag, labels.data, affinities.data, offsets, labels.shape,
                begin, end, per_thread_stats[thread_id]
            );
        };
        for (std::size_t thread_id = 1; thread_id < n_threads; ++thread_id) {
            threads.emplace_back(run_chunk, thread_id);
        }
        run_chunk(0);
        for (auto &thread : threads) {
            thread.join();
        }
        auto stats = detail_features::merge_stats(per_thread_stats, static_cast<std::size_t>(rag.number_of_edges()));
        detail_features::write_simple_features(stats, out);
    }
}

} // namespace bioimage_cpp::graph
