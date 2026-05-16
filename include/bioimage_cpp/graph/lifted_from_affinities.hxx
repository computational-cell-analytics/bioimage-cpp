#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/edge_hash.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/threading.hxx"
#include "bioimage_cpp/graph/feature_accumulation.hxx"
#include "bioimage_cpp/graph/region_adjacency_graph.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph {

namespace detail_lifted {

// An offset is "long-range" if it moves by more than a single grid step along
// any axis or by a step on more than one axis. 1-hop offsets (sum |o| == 1)
// already correspond to local RAG edges; skipping them keeps lifted-edge
// discovery focused on edges that the local RAG cannot represent.
inline bool is_long_range(const std::vector<std::ptrdiff_t> &offset) {
    std::ptrdiff_t l1 = 0;
    for (const auto value : offset) {
        l1 += value < 0 ? -value : value;
    }
    return l1 > 1;
}

inline void validate_affinity_inputs(
    const ConstArrayView<double> &affinities_or_dummy,
    const std::vector<std::ptrdiff_t> &labels_shape,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const bool has_affinities
) {
    if (has_affinities) {
        if (affinities_or_dummy.ndim() != static_cast<std::ptrdiff_t>(labels_shape.size()) + 1) {
            throw std::invalid_argument(
                "affinities must have shape (channels, *labels.shape)"
            );
        }
        if (static_cast<std::size_t>(affinities_or_dummy.shape[0]) != offsets.size()) {
            throw std::invalid_argument(
                "offsets length must match affinities channel count"
            );
        }
        for (std::size_t axis = 0; axis < labels_shape.size(); ++axis) {
            if (affinities_or_dummy.shape[axis + 1] != labels_shape[axis]) {
                throw std::invalid_argument(
                    "affinities spatial shape must match labels shape"
                );
            }
        }
    }
    for (const auto &offset : offsets) {
        if (offset.size() != labels_shape.size()) {
            throw std::invalid_argument("each offset must match labels ndim");
        }
    }
}

template <class LabelT>
void discover_lifted_2d_chunk(
    const RegionAdjacencyGraph &rag,
    const LabelT *labels,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::vector<std::size_t> &long_range_channels,
    const std::size_t height,
    const std::size_t width,
    const std::size_t y_begin,
    const std::size_t y_end,
    std::unordered_set<bioimage_cpp::detail::Edge, bioimage_cpp::detail::EdgeHash> &out
) {
    for (const auto channel : long_range_channels) {
        const auto &off = offsets[channel];
        detail_features::sweep_offset_box_2d(
            off[0], off[1], height, width, y_begin, y_end,
            [&](const std::uint64_t node, const std::uint64_t target) {
                const auto u = detail_features::label_at(labels, node);
                const auto v = detail_features::label_at(labels, target);
                if (u == v) {
                    return;
                }
                if (rag.find_edge(u, v) >= 0) {
                    return;
                }
                out.insert(bioimage_cpp::detail::edge_key(u, v));
            }
        );
    }
}

template <class LabelT>
void discover_lifted_3d_chunk(
    const RegionAdjacencyGraph &rag,
    const LabelT *labels,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::vector<std::size_t> &long_range_channels,
    const std::size_t depth,
    const std::size_t height,
    const std::size_t width,
    const std::size_t z_begin,
    const std::size_t z_end,
    std::unordered_set<bioimage_cpp::detail::Edge, bioimage_cpp::detail::EdgeHash> &out
) {
    for (const auto channel : long_range_channels) {
        const auto &off = offsets[channel];
        detail_features::sweep_offset_box_3d(
            off[0], off[1], off[2], depth, height, width, z_begin, z_end,
            [&](const std::uint64_t node, const std::uint64_t target) {
                const auto u = detail_features::label_at(labels, node);
                const auto v = detail_features::label_at(labels, target);
                if (u == v) {
                    return;
                }
                if (rag.find_edge(u, v) >= 0) {
                    return;
                }
                out.insert(bioimage_cpp::detail::edge_key(u, v));
            }
        );
    }
}

} // namespace detail_lifted

// Discover lifted edges implied by long-range offsets on the affinity grid.
//
// Walks every grid coordinate together with each long-range offset (offsets
// whose L1 norm is > 1; 1-hop offsets are silently skipped). When labels at
// (p, p + offset) differ and the (u, v) pair is not already a local RAG edge,
// (u, v) is recorded as a lifted edge. Returns the deduplicated set sorted
// lexicographically with `u < v`.
template <class LabelT>
std::vector<bioimage_cpp::detail::Edge> lifted_edges_from_offsets(
    const RegionAdjacencyGraph &rag,
    const ConstArrayView<LabelT> &labels,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::size_t number_of_threads
) {
    if (labels.ndim() != 2 && labels.ndim() != 3) {
        throw std::invalid_argument("labels must be a 2D or 3D array");
    }
    detail_lifted::validate_affinity_inputs(
        ConstArrayView<double>{nullptr, {}, {}},
        labels.shape,
        offsets,
        /*has_affinities=*/false
    );

    std::vector<std::size_t> long_range_channels;
    long_range_channels.reserve(offsets.size());
    for (std::size_t channel = 0; channel < offsets.size(); ++channel) {
        if (detail_lifted::is_long_range(offsets[channel])) {
            long_range_channels.push_back(channel);
        }
    }
    if (long_range_channels.empty()) {
        return {};
    }

    const auto work_items = static_cast<std::size_t>(labels.shape[0]);
    const auto n_threads = bioimage_cpp::detail::normalize_thread_count(
        number_of_threads, work_items
    );

    using EdgeSet = std::unordered_set<
        bioimage_cpp::detail::Edge, bioimage_cpp::detail::EdgeHash
    >;
    std::vector<EdgeSet> per_thread(n_threads);

    bioimage_cpp::detail::parallel_for_chunks(
        n_threads,
        work_items,
        [&](const std::size_t thread_id, const std::size_t begin, const std::size_t end) {
            if (labels.ndim() == 2) {
                detail_lifted::discover_lifted_2d_chunk(
                    rag, labels.data, offsets, long_range_channels,
                    static_cast<std::size_t>(labels.shape[0]),
                    static_cast<std::size_t>(labels.shape[1]),
                    begin, end, per_thread[thread_id]
                );
            } else {
                detail_lifted::discover_lifted_3d_chunk(
                    rag, labels.data, offsets, long_range_channels,
                    static_cast<std::size_t>(labels.shape[0]),
                    static_cast<std::size_t>(labels.shape[1]),
                    static_cast<std::size_t>(labels.shape[2]),
                    begin, end, per_thread[thread_id]
                );
            }
        }
    );

    EdgeSet merged;
    std::size_t total = 0;
    for (const auto &set : per_thread) {
        total += set.size();
    }
    merged.reserve(total);
    for (auto &set : per_thread) {
        merged.insert(set.begin(), set.end());
    }

    std::vector<bioimage_cpp::detail::Edge> result(merged.begin(), merged.end());
    std::sort(result.begin(), result.end());
    return result;
}

namespace detail_lifted {

template <class LabelT, class ValueT, class Stats>
void scan_lifted_affinity_2d_chunk(
    const std::unordered_map<bioimage_cpp::detail::Edge, std::size_t, bioimage_cpp::detail::EdgeHash>
        &lifted_index,
    const LabelT *labels,
    const ValueT *affinities,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::vector<std::size_t> &long_range_channels,
    const std::size_t height,
    const std::size_t width,
    const std::size_t y_begin,
    const std::size_t y_end,
    std::vector<Stats> &stats
) {
    const auto number_of_nodes = static_cast<std::uint64_t>(height * width);
    for (const auto channel : long_range_channels) {
        const auto &off = offsets[channel];
        const auto channel_offset =
            static_cast<std::uint64_t>(channel) * number_of_nodes;
        detail_features::sweep_offset_box_2d(
            off[0], off[1], height, width, y_begin, y_end,
            [&](const std::uint64_t node, const std::uint64_t target) {
                const auto u = detail_features::label_at(labels, node);
                const auto v = detail_features::label_at(labels, target);
                if (u == v) {
                    return;
                }
                const auto found = lifted_index.find(
                    bioimage_cpp::detail::edge_key(u, v)
                );
                if (found == lifted_index.end()) {
                    return;
                }
                stats[found->second].add(affinities[channel_offset + node]);
            }
        );
    }
}

template <class LabelT, class ValueT, class Stats>
void scan_lifted_affinity_3d_chunk(
    const std::unordered_map<bioimage_cpp::detail::Edge, std::size_t, bioimage_cpp::detail::EdgeHash>
        &lifted_index,
    const LabelT *labels,
    const ValueT *affinities,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::vector<std::size_t> &long_range_channels,
    const std::size_t depth,
    const std::size_t height,
    const std::size_t width,
    const std::size_t z_begin,
    const std::size_t z_end,
    std::vector<Stats> &stats
) {
    const auto slice_size = height * width;
    const auto number_of_nodes = static_cast<std::uint64_t>(depth * slice_size);
    for (const auto channel : long_range_channels) {
        const auto &off = offsets[channel];
        const auto channel_offset =
            static_cast<std::uint64_t>(channel) * number_of_nodes;
        detail_features::sweep_offset_box_3d(
            off[0], off[1], off[2], depth, height, width, z_begin, z_end,
            [&](const std::uint64_t node, const std::uint64_t target) {
                const auto u = detail_features::label_at(labels, node);
                const auto v = detail_features::label_at(labels, target);
                if (u == v) {
                    return;
                }
                const auto found = lifted_index.find(
                    bioimage_cpp::detail::edge_key(u, v)
                );
                if (found == lifted_index.end()) {
                    return;
                }
                stats[found->second].add(affinities[channel_offset + node]);
            }
        );
    }
}

} // namespace detail_lifted

// Accumulate affinity statistics onto a caller-supplied lifted edge set.
//
// `lifted_uvs` lists the (u, v) pairs to bin into, one row per lifted edge;
// it is typically the output of `lifted_edges_from_offsets`. Pixel pairs
// (p, p + offset) whose endpoints differ and whose (u, v) appears in
// `lifted_uvs` contribute their affinity value to that lifted edge's stats.
// Pairs that hit a local-only or unknown edge are silently skipped, so a
// local edge that happens to be reachable via a long-range offset is not
// contaminated by long-range affinities.
//
// 1-hop offsets are skipped automatically so callers can pass the full
// offset list without pre-filtering.
template <class LabelT, class ValueT>
void accumulate_lifted_affinity_features(
    const ConstArrayView<LabelT> &labels,
    const ConstArrayView<ValueT> &affinities,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::vector<bioimage_cpp::detail::Edge> &lifted_uvs,
    const bool compute_complex_features,
    const std::size_t number_of_threads,
    const ArrayView<double> &out
) {
    if (labels.ndim() != 2 && labels.ndim() != 3) {
        throw std::invalid_argument("labels must be a 2D or 3D array");
    }
    if (affinities.ndim() != labels.ndim() + 1) {
        throw std::invalid_argument(
            "affinities must have shape (channels, *labels.shape)"
        );
    }
    if (static_cast<std::size_t>(affinities.shape[0]) != offsets.size()) {
        throw std::invalid_argument(
            "offsets length must match affinities channel count"
        );
    }
    for (std::size_t axis = 0; axis < labels.shape.size(); ++axis) {
        if (affinities.shape[axis + 1] != labels.shape[axis]) {
            throw std::invalid_argument(
                "affinities spatial shape must match labels shape"
            );
        }
    }
    for (const auto &offset : offsets) {
        if (offset.size() != static_cast<std::size_t>(labels.ndim())) {
            throw std::invalid_argument("each offset must match labels ndim");
        }
    }

    const auto expected_features = compute_complex_features ? 12 : 2;
    if (out.shape != std::vector<std::ptrdiff_t>{
            static_cast<std::ptrdiff_t>(lifted_uvs.size()), expected_features}) {
        throw std::invalid_argument(
            "out shape must be (number_of_lifted_edges, number_of_features)"
        );
    }

    std::vector<std::size_t> long_range_channels;
    long_range_channels.reserve(offsets.size());
    for (std::size_t channel = 0; channel < offsets.size(); ++channel) {
        if (detail_lifted::is_long_range(offsets[channel])) {
            long_range_channels.push_back(channel);
        }
    }

    std::unordered_map<bioimage_cpp::detail::Edge, std::size_t, bioimage_cpp::detail::EdgeHash>
        lifted_index;
    lifted_index.reserve(lifted_uvs.size());
    for (std::size_t i = 0; i < lifted_uvs.size(); ++i) {
        const auto key = bioimage_cpp::detail::edge_key(lifted_uvs[i].first, lifted_uvs[i].second);
        if (!lifted_index.emplace(key, i).second) {
            throw std::invalid_argument("lifted_uvs must not contain duplicate edges");
        }
    }

    const auto number_of_lifted = lifted_uvs.size();
    if (long_range_channels.empty() || number_of_lifted == 0) {
        if (compute_complex_features) {
            std::vector<detail_features::ComplexStats> empty(number_of_lifted);
            detail_features::write_complex_features(empty, out);
        } else {
            std::vector<detail_features::SimpleStats> empty(number_of_lifted);
            detail_features::write_simple_features(empty, out);
        }
        return;
    }

    const auto work_items = static_cast<std::size_t>(labels.shape[0]);
    const auto n_threads = bioimage_cpp::detail::normalize_thread_count(
        number_of_threads, work_items
    );

    const auto run_scan = [&](auto &per_thread_stats) {
        bioimage_cpp::detail::parallel_for_chunks(
            n_threads,
            work_items,
            [&](const std::size_t thread_id, const std::size_t begin, const std::size_t end) {
                if (labels.ndim() == 2) {
                    detail_lifted::scan_lifted_affinity_2d_chunk(
                        lifted_index, labels.data, affinities.data,
                        offsets, long_range_channels,
                        static_cast<std::size_t>(labels.shape[0]),
                        static_cast<std::size_t>(labels.shape[1]),
                        begin, end, per_thread_stats[thread_id]
                    );
                } else {
                    detail_lifted::scan_lifted_affinity_3d_chunk(
                        lifted_index, labels.data, affinities.data,
                        offsets, long_range_channels,
                        static_cast<std::size_t>(labels.shape[0]),
                        static_cast<std::size_t>(labels.shape[1]),
                        static_cast<std::size_t>(labels.shape[2]),
                        begin, end, per_thread_stats[thread_id]
                    );
                }
            }
        );
    };

    if (compute_complex_features) {
        std::vector<std::vector<detail_features::ComplexStats>> per_thread_stats(
            n_threads,
            std::vector<detail_features::ComplexStats>(number_of_lifted)
        );
        run_scan(per_thread_stats);
        auto stats = detail_features::merge_stats(per_thread_stats, number_of_lifted);
        detail_features::write_complex_features(stats, out);
    } else {
        std::vector<std::vector<detail_features::SimpleStats>> per_thread_stats(
            n_threads,
            std::vector<detail_features::SimpleStats>(number_of_lifted)
        );
        run_scan(per_thread_stats);
        auto stats = detail_features::merge_stats(per_thread_stats, number_of_lifted);
        detail_features::write_simple_features(stats, out);
    }
}

} // namespace bioimage_cpp::graph
