#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/edge_hash.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/label_cast.hxx"
#include "bioimage_cpp/detail/threading.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Distributed region-adjacency-graph + edge-feature primitives.
//
// These functions operate on a single (haloed) label block and extract the
// edges / partial edge statistics that the block *owns*, so that per-block
// results can later be merged into a whole-volume result by `merge.hxx`.
// Orchestration (block iteration, halo sizing, I/O, hierarchical merging) lives
// in Python and is intentionally not implemented here.
//
// Ownership rule: a block owns the pixel-pairs whose *reference pixel* lies in
// the block's inner (non-halo) box `[own_begin, own_begin + own_shape)`; the
// neighbor pixel is read from the passed (outer, haloed) array. Because inner
// boxes tile the volume, every contributing pair is counted exactly once, so
// per-block partial statistics (count/sum/sum_of_squares add, min/max reduce)
// reconstruct the whole-volume statistics exactly. The reference pixel is the
// lower pixel of a forward nearest-neighbor edge (region graph / edge map) or
// the pixel an affinity value is stored at (affinities).
//
// Halo requirement (caller's responsibility — not detectable here): the outer
// array must extend at least one pixel past the owned box on the forward faces
// for nearest-neighbor edges, and at least `max |offset component|` past the
// owned box on the relevant faces for affinities. If the halo is too small,
// owned pairs whose neighbor falls outside the block are silently dropped.
namespace bioimage_cpp::graph::distributed {

// Per-edge partial statistics accumulated over the pixels a block owns. Layout
// mirrors the serialized `(n_edges, 5)` array: count, sum, sum_of_squares, min,
// max. `count/min/max` are exact under any thread/block ordering; the
// floating-point `sum/sum_of_squares` (and hence mean/std after finalization)
// are reproducible only for a fixed thread count and merge order.
struct PartialStats {
    double count = 0.0;
    double sum = 0.0;
    double sum_of_squares = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();

    void add(const double value) {
        sum += value;
        sum_of_squares += value * value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        count += 1.0;
    }

    void merge(const PartialStats &other) {
        if (other.count == 0.0) {
            return;
        }
        sum += other.sum;
        sum_of_squares += other.sum_of_squares;
        minimum = std::min(minimum, other.minimum);
        maximum = std::max(maximum, other.maximum);
        count += other.count;
    }
};

// A block's owned edges together with their aligned partial statistics. `edges`
// is sorted-unique with `u < v`; `stats` is row-major `(edges.size(), 5)` with
// columns [count, sum, sum_of_squares, min, max] and row `i` describing
// `edges[i]`.
struct BlockEdgeStats {
    std::vector<bioimage_cpp::detail::Edge> edges;
    std::vector<double> stats;
};

namespace detail_block {

using bioimage_cpp::detail::checked_label_to_node;
using bioimage_cpp::detail::Edge;
using bioimage_cpp::detail::EdgeHash;
using bioimage_cpp::detail::edge_key;
using bioimage_cpp::detail::valid_axis_range;

using EdgeSet = std::unordered_set<Edge, EdgeHash>;
using EdgeStatsMap = std::unordered_map<Edge, PartialStats, EdgeHash>;

inline std::vector<std::size_t> to_size_dims(const std::vector<std::ptrdiff_t> &shape) {
    std::vector<std::size_t> dims(shape.size());
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        dims[axis] = static_cast<std::size_t>(shape[axis]);
    }
    return dims;
}

// Forward nearest-neighbor unit offsets, one per axis (2D: (1,0),(0,1);
// 3D: (1,0,0),(0,1,0),(0,0,1)). These are the edges of the region graph.
inline std::vector<std::vector<std::ptrdiff_t>> forward_nn_offsets(const std::size_t ndim) {
    std::vector<std::vector<std::ptrdiff_t>> offsets;
    offsets.reserve(ndim);
    for (std::size_t axis = 0; axis < ndim; ++axis) {
        std::vector<std::ptrdiff_t> offset(ndim, 0);
        offset[axis] = 1;
        offsets.push_back(std::move(offset));
    }
    return offsets;
}

inline void validate_owned_box(
    const std::vector<std::ptrdiff_t> &shape,
    const std::vector<std::int64_t> &own_begin,
    const std::vector<std::int64_t> &own_shape
) {
    const auto ndim = shape.size();
    if (ndim != 2 && ndim != 3) {
        throw std::invalid_argument(
            "labels must be a 2D or 3D array, got ndim=" + std::to_string(ndim)
        );
    }
    if (own_begin.size() != ndim) {
        throw std::invalid_argument("own_begin length must match labels ndim");
    }
    if (own_shape.size() != ndim) {
        throw std::invalid_argument("own_shape length must match labels ndim");
    }
    for (std::size_t axis = 0; axis < ndim; ++axis) {
        if (own_begin[axis] < 0) {
            throw std::invalid_argument("own_begin values must be non-negative");
        }
        if (own_shape[axis] <= 0) {
            throw std::invalid_argument("own_shape values must be positive");
        }
        if (own_begin[axis] + own_shape[axis] > shape[axis]) {
            throw std::invalid_argument(
                "owned box must lie within the block (own_begin + own_shape <= block shape)"
            );
        }
    }
}

// Sweep reference nodes over the owned box on a 2D grid, calling
// `body(node, target)` with flat C-order indices into the outer array for each
// reference node whose `+offset` neighbor stays inside the outer array. Axis 0
// is additionally restricted to the absolute slab `[slab_begin, slab_end)`
// (the caller's thread chunk, already inside the owned box).
template <class Body>
void sweep_owned_box_2d(
    const std::ptrdiff_t dy,
    const std::ptrdiff_t dx,
    const std::size_t outer_h,
    const std::size_t outer_w,
    const std::int64_t own_begin_y,
    const std::int64_t own_begin_x,
    const std::int64_t own_shape_y,
    const std::int64_t own_shape_x,
    const std::size_t slab_begin,
    const std::size_t slab_end,
    const Body &body
) {
    std::size_t y_lo_v, y_hi_v, x_lo_v, x_hi_v;
    valid_axis_range(dy, outer_h, y_lo_v, y_hi_v);
    valid_axis_range(dx, outer_w, x_lo_v, x_hi_v);

    const auto y_lo = std::max({y_lo_v, static_cast<std::size_t>(own_begin_y), slab_begin});
    const auto y_hi = std::min({y_hi_v, static_cast<std::size_t>(own_begin_y + own_shape_y), slab_end});
    const auto x_lo = std::max(x_lo_v, static_cast<std::size_t>(own_begin_x));
    const auto x_hi = std::min(x_hi_v, static_cast<std::size_t>(own_begin_x + own_shape_x));
    if (y_lo >= y_hi || x_lo >= x_hi) {
        return;
    }

    const auto offset_stride = dy * static_cast<std::ptrdiff_t>(outer_w) + dx;
    for (std::size_t y = y_lo; y < y_hi; ++y) {
        const auto row_offset = y * outer_w;
        for (std::size_t x = x_lo; x < x_hi; ++x) {
            const auto node = row_offset + x;
            const auto target = static_cast<std::uint64_t>(
                static_cast<std::ptrdiff_t>(node) + offset_stride
            );
            body(static_cast<std::uint64_t>(node), target);
        }
    }
}

// 3D variant of `sweep_owned_box_2d`.
template <class Body>
void sweep_owned_box_3d(
    const std::ptrdiff_t dz,
    const std::ptrdiff_t dy,
    const std::ptrdiff_t dx,
    const std::size_t outer_d,
    const std::size_t outer_h,
    const std::size_t outer_w,
    const std::int64_t own_begin_z,
    const std::int64_t own_begin_y,
    const std::int64_t own_begin_x,
    const std::int64_t own_shape_z,
    const std::int64_t own_shape_y,
    const std::int64_t own_shape_x,
    const std::size_t slab_begin,
    const std::size_t slab_end,
    const Body &body
) {
    std::size_t z_lo_v, z_hi_v, y_lo_v, y_hi_v, x_lo_v, x_hi_v;
    valid_axis_range(dz, outer_d, z_lo_v, z_hi_v);
    valid_axis_range(dy, outer_h, y_lo_v, y_hi_v);
    valid_axis_range(dx, outer_w, x_lo_v, x_hi_v);

    const auto z_lo = std::max({z_lo_v, static_cast<std::size_t>(own_begin_z), slab_begin});
    const auto z_hi = std::min({z_hi_v, static_cast<std::size_t>(own_begin_z + own_shape_z), slab_end});
    const auto y_lo = std::max(y_lo_v, static_cast<std::size_t>(own_begin_y));
    const auto y_hi = std::min(y_hi_v, static_cast<std::size_t>(own_begin_y + own_shape_y));
    const auto x_lo = std::max(x_lo_v, static_cast<std::size_t>(own_begin_x));
    const auto x_hi = std::min(x_hi_v, static_cast<std::size_t>(own_begin_x + own_shape_x));
    if (z_lo >= z_hi || y_lo >= y_hi || x_lo >= x_hi) {
        return;
    }

    const auto slice_size = outer_h * outer_w;
    const auto offset_stride =
        dz * static_cast<std::ptrdiff_t>(slice_size) +
        dy * static_cast<std::ptrdiff_t>(outer_w) + dx;
    for (std::size_t z = z_lo; z < z_hi; ++z) {
        const auto slice_offset = z * slice_size;
        for (std::size_t y = y_lo; y < y_hi; ++y) {
            const auto row_offset = slice_offset + y * outer_w;
            for (std::size_t x = x_lo; x < x_hi; ++x) {
                const auto node = row_offset + x;
                const auto target = static_cast<std::uint64_t>(
                    static_cast<std::ptrdiff_t>(node) + offset_stride
                );
                body(static_cast<std::uint64_t>(node), target);
            }
        }
    }
}

// Dispatch the owned-box sweep for one offset over the correct grid rank.
template <class Body>
void sweep_owned_box(
    const std::vector<std::ptrdiff_t> &offset,
    const std::vector<std::size_t> &outer_dims,
    const std::vector<std::int64_t> &own_begin,
    const std::vector<std::int64_t> &own_shape,
    const std::size_t slab_begin,
    const std::size_t slab_end,
    const Body &body
) {
    if (outer_dims.size() == 2) {
        sweep_owned_box_2d(
            offset[0], offset[1], outer_dims[0], outer_dims[1],
            own_begin[0], own_begin[1], own_shape[0], own_shape[1],
            slab_begin, slab_end, body
        );
    } else {
        sweep_owned_box_3d(
            offset[0], offset[1], offset[2], outer_dims[0], outer_dims[1], outer_dims[2],
            own_begin[0], own_begin[1], own_begin[2],
            own_shape[0], own_shape[1], own_shape[2],
            slab_begin, slab_end, body
        );
    }
}

// Collect the owned region-graph edges of one thread's slab into `out`.
template <class LabelT>
void collect_edges_chunk(
    const LabelT *labels,
    const std::vector<std::size_t> &outer_dims,
    const std::vector<std::int64_t> &own_begin,
    const std::vector<std::int64_t> &own_shape,
    const std::size_t slab_begin,
    const std::size_t slab_end,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    EdgeSet &out
) {
    for (const auto &offset : offsets) {
        sweep_owned_box(
            offset, outer_dims, own_begin, own_shape, slab_begin, slab_end,
            [&](const std::uint64_t node, const std::uint64_t target) {
                const auto u = checked_label_to_node(labels[node]);
                const auto v = checked_label_to_node(labels[target]);
                if (u != v) {
                    out.insert(edge_key(u, v));
                }
            }
        );
    }
}

// Accumulate one thread's owned partial statistics into `out`. `value_fn`
// receives `(offset_index, node, target)` and returns the value to accumulate
// on the edge between the labels at `node` and `target`.
template <class LabelT, class ValueFn>
void accumulate_stats_chunk(
    const LabelT *labels,
    const std::vector<std::size_t> &outer_dims,
    const std::vector<std::int64_t> &own_begin,
    const std::vector<std::int64_t> &own_shape,
    const std::size_t slab_begin,
    const std::size_t slab_end,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const ValueFn &value_fn,
    EdgeStatsMap &out
) {
    for (std::size_t offset_index = 0; offset_index < offsets.size(); ++offset_index) {
        sweep_owned_box(
            offsets[offset_index], outer_dims, own_begin, own_shape,
            slab_begin, slab_end,
            [&](const std::uint64_t node, const std::uint64_t target) {
                const auto u = checked_label_to_node(labels[node]);
                const auto v = checked_label_to_node(labels[target]);
                if (u != v) {
                    out[edge_key(u, v)].add(value_fn(offset_index, node, target));
                }
            }
        );
    }
}

inline std::vector<Edge> merge_edge_sets(std::vector<EdgeSet> &per_thread) {
    EdgeSet merged;
    for (const auto &edges : per_thread) {
        merged.insert(edges.begin(), edges.end());
    }
    std::vector<Edge> sorted_edges(merged.begin(), merged.end());
    std::sort(sorted_edges.begin(), sorted_edges.end());
    return sorted_edges;
}

inline BlockEdgeStats merge_stats_maps(std::vector<EdgeStatsMap> &per_thread) {
    EdgeStatsMap merged;
    for (auto &thread_map : per_thread) {
        for (const auto &[edge, stats] : thread_map) {
            merged[edge].merge(stats);
        }
    }

    std::vector<Edge> sorted_edges;
    sorted_edges.reserve(merged.size());
    for (const auto &[edge, stats] : merged) {
        sorted_edges.push_back(edge);
    }
    std::sort(sorted_edges.begin(), sorted_edges.end());

    BlockEdgeStats result;
    result.edges = std::move(sorted_edges);
    result.stats.reserve(result.edges.size() * 5);
    for (const auto &edge : result.edges) {
        const auto &stats = merged[edge];
        result.stats.push_back(stats.count);
        result.stats.push_back(stats.sum);
        result.stats.push_back(stats.sum_of_squares);
        result.stats.push_back(stats.minimum);
        result.stats.push_back(stats.maximum);
    }
    return result;
}

// Common driver: run `chunk_fn(thread_id, slab_begin, slab_end, container)`
// over the owned box's axis-0 extent split across threads.
template <class Container, class ChunkFn>
std::vector<Container> run_owned_scan(
    const std::vector<std::int64_t> &own_begin,
    const std::vector<std::int64_t> &own_shape,
    const std::size_t number_of_threads,
    const ChunkFn &chunk_fn
) {
    const auto work_items = static_cast<std::size_t>(own_shape[0]);
    const auto n_threads = bioimage_cpp::detail::normalize_thread_count(
        number_of_threads, work_items
    );
    const auto axis0_begin = static_cast<std::size_t>(own_begin[0]);

    std::vector<Container> per_thread(n_threads);
    bioimage_cpp::detail::parallel_for_chunks(
        n_threads,
        work_items,
        [&](const std::size_t thread_id, const std::size_t begin, const std::size_t end) {
            chunk_fn(axis0_begin + begin, axis0_begin + end, per_thread[thread_id]);
        }
    );
    return per_thread;
}

inline void validate_affinities(
    const std::vector<std::ptrdiff_t> &labels_shape,
    const std::vector<std::ptrdiff_t> &affinities_shape,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets
) {
    const auto ndim = labels_shape.size();
    if (affinities_shape.size() != ndim + 1) {
        throw std::invalid_argument("affinities must have shape (channels, *labels.shape)");
    }
    if (static_cast<std::size_t>(affinities_shape[0]) != offsets.size()) {
        throw std::invalid_argument("offsets length must match affinities channel count");
    }
    for (std::size_t axis = 0; axis < ndim; ++axis) {
        if (affinities_shape[axis + 1] != labels_shape[axis]) {
            throw std::invalid_argument("affinities spatial shape must match labels shape");
        }
    }
    for (const auto &offset : offsets) {
        if (offset.size() != ndim) {
            throw std::invalid_argument("each offset must have length matching labels ndim");
        }
    }
}

} // namespace detail_block

// Extract the region-adjacency edges a block owns. Returns a sorted-unique edge
// list (`u < v`) using global label ids; concatenating these across blocks and
// passing them through `merge_edges` yields the whole-volume edge set.
template <class LabelT>
std::vector<bioimage_cpp::detail::Edge> block_region_adjacency_edges(
    const ConstArrayView<LabelT> &labels_outer,
    const std::vector<std::int64_t> &own_begin,
    const std::vector<std::int64_t> &own_shape,
    const std::size_t number_of_threads
) {
    detail_block::validate_owned_box(labels_outer.shape, own_begin, own_shape);
    const auto outer_dims = detail_block::to_size_dims(labels_outer.shape);
    const auto offsets = detail_block::forward_nn_offsets(outer_dims.size());
    const auto *labels = labels_outer.data;

    auto per_thread = detail_block::run_owned_scan<detail_block::EdgeSet>(
        own_begin, own_shape, number_of_threads,
        [&](const std::size_t slab_begin, const std::size_t slab_end,
            detail_block::EdgeSet &out) {
            detail_block::collect_edges_chunk(
                labels, outer_dims, own_begin, own_shape, slab_begin, slab_end,
                offsets, out
            );
        }
    );
    return detail_block::merge_edge_sets(per_thread);
}

// Accumulate the owned partial statistics of a scalar edge map. The value on an
// edge is the average of the two endpoint pixel values, `0.5 * (map[node] +
// map[neighbor])`, matching the in-core `accumulate_edge_map_features`. The
// returned edges match `block_region_adjacency_edges` for the same block.
template <class LabelT>
BlockEdgeStats block_edge_map_stats(
    const ConstArrayView<LabelT> &labels_outer,
    const ConstArrayView<double> &edge_map_outer,
    const std::vector<std::int64_t> &own_begin,
    const std::vector<std::int64_t> &own_shape,
    const std::size_t number_of_threads
) {
    detail_block::validate_owned_box(labels_outer.shape, own_begin, own_shape);
    if (edge_map_outer.shape != labels_outer.shape) {
        throw std::invalid_argument("edge_map shape must match labels shape");
    }
    const auto outer_dims = detail_block::to_size_dims(labels_outer.shape);
    const auto offsets = detail_block::forward_nn_offsets(outer_dims.size());
    const auto *labels = labels_outer.data;
    const auto *edge_map = edge_map_outer.data;

    const auto value_fn = [&](const std::size_t, const std::uint64_t node,
                              const std::uint64_t target) {
        return 0.5 * (edge_map[node] + edge_map[target]);
    };

    auto per_thread = detail_block::run_owned_scan<detail_block::EdgeStatsMap>(
        own_begin, own_shape, number_of_threads,
        [&](const std::size_t slab_begin, const std::size_t slab_end,
            detail_block::EdgeStatsMap &out) {
            detail_block::accumulate_stats_chunk(
                labels, outer_dims, own_begin, own_shape, slab_begin, slab_end,
                offsets, value_fn, out
            );
        }
    );
    return detail_block::merge_stats_maps(per_thread);
}

// Accumulate the owned partial statistics of affinity channels. `affinities`
// has shape `(len(offsets), *labels.shape)`; the value on an edge is the
// affinity stored at the reference node, `affinities[channel * n_nodes + node]`,
// matching the in-core `accumulate_affinity_features`. Values from all offsets
// are aggregated per `(u, v)`, so a nearest-neighbor edge also hit by a
// long-range offset receives that offset's value too. Long-range-only pairs are
// included here and dropped at merge time if absent from the global graph.
template <class LabelT>
BlockEdgeStats block_affinity_stats(
    const ConstArrayView<LabelT> &labels_outer,
    const ConstArrayView<double> &affinities_outer,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::vector<std::int64_t> &own_begin,
    const std::vector<std::int64_t> &own_shape,
    const std::size_t number_of_threads
) {
    detail_block::validate_owned_box(labels_outer.shape, own_begin, own_shape);
    detail_block::validate_affinities(labels_outer.shape, affinities_outer.shape, offsets);
    const auto outer_dims = detail_block::to_size_dims(labels_outer.shape);
    const auto *labels = labels_outer.data;
    const auto *affinities = affinities_outer.data;

    std::uint64_t number_of_nodes = 1;
    for (const auto dim : outer_dims) {
        number_of_nodes *= static_cast<std::uint64_t>(dim);
    }

    const auto value_fn = [&](const std::size_t offset_index, const std::uint64_t node,
                              const std::uint64_t) {
        const auto channel_offset =
            static_cast<std::uint64_t>(offset_index) * number_of_nodes;
        return affinities[channel_offset + node];
    };

    auto per_thread = detail_block::run_owned_scan<detail_block::EdgeStatsMap>(
        own_begin, own_shape, number_of_threads,
        [&](const std::size_t slab_begin, const std::size_t slab_end,
            detail_block::EdgeStatsMap &out) {
            detail_block::accumulate_stats_chunk(
                labels, outer_dims, own_begin, own_shape, slab_begin, slab_end,
                offsets, value_fn, out
            );
        }
    );
    return detail_block::merge_stats_maps(per_thread);
}

} // namespace bioimage_cpp::graph::distributed
