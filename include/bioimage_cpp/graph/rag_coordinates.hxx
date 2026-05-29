#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/threading.hxx"
#include "bioimage_cpp/graph/region_adjacency_graph.hxx"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace bioimage_cpp::graph {

namespace detail_rag_coordinates {

// Visit every forward-neighbor contact (a pair of adjacent pixels with
// different labels) whose lower pixel lies in rows [begin, end) of a 2D label
// image. `on_contact(edge_id, low_flat, high_flat)` is called for each, where
// `low_flat` is the current pixel and `high_flat` the +axis neighbor.
template <class T, class OnContact>
void scan_contacts_2d(
    const T *data,
    const std::size_t height,
    const std::size_t width,
    const std::size_t y_begin,
    const std::size_t y_end,
    const RegionAdjacencyGraph &rag,
    OnContact &&on_contact
) {
    for (std::size_t y = y_begin; y < y_end; ++y) {
        const auto row_offset = y * width;
        for (std::size_t x = 0; x < width; ++x) {
            const auto pixel = row_offset + x;
            const auto u = static_cast<std::uint64_t>(data[pixel]);
            if (x + 1 < width) {
                const auto v = static_cast<std::uint64_t>(data[pixel + 1]);
                if (u != v) {
                    const auto edge = rag.find_edge(u, v);
                    if (edge >= 0) {
                        on_contact(static_cast<std::uint64_t>(edge), pixel, pixel + 1);
                    }
                }
            }
            if (y + 1 < height) {
                const auto v = static_cast<std::uint64_t>(data[pixel + width]);
                if (u != v) {
                    const auto edge = rag.find_edge(u, v);
                    if (edge >= 0) {
                        on_contact(static_cast<std::uint64_t>(edge), pixel, pixel + width);
                    }
                }
            }
        }
    }
}

// 3D analogue of scan_contacts_2d; chunks by the leading (z) axis.
template <class T, class OnContact>
void scan_contacts_3d(
    const T *data,
    const std::size_t depth,
    const std::size_t height,
    const std::size_t width,
    const std::size_t z_begin,
    const std::size_t z_end,
    const RegionAdjacencyGraph &rag,
    OnContact &&on_contact
) {
    const auto slice_size = height * width;
    for (std::size_t z = z_begin; z < z_end; ++z) {
        const auto slice_offset = z * slice_size;
        for (std::size_t y = 0; y < height; ++y) {
            const auto row_offset = slice_offset + y * width;
            for (std::size_t x = 0; x < width; ++x) {
                const auto pixel = row_offset + x;
                const auto u = static_cast<std::uint64_t>(data[pixel]);
                if (x + 1 < width) {
                    const auto v = static_cast<std::uint64_t>(data[pixel + 1]);
                    if (u != v) {
                        const auto edge = rag.find_edge(u, v);
                        if (edge >= 0) {
                            on_contact(static_cast<std::uint64_t>(edge), pixel, pixel + 1);
                        }
                    }
                }
                if (y + 1 < height) {
                    const auto v = static_cast<std::uint64_t>(data[pixel + width]);
                    if (u != v) {
                        const auto edge = rag.find_edge(u, v);
                        if (edge >= 0) {
                            on_contact(static_cast<std::uint64_t>(edge), pixel, pixel + width);
                        }
                    }
                }
                if (z + 1 < depth) {
                    const auto v = static_cast<std::uint64_t>(data[pixel + slice_size]);
                    if (u != v) {
                        const auto edge = rag.find_edge(u, v);
                        if (edge >= 0) {
                            on_contact(static_cast<std::uint64_t>(edge), pixel, pixel + slice_size);
                        }
                    }
                }
            }
        }
    }
}

} // namespace detail_rag_coordinates

// Maps the edges of a RegionAdjacencyGraph back to the pixel/voxel coordinates
// of the boundary between the two adjacent regions. Scans the label volume once
// at construction and caches a CSR-style mapping so the result can be reused
// across many `edges_to_volume` calls.
//
// Each boundary "contact" — a pair of directly adjacent pixels with different
// labels — contributes two consecutive flat (C-order) pixel indices to its
// edge: the lower-coordinate pixel followed by its +axis neighbor. Within an
// edge's slice even local indices are therefore lower-side pixels and odd local
// indices are higher-side. The `edge_direction` parameter selects which side(s)
// to report: 0 = both (default), 1 = lower-side only, 2 = higher-side only.
class RagCoordinates {
public:
    template <class T>
    RagCoordinates(
        const RegionAdjacencyGraph &rag,
        const ConstArrayView<T> &labels,
        const std::size_t number_of_threads
    ) {
        if (labels.ndim() != 2 && labels.ndim() != 3) {
            throw std::invalid_argument(
                "labels must be a 2D or 3D array, got ndim=" +
                std::to_string(labels.ndim())
            );
        }
        if (static_cast<std::size_t>(labels.ndim()) != rag.shape().size()) {
            throw std::invalid_argument("labels ndim must match rag shape");
        }
        for (std::size_t axis = 0; axis < rag.shape().size(); ++axis) {
            if (labels.shape[axis] != static_cast<std::ptrdiff_t>(rag.shape()[axis])) {
                throw std::invalid_argument("labels shape must match rag shape");
            }
        }

        shape_ = rag.shape();
        const auto n_edges = static_cast<std::size_t>(rag.number_of_edges());

        const auto work_items = static_cast<std::size_t>(labels.shape[0]);
        const auto n_threads =
            bioimage_cpp::detail::normalize_thread_count(number_of_threads, work_items);

        // Pass 1: count contacts per edge, per thread (each contact = 2 points).
        std::vector<std::vector<std::uint64_t>> per_thread_counts(
            n_threads, std::vector<std::uint64_t>(n_edges, 0)
        );
        run_scan(labels, rag, n_threads, [&per_thread_counts](std::size_t thread_id) {
            std::uint64_t *counts = per_thread_counts[thread_id].data();
            return [counts](std::uint64_t edge, std::uint64_t, std::uint64_t) {
                counts[edge] += 2;
            };
        });

        // Build CSR offsets (in points) and per-thread write bases within each
        // edge slice so the fill preserves global scan order deterministically.
        offsets_.assign(n_edges + 1, 0);
        std::vector<std::uint64_t> edge_total(n_edges, 0);
        for (std::size_t edge = 0; edge < n_edges; ++edge) {
            std::uint64_t total = 0;
            for (std::size_t thread_id = 0; thread_id < n_threads; ++thread_id) {
                total += per_thread_counts[thread_id][edge];
            }
            edge_total[edge] = total;
            offsets_[edge + 1] = offsets_[edge] + total;
        }
        points_.assign(static_cast<std::size_t>(offsets_[n_edges]), 0);

        // cursor[thread][edge] = next write position (point index) for that
        // thread within the edge's slice.
        std::vector<std::vector<std::uint64_t>> cursor(
            n_threads, std::vector<std::uint64_t>(n_edges, 0)
        );
        for (std::size_t edge = 0; edge < n_edges; ++edge) {
            std::uint64_t base = offsets_[edge];
            for (std::size_t thread_id = 0; thread_id < n_threads; ++thread_id) {
                cursor[thread_id][edge] = base;
                base += per_thread_counts[thread_id][edge];
            }
        }

        // Pass 2: fill points at the computed cursors.
        std::uint64_t *points = points_.data();
        run_scan(labels, rag, n_threads, [&cursor, points](std::size_t thread_id) {
            std::uint64_t *cur = cursor[thread_id].data();
            return [cur, points](std::uint64_t edge, std::uint64_t low, std::uint64_t high) {
                points[cur[edge]++] = low;
                points[cur[edge]++] = high;
            };
        });
    }

    [[nodiscard]] std::size_t ndim() const {
        return shape_.size();
    }

    [[nodiscard]] const std::vector<std::uint64_t> &shape() const {
        return shape_;
    }

    [[nodiscard]] std::uint64_t number_of_edges() const {
        return offsets_.empty() ? 0 : static_cast<std::uint64_t>(offsets_.size() - 1);
    }

    // Number of stored points (= 2 * number of contacts) per edge.
    [[nodiscard]] std::vector<std::uint64_t> storage_lengths() const {
        const auto n_edges = static_cast<std::size_t>(number_of_edges());
        std::vector<std::uint64_t> lengths(n_edges, 0);
        for (std::size_t edge = 0; edge < n_edges; ++edge) {
            lengths[edge] = offsets_[edge + 1] - offsets_[edge];
        }
        return lengths;
    }

    // Decoded boundary coordinates of one edge, flattened as n_points * ndim in
    // C-order. `edge_direction` selects which side(s) are reported.
    [[nodiscard]] std::vector<std::uint64_t> edge_coordinates(
        const std::uint64_t edge,
        const int edge_direction
    ) const {
        check_edge(edge);
        check_edge_direction(edge_direction);
        const auto strides = make_strides();
        const auto begin = offsets_[edge];
        const auto end = offsets_[edge + 1];

        std::vector<std::uint64_t> coordinates;
        for (auto index = begin; index < end; ++index) {
            if (!selected(index - begin, edge_direction)) {
                continue;
            }
            decode(points_[index], strides, coordinates);
        }
        return coordinates;
    }

    // Paint per-edge values into `out` (shape == this->shape()). Every pixel is
    // first set to `ignore_value`; then each selected boundary point receives
    // its edge's value. Painting is sequential in ascending edge id, so where
    // the boundaries of several edges coincide on the same pixel the highest
    // edge id wins — a deterministic, race-free rule.
    template <class V>
    void edges_to_volume(
        const std::vector<V> &edge_values,
        ArrayView<V> &out,
        const int edge_direction,
        const V ignore_value
    ) const {
        const auto n_edges = static_cast<std::size_t>(number_of_edges());
        if (edge_values.size() != n_edges) {
            throw std::invalid_argument("edge_values length must match number_of_edges");
        }
        check_edge_direction(edge_direction);
        if (static_cast<std::size_t>(out.ndim()) != shape_.size()) {
            throw std::invalid_argument("out ndim must match rag shape");
        }
        std::size_t total = 1;
        for (std::size_t axis = 0; axis < shape_.size(); ++axis) {
            if (out.shape[axis] != static_cast<std::ptrdiff_t>(shape_[axis])) {
                throw std::invalid_argument("out shape must match rag shape");
            }
            total *= shape_[axis];
        }

        auto *data = out.data;
        for (std::size_t index = 0; index < total; ++index) {
            data[index] = ignore_value;
        }

        for (std::size_t edge = 0; edge < n_edges; ++edge) {
            const auto value = edge_values[edge];
            const auto slice_begin = offsets_[edge];
            const auto slice_end = offsets_[edge + 1];
            for (auto index = slice_begin; index < slice_end; ++index) {
                if (selected(index - slice_begin, edge_direction)) {
                    data[points_[index]] = value;
                }
            }
        }
    }

private:
    template <class T, class MakeCallback>
    void run_scan(
        const ConstArrayView<T> &labels,
        const RegionAdjacencyGraph &rag,
        const std::size_t n_threads,
        MakeCallback &&make_callback
    ) const {
        namespace drc = detail_rag_coordinates;
        const auto work_items = static_cast<std::size_t>(labels.shape[0]);
        bioimage_cpp::detail::parallel_for_chunks(
            n_threads,
            work_items,
            [&](std::size_t thread_id, std::size_t begin, std::size_t end) {
                auto on_contact = make_callback(thread_id);
                if (labels.ndim() == 2) {
                    drc::scan_contacts_2d(
                        labels.data,
                        static_cast<std::size_t>(labels.shape[0]),
                        static_cast<std::size_t>(labels.shape[1]),
                        begin,
                        end,
                        rag,
                        on_contact
                    );
                } else {
                    drc::scan_contacts_3d(
                        labels.data,
                        static_cast<std::size_t>(labels.shape[0]),
                        static_cast<std::size_t>(labels.shape[1]),
                        static_cast<std::size_t>(labels.shape[2]),
                        begin,
                        end,
                        rag,
                        on_contact
                    );
                }
            }
        );
    }

    [[nodiscard]] std::vector<std::ptrdiff_t> make_strides() const {
        std::vector<std::ptrdiff_t> shape(shape_.size());
        for (std::size_t axis = 0; axis < shape_.size(); ++axis) {
            shape[axis] = static_cast<std::ptrdiff_t>(shape_[axis]);
        }
        return bioimage_cpp::detail::c_order_strides(shape);
    }

    void decode(
        const std::uint64_t flat,
        const std::vector<std::ptrdiff_t> &strides,
        std::vector<std::uint64_t> &out
    ) const {
        auto remainder = flat;
        for (std::size_t axis = 0; axis < shape_.size(); ++axis) {
            const auto stride = static_cast<std::uint64_t>(strides[axis]);
            out.push_back(remainder / stride);
            remainder %= stride;
        }
    }

    static bool selected(const std::uint64_t local_index, const int edge_direction) {
        if (edge_direction == 0) {
            return true;
        }
        if (edge_direction == 1) {
            return (local_index % 2) == 0;
        }
        return (local_index % 2) == 1;
    }

    static void check_edge_direction(const int edge_direction) {
        if (edge_direction < 0 || edge_direction > 2) {
            throw std::invalid_argument("edge_direction must be 0, 1, or 2");
        }
    }

    void check_edge(const std::uint64_t edge) const {
        if (edge >= number_of_edges()) {
            throw std::invalid_argument("edge id out of range");
        }
    }

    std::vector<std::uint64_t> shape_;
    std::vector<std::uint64_t> offsets_;
    std::vector<std::uint64_t> points_;
};

} // namespace bioimage_cpp::graph
