#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/threading.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bioimage_cpp::distance {

namespace detail {

template <class PointT>
inline std::ptrdiff_t point_to_flat(
    const PointT *coord_row,
    const std::vector<std::ptrdiff_t> &strides,
    std::ptrdiff_t ndim
) {
    std::ptrdiff_t flat = 0;
    for (std::ptrdiff_t d = 0; d < ndim; ++d) {
        flat += static_cast<std::ptrdiff_t>(coord_row[d]) *
                strides[static_cast<std::size_t>(d)];
    }
    return flat;
}

} // namespace detail

// Non-maximum suppression of candidate points by a distance map.
//
// For each input point p_i, let d_i = distance_map[p_i]. Among all input
// points (including i itself) within Euclidean distance d_i of p_i, the one
// with the largest distance_map value is selected. The unique set of selected
// indices is returned in ascending order via `kept_indices`.
//
// Matches `nifty.filters.nonMaximumDistanceSuppression`, including its float
// arithmetic: coordinate differences and their squared sum accumulate in
// float, the Euclidean distance is `float(sqrt(sum))`, and the neighborhood
// test compares that distance directly against d_i. Replicating this exactly
// (rather than comparing squared distances) keeps boundary ties identical to
// nifty.
//
// Complexity: O(N^2) time and O(number_of_threads * N) auxiliary memory.
template <class PointT>
inline void non_maximum_distance_suppression(
    const ConstArrayView<float> &distance_map,
    const ConstArrayView<PointT> &points,
    std::vector<std::size_t> &kept_indices,
    const std::size_t number_of_threads = 1
) {
    if (distance_map.ndim() < 1) {
        throw std::invalid_argument(
            "distance_map must have ndim >= 1, got ndim=0"
        );
    }
    if (points.ndim() != 2) {
        throw std::invalid_argument(
            "points must have ndim == 2, got ndim=" + std::to_string(points.ndim())
        );
    }
    const auto n_points = points.shape[0];
    const auto coord_ndim = points.shape[1];
    if (coord_ndim != distance_map.ndim()) {
        throw std::invalid_argument(
            "points second axis must match distance_map ndim, got points.shape[1]=" +
            std::to_string(coord_ndim) + ", distance_map.ndim()=" +
            std::to_string(distance_map.ndim())
        );
    }

    kept_indices.clear();
    if (n_points == 0) {
        return;
    }

    const auto strides = bioimage_cpp::detail::c_order_strides(distance_map.shape);
    const auto n = static_cast<std::size_t>(n_points);

    // Precompute flat index and distance value at each point. Point coordinates
    // are validated against distance_map bounds in the binding layer before this
    // is called (see bindings/distance.cxx).
    std::vector<float> point_dist(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto *row =
            points.data + static_cast<std::ptrdiff_t>(i) * coord_ndim;
        const auto flat = detail::point_to_flat(row, strides, coord_ndim);
        point_dist[i] = distance_map.data[flat];
    }

    const auto threads = bioimage_cpp::detail::normalize_thread_count(
        number_of_threads, n
    );
    constexpr auto no_candidate = std::numeric_limits<std::size_t>::max();
    std::vector<std::vector<std::size_t>> local_best(
        threads, std::vector<std::size_t>(n, no_candidate)
    );
    const auto consider = [&](std::size_t &current, const std::size_t candidate) {
        const float value = point_dist[candidate];
        if (current == no_candidate) {
            if (value > -std::numeric_limits<float>::infinity()) {
                current = candidate;
            }
            return;
        }
        const float current_value = point_dist[current];
        if (value > current_value || (value == current_value && candidate < current)) {
            current = candidate;
        }
    };

    bioimage_cpp::detail::parallel_for_chunks(
        threads, threads,
        [&](const std::size_t thread_id, const std::size_t, const std::size_t) {
            auto &best = local_best[thread_id];
            for (std::size_t i = thread_id; i < n; i += threads) {
                const auto *row_i =
                    points.data + static_cast<std::ptrdiff_t>(i) * coord_ndim;
                for (std::size_t j = i + 1; j < n; ++j) {
                    const auto *row_j =
                        points.data + static_cast<std::ptrdiff_t>(j) * coord_ndim;
                    float sum_sq = 0.0f;
                    for (std::ptrdiff_t d = 0; d < coord_ndim; ++d) {
                        const float diff = static_cast<float>(row_i[d]) -
                                           static_cast<float>(row_j[d]);
                        sum_sq += diff * diff;
                    }
                    const auto distance = static_cast<float>(
                        std::sqrt(static_cast<double>(sum_sq))
                    );
                    if (!(distance > point_dist[i])) {
                        consider(best[i], j);
                    }
                    if (!(distance > point_dist[j])) {
                        consider(best[j], i);
                    }
                }
            }
        }
    );

    std::vector<std::size_t> bests;
    bests.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t best_idx = no_candidate;
        if (!(0.0f > point_dist[i])) {
            consider(best_idx, i);
        }
        for (std::size_t thread = 0; thread < threads; ++thread) {
            const auto candidate = local_best[thread][i];
            if (candidate != no_candidate) {
                consider(best_idx, candidate);
            }
        }
        bests.push_back(best_idx == no_candidate ? i : best_idx);
    }

    std::sort(bests.begin(), bests.end());
    bests.erase(std::unique(bests.begin(), bests.end()), bests.end());
    kept_indices = std::move(bests);
}

} // namespace bioimage_cpp::distance
