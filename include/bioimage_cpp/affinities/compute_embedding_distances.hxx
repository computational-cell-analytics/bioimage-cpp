#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/threading.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// Pairwise distances between an embedding tensor and itself under a set of
// spatial offsets. Mirrors affogato's `compute_embedding_distances_impl`
// without the xtensor dependency, with the norm selected by an enum branched
// once per offset (the inner spatial loop has no per-pixel branch on norm).
//
// Input  values    : shape (C, *spatial), C-contiguous.
// Output distances : shape (n_offsets, *spatial). Out-of-bounds positions
//                    are left at 0.0; in-bounds positions hold
//                    norm(values[:, p], values[:, p + offset]).
namespace bioimage_cpp::affinities {

enum class EmbeddingNorm { L1, L2, Cosine };

namespace detail {

template <class ValueT>
inline ValueT embedding_distance_l1(
    const ValueT *a, const ValueT *b, std::ptrdiff_t n_channels,
    std::ptrdiff_t channel_stride
) {
    double acc = 0.0;
    for (std::ptrdiff_t c = 0; c < n_channels; ++c) {
        const double diff = static_cast<double>(a[c * channel_stride])
                          - static_cast<double>(b[c * channel_stride]);
        acc += std::abs(diff);
    }
    return static_cast<ValueT>(acc);
}

template <class ValueT>
inline ValueT embedding_distance_l2(
    const ValueT *a, const ValueT *b, std::ptrdiff_t n_channels,
    std::ptrdiff_t channel_stride
) {
    double acc = 0.0;
    for (std::ptrdiff_t c = 0; c < n_channels; ++c) {
        const double diff = static_cast<double>(a[c * channel_stride])
                          - static_cast<double>(b[c * channel_stride]);
        acc += diff * diff;
    }
    return static_cast<ValueT>(std::sqrt(acc));
}

template <class ValueT>
inline ValueT embedding_distance_cosine(
    const ValueT *a, const ValueT *b, std::ptrdiff_t n_channels,
    std::ptrdiff_t channel_stride
) {
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (std::ptrdiff_t c = 0; c < n_channels; ++c) {
        const double va = static_cast<double>(a[c * channel_stride]);
        const double vb = static_cast<double>(b[c * channel_stride]);
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }
    return static_cast<ValueT>(1.0 - dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
}

} // namespace detail

// Embedding distances on a 2D spatial grid. Preconditions (validated in the
// binding layer):
//   * values.ndim() == 3 with shape (C, H, W)
//   * distances.shape == {n_offsets, H, W}
//   * each entry of offsets has length 2
template <class ValueT>
void compute_embedding_distances_2d(
    const ConstArrayView<ValueT> &values,
    const std::vector<std::array<std::ptrdiff_t, 2>> &offsets,
    const ArrayView<ValueT> &distances,
    EmbeddingNorm norm,
    const std::size_t number_of_threads = 1
) {
    const auto n_channels = values.shape[0];
    const auto height = values.shape[1];
    const auto width = values.shape[2];
    const auto plane = height * width;
    const auto number_of_offsets = offsets.size();

    const auto n_threads = ::bioimage_cpp::detail::normalize_thread_count(
        number_of_threads, number_of_offsets
    );

    ::bioimage_cpp::detail::parallel_for_chunks(
        n_threads,
        number_of_offsets,
        [&](const std::size_t, const std::size_t begin, const std::size_t end) {
            for (std::size_t oi = begin; oi < end; ++oi) {
                const auto dy = offsets[oi][0];
                const auto dx = offsets[oi][1];

                ValueT * const dist_channel =
                    distances.data + static_cast<std::ptrdiff_t>(oi) * plane;
                std::fill_n(dist_channel, plane, ValueT{0});

                const auto y_begin = std::max<std::ptrdiff_t>(0, -dy);
                const auto y_end = height - std::max<std::ptrdiff_t>(0, dy);
                const auto x_begin = std::max<std::ptrdiff_t>(0, -dx);
                const auto x_end = width - std::max<std::ptrdiff_t>(0, dx);
                if (y_begin >= y_end || x_begin >= x_end) {
                    continue;
                }

                auto run_norm = [&](auto kernel) {
                    for (std::ptrdiff_t y = y_begin; y < y_end; ++y) {
                        const auto ny = y + dy;
                        ValueT * const out_row = dist_channel + y * width;
                        for (std::ptrdiff_t x = x_begin; x < x_end; ++x) {
                            const auto nx = x + dx;
                            const ValueT * const a = values.data + y * width + x;
                            const ValueT * const b = values.data + ny * width + nx;
                            out_row[x] = kernel(a, b, n_channels, plane);
                        }
                    }
                };

                switch (norm) {
                    case EmbeddingNorm::L1:
                        run_norm(&detail::embedding_distance_l1<ValueT>);
                        break;
                    case EmbeddingNorm::L2:
                        run_norm(&detail::embedding_distance_l2<ValueT>);
                        break;
                    case EmbeddingNorm::Cosine:
                        run_norm(&detail::embedding_distance_cosine<ValueT>);
                        break;
                }
            }
        }
    );
}

// Embedding distances on a 3D spatial grid. Preconditions identical to the
// 2D case with one extra spatial axis (values shape == (C, D, H, W)).
template <class ValueT>
void compute_embedding_distances_3d(
    const ConstArrayView<ValueT> &values,
    const std::vector<std::array<std::ptrdiff_t, 3>> &offsets,
    const ArrayView<ValueT> &distances,
    EmbeddingNorm norm,
    const std::size_t number_of_threads = 1
) {
    const auto n_channels = values.shape[0];
    const auto depth = values.shape[1];
    const auto height = values.shape[2];
    const auto width = values.shape[3];
    const auto plane = height * width;
    const auto volume = depth * plane;
    const auto number_of_offsets = offsets.size();

    const auto n_threads = ::bioimage_cpp::detail::normalize_thread_count(
        number_of_threads, number_of_offsets
    );

    ::bioimage_cpp::detail::parallel_for_chunks(
        n_threads,
        number_of_offsets,
        [&](const std::size_t, const std::size_t begin, const std::size_t end) {
            for (std::size_t oi = begin; oi < end; ++oi) {
                const auto dz = offsets[oi][0];
                const auto dy = offsets[oi][1];
                const auto dx = offsets[oi][2];

                ValueT * const dist_channel =
                    distances.data + static_cast<std::ptrdiff_t>(oi) * volume;
                std::fill_n(dist_channel, volume, ValueT{0});

                const auto z_begin = std::max<std::ptrdiff_t>(0, -dz);
                const auto z_end = depth - std::max<std::ptrdiff_t>(0, dz);
                const auto y_begin = std::max<std::ptrdiff_t>(0, -dy);
                const auto y_end = height - std::max<std::ptrdiff_t>(0, dy);
                const auto x_begin = std::max<std::ptrdiff_t>(0, -dx);
                const auto x_end = width - std::max<std::ptrdiff_t>(0, dx);
                if (z_begin >= z_end || y_begin >= y_end || x_begin >= x_end) {
                    continue;
                }

                auto run_norm = [&](auto kernel) {
                    for (std::ptrdiff_t z = z_begin; z < z_end; ++z) {
                        const auto nz = z + dz;
                        ValueT * const out_slab = dist_channel + z * plane;
                        for (std::ptrdiff_t y = y_begin; y < y_end; ++y) {
                            const auto ny = y + dy;
                            ValueT * const out_row = out_slab + y * width;
                            for (std::ptrdiff_t x = x_begin; x < x_end; ++x) {
                                const auto nx = x + dx;
                                const ValueT * const a =
                                    values.data + z * plane + y * width + x;
                                const ValueT * const b =
                                    values.data + nz * plane + ny * width + nx;
                                out_row[x] = kernel(a, b, n_channels, volume);
                            }
                        }
                    }
                };

                switch (norm) {
                    case EmbeddingNorm::L1:
                        run_norm(&detail::embedding_distance_l1<ValueT>);
                        break;
                    case EmbeddingNorm::L2:
                        run_norm(&detail::embedding_distance_l2<ValueT>);
                        break;
                    case EmbeddingNorm::Cosine:
                        run_norm(&detail::embedding_distance_cosine<ValueT>);
                        break;
                }
            }
        }
    );
}

} // namespace bioimage_cpp::affinities
