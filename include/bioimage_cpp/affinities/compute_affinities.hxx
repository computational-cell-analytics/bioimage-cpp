#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/threading.hxx"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

// Pairwise affinity computation from a label volume. Mirrors affogato's
// `affinities/affinities.hxx::compute_affinities` but without the xtensor
// dependency, with proper input validation at the binding boundary, and with
// optional output mask + optional per-offset parallelism.
//
// For each spatial coordinate ``c`` and offset index ``oi``,
// ``affs[oi, c] = 1`` iff ``labels[c] == labels[c + offsets[oi]]``, else 0.
// ``mask[oi, c] = 1`` iff the offset stays in bounds and neither endpoint
// equals ``ignore_label``; out-of-bounds and ignore-label positions produce
// ``affs = 0`` and ``mask = 0``.
namespace bioimage_cpp::affinities {

// Boolean affinities on a 2D label volume. Preconditions (validated in the
// binding layer):
//   * labels.ndim() == 2
//   * affs.shape == {n_offsets, labels.shape[0], labels.shape[1]}
//   * if mask is non-null: mask->shape == affs.shape
//   * each entry of offsets has length 2
template <class LabelT, class AffT>
void compute_affinities_2d(
    const ConstArrayView<LabelT> &labels,
    const std::vector<std::array<std::ptrdiff_t, 2>> &offsets,
    const ArrayView<AffT> &affs,
    const ArrayView<std::uint8_t> *mask,
    const std::optional<LabelT> ignore_label,
    const std::size_t number_of_threads = 1
) {
    const auto height = labels.shape[0];
    const auto width = labels.shape[1];
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

                AffT * const affs_channel =
                    affs.data + static_cast<std::ptrdiff_t>(oi) * plane;
                std::uint8_t * const mask_channel = (mask != nullptr)
                    ? mask->data + static_cast<std::ptrdiff_t>(oi) * plane
                    : nullptr;

                std::fill_n(affs_channel, plane, AffT{0});
                if (mask_channel != nullptr) {
                    std::fill_n(mask_channel, plane, std::uint8_t{0});
                }

                // Sub-rectangle where (y, x) AND (y+dy, x+dx) are both in bounds.
                const auto y_begin = std::max<std::ptrdiff_t>(0, -dy);
                const auto y_end = height - std::max<std::ptrdiff_t>(0, dy);
                const auto x_begin = std::max<std::ptrdiff_t>(0, -dx);
                const auto x_end = width - std::max<std::ptrdiff_t>(0, dx);
                if (y_begin >= y_end || x_begin >= x_end) {
                    continue;
                }

                for (std::ptrdiff_t y = y_begin; y < y_end; ++y) {
                    const auto ny = y + dy;
                    const LabelT * const row = labels.data + y * width;
                    const LabelT * const neighbor_row = labels.data + ny * width;
                    AffT * const out_row = affs_channel + y * width;
                    std::uint8_t * const mask_row = (mask_channel != nullptr)
                        ? mask_channel + y * width : nullptr;
                    for (std::ptrdiff_t x = x_begin; x < x_end; ++x) {
                        const LabelT a = row[x];
                        const LabelT b = neighbor_row[x + dx];
                        if (ignore_label.has_value()) {
                            const LabelT ig = *ignore_label;
                            if (a == ig || b == ig) {
                                continue; // affs/mask already 0
                            }
                        }
                        out_row[x] = (a == b) ? AffT{1} : AffT{0};
                        if (mask_row != nullptr) {
                            mask_row[x] = 1;
                        }
                    }
                }
            }
        }
    );
}

// Boolean affinities on a 3D label volume. Preconditions identical to the 2D
// case with one extra axis.
template <class LabelT, class AffT>
void compute_affinities_3d(
    const ConstArrayView<LabelT> &labels,
    const std::vector<std::array<std::ptrdiff_t, 3>> &offsets,
    const ArrayView<AffT> &affs,
    const ArrayView<std::uint8_t> *mask,
    const std::optional<LabelT> ignore_label,
    const std::size_t number_of_threads = 1
) {
    const auto depth = labels.shape[0];
    const auto height = labels.shape[1];
    const auto width = labels.shape[2];
    const auto volume = depth * height * width;
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
                const auto dz = offsets[oi][0];
                const auto dy = offsets[oi][1];
                const auto dx = offsets[oi][2];

                AffT * const affs_channel =
                    affs.data + static_cast<std::ptrdiff_t>(oi) * volume;
                std::uint8_t * const mask_channel = (mask != nullptr)
                    ? mask->data + static_cast<std::ptrdiff_t>(oi) * volume
                    : nullptr;

                std::fill_n(affs_channel, volume, AffT{0});
                if (mask_channel != nullptr) {
                    std::fill_n(mask_channel, volume, std::uint8_t{0});
                }

                const auto z_begin = std::max<std::ptrdiff_t>(0, -dz);
                const auto z_end = depth - std::max<std::ptrdiff_t>(0, dz);
                const auto y_begin = std::max<std::ptrdiff_t>(0, -dy);
                const auto y_end = height - std::max<std::ptrdiff_t>(0, dy);
                const auto x_begin = std::max<std::ptrdiff_t>(0, -dx);
                const auto x_end = width - std::max<std::ptrdiff_t>(0, dx);
                if (z_begin >= z_end || y_begin >= y_end || x_begin >= x_end) {
                    continue;
                }

                for (std::ptrdiff_t z = z_begin; z < z_end; ++z) {
                    const auto nz = z + dz;
                    const LabelT * const slab = labels.data + z * plane;
                    const LabelT * const neighbor_slab = labels.data + nz * plane;
                    AffT * const out_slab = affs_channel + z * plane;
                    std::uint8_t * const mask_slab = (mask_channel != nullptr)
                        ? mask_channel + z * plane : nullptr;
                    for (std::ptrdiff_t y = y_begin; y < y_end; ++y) {
                        const auto ny = y + dy;
                        const LabelT * const row = slab + y * width;
                        const LabelT * const neighbor_row = neighbor_slab + ny * width;
                        AffT * const out_row = out_slab + y * width;
                        std::uint8_t * const mask_row = (mask_slab != nullptr)
                            ? mask_slab + y * width : nullptr;
                        for (std::ptrdiff_t x = x_begin; x < x_end; ++x) {
                            const LabelT a = row[x];
                            const LabelT b = neighbor_row[x + dx];
                            if (ignore_label.has_value()) {
                                const LabelT ig = *ignore_label;
                                if (a == ig || b == ig) {
                                    continue;
                                }
                            }
                            out_row[x] = (a == b) ? AffT{1} : AffT{0};
                            if (mask_row != nullptr) {
                                mask_row[x] = 1;
                            }
                        }
                    }
                }
            }
        }
    );
}

} // namespace bioimage_cpp::affinities
