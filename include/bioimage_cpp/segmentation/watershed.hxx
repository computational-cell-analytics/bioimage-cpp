#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/profile.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace bioimage_cpp {

namespace detail_ws {

// Bucket-queue marker watershed. Quantises the heightmap to `n_levels`
// uint16 levels and floods one level at a time with Meyer-style monotone
// semantics: a neighbour pushed from level L always lands at
// max(level[neighbour], L). This is O(N + n_levels) — effectively O(N) for
// typical bioimage volumes — and replaces the std::priority_queue path that
// the first version of this function used.
//
// Tie-breaking on equal levels (including ULP-level differences in the
// quantised range) is intentionally unspecified, matching the documented
// API contract.

constexpr std::size_t k_n_levels = 65536;

template <class HeightT, class LabelT>
void watershed_2d(
    const ConstArrayView<HeightT> &image,
    const ConstArrayView<LabelT> &markers,
    const ConstArrayView<std::uint8_t> &mask,
    const ArrayView<LabelT> &out
) {
    BIOIMAGE_PROFILE_INIT(profile);

    const bool has_mask = !mask.shape.empty();
    const std::int64_t Y = image.shape[0];
    const std::int64_t X = image.shape[1];
    const std::uint64_t N =
        static_cast<std::uint64_t>(Y) * static_cast<std::uint64_t>(X);

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "init_output");
        for (std::uint64_t node = 0; node < N; ++node) {
            out.data[node] = LabelT{0};
        }
    }

    HeightT h_min{};
    HeightT h_max{};
    bool any_valid = false;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "range");
        for (std::uint64_t node = 0; node < N; ++node) {
            if (has_mask && mask.data[node] == 0) {
                continue;
            }
            const HeightT v = image.data[node];
            if (!any_valid) {
                h_min = v;
                h_max = v;
                any_valid = true;
            } else {
                if (v < h_min) {
                    h_min = v;
                }
                if (v > h_max) {
                    h_max = v;
                }
            }
        }
    }
    if (!any_valid) {
        BIOIMAGE_PROFILE_REPORT(profile);
        return;
    }

    std::vector<std::uint16_t> levels(static_cast<std::size_t>(N), 0);
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "quantize");
        const double range = static_cast<double>(h_max) - static_cast<double>(h_min);
        if (range > 0.0) {
            const double scale = static_cast<double>(k_n_levels - 1) / range;
            const double max_level = static_cast<double>(k_n_levels - 1);
            for (std::uint64_t node = 0; node < N; ++node) {
                if (has_mask && mask.data[node] == 0) {
                    continue;
                }
                double q =
                    (static_cast<double>(image.data[node]) - static_cast<double>(h_min)) * scale;
                if (q < 0.0) {
                    q = 0.0;
                } else if (q > max_level) {
                    q = max_level;
                }
                levels[node] = static_cast<std::uint16_t>(q);
            }
        }
    }

    std::vector<std::vector<std::uint64_t>> buckets(k_n_levels);
    std::size_t current_level = k_n_levels;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "seed_pass");
        for (std::uint64_t node = 0; node < N; ++node) {
            if (has_mask && mask.data[node] == 0) {
                continue;
            }
            const auto marker = markers.data[node];
            if (marker != LabelT{0}) {
                out.data[node] = marker;
                const std::size_t lvl = levels[node];
                buckets[lvl].push_back(node);
                if (lvl < current_level) {
                    current_level = lvl;
                }
            }
        }
    }

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "main_loop");
        while (current_level < k_n_levels) {
            auto &bucket = buckets[current_level];
            while (!bucket.empty()) {
                const std::uint64_t node = bucket.back();
                bucket.pop_back();
                const auto label = out.data[node];
                const std::int64_t y = static_cast<std::int64_t>(node) / X;
                const std::int64_t x = static_cast<std::int64_t>(node) - y * X;

                if (y > 0) {
                    const std::uint64_t n = node - static_cast<std::uint64_t>(X);
                    if ((!has_mask || mask.data[n] != 0) && out.data[n] == LabelT{0}) {
                        out.data[n] = label;
                        std::size_t lvl = levels[n];
                        if (lvl < current_level) lvl = current_level;
                        buckets[lvl].push_back(n);
                    }
                }
                if (y + 1 < Y) {
                    const std::uint64_t n = node + static_cast<std::uint64_t>(X);
                    if ((!has_mask || mask.data[n] != 0) && out.data[n] == LabelT{0}) {
                        out.data[n] = label;
                        std::size_t lvl = levels[n];
                        if (lvl < current_level) lvl = current_level;
                        buckets[lvl].push_back(n);
                    }
                }
                if (x > 0) {
                    const std::uint64_t n = node - 1;
                    if ((!has_mask || mask.data[n] != 0) && out.data[n] == LabelT{0}) {
                        out.data[n] = label;
                        std::size_t lvl = levels[n];
                        if (lvl < current_level) lvl = current_level;
                        buckets[lvl].push_back(n);
                    }
                }
                if (x + 1 < X) {
                    const std::uint64_t n = node + 1;
                    if ((!has_mask || mask.data[n] != 0) && out.data[n] == LabelT{0}) {
                        out.data[n] = label;
                        std::size_t lvl = levels[n];
                        if (lvl < current_level) lvl = current_level;
                        buckets[lvl].push_back(n);
                    }
                }
            }
            ++current_level;
        }
    }

    BIOIMAGE_PROFILE_REPORT(profile);
}

template <class HeightT, class LabelT>
void watershed_3d(
    const ConstArrayView<HeightT> &image,
    const ConstArrayView<LabelT> &markers,
    const ConstArrayView<std::uint8_t> &mask,
    const ArrayView<LabelT> &out
) {
    BIOIMAGE_PROFILE_INIT(profile);

    const bool has_mask = !mask.shape.empty();
    const std::int64_t Z = image.shape[0];
    const std::int64_t Y = image.shape[1];
    const std::int64_t X = image.shape[2];
    const std::int64_t YX = Y * X;
    const std::uint64_t N =
        static_cast<std::uint64_t>(Z) * static_cast<std::uint64_t>(YX);

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "init_output");
        for (std::uint64_t node = 0; node < N; ++node) {
            out.data[node] = LabelT{0};
        }
    }

    HeightT h_min{};
    HeightT h_max{};
    bool any_valid = false;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "range");
        for (std::uint64_t node = 0; node < N; ++node) {
            if (has_mask && mask.data[node] == 0) {
                continue;
            }
            const HeightT v = image.data[node];
            if (!any_valid) {
                h_min = v;
                h_max = v;
                any_valid = true;
            } else {
                if (v < h_min) {
                    h_min = v;
                }
                if (v > h_max) {
                    h_max = v;
                }
            }
        }
    }
    if (!any_valid) {
        BIOIMAGE_PROFILE_REPORT(profile);
        return;
    }

    std::vector<std::uint16_t> levels(static_cast<std::size_t>(N), 0);
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "quantize");
        const double range = static_cast<double>(h_max) - static_cast<double>(h_min);
        if (range > 0.0) {
            const double scale = static_cast<double>(k_n_levels - 1) / range;
            const double max_level = static_cast<double>(k_n_levels - 1);
            for (std::uint64_t node = 0; node < N; ++node) {
                if (has_mask && mask.data[node] == 0) {
                    continue;
                }
                double q =
                    (static_cast<double>(image.data[node]) - static_cast<double>(h_min)) * scale;
                if (q < 0.0) {
                    q = 0.0;
                } else if (q > max_level) {
                    q = max_level;
                }
                levels[node] = static_cast<std::uint16_t>(q);
            }
        }
    }

    std::vector<std::vector<std::uint64_t>> buckets(k_n_levels);
    std::size_t current_level = k_n_levels;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "seed_pass");
        for (std::uint64_t node = 0; node < N; ++node) {
            if (has_mask && mask.data[node] == 0) {
                continue;
            }
            const auto marker = markers.data[node];
            if (marker != LabelT{0}) {
                out.data[node] = marker;
                const std::size_t lvl = levels[node];
                buckets[lvl].push_back(node);
                if (lvl < current_level) {
                    current_level = lvl;
                }
            }
        }
    }

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "main_loop");
        while (current_level < k_n_levels) {
            auto &bucket = buckets[current_level];
            while (!bucket.empty()) {
                const std::uint64_t node = bucket.back();
                bucket.pop_back();
                const auto label = out.data[node];
                const std::int64_t z = static_cast<std::int64_t>(node) / YX;
                const std::int64_t rem = static_cast<std::int64_t>(node) - z * YX;
                const std::int64_t y = rem / X;
                const std::int64_t x = rem - y * X;

                if (z > 0) {
                    const std::uint64_t n = node - static_cast<std::uint64_t>(YX);
                    if ((!has_mask || mask.data[n] != 0) && out.data[n] == LabelT{0}) {
                        out.data[n] = label;
                        std::size_t lvl = levels[n];
                        if (lvl < current_level) lvl = current_level;
                        buckets[lvl].push_back(n);
                    }
                }
                if (z + 1 < Z) {
                    const std::uint64_t n = node + static_cast<std::uint64_t>(YX);
                    if ((!has_mask || mask.data[n] != 0) && out.data[n] == LabelT{0}) {
                        out.data[n] = label;
                        std::size_t lvl = levels[n];
                        if (lvl < current_level) lvl = current_level;
                        buckets[lvl].push_back(n);
                    }
                }
                if (y > 0) {
                    const std::uint64_t n = node - static_cast<std::uint64_t>(X);
                    if ((!has_mask || mask.data[n] != 0) && out.data[n] == LabelT{0}) {
                        out.data[n] = label;
                        std::size_t lvl = levels[n];
                        if (lvl < current_level) lvl = current_level;
                        buckets[lvl].push_back(n);
                    }
                }
                if (y + 1 < Y) {
                    const std::uint64_t n = node + static_cast<std::uint64_t>(X);
                    if ((!has_mask || mask.data[n] != 0) && out.data[n] == LabelT{0}) {
                        out.data[n] = label;
                        std::size_t lvl = levels[n];
                        if (lvl < current_level) lvl = current_level;
                        buckets[lvl].push_back(n);
                    }
                }
                if (x > 0) {
                    const std::uint64_t n = node - 1;
                    if ((!has_mask || mask.data[n] != 0) && out.data[n] == LabelT{0}) {
                        out.data[n] = label;
                        std::size_t lvl = levels[n];
                        if (lvl < current_level) lvl = current_level;
                        buckets[lvl].push_back(n);
                    }
                }
                if (x + 1 < X) {
                    const std::uint64_t n = node + 1;
                    if ((!has_mask || mask.data[n] != 0) && out.data[n] == LabelT{0}) {
                        out.data[n] = label;
                        std::size_t lvl = levels[n];
                        if (lvl < current_level) lvl = current_level;
                        buckets[lvl].push_back(n);
                    }
                }
            }
            ++current_level;
        }
    }

    BIOIMAGE_PROFILE_REPORT(profile);
}

} // namespace detail_ws

// Marker-controlled flooding watershed on a 2D or 3D image.
//
// `image` is the heightmap. `markers` carries non-zero seed labels that get
// propagated to neighbouring pixels in order of increasing height. If `mask`
// is non-empty, only pixels with a non-zero mask value participate; the
// remaining pixels stay 0 in the output. Connectivity is 1 (axis-aligned
// 4-neighbours in 2D, 6-neighbours in 3D).
//
// Internally uses a 65536-bucket queue with Meyer-style monotone flooding.
// Tie-breaking on equal heights (including 1-ULP differences within a
// quantisation bucket) is unspecified, as documented in the Python wrapper.
template <class HeightT, class LabelT>
void watershed(
    const ConstArrayView<HeightT> &image,
    const ConstArrayView<LabelT> &markers,
    const ConstArrayView<std::uint8_t> &mask,
    const ArrayView<LabelT> &out
) {
    if (image.ndim() != 2 && image.ndim() != 3) {
        throw std::invalid_argument(
            "image must have ndim 2 or 3, got ndim=" + std::to_string(image.ndim())
        );
    }
    if (markers.shape != image.shape) {
        throw std::invalid_argument("markers shape must match image shape");
    }
    if (out.shape != image.shape) {
        throw std::invalid_argument("out shape must match image shape");
    }
    const bool has_mask = !mask.shape.empty();
    if (has_mask && mask.shape != image.shape) {
        throw std::invalid_argument("mask shape must match image shape");
    }

    if (image.ndim() == 2) {
        detail_ws::watershed_2d<HeightT, LabelT>(image, markers, mask, out);
    } else {
        detail_ws::watershed_3d<HeightT, LabelT>(image, markers, mask, out);
    }
}

} // namespace bioimage_cpp
