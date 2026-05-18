#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/profile.hxx"

#include <algorithm>
#include <array>
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

// Bucket entry for affinity-based watershed. Each entry is a pending claim
// for `target` by `source_label`; the actual labelling happens at pop time
// because competing claims with different priorities can target the same
// pixel.
template <class LabelT>
struct AffinityWatershedEntry {
    std::uint64_t target;
    LabelT source_label;
};

template <int Sign, class AffT, class LabelT>
void watershed_2d_from_affinities(
    const ConstArrayView<AffT> &affinities,
    const std::array<std::size_t, 2> &channel_for_axis,
    const ConstArrayView<LabelT> &markers,
    const ConstArrayView<std::uint8_t> &mask,
    const ArrayView<LabelT> &out
) {
    BIOIMAGE_PROFILE_INIT(profile);

    const bool has_mask = !mask.shape.empty();
    const std::size_t C = static_cast<std::size_t>(affinities.shape[0]);
    const std::int64_t Y = affinities.shape[1];
    const std::int64_t X = affinities.shape[2];
    const std::uint64_t N =
        static_cast<std::uint64_t>(Y) * static_cast<std::uint64_t>(X);

    const std::size_t cy = channel_for_axis[0];
    const std::size_t cx = channel_for_axis[1];

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "init_output");
        for (std::uint64_t node = 0; node < N; ++node) {
            out.data[node] = LabelT{0};
        }
    }

    AffT a_min{};
    AffT a_max{};
    bool any_valid = false;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "range");
        const std::size_t total = C * static_cast<std::size_t>(N);
        for (std::size_t i = 0; i < total; ++i) {
            if (has_mask) {
                const std::uint64_t node = static_cast<std::uint64_t>(i) % N;
                if (mask.data[node] == 0) {
                    continue;
                }
            }
            const AffT v = affinities.data[i];
            if (!any_valid) {
                a_min = v;
                a_max = v;
                any_valid = true;
            } else {
                if (v < a_min) a_min = v;
                if (v > a_max) a_max = v;
            }
        }
    }
    if (!any_valid) {
        BIOIMAGE_PROFILE_REPORT(profile);
        return;
    }

    std::vector<std::uint16_t> levels(C * static_cast<std::size_t>(N), 0);
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "quantize");
        const double a_max_d = static_cast<double>(a_max);
        const double range = a_max_d - static_cast<double>(a_min);
        if (range > 0.0) {
            const double scale = static_cast<double>(k_n_levels - 1) / range;
            const double max_level = static_cast<double>(k_n_levels - 1);
            const std::size_t total = C * static_cast<std::size_t>(N);
            for (std::size_t i = 0; i < total; ++i) {
                double q =
                    (a_max_d - static_cast<double>(affinities.data[i])) * scale;
                if (q < 0.0) {
                    q = 0.0;
                } else if (q > max_level) {
                    q = max_level;
                }
                levels[i] = static_cast<std::uint16_t>(q);
            }
        }
    }

    using Entry = AffinityWatershedEntry<LabelT>;
    std::vector<std::vector<Entry>> buckets(k_n_levels);
    std::size_t current_level = k_n_levels;

    auto try_push_seed = [&](std::uint64_t neighbor, LabelT label, std::size_t edge_lvl) {
        if (has_mask && mask.data[neighbor] == 0) return;
        if (out.data[neighbor] != LabelT{0}) return;
        buckets[edge_lvl].push_back(Entry{neighbor, label});
        if (edge_lvl < current_level) current_level = edge_lvl;
    };
    auto try_push_main = [&](std::uint64_t neighbor, LabelT label, std::size_t edge_lvl) {
        if (has_mask && mask.data[neighbor] == 0) return;
        if (out.data[neighbor] != LabelT{0}) return;
        if (edge_lvl < current_level) edge_lvl = current_level;
        buckets[edge_lvl].push_back(Entry{neighbor, label});
    };

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "seed_pass");
        for (std::uint64_t node = 0; node < N; ++node) {
            if (has_mask && mask.data[node] == 0) continue;
            const auto marker = markers.data[node];
            if (marker == LabelT{0}) continue;
            out.data[node] = marker;

            const std::int64_t y = static_cast<std::int64_t>(node) / X;
            const std::int64_t x = static_cast<std::int64_t>(node) - y * X;

            if constexpr (Sign < 0) {
                if (y > 0) {
                    const std::uint64_t n = node - static_cast<std::uint64_t>(X);
                    try_push_seed(n, marker, levels[cy * N + node]);
                }
                if (y + 1 < Y) {
                    const std::uint64_t n = node + static_cast<std::uint64_t>(X);
                    try_push_seed(n, marker, levels[cy * N + n]);
                }
                if (x > 0) {
                    const std::uint64_t n = node - 1;
                    try_push_seed(n, marker, levels[cx * N + node]);
                }
                if (x + 1 < X) {
                    const std::uint64_t n = node + 1;
                    try_push_seed(n, marker, levels[cx * N + n]);
                }
            } else {
                if (y + 1 < Y) {
                    const std::uint64_t n = node + static_cast<std::uint64_t>(X);
                    try_push_seed(n, marker, levels[cy * N + node]);
                }
                if (y > 0) {
                    const std::uint64_t n = node - static_cast<std::uint64_t>(X);
                    try_push_seed(n, marker, levels[cy * N + n]);
                }
                if (x + 1 < X) {
                    const std::uint64_t n = node + 1;
                    try_push_seed(n, marker, levels[cx * N + node]);
                }
                if (x > 0) {
                    const std::uint64_t n = node - 1;
                    try_push_seed(n, marker, levels[cx * N + n]);
                }
            }
        }
    }

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "main_loop");
        while (current_level < k_n_levels) {
            auto &bucket = buckets[current_level];
            while (!bucket.empty()) {
                const Entry e = bucket.back();
                bucket.pop_back();
                const std::uint64_t u = e.target;
                if (out.data[u] != LabelT{0}) continue;
                out.data[u] = e.source_label;
                const LabelT label = e.source_label;

                const std::int64_t y = static_cast<std::int64_t>(u) / X;
                const std::int64_t x = static_cast<std::int64_t>(u) - y * X;

                if constexpr (Sign < 0) {
                    if (y > 0) {
                        const std::uint64_t n = u - static_cast<std::uint64_t>(X);
                        try_push_main(n, label, levels[cy * N + u]);
                    }
                    if (y + 1 < Y) {
                        const std::uint64_t n = u + static_cast<std::uint64_t>(X);
                        try_push_main(n, label, levels[cy * N + n]);
                    }
                    if (x > 0) {
                        const std::uint64_t n = u - 1;
                        try_push_main(n, label, levels[cx * N + u]);
                    }
                    if (x + 1 < X) {
                        const std::uint64_t n = u + 1;
                        try_push_main(n, label, levels[cx * N + n]);
                    }
                } else {
                    if (y + 1 < Y) {
                        const std::uint64_t n = u + static_cast<std::uint64_t>(X);
                        try_push_main(n, label, levels[cy * N + u]);
                    }
                    if (y > 0) {
                        const std::uint64_t n = u - static_cast<std::uint64_t>(X);
                        try_push_main(n, label, levels[cy * N + n]);
                    }
                    if (x + 1 < X) {
                        const std::uint64_t n = u + 1;
                        try_push_main(n, label, levels[cx * N + u]);
                    }
                    if (x > 0) {
                        const std::uint64_t n = u - 1;
                        try_push_main(n, label, levels[cx * N + n]);
                    }
                }
            }
            ++current_level;
        }
    }

    BIOIMAGE_PROFILE_REPORT(profile);
}

template <int Sign, class AffT, class LabelT>
void watershed_3d_from_affinities(
    const ConstArrayView<AffT> &affinities,
    const std::array<std::size_t, 3> &channel_for_axis,
    const ConstArrayView<LabelT> &markers,
    const ConstArrayView<std::uint8_t> &mask,
    const ArrayView<LabelT> &out
) {
    BIOIMAGE_PROFILE_INIT(profile);

    const bool has_mask = !mask.shape.empty();
    const std::size_t C = static_cast<std::size_t>(affinities.shape[0]);
    const std::int64_t Z = affinities.shape[1];
    const std::int64_t Y = affinities.shape[2];
    const std::int64_t X = affinities.shape[3];
    const std::int64_t YX = Y * X;
    const std::uint64_t N =
        static_cast<std::uint64_t>(Z) * static_cast<std::uint64_t>(YX);

    const std::size_t cz = channel_for_axis[0];
    const std::size_t cy = channel_for_axis[1];
    const std::size_t cx = channel_for_axis[2];

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "init_output");
        for (std::uint64_t node = 0; node < N; ++node) {
            out.data[node] = LabelT{0};
        }
    }

    AffT a_min{};
    AffT a_max{};
    bool any_valid = false;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "range");
        const std::size_t total = C * static_cast<std::size_t>(N);
        for (std::size_t i = 0; i < total; ++i) {
            if (has_mask) {
                const std::uint64_t node = static_cast<std::uint64_t>(i) % N;
                if (mask.data[node] == 0) {
                    continue;
                }
            }
            const AffT v = affinities.data[i];
            if (!any_valid) {
                a_min = v;
                a_max = v;
                any_valid = true;
            } else {
                if (v < a_min) a_min = v;
                if (v > a_max) a_max = v;
            }
        }
    }
    if (!any_valid) {
        BIOIMAGE_PROFILE_REPORT(profile);
        return;
    }

    std::vector<std::uint16_t> levels(C * static_cast<std::size_t>(N), 0);
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "quantize");
        const double a_max_d = static_cast<double>(a_max);
        const double range = a_max_d - static_cast<double>(a_min);
        if (range > 0.0) {
            const double scale = static_cast<double>(k_n_levels - 1) / range;
            const double max_level = static_cast<double>(k_n_levels - 1);
            const std::size_t total = C * static_cast<std::size_t>(N);
            for (std::size_t i = 0; i < total; ++i) {
                double q =
                    (a_max_d - static_cast<double>(affinities.data[i])) * scale;
                if (q < 0.0) {
                    q = 0.0;
                } else if (q > max_level) {
                    q = max_level;
                }
                levels[i] = static_cast<std::uint16_t>(q);
            }
        }
    }

    using Entry = AffinityWatershedEntry<LabelT>;
    std::vector<std::vector<Entry>> buckets(k_n_levels);
    std::size_t current_level = k_n_levels;

    auto try_push_seed = [&](std::uint64_t neighbor, LabelT label, std::size_t edge_lvl) {
        if (has_mask && mask.data[neighbor] == 0) return;
        if (out.data[neighbor] != LabelT{0}) return;
        buckets[edge_lvl].push_back(Entry{neighbor, label});
        if (edge_lvl < current_level) current_level = edge_lvl;
    };
    auto try_push_main = [&](std::uint64_t neighbor, LabelT label, std::size_t edge_lvl) {
        if (has_mask && mask.data[neighbor] == 0) return;
        if (out.data[neighbor] != LabelT{0}) return;
        if (edge_lvl < current_level) edge_lvl = current_level;
        buckets[edge_lvl].push_back(Entry{neighbor, label});
    };

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "seed_pass");
        for (std::uint64_t node = 0; node < N; ++node) {
            if (has_mask && mask.data[node] == 0) continue;
            const auto marker = markers.data[node];
            if (marker == LabelT{0}) continue;
            out.data[node] = marker;

            const std::int64_t z = static_cast<std::int64_t>(node) / YX;
            const std::int64_t rem = static_cast<std::int64_t>(node) - z * YX;
            const std::int64_t y = rem / X;
            const std::int64_t x = rem - y * X;

            if constexpr (Sign < 0) {
                if (z > 0) {
                    const std::uint64_t n = node - static_cast<std::uint64_t>(YX);
                    try_push_seed(n, marker, levels[cz * N + node]);
                }
                if (z + 1 < Z) {
                    const std::uint64_t n = node + static_cast<std::uint64_t>(YX);
                    try_push_seed(n, marker, levels[cz * N + n]);
                }
                if (y > 0) {
                    const std::uint64_t n = node - static_cast<std::uint64_t>(X);
                    try_push_seed(n, marker, levels[cy * N + node]);
                }
                if (y + 1 < Y) {
                    const std::uint64_t n = node + static_cast<std::uint64_t>(X);
                    try_push_seed(n, marker, levels[cy * N + n]);
                }
                if (x > 0) {
                    const std::uint64_t n = node - 1;
                    try_push_seed(n, marker, levels[cx * N + node]);
                }
                if (x + 1 < X) {
                    const std::uint64_t n = node + 1;
                    try_push_seed(n, marker, levels[cx * N + n]);
                }
            } else {
                if (z + 1 < Z) {
                    const std::uint64_t n = node + static_cast<std::uint64_t>(YX);
                    try_push_seed(n, marker, levels[cz * N + node]);
                }
                if (z > 0) {
                    const std::uint64_t n = node - static_cast<std::uint64_t>(YX);
                    try_push_seed(n, marker, levels[cz * N + n]);
                }
                if (y + 1 < Y) {
                    const std::uint64_t n = node + static_cast<std::uint64_t>(X);
                    try_push_seed(n, marker, levels[cy * N + node]);
                }
                if (y > 0) {
                    const std::uint64_t n = node - static_cast<std::uint64_t>(X);
                    try_push_seed(n, marker, levels[cy * N + n]);
                }
                if (x + 1 < X) {
                    const std::uint64_t n = node + 1;
                    try_push_seed(n, marker, levels[cx * N + node]);
                }
                if (x > 0) {
                    const std::uint64_t n = node - 1;
                    try_push_seed(n, marker, levels[cx * N + n]);
                }
            }
        }
    }

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "main_loop");
        while (current_level < k_n_levels) {
            auto &bucket = buckets[current_level];
            while (!bucket.empty()) {
                const Entry e = bucket.back();
                bucket.pop_back();
                const std::uint64_t u = e.target;
                if (out.data[u] != LabelT{0}) continue;
                out.data[u] = e.source_label;
                const LabelT label = e.source_label;

                const std::int64_t z = static_cast<std::int64_t>(u) / YX;
                const std::int64_t rem = static_cast<std::int64_t>(u) - z * YX;
                const std::int64_t y = rem / X;
                const std::int64_t x = rem - y * X;

                if constexpr (Sign < 0) {
                    if (z > 0) {
                        const std::uint64_t n = u - static_cast<std::uint64_t>(YX);
                        try_push_main(n, label, levels[cz * N + u]);
                    }
                    if (z + 1 < Z) {
                        const std::uint64_t n = u + static_cast<std::uint64_t>(YX);
                        try_push_main(n, label, levels[cz * N + n]);
                    }
                    if (y > 0) {
                        const std::uint64_t n = u - static_cast<std::uint64_t>(X);
                        try_push_main(n, label, levels[cy * N + u]);
                    }
                    if (y + 1 < Y) {
                        const std::uint64_t n = u + static_cast<std::uint64_t>(X);
                        try_push_main(n, label, levels[cy * N + n]);
                    }
                    if (x > 0) {
                        const std::uint64_t n = u - 1;
                        try_push_main(n, label, levels[cx * N + u]);
                    }
                    if (x + 1 < X) {
                        const std::uint64_t n = u + 1;
                        try_push_main(n, label, levels[cx * N + n]);
                    }
                } else {
                    if (z + 1 < Z) {
                        const std::uint64_t n = u + static_cast<std::uint64_t>(YX);
                        try_push_main(n, label, levels[cz * N + u]);
                    }
                    if (z > 0) {
                        const std::uint64_t n = u - static_cast<std::uint64_t>(YX);
                        try_push_main(n, label, levels[cz * N + n]);
                    }
                    if (y + 1 < Y) {
                        const std::uint64_t n = u + static_cast<std::uint64_t>(X);
                        try_push_main(n, label, levels[cy * N + u]);
                    }
                    if (y > 0) {
                        const std::uint64_t n = u - static_cast<std::uint64_t>(X);
                        try_push_main(n, label, levels[cy * N + n]);
                    }
                    if (x + 1 < X) {
                        const std::uint64_t n = u + 1;
                        try_push_main(n, label, levels[cx * N + u]);
                    }
                    if (x > 0) {
                        const std::uint64_t n = u - 1;
                        try_push_main(n, label, levels[cx * N + n]);
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

// Marker-controlled watershed driven by nearest-neighbour edge affinities.
//
// `affinities` has shape `(C, *spatial)` where `C == spatial_ndim` and each
// channel encodes the affinity of one axis-aligned edge per pixel. `offsets`
// must contain exactly `C` nearest-neighbour offsets (each with one entry
// ±1 and the rest zero), all with the same sign, and together covering every
// spatial axis exactly once. Higher affinity = stronger bond = processed
// first.
template <class AffT, class LabelT>
void watershed_from_affinities(
    const ConstArrayView<AffT> &affinities,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const ConstArrayView<LabelT> &markers,
    const ConstArrayView<std::uint8_t> &mask,
    const ArrayView<LabelT> &out
) {
    if (affinities.ndim() != 3 && affinities.ndim() != 4) {
        throw std::invalid_argument(
            "affinities must have ndim 3 or 4, got ndim=" + std::to_string(affinities.ndim())
        );
    }
    const auto spatial_ndim = static_cast<std::size_t>(affinities.ndim() - 1);
    if (static_cast<std::size_t>(affinities.shape[0]) != spatial_ndim) {
        throw std::invalid_argument(
            "affinities channel count must equal spatial ndim, got channels=" +
            std::to_string(affinities.shape[0]) + ", spatial ndim=" +
            std::to_string(spatial_ndim)
        );
    }

    std::vector<std::ptrdiff_t> spatial_shape(
        affinities.shape.begin() + 1,
        affinities.shape.end()
    );
    if (markers.shape != spatial_shape) {
        throw std::invalid_argument("markers shape must match affinities spatial shape");
    }
    if (out.shape != spatial_shape) {
        throw std::invalid_argument("out shape must match affinities spatial shape");
    }
    const bool has_mask = !mask.shape.empty();
    if (has_mask && mask.shape != spatial_shape) {
        throw std::invalid_argument("mask shape must match affinities spatial shape");
    }

    if (offsets.size() != spatial_ndim) {
        throw std::invalid_argument(
            "offsets count must equal affinities channel count, got offsets=" +
            std::to_string(offsets.size()) + ", channels=" +
            std::to_string(spatial_ndim)
        );
    }

    int sign = 0;
    std::array<std::size_t, 3> channel_for_axis{};
    std::array<bool, 3> axis_seen{false, false, false};
    for (std::size_t c = 0; c < spatial_ndim; ++c) {
        if (offsets[c].size() != spatial_ndim) {
            throw std::invalid_argument(
                "each offset must have length matching the spatial ndim, got spatial ndim=" +
                std::to_string(spatial_ndim)
            );
        }
        std::size_t nonzero_axis = 0;
        int nonzero_count = 0;
        int channel_sign = 0;
        for (std::size_t a = 0; a < spatial_ndim; ++a) {
            const auto v = offsets[c][a];
            if (v == 0) continue;
            ++nonzero_count;
            nonzero_axis = a;
            if (v == 1) channel_sign = 1;
            else if (v == -1) channel_sign = -1;
            else nonzero_count = -1;
        }
        if (nonzero_count != 1) {
            throw std::invalid_argument(
                "each offset must be a nearest-neighbour offset (one entry of value +1 or -1, rest 0)"
            );
        }
        if (sign == 0) {
            sign = channel_sign;
        } else if (sign != channel_sign) {
            throw std::invalid_argument(
                "all offsets must have the same sign (positive or negative direction), "
                "mixing positive and negative is not supported"
            );
        }
        if (axis_seen[nonzero_axis]) {
            throw std::invalid_argument(
                "each spatial axis must be covered by exactly one offset; "
                "axis " + std::to_string(nonzero_axis) + " is covered more than once"
            );
        }
        axis_seen[nonzero_axis] = true;
        channel_for_axis[nonzero_axis] = c;
    }
    for (std::size_t a = 0; a < spatial_ndim; ++a) {
        if (!axis_seen[a]) {
            throw std::invalid_argument(
                "each spatial axis must be covered by exactly one offset; "
                "axis " + std::to_string(a) + " is not covered"
            );
        }
    }

    if (spatial_ndim == 2) {
        const std::array<std::size_t, 2> cfa{channel_for_axis[0], channel_for_axis[1]};
        if (sign < 0) {
            detail_ws::watershed_2d_from_affinities<-1, AffT, LabelT>(
                affinities, cfa, markers, mask, out
            );
        } else {
            detail_ws::watershed_2d_from_affinities<+1, AffT, LabelT>(
                affinities, cfa, markers, mask, out
            );
        }
    } else {
        const std::array<std::size_t, 3> cfa{
            channel_for_axis[0], channel_for_axis[1], channel_for_axis[2]
        };
        if (sign < 0) {
            detail_ws::watershed_3d_from_affinities<-1, AffT, LabelT>(
                affinities, cfa, markers, mask, out
            );
        } else {
            detail_ws::watershed_3d_from_affinities<+1, AffT, LabelT>(
                affinities, cfa, markers, mask, out
            );
        }
    }
}

} // namespace bioimage_cpp
