#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/detail/threading.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace bioimage_cpp::distance {

namespace detail {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

using SquaredDistanceConsumer = void (*)(
    const double *, std::size_t, void *
);

inline std::ptrdiff_t number_of_elements(const std::vector<std::ptrdiff_t> &shape) {
    std::ptrdiff_t n = 1;
    for (const auto axis_size : shape) {
        n *= axis_size;
    }
    return n;
}

// Per-thread scratch space for the 1D EDT and the gather/scatter buffers.
struct Edt1DWorkspace {
    std::vector<double> f;
    std::vector<std::int32_t> old_feature_coord;  // row-major (feature_axes, line_length)
    std::vector<double> distance;
    std::vector<std::int32_t> source;
    std::vector<std::int32_t> envelope_v;
    std::vector<double> envelope_z;

    void ensure(std::ptrdiff_t line_length, std::ptrdiff_t feature_axes) {
        const auto line_sz = static_cast<std::size_t>(line_length);
        if (f.size() < line_sz) {
            f.resize(line_sz);
            distance.resize(line_sz);
            source.resize(line_sz);
            envelope_v.resize(line_sz);
            envelope_z.resize(line_sz + 1);
        }
        const auto coord_sz =
            static_cast<std::size_t>(line_length) * static_cast<std::size_t>(feature_axes);
        if (old_feature_coord.size() < coord_sz) {
            old_feature_coord.resize(coord_sz);
        }
    }
};

// Felzenszwalb–Huttenlocher 1D squared distance transform of length n along
// one axis. Reads f from ws.f[0..n-1], writes the squared-distance result to
// ws.distance[0..n-1] and the per-position argmin source index to
// ws.source[0..n-1]. Source is -1 for positions that have no finite parabola
// in the envelope (only when every entry in f is +infinity).
//
// `Isotropic = true` skips the squared_spacing multiplications when the
// per-axis sampling is 1.0; the compiler then folds away the constant
// multiplications in the inner loop. Hot path: most bioimage workflows do not
// pass an explicit sampling.
template <bool Isotropic>
inline void edt_1d_squared_impl(
    Edt1DWorkspace &ws, std::ptrdiff_t n, double squared_spacing
) {
    auto &f = ws.f;
    auto &distance = ws.distance;
    auto &source = ws.source;
    auto &v = ws.envelope_v;
    auto &z = ws.envelope_z;

    std::ptrdiff_t start = 0;
    while (start < n && f[static_cast<std::size_t>(start)] == kInfinity) {
        ++start;
    }
    if (start == n) {
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            distance[static_cast<std::size_t>(i)] = kInfinity;
            source[static_cast<std::size_t>(i)] = -1;
        }
        return;
    }

    std::ptrdiff_t k = 0;
    v[0] = static_cast<std::int32_t>(start);
    z[0] = -kInfinity;
    z[1] = kInfinity;
    for (std::ptrdiff_t q = start + 1; q < n; ++q) {
        const double fq = f[static_cast<std::size_t>(q)];
        if (fq == kInfinity) {
            continue;
        }
        const double q_d = static_cast<double>(q);
        double s = 0.0;
        while (true) {
            const std::int32_t vk = v[static_cast<std::size_t>(k)];
            const double fvk = f[static_cast<std::size_t>(vk)];
            const double vk_d = static_cast<double>(vk);
            if constexpr (Isotropic) {
                s = ((fq + q_d * q_d) - (fvk + vk_d * vk_d)) /
                    (2.0 * (q_d - vk_d));
            } else {
                s = ((fq + squared_spacing * q_d * q_d) -
                     (fvk + squared_spacing * vk_d * vk_d)) /
                    (2.0 * squared_spacing * (q_d - vk_d));
            }
            if (s > z[static_cast<std::size_t>(k)]) {
                break;
            }
            if (k == 0) {
                v[0] = static_cast<std::int32_t>(q);
                z[1] = kInfinity;
                s = -kInfinity;
                break;
            }
            --k;
        }
        if (s == -kInfinity) {
            continue;
        }
        ++k;
        v[static_cast<std::size_t>(k)] = static_cast<std::int32_t>(q);
        z[static_cast<std::size_t>(k)] = s;
        z[static_cast<std::size_t>(k + 1)] = kInfinity;
    }

    std::ptrdiff_t kk = 0;
    for (std::ptrdiff_t q = 0; q < n; ++q) {
        const double q_d = static_cast<double>(q);
        while (z[static_cast<std::size_t>(kk + 1)] < q_d) {
            ++kk;
        }
        const std::int32_t vk = v[static_cast<std::size_t>(kk)];
        const double diff = q_d - static_cast<double>(vk);
        if constexpr (Isotropic) {
            distance[static_cast<std::size_t>(q)] = diff * diff + f[static_cast<std::size_t>(vk)];
        } else {
            distance[static_cast<std::size_t>(q)] =
                squared_spacing * diff * diff + f[static_cast<std::size_t>(vk)];
        }
        source[static_cast<std::size_t>(q)] = vk;
    }
}

inline void edt_1d_squared(Edt1DWorkspace &ws, std::ptrdiff_t n, double squared_spacing) {
    edt_1d_squared_impl<false>(ws, n, squared_spacing);
}

inline void edt_1d_squared_iso(Edt1DWorkspace &ws, std::ptrdiff_t n) {
    edt_1d_squared_impl<true>(ws, n, 1.0);
}

// Exact first-axis transform for binary isotropic input. Equal-distance ties
// stay with the lower feature coordinate, as in edt_1d_squared_iso.
inline void binary_edt_1d_squared_iso(
    Edt1DWorkspace &ws,
    const std::uint8_t *input,
    const std::ptrdiff_t base,
    const std::ptrdiff_t stride,
    const std::ptrdiff_t n
) {
    std::ptrdiff_t number_of_features = 0;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        if (input[base + i * stride] == 0) {
            ws.envelope_v[static_cast<std::size_t>(number_of_features++)] =
                static_cast<std::int32_t>(i);
        }
    }

    if (number_of_features == 0) {
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            ws.distance[static_cast<std::size_t>(i)] = kInfinity;
            ws.source[static_cast<std::size_t>(i)] = -1;
        }
        return;
    }
    if (number_of_features == n) {
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            ws.distance[static_cast<std::size_t>(i)] = 0.0;
            ws.source[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
        }
        return;
    }

    if (number_of_features < 1 + (n - 1) / 8) {
        std::ptrdiff_t feature = 0;
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            while (
                feature + 1 < number_of_features &&
                2 * static_cast<std::int64_t>(i) >
                    static_cast<std::int64_t>(
                        ws.envelope_v[static_cast<std::size_t>(feature)]
                    ) +
                    static_cast<std::int64_t>(
                        ws.envelope_v[static_cast<std::size_t>(feature + 1)]
                    )
            ) {
                ++feature;
            }
            const auto nearest = ws.envelope_v[static_cast<std::size_t>(feature)];
            const double diff = static_cast<double>(i - nearest);
            ws.distance[static_cast<std::size_t>(i)] = diff * diff;
            ws.source[static_cast<std::size_t>(i)] = nearest;
        }
        return;
    }

    std::ptrdiff_t begin = 0;
    for (std::ptrdiff_t feature = 0; feature < number_of_features; ++feature) {
        const auto nearest = ws.envelope_v[static_cast<std::size_t>(feature)];
        std::ptrdiff_t end = n;
        if (feature + 1 < number_of_features) {
            const auto next = ws.envelope_v[static_cast<std::size_t>(feature + 1)];
            end = static_cast<std::ptrdiff_t>(
                (static_cast<std::int64_t>(nearest) + static_cast<std::int64_t>(next)) /
                    2 +
                1
            );
        }
        for (auto i = begin; i < end; ++i) {
            const double diff = static_cast<double>(i - nearest);
            ws.distance[static_cast<std::size_t>(i)] = diff * diff;
            ws.source[static_cast<std::size_t>(i)] = nearest;
        }
        begin = end;
    }
}

inline bool squared_distances_fit_exact_float(
    const std::vector<std::ptrdiff_t> &shape
) {
    constexpr std::uint64_t max_exact_integer =
        std::uint64_t{1} << std::numeric_limits<float>::digits;
    constexpr std::uint64_t max_exact_delta = 4096;
    std::uint64_t remaining = max_exact_integer;
    for (const auto axis_size : shape) {
        if (axis_size <= 0) {
            return false;
        }
        const auto delta = static_cast<std::uint64_t>(axis_size - 1);
        if (delta > max_exact_delta) {
            return false;
        }
        const auto squared = delta * delta;
        if (squared > remaining) {
            return false;
        }
        remaining -= squared;
    }
    return true;
}

inline void unravel(
    std::ptrdiff_t flat,
    const std::vector<std::ptrdiff_t> &strides,
    std::ptrdiff_t ndim,
    std::int32_t *out
) {
    for (std::ptrdiff_t ax = 0; ax < ndim; ++ax) {
        const auto stride = strides[static_cast<std::size_t>(ax)];
        const auto coord = flat / stride;
        flat -= coord * stride;
        out[ax] = static_cast<std::int32_t>(coord);
    }
}

} // namespace detail

struct DistanceTransformOutputs {
    ArrayView<float> distances;       // shape (*input.shape); nullptr to skip.
    ArrayView<std::int32_t> indices;  // shape (ndim, *input.shape); nullptr to skip.
    ArrayView<float> vectors;         // shape (*input.shape, ndim); nullptr to skip.
};

// Exact Euclidean distance transform of a binary input. Background pixels are
// those equal to zero. Uses the separable Felzenszwalb–Huttenlocher algorithm:
// one 1D squared-EDT sweep per spatial axis, each sweep processing all lines
// along that axis. Complexity is O(N * ndim) with N = total number of pixels.
//
// `n_threads` accepts the usual convention: 0 = hardware concurrency,
// >=1 = explicit thread count. Threading splits the orthogonal lines of each
// axis sweep across threads via detail::parallel_for_chunks; the per-axis sweep
// is a barrier (the next axis depends on the current axis result).
inline void detail_distance_transform_impl(
    const ConstArrayView<std::uint8_t> &input,
    const std::vector<double> &sampling,
    const DistanceTransformOutputs &outputs,
    const std::size_t n_threads,
    std::unique_ptr<double[]> initialized_squared_distance,
    const detail::SquaredDistanceConsumer squared_distance_consumer,
    void *squared_distance_consumer_context
) {
    const auto ndim = input.ndim();
    if (ndim < 1) {
        throw std::invalid_argument("input must have ndim >= 1, got ndim=0");
    }
    if (sampling.size() != static_cast<std::size_t>(ndim)) {
        throw std::invalid_argument(
            "sampling must have length matching input ndim, got ndim=" +
            std::to_string(ndim) + ", sampling length=" + std::to_string(sampling.size())
        );
    }
    for (std::size_t axis = 0; axis < sampling.size(); ++axis) {
        if (!(std::isfinite(sampling[axis]) && sampling[axis] > 0.0)) {
            throw std::invalid_argument(
                "sampling values must be positive and finite, got sampling[" +
                std::to_string(axis) + "]=" + std::to_string(sampling[axis])
            );
        }
    }

    const auto n = detail::number_of_elements(input.shape);
    if (n == 0) {
        return;
    }
    const auto strides = bioimage_cpp::detail::c_order_strides(input.shape);

    const bool want_distances = outputs.distances.data != nullptr;
    const bool want_indices = outputs.indices.data != nullptr;
    const bool want_vectors = outputs.vectors.data != nullptr;
    const bool use_initialized_squared =
        initialized_squared_distance != nullptr;
    const bool track_feature = want_indices || want_vectors;
    bool is_isotropic = true;
    for (std::size_t axis = 0; axis < sampling.size(); ++axis) {
        if (sampling[axis] != 1.0) {
            is_isotropic = false;
            break;
        }
    }

    BIOIMAGE_PROFILE_INIT(profiler)

    // Detect the all-foreground case and sample the binary fast-path density.
    // SciPy reports all-foreground results against a virtual background row at
    // axis-0 coordinate -1; we mirror that convention.
    constexpr std::ptrdiff_t binary_sample_limit = 4096;
    bool has_background = use_initialized_squared;
    std::ptrdiff_t sampled_background = 0;
    std::ptrdiff_t sample_size = 0;
    if (!use_initialized_squared) {
        BIOIMAGE_PROFILE_SCOPE(profiler, "scan_for_bg")
        sample_size = std::min(n, binary_sample_limit);
        for (std::ptrdiff_t i = 0; i < sample_size; ++i) {
            if (input.data[i] == 0) {
                ++sampled_background;
                has_background = true;
            }
        }
        if (!has_background) {
            for (std::ptrdiff_t i = sample_size; i < n; ++i) {
                if (input.data[i] == 0) {
                    has_background = true;
                    break;
                }
            }
        }
    }

    if (!has_background) {
        // Virtual feature is at axis-0 coord -1, all other axes 0. Distances,
        // indices, and vectors are all computed against this single fixed
        // feature point — matches scipy.ndimage.distance_transform_edt.
        std::vector<std::int32_t> coords(static_cast<std::size_t>(ndim), 0);
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            detail::unravel(i, strides, ndim, coords.data());
            double squared = 0.0;
            for (std::ptrdiff_t ax = 0; ax < ndim; ++ax) {
                const double feature_coord = (ax == 0) ? -1.0 : 0.0;
                const double diff =
                    (feature_coord - static_cast<double>(coords[static_cast<std::size_t>(ax)])) *
                    sampling[static_cast<std::size_t>(ax)];
                squared += diff * diff;
                if (want_vectors) {
                    outputs.vectors.data[i * ndim + ax] = static_cast<float>(diff);
                }
                if (want_indices) {
                    outputs.indices.data[ax * n + i] = (ax == 0) ? -1 : 0;
                }
            }
            if (want_distances) {
                outputs.distances.data[i] = static_cast<float>(std::sqrt(squared));
            }
        }
        return;
    }

    const bool use_binary_first_sweep =
        !use_initialized_squared && is_isotropic &&
        sampled_background >= (sample_size + 7) / 8;
    const bool use_float_squared =
        !use_initialized_squared && squared_distance_consumer == nullptr &&
        is_isotropic && detail::squared_distances_fit_exact_float(input.shape);

    std::unique_ptr<double[]> squared_distance_double = use_initialized_squared
        ? std::move(initialized_squared_distance)
        : nullptr;
    std::unique_ptr<float[]> squared_distance_float;
    if (use_float_squared) {
        squared_distance_float = std::make_unique_for_overwrite<float[]>(
            static_cast<std::size_t>(n)
        );
    } else if (squared_distance_double == nullptr) {
        squared_distance_double = std::make_unique_for_overwrite<double[]>(
            static_cast<std::size_t>(n)
        );
    }

    const auto run_sweeps = [&]<class SquaredDistance>(
        SquaredDistance *squared_distance
    ) {
        // Indices can hold feature coordinates during the sweeps. Vector-only
        // calls use one uninitialized coordinate buffer with the same layout.
        std::vector<std::int32_t *> feature_coord;
        std::unique_ptr<std::int32_t[]> owned_feature_coord;
        if (track_feature) {
            feature_coord.resize(static_cast<std::size_t>(ndim));
            if (want_indices) {
                for (std::ptrdiff_t ax = 0; ax < ndim; ++ax) {
                    feature_coord[static_cast<std::size_t>(ax)] =
                        outputs.indices.data + ax * n;
                }
            } else {
                owned_feature_coord = std::make_unique_for_overwrite<std::int32_t[]>(
                    static_cast<std::size_t>(ndim) * static_cast<std::size_t>(n)
                );
                for (std::ptrdiff_t ax = 0; ax < ndim; ++ax) {
                    feature_coord[static_cast<std::size_t>(ax)] =
                        owned_feature_coord.get() + ax * n;
                }
            }
        }

        for (std::ptrdiff_t ax = 0; ax < ndim; ++ax) {
            BIOIMAGE_PROFILE_SCOPE(profiler, "sweep_axis")
            const std::ptrdiff_t line_length = input.shape[static_cast<std::size_t>(ax)];
            if (line_length <= 0) {
                continue;
            }
            const std::ptrdiff_t stride = strides[static_cast<std::size_t>(ax)];
            const std::ptrdiff_t inner_count = stride;
            const std::ptrdiff_t axis_block = line_length * stride;
            const std::ptrdiff_t outer_count = (axis_block == 0) ? 0 : n / axis_block;
            const std::size_t n_lines =
                static_cast<std::size_t>(outer_count) * static_cast<std::size_t>(inner_count);
            if (n_lines == 0) {
                continue;
            }
            const double sampling_ax = sampling[static_cast<std::size_t>(ax)];
            const double squared_spacing = sampling_ax * sampling_ax;
            const std::ptrdiff_t feature_axes_in = track_feature ? ax : 0;

            const auto process_line = [&](std::size_t line_id, detail::Edt1DWorkspace &ws) {
                const auto outer_idx = static_cast<std::ptrdiff_t>(line_id) / inner_count;
                const auto inner_idx = static_cast<std::ptrdiff_t>(line_id) % inner_count;
                const std::ptrdiff_t base = outer_idx * axis_block + inner_idx;
                ws.ensure(line_length, feature_axes_in);
                const bool use_binary_line = ax == 0 && use_binary_first_sweep;

                // The first-axis gather initializes the uninitialized squared-
                // distance buffer directly from the input. Later axes gather the
                // preceding sweep, avoiding a redundant full-volume init pass.
                if (!use_binary_line) {
                    if (ax == 0 && !use_initialized_squared) {
                        for (std::ptrdiff_t i = 0; i < line_length; ++i) {
                            const auto index = static_cast<std::size_t>(base + i * stride);
                            ws.f[static_cast<std::size_t>(i)] =
                                input.data[index] == 0 ? 0.0 : detail::kInfinity;
                        }
                    } else {
                        for (std::ptrdiff_t i = 0; i < line_length; ++i) {
                            const auto index = static_cast<std::size_t>(base + i * stride);
                            ws.f[static_cast<std::size_t>(i)] =
                                static_cast<double>(squared_distance[index]);
                        }
                    }
                }
                // Gather already-tracked feature coords (axes < ax).
                for (std::ptrdiff_t a = 0; a < feature_axes_in; ++a) {
                    const auto *src = feature_coord[static_cast<std::size_t>(a)];
                    auto *dst = ws.old_feature_coord.data() + a * line_length;
                    for (std::ptrdiff_t i = 0; i < line_length; ++i) {
                        dst[i] = src[base + i * stride];
                    }
                }

                if (use_binary_line) {
                    detail::binary_edt_1d_squared_iso(
                        ws, input.data, base, stride, line_length
                    );
                } else if (is_isotropic) {
                    detail::edt_1d_squared_iso(ws, line_length);
                } else {
                    detail::edt_1d_squared(ws, line_length, squared_spacing);
                }

                // Scatter squared distances back.
                for (std::ptrdiff_t i = 0; i < line_length; ++i) {
                    squared_distance[static_cast<std::size_t>(base + i * stride)] =
                        ws.distance[static_cast<std::size_t>(i)];
                }
                if (!track_feature) {
                    return;
                }
                // Scatter feature coords for axes < ax via source[i].
                for (std::ptrdiff_t a = 0; a < ax; ++a) {
                    auto *dst = feature_coord[static_cast<std::size_t>(a)];
                    const auto *src = ws.old_feature_coord.data() + a * line_length;
                    for (std::ptrdiff_t i = 0; i < line_length; ++i) {
                        const auto s = ws.source[static_cast<std::size_t>(i)];
                        if (s >= 0) {
                            dst[base + i * stride] = src[s];
                        }
                    }
                }
                // Axis ax's feature coord is the parabola minimizer position itself.
                {
                    auto *dst = feature_coord[static_cast<std::size_t>(ax)];
                    for (std::ptrdiff_t i = 0; i < line_length; ++i) {
                        const auto s = ws.source[static_cast<std::size_t>(i)];
                        if (s >= 0) {
                            dst[base + i * stride] = s;
                        }
                    }
                }
            };

            const auto resolved_threads =
                bioimage_cpp::detail::normalize_thread_count(n_threads, n_lines);
            if (resolved_threads <= 1) {
                detail::Edt1DWorkspace ws;
                for (std::size_t line_id = 0; line_id < n_lines; ++line_id) {
                    process_line(line_id, ws);
                }
            } else {
                std::vector<detail::Edt1DWorkspace> per_thread(resolved_threads);
                bioimage_cpp::detail::parallel_for_chunks(
                    resolved_threads,
                    n_lines,
                    [&](std::size_t thread_id, std::size_t begin, std::size_t end) {
                        auto &ws = per_thread[thread_id];
                        for (std::size_t line_id = begin; line_id < end; ++line_id) {
                            process_line(line_id, ws);
                        }
                    }
                );
            }
        }

        // Indices already contain their final values. Distances and vectors
        // stream over flat indices in C-order without per-pixel division.
        if (want_distances) {
            BIOIMAGE_PROFILE_SCOPE(profiler, "output_distances")
            const auto output_threads = bioimage_cpp::detail::normalize_thread_count(
                n_threads, static_cast<std::size_t>(n)
            );
            const auto write_distances = [&](const std::size_t begin, const std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    outputs.distances.data[i] =
                        static_cast<float>(
                            std::sqrt(static_cast<double>(squared_distance[i]))
                        );
                }
            };
            if (output_threads <= 1) {
                write_distances(0, static_cast<std::size_t>(n));
            } else {
                bioimage_cpp::detail::parallel_for_chunks(
                    output_threads,
                    static_cast<std::size_t>(n),
                    [&](const std::size_t, const std::size_t begin, const std::size_t end) {
                        write_distances(begin, end);
                    }
                );
            }
        }

        if (want_vectors) {
            BIOIMAGE_PROFILE_SCOPE(profiler, "output_vectors")
            std::vector<std::int32_t> coord(static_cast<std::size_t>(ndim), 0);
            std::vector<std::int32_t> shape_i32(static_cast<std::size_t>(ndim), 0);
            for (std::ptrdiff_t ax = 0; ax < ndim; ++ax) {
                shape_i32[static_cast<std::size_t>(ax)] =
                    static_cast<std::int32_t>(input.shape[static_cast<std::size_t>(ax)]);
            }
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                auto *dst = outputs.vectors.data + i * ndim;
                for (std::ptrdiff_t ax = 0; ax < ndim; ++ax) {
                    const double diff =
                        static_cast<double>(
                            feature_coord[static_cast<std::size_t>(ax)][static_cast<std::size_t>(i)] -
                            coord[static_cast<std::size_t>(ax)]
                        ) *
                        sampling[static_cast<std::size_t>(ax)];
                    dst[ax] = static_cast<float>(diff);
                }
                // Increment coord in C-order (innermost axis fastest).
                for (std::ptrdiff_t ax = ndim - 1; ax >= 0; --ax) {
                    auto &c = coord[static_cast<std::size_t>(ax)];
                    if (++c < shape_i32[static_cast<std::size_t>(ax)]) {
                        break;
                    }
                    c = 0;
                }
            }
        }
    };

    if (use_float_squared) {
        run_sweeps(squared_distance_float.get());
    } else {
        run_sweeps(squared_distance_double.get());
    }

    if (squared_distance_consumer != nullptr) {
        BIOIMAGE_PROFILE_SCOPE(profiler, "squared_distance_consumer")
        squared_distance_consumer(
            squared_distance_double.get(), static_cast<std::size_t>(n),
            squared_distance_consumer_context
        );
    }
    BIOIMAGE_PROFILE_REPORT(profiler)
}

inline void distance_transform(
    const ConstArrayView<std::uint8_t> &input,
    const std::vector<double> &sampling,
    const DistanceTransformOutputs &outputs,
    const std::size_t n_threads = 1
) {
    detail_distance_transform_impl(
        input, sampling, outputs, n_threads, nullptr, nullptr, nullptr
    );
}

namespace detail {

inline void distance_transform_from_squared(
    const std::vector<std::ptrdiff_t> &shape,
    const std::vector<double> &sampling,
    std::unique_ptr<double[]> initialized_squared_distance,
    const SquaredDistanceConsumer consumer,
    void *consumer_context,
    const std::size_t n_threads
) {
    if (initialized_squared_distance == nullptr) {
        throw std::invalid_argument(
            "initialized squared-distance buffer must not be null"
        );
    }
    ConstArrayView<std::uint8_t> shape_only_input{nullptr, shape, {}};
    detail_distance_transform_impl(
        shape_only_input, sampling, {}, n_threads,
        std::move(initialized_squared_distance), consumer, consumer_context
    );
}

} // namespace detail

} // namespace bioimage_cpp::distance
