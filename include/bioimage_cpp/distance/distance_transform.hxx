#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/detail/threading.hxx"

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
inline void distance_transform(
    const ConstArrayView<std::uint8_t> &input,
    const std::vector<double> &sampling,
    const DistanceTransformOutputs &outputs,
    std::size_t n_threads = 1
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
    const bool track_feature = want_indices || want_vectors;
    bool is_isotropic = true;
    for (std::size_t axis = 0; axis < sampling.size(); ++axis) {
        if (sampling[axis] != 1.0) {
            is_isotropic = false;
            break;
        }
    }

    BIOIMAGE_PROFILE_INIT(profiler)

    // Detect the all-foreground (no background) case. SciPy reports distances
    // and indices against a virtual background row at axis-0 coordinate -1; we
    // mirror that convention so callers can switch between SciPy and us
    // without surprises.
    bool has_background = false;
    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "scan_for_bg")
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            if (input.data[i] == 0) {
                has_background = true;
                break;
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

    // Squared sampled distance buffer. ndim per-axis feature-coord buffers
    // (int32) replace the previous flat int64 feature index — this lets the
    // output pass materialize indices/vectors without re-unraveling per pixel.
    auto squared_distance = std::make_unique_for_overwrite<double[]>(
        static_cast<std::size_t>(n)
    );
    std::vector<std::vector<std::int32_t>> feature_coord;
    if (track_feature) {
        feature_coord.resize(static_cast<std::size_t>(ndim));
        for (auto &arr : feature_coord) {
            arr.assign(static_cast<std::size_t>(n), 0);
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

            // The first-axis gather initializes the uninitialized squared-
            // distance buffer directly from the input. Later axes gather the
            // preceding sweep, avoiding a redundant full-volume init pass.
            if (ax == 0) {
                for (std::ptrdiff_t i = 0; i < line_length; ++i) {
                    const auto index = static_cast<std::size_t>(base + i * stride);
                    ws.f[static_cast<std::size_t>(i)] =
                        input.data[index] == 0 ? 0.0 : detail::kInfinity;
                }
            } else {
                for (std::ptrdiff_t i = 0; i < line_length; ++i) {
                    const auto index = static_cast<std::size_t>(base + i * stride);
                    ws.f[static_cast<std::size_t>(i)] = squared_distance[index];
                }
            }
            // Gather already-tracked feature coords (axes < ax).
            for (std::ptrdiff_t a = 0; a < feature_axes_in; ++a) {
                const auto *src = feature_coord[static_cast<std::size_t>(a)].data();
                auto *dst = ws.old_feature_coord.data() + a * line_length;
                for (std::ptrdiff_t i = 0; i < line_length; ++i) {
                    dst[i] = src[base + i * stride];
                }
            }

            if (is_isotropic) {
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
                auto *dst = feature_coord[static_cast<std::size_t>(a)].data();
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
                auto *dst = feature_coord[static_cast<std::size_t>(ax)].data();
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

    // Output materialization. All three branches stream over flat indices in
    // C-order with no integer divisions per pixel: indices come straight from
    // per-axis feature buffers, vectors use an incremental coord counter.
    if (want_distances) {
        BIOIMAGE_PROFILE_SCOPE(profiler, "output_distances")
        const auto output_threads = bioimage_cpp::detail::normalize_thread_count(
            n_threads, static_cast<std::size_t>(n)
        );
        const auto write_distances = [&](const std::size_t begin, const std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                outputs.distances.data[i] =
                    static_cast<float>(std::sqrt(squared_distance[i]));
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

    if (want_indices) {
        BIOIMAGE_PROFILE_SCOPE(profiler, "output_indices")
        for (std::ptrdiff_t ax = 0; ax < ndim; ++ax) {
            const auto *src = feature_coord[static_cast<std::size_t>(ax)].data();
            auto *dst = outputs.indices.data + ax * n;
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                dst[i] = src[i];
            }
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
    BIOIMAGE_PROFILE_REPORT(profiler)
}

} // namespace bioimage_cpp::distance
