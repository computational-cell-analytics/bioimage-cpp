#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/detail/threading.hxx"
#include "bioimage_cpp/distance/distance_transform.hxx"
#include "bioimage_cpp/skeleton/detail/components.hxx"
#include "bioimage_cpp/skeleton/teasar.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace bioimage_cpp::skeleton::distributed {

struct BlockFace {
    std::size_t axis = 0;
    bool high = false;

    auto operator<=>(const BlockFace &) const = default;
};

namespace detail_border {

inline std::int64_t checked_global_coordinate(
    const std::int64_t origin,
    const std::ptrdiff_t local
) {
    if (local < 0) {
        throw std::runtime_error("local border coordinate became negative");
    }
    if (
        static_cast<std::uint64_t>(local) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
    ) {
        throw std::overflow_error("local border coordinate exceeds int64 range");
    }
    const auto local64 = static_cast<std::int64_t>(local);
    if (origin > std::numeric_limits<std::int64_t>::max() - local64) {
        throw std::overflow_error("global border coordinate overflows int64");
    }
    return origin + local64;
}

inline void validate_common(
    const std::vector<std::ptrdiff_t> &shape,
    const std::array<double, 3> &spacing,
    const std::vector<BlockFace> &faces
) {
    if (shape.size() != 3) {
        throw std::invalid_argument("block input must have exactly three dimensions");
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (shape[axis] < 0) {
            throw std::invalid_argument("block shape entries must be non-negative");
        }
        if (!(std::isfinite(spacing[axis]) && spacing[axis] > 0.0)) {
            throw std::invalid_argument("spacing values must be positive and finite");
        }
    }
    for (const auto &face : faces) {
        if (face.axis >= 3) {
            throw std::invalid_argument("face axis must be in [0, 3)");
        }
    }
}

template <class LabelT, bool Binary>
std::vector<LabeledVoxelTarget<LabelT>> border_targets_impl(
    const ConstArrayView<LabelT> &input,
    const LabelT background,
    std::vector<BlockFace> faces,
    const std::array<std::int64_t, 3> &origin,
    const std::array<double, 3> &spacing,
    const std::size_t number_of_threads
) {
    static_assert(std::is_integral_v<LabelT> && !std::is_same_v<LabelT, bool>);
    validate_common(input.shape, spacing, faces);
    std::sort(faces.begin(), faces.end());
    faces.erase(std::unique(faces.begin(), faces.end()), faces.end());

    BIOIMAGE_PROFILE_INIT(profile)
    std::vector<LabeledVoxelTarget<LabelT>> targets;
    for (const auto &face : faces) {
        if (
            input.shape[face.axis] == 0 ||
            input.shape[(face.axis + 1) % 3] == 0 ||
            input.shape[(face.axis + 2) % 3] == 0
        ) {
            continue;
        }
        std::array<std::size_t, 2> face_axes{};
        std::size_t cursor = 0;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (axis != face.axis) {
                face_axes[cursor++] = axis;
            }
        }
        const auto height = input.shape[face_axes[0]];
        const auto width = input.shape[face_axes[1]];
        const auto face_size = detail::checked_multiply_size(
            static_cast<std::size_t>(height),
            static_cast<std::size_t>(width),
            "face size overflows size_t"
        );
        std::vector<LabelT> packed(face_size);
        std::array<std::ptrdiff_t, 3> local{};
        local[face.axis] = face.high ? input.shape[face.axis] - 1 : 0;
        for (std::ptrdiff_t first = 0; first < height; ++first) {
            local[face_axes[0]] = first;
            for (std::ptrdiff_t second = 0; second < width; ++second) {
                local[face_axes[1]] = second;
                const auto input_index = static_cast<std::size_t>(
                    (local[0] * input.shape[1] + local[1]) * input.shape[2] +
                    local[2]
                );
                auto value = input.data[input_index];
                if constexpr (Binary) {
                    value = value == background ? LabelT{0} : LabelT{1};
                }
                packed[
                    static_cast<std::size_t>(first * width + second)
                ] = value;
            }
        }

        ConstArrayView<LabelT> packed_view{
            packed.data(), {1, height, width}, {}
        };
        auto components = [&] {
            if constexpr (Binary) {
                return detail::extract_components<LabelT, true>(
                    packed_view, LabelT{0}, profile
                );
            } else {
                return detail::extract_labeled_components(
                    packed_view, background, profile
                );
            }
        }();

        for (const auto &component : components.components) {
            const auto component_height = component.end[1] - component.begin[1];
            const auto component_width = component.end[2] - component.begin[2];
            const std::vector<std::ptrdiff_t> padded_shape{
                component_height + 2, component_width + 2
            };
            const auto padded_size = detail::checked_shape_size(
                padded_shape, "padded face component size overflows size_t"
            );
            std::vector<std::uint8_t> component_mask(padded_size, 0);
            const auto padded_width = padded_shape[1];
            long double first_sum = 0.0L;
            long double second_sum = 0.0L;
            std::size_t count = 0;
            for (std::size_t offset = 0;
                 offset < component.number_of_runs; ++offset) {
                const auto run_id = components.component_run_ids[
                    component.run_offset + offset
                ];
                const auto &run = components.runs[run_id];
                const auto row = run.y - component.begin[1] + 1;
                for (auto x = run.x_begin; x <= run.x_end; ++x) {
                    const auto column = x - component.begin[2] + 1;
                    component_mask[static_cast<std::size_t>(
                        row * padded_width + column
                    )] = 1;
                    first_sum += static_cast<long double>(run.y);
                    second_sum += static_cast<long double>(x);
                    ++count;
                }
            }
            if (count == 0) {
                throw std::runtime_error("face component has no foreground pixels");
            }

            std::vector<float> distances(padded_size, 0.0f);
            ConstArrayView<std::uint8_t> component_view{
                component_mask.data(), padded_shape, {}
            };
            ArrayView<float> distance_view{
                distances.data(), padded_shape, {}
            };
            distance::distance_transform(
                component_view,
                {spacing[face_axes[0]], spacing[face_axes[1]]},
                {distance_view, {}, {}},
                bioimage_cpp::detail::normalize_thread_count(
                    number_of_threads, count
                )
            );

            const auto mean_first = first_sum / static_cast<long double>(count);
            const auto mean_second = second_sum / static_cast<long double>(count);
            const auto face_center_first =
                static_cast<long double>(height) / 2.0L;
            const auto face_center_second =
                static_cast<long double>(width) / 2.0L;
            // Resolve half-grid centroids toward the centre of the full face.
            // This makes the choice invariant when neighboring blocks view the
            // shared plane from opposite sides.
            const auto centroid_first = mean_first >= face_center_first
                ? std::floor(mean_first) : std::ceil(mean_first);
            const auto centroid_second = mean_second >= face_center_second
                ? std::floor(mean_second) : std::ceil(mean_second);
            float best_distance = -1.0f;
            long double best_centroid_distance =
                std::numeric_limits<long double>::infinity();
            long double best_face_center_distance =
                std::numeric_limits<long double>::infinity();
            long double best_corner_distance =
                std::numeric_limits<long double>::infinity();
            long double best_edge_distance =
                std::numeric_limits<long double>::infinity();
            VoxelCoordinate best_coordinate{};
            bool have_best = false;
            for (std::size_t offset = 0;
                 offset < component.number_of_runs; ++offset) {
                const auto run_id = components.component_run_ids[
                    component.run_offset + offset
                ];
                const auto &run = components.runs[run_id];
                const auto row = run.y - component.begin[1] + 1;
                for (auto x = run.x_begin; x <= run.x_end; ++x) {
                    const auto column = x - component.begin[2] + 1;
                    const auto edt = distances[static_cast<std::size_t>(
                        row * padded_width + column
                    )];
                    const auto first_delta =
                        (static_cast<long double>(run.y) - centroid_first) *
                        static_cast<long double>(spacing[face_axes[0]]);
                    const auto second_delta =
                        (static_cast<long double>(x) - centroid_second) *
                        static_cast<long double>(spacing[face_axes[1]]);
                    const auto centroid_distance =
                        first_delta * first_delta + second_delta * second_delta;
                    const auto face_first_delta =
                        (static_cast<long double>(run.y) - face_center_first) *
                        static_cast<long double>(spacing[face_axes[0]]);
                    const auto face_second_delta =
                        (static_cast<long double>(x) - face_center_second) *
                        static_cast<long double>(spacing[face_axes[1]]);
                    const auto face_center_distance =
                        face_first_delta * face_first_delta +
                        face_second_delta * face_second_delta;
                    const auto corner_distance = [&] {
                        long double result =
                            std::numeric_limits<long double>::infinity();
                        for (const auto corner_first : {
                                 -0.5L,
                                 static_cast<long double>(height) - 0.5L,
                             }) {
                            for (const auto corner_second : {
                                     -0.5L,
                                     static_cast<long double>(width) - 0.5L,
                                 }) {
                                const auto first =
                                    (static_cast<long double>(run.y) - corner_first) *
                                    static_cast<long double>(spacing[face_axes[0]]);
                                const auto second =
                                    (static_cast<long double>(x) - corner_second) *
                                    static_cast<long double>(spacing[face_axes[1]]);
                                result = std::min(
                                    result, first * first + second * second
                                );
                            }
                        }
                        return result;
                    }();
                    const auto edge_distance = std::min({
                        static_cast<long double>(spacing[face_axes[0]]) *
                            (static_cast<long double>(run.y) + 0.5L),
                        static_cast<long double>(spacing[face_axes[0]]) *
                            (static_cast<long double>(height) - 0.5L -
                             static_cast<long double>(run.y)),
                        static_cast<long double>(spacing[face_axes[1]]) *
                            (static_cast<long double>(x) + 0.5L),
                        static_cast<long double>(spacing[face_axes[1]]) *
                            (static_cast<long double>(width) - 0.5L -
                             static_cast<long double>(x)),
                    });
                    local[face_axes[0]] = run.y;
                    local[face_axes[1]] = x;
                    VoxelCoordinate global{};
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        global[axis] = checked_global_coordinate(
                            origin[axis], local[axis]
                        );
                    }
                    if (
                        !have_best || edt > best_distance ||
                        (edt == best_distance &&
                         (centroid_distance < best_centroid_distance ||
                          (centroid_distance == best_centroid_distance &&
                           (face_center_distance < best_face_center_distance ||
                            (face_center_distance == best_face_center_distance &&
                             (corner_distance < best_corner_distance ||
                              (corner_distance == best_corner_distance &&
                               (edge_distance < best_edge_distance ||
                                (edge_distance == best_edge_distance &&
                                 global < best_coordinate)))))))))
                    ) {
                        have_best = true;
                        best_distance = edt;
                        best_centroid_distance = centroid_distance;
                        best_face_center_distance = face_center_distance;
                        best_corner_distance = corner_distance;
                        best_edge_distance = edge_distance;
                        best_coordinate = global;
                    }
                }
            }
            if (!have_best) {
                throw std::runtime_error("failed to select a face target");
            }
            targets.push_back({
                Binary ? LabelT{1} : component.label,
                best_coordinate,
            });
        }
    }

    std::sort(
        targets.begin(), targets.end(),
        [](const auto &first, const auto &second) {
            if (first.label != second.label) {
                return first.label < second.label;
            }
            return first.coordinate < second.coordinate;
        }
    );
    targets.erase(
        std::unique(
            targets.begin(), targets.end(),
            [](const auto &first, const auto &second) {
                return first.label == second.label &&
                    first.coordinate == second.coordinate;
            }
        ),
        targets.end()
    );
    BIOIMAGE_PROFILE_REPORT(profile)
    return targets;
}

} // namespace detail_border

inline std::vector<VoxelCoordinate> block_border_targets(
    const ConstArrayView<std::uint8_t> &mask,
    std::vector<BlockFace> faces,
    const std::array<std::int64_t, 3> &origin,
    const std::array<double, 3> &spacing,
    const std::size_t number_of_threads = 1
) {
    auto labeled = detail_border::border_targets_impl<std::uint8_t, true>(
        mask, std::uint8_t{0}, std::move(faces), origin, spacing,
        number_of_threads
    );
    std::vector<VoxelCoordinate> targets;
    targets.reserve(labeled.size());
    for (const auto &target : labeled) {
        targets.push_back(target.coordinate);
    }
    return targets;
}

template <class LabelT>
std::vector<LabeledVoxelTarget<LabelT>> block_border_targets_labels(
    const ConstArrayView<LabelT> &labels,
    const LabelT background,
    std::vector<BlockFace> faces,
    const std::array<std::int64_t, 3> &origin,
    const std::array<double, 3> &spacing,
    const std::size_t number_of_threads = 1
) {
    return detail_border::border_targets_impl<LabelT, false>(
        labels, background, std::move(faces), origin, spacing,
        number_of_threads
    );
}

} // namespace bioimage_cpp::skeleton::distributed
