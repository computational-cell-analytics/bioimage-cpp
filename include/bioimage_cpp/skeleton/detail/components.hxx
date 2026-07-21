#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/util/union_find.hxx"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bioimage_cpp::skeleton::detail {

inline std::size_t checked_add_size(
    const std::size_t first,
    const std::size_t second,
    const char *what
) {
    if (first > std::numeric_limits<std::size_t>::max() - second) {
        throw std::overflow_error(what);
    }
    return first + second;
}

inline std::size_t checked_multiply_size(
    const std::size_t first,
    const std::size_t second,
    const char *what
) {
    if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) {
        throw std::overflow_error(what);
    }
    return first * second;
}

inline std::size_t checked_shape_size(
    const std::vector<std::ptrdiff_t> &shape,
    const char *what
) {
    std::size_t size = 1;
    for (const auto extent : shape) {
        if (extent < 0) {
            throw std::invalid_argument("shape entries must be non-negative");
        }
        size = checked_multiply_size(
            size, static_cast<std::size_t>(extent), what
        );
    }
    return size;
}

template <class LabelT>
struct ComponentRun {
    LabelT label{};
    std::ptrdiff_t z = 0;
    std::ptrdiff_t y = 0;
    std::ptrdiff_t x_begin = 0;
    std::ptrdiff_t x_end = 0; // inclusive
};

template <class LabelT>
struct ComponentDescriptor {
    LabelT label{};
    std::array<std::ptrdiff_t, 3> begin{}; // inclusive
    std::array<std::ptrdiff_t, 3> end{};   // exclusive
    std::array<std::ptrdiff_t, 3> first_coordinate{};
    std::uint64_t voxel_count = 0;
    std::uint64_t first_flat_index = 0;
    std::size_t run_offset = 0;
    std::size_t number_of_runs = 0;
};

template <class LabelT>
struct ComponentSet {
    std::array<std::ptrdiff_t, 3> input_shape{};
    std::vector<ComponentRun<LabelT>> runs;
    std::vector<std::size_t> row_offsets;
    std::vector<ComponentDescriptor<LabelT>> components;
    std::vector<std::size_t> component_run_ids;
    std::size_t foreground_count = 0;
};

struct PreparedTeasarComponent {
    std::vector<std::ptrdiff_t> padded_shape;
    std::vector<std::uint8_t> padded_mask;
    // Optional EDT-only mask. The path domain remains padded_mask; this mask
    // extends foreground across artificial processing-block cuts so those
    // cuts do not become false object boundaries.
    std::vector<std::uint8_t> distance_mask;
    std::array<std::ptrdiff_t, 3> input_origin{};
    std::size_t foreground_count = 0;
    std::vector<std::size_t> required_target_voxels;
    std::size_t required_root_voxel = std::numeric_limits<std::size_t>::max();
};

struct OpenBlockFaces {
    // low/high pairs for axes 0, 1, and 2.
    std::array<bool, 6> values{};

    bool operator()(const std::size_t axis, const bool high) const {
        return values[2 * axis + static_cast<std::size_t>(high)];
    }

    bool any() const {
        return std::any_of(values.begin(), values.end(), [](const bool value) {
            return value;
        });
    }
};

inline bool intervals_within_one(
    const std::ptrdiff_t first_begin,
    const std::ptrdiff_t first_end,
    const std::ptrdiff_t second_begin,
    const std::ptrdiff_t second_end
) {
    if (first_end < second_begin) {
        return second_begin - first_end <= 1;
    }
    if (second_end < first_begin) {
        return first_begin - second_end <= 1;
    }
    return true;
}

template <class LabelT, bool Binary, class Profiler>
ComponentSet<LabelT> extract_components(
    const ConstArrayView<LabelT> &input,
    const LabelT background,
    Profiler &profiler
) {
    if (input.shape.size() != 3) {
        throw std::invalid_argument(
            "component input must have exactly three dimensions"
        );
    }
    for (const auto extent : input.shape) {
        if (extent < 0) {
            throw std::invalid_argument("component input shape must be non-negative");
        }
    }

    const auto z_size = static_cast<std::size_t>(input.shape[0]);
    const auto y_size = static_cast<std::size_t>(input.shape[1]);
    const auto x_size = static_cast<std::size_t>(input.shape[2]);
    const auto row_count = checked_multiply_size(
        z_size, y_size, "component row count overflows size_t"
    );
    (void)checked_multiply_size(
        row_count, x_size, "component input size overflows size_t"
    );

    ComponentSet<LabelT> result;
    result.input_shape = {
        static_cast<std::ptrdiff_t>(z_size),
        static_cast<std::ptrdiff_t>(y_size),
        static_cast<std::ptrdiff_t>(x_size),
    };
    result.row_offsets.resize(
        checked_add_size(row_count, 1, "component row offsets overflow size_t"),
        0
    );

    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "component_scan")
        for (std::size_t z = 0; z < z_size; ++z) {
            for (std::size_t y = 0; y < y_size; ++y) {
                const auto row = z * y_size + y;
                const auto row_begin = row * x_size;
                result.row_offsets[row] = result.runs.size();
                std::size_t x = 0;
                while (x < x_size) {
                    const auto value = input.data[row_begin + x];
                    if (value == background) {
                        ++x;
                        continue;
                    }
                    const auto run_begin = x;
                    ++x;
                    if constexpr (Binary) {
                        while (x < x_size && input.data[row_begin + x] != background) {
                            ++x;
                        }
                    } else {
                        while (x < x_size && input.data[row_begin + x] == value) {
                            ++x;
                        }
                    }
                    result.runs.push_back({
                        Binary ? LabelT{1} : value,
                        static_cast<std::ptrdiff_t>(z),
                        static_cast<std::ptrdiff_t>(y),
                        static_cast<std::ptrdiff_t>(run_begin),
                        static_cast<std::ptrdiff_t>(x - 1),
                    });
                    result.foreground_count = checked_add_size(
                        result.foreground_count,
                        x - run_begin,
                        "component foreground count overflows size_t"
                    );
                }
            }
        }
    }
    result.row_offsets[row_count] = result.runs.size();
    if (result.runs.empty()) {
        return result;
    }
    if (result.runs.size() > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("component run count exceeds uint64 range");
    }

    util::UnionFind union_find(result.runs.size());
    const auto union_rows = [&] (
        const std::size_t current_row,
        const std::size_t previous_row
    ) {
        const auto current_begin = result.row_offsets[current_row];
        const auto current_end = result.row_offsets[current_row + 1];
        const auto previous_begin = result.row_offsets[previous_row];
        const auto previous_end = result.row_offsets[previous_row + 1];
        std::size_t first_candidate = previous_begin;
        for (auto current = current_begin; current < current_end; ++current) {
            const auto &run = result.runs[current];
            while (
                first_candidate < previous_end &&
                result.runs[first_candidate].x_end < run.x_begin &&
                run.x_begin - result.runs[first_candidate].x_end > 1
            ) {
                ++first_candidate;
            }
            for (auto previous = first_candidate; previous < previous_end; ++previous) {
                const auto &candidate = result.runs[previous];
                if (
                    run.x_end < candidate.x_begin &&
                    candidate.x_begin - run.x_end > 1
                ) {
                    break;
                }
                if constexpr (!Binary) {
                    if (run.label != candidate.label) {
                        continue;
                    }
                }
                if (intervals_within_one(
                        run.x_begin, run.x_end,
                        candidate.x_begin, candidate.x_end)) {
                    union_find.merge(
                        static_cast<std::uint64_t>(current),
                        static_cast<std::uint64_t>(previous)
                    );
                }
            }
        }
    };

    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "component_union")
        for (std::size_t z = 0; z < z_size; ++z) {
            for (std::size_t y = 0; y < y_size; ++y) {
                const auto row = z * y_size + y;
                if (y > 0) {
                    union_rows(row, row - 1);
                }
                if (z > 0) {
                    const auto previous_slice = (z - 1) * y_size;
                    if (y > 0) {
                        union_rows(row, previous_slice + y - 1);
                    }
                    union_rows(row, previous_slice + y);
                    if (y + 1 < y_size) {
                        union_rows(row, previous_slice + y + 1);
                    }
                }
            }
        }
    }

    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "component_descriptors")
        std::unordered_map<std::uint64_t, std::size_t> component_of_root;
        component_of_root.reserve(result.runs.size());
        std::vector<std::size_t> component_of_run(result.runs.size());
        for (std::size_t run_id = 0; run_id < result.runs.size(); ++run_id) {
            const auto root = union_find.find(static_cast<std::uint64_t>(run_id));
            auto [it, inserted] = component_of_root.emplace(
                root, result.components.size()
            );
            const auto component_id = it->second;
            const auto &run = result.runs[run_id];
            const auto run_length = static_cast<std::uint64_t>(
                run.x_end - run.x_begin + 1
            );
            if (inserted) {
                const auto first_flat = checked_add_size(
                    checked_multiply_size(
                        checked_add_size(
                            checked_multiply_size(
                                static_cast<std::size_t>(run.z), y_size,
                                "component first index overflows size_t"
                            ),
                            static_cast<std::size_t>(run.y),
                            "component first index overflows size_t"
                        ),
                        x_size,
                        "component first index overflows size_t"
                    ),
                    static_cast<std::size_t>(run.x_begin),
                    "component first index overflows size_t"
                );
                result.components.push_back({
                    run.label,
                    {run.z, run.y, run.x_begin},
                    {run.z + 1, run.y + 1, run.x_end + 1},
                    {run.z, run.y, run.x_begin},
                    run_length,
                    static_cast<std::uint64_t>(first_flat),
                    0,
                    1,
                });
            } else {
                auto &component = result.components[component_id];
                if constexpr (!Binary) {
                    if (component.label != run.label) {
                        throw std::runtime_error(
                            "component union mixed distinct semantic labels"
                        );
                    }
                }
                component.begin[0] = std::min(component.begin[0], run.z);
                component.begin[1] = std::min(component.begin[1], run.y);
                component.begin[2] = std::min(component.begin[2], run.x_begin);
                component.end[0] = std::max(component.end[0], run.z + 1);
                component.end[1] = std::max(component.end[1], run.y + 1);
                component.end[2] = std::max(component.end[2], run.x_end + 1);
                if (
                    component.voxel_count >
                    std::numeric_limits<std::uint64_t>::max() - run_length
                ) {
                    throw std::overflow_error(
                        "component voxel count overflows uint64"
                    );
                }
                component.voxel_count += run_length;
                component.number_of_runs = checked_add_size(
                    component.number_of_runs, 1,
                    "component run count overflows size_t"
                );
            }
            component_of_run[run_id] = component_id;
        }

        std::size_t run_offset = 0;
        for (auto &component : result.components) {
            component.run_offset = run_offset;
            run_offset = checked_add_size(
                run_offset, component.number_of_runs,
                "component membership offsets overflow size_t"
            );
        }
        result.component_run_ids.resize(run_offset);
        std::vector<std::size_t> cursors(result.components.size());
        for (std::size_t component_id = 0;
             component_id < result.components.size(); ++component_id) {
            cursors[component_id] = result.components[component_id].run_offset;
        }
        for (std::size_t run_id = 0; run_id < result.runs.size(); ++run_id) {
            const auto component_id = component_of_run[run_id];
            result.component_run_ids[cursors[component_id]++] = run_id;
        }
    }
    return result;
}

template <class Profiler>
inline ComponentSet<std::uint8_t> extract_binary_components(
    const ConstArrayView<std::uint8_t> &mask,
    Profiler &profiler
) {
    return extract_components<std::uint8_t, true>(
        mask, std::uint8_t{0}, profiler
    );
}

template <class LabelT, class Profiler>
ComponentSet<LabelT> extract_labeled_components(
    const ConstArrayView<LabelT> &labels,
    const LabelT background,
    Profiler &profiler
) {
    static_assert(std::is_integral_v<LabelT> && !std::is_same_v<LabelT, bool>);
    return extract_components<LabelT, false>(labels, background, profiler);
}

template <class LabelT>
std::size_t padded_component_volume(
    const ComponentDescriptor<LabelT> &component
) {
    std::size_t volume = 1;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto extent = static_cast<std::size_t>(
            component.end[axis] - component.begin[axis]
        );
        volume = checked_multiply_size(
            volume,
            checked_add_size(extent, 2, "component padding overflows size_t"),
            "padded component volume overflows size_t"
        );
    }
    return volume;
}

template <class LabelT>
PreparedTeasarComponent prepare_component(
    const ComponentSet<LabelT> &components,
    const std::size_t component_id,
    const std::vector<std::array<std::ptrdiff_t, 3>> &required_targets = {},
    const OpenBlockFaces *open_faces = nullptr
) {
    const auto &component = components.components.at(component_id);
    PreparedTeasarComponent prepared;
    prepared.input_origin = component.begin;
    prepared.foreground_count = static_cast<std::size_t>(component.voxel_count);
    prepared.padded_shape.reserve(3);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto extent = component.end[axis] - component.begin[axis];
        if (extent > std::numeric_limits<std::ptrdiff_t>::max() - 2) {
            throw std::overflow_error("padded component shape overflows ptrdiff_t");
        }
        prepared.padded_shape.push_back(extent + 2);
    }
    const auto padded_size = checked_shape_size(
        prepared.padded_shape, "padded component size overflows size_t"
    );
    prepared.padded_mask.assign(padded_size, std::uint8_t{0});
    const auto sy = static_cast<std::size_t>(prepared.padded_shape[2]);
    const auto sz = checked_multiply_size(
        static_cast<std::size_t>(prepared.padded_shape[1]), sy,
        "padded component stride overflows size_t"
    );
    for (std::size_t offset = 0; offset < component.number_of_runs; ++offset) {
        const auto run_id = components.component_run_ids[
            component.run_offset + offset
        ];
        const auto &run = components.runs[run_id];
        const auto z = static_cast<std::size_t>(
            run.z - component.begin[0] + 1
        );
        const auto y = static_cast<std::size_t>(
            run.y - component.begin[1] + 1
        );
        const auto x_begin = static_cast<std::size_t>(
            run.x_begin - component.begin[2] + 1
        );
        const auto x_end = static_cast<std::size_t>(
            run.x_end - component.begin[2] + 1
        );
        const auto row_begin = z * sz + y * sy;
        std::fill(
            prepared.padded_mask.begin() +
                static_cast<std::ptrdiff_t>(row_begin + x_begin),
            prepared.padded_mask.begin() +
                static_cast<std::ptrdiff_t>(row_begin + x_end + 1),
            std::uint8_t{1}
        );
    }

    // The zero halo in padded_mask is part of the TEASAR path-domain
    // contract. For block calls, construct a separate distance-only mask by
    // copying foreground through declared artificial cuts. The EDT does not
    // treat array exteriors as background, so one copied layer is sufficient.
    // Corners and edges are extended only if every participating face is open.
    const auto input_size = checked_multiply_size(
        checked_multiply_size(
            static_cast<std::size_t>(components.input_shape[0]),
            static_cast<std::size_t>(components.input_shape[1]),
            "component input size overflows size_t"
        ),
        static_cast<std::size_t>(components.input_shape[2]),
        "component input size overflows size_t"
    );
    const bool all_foreground = components.components.size() == 1 &&
        static_cast<std::size_t>(component.voxel_count) == input_size;
    if (open_faces != nullptr && open_faces->any() && !all_foreground) {
        std::array<bool, 6> extend{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            extend[2 * axis] = (*open_faces)(axis, false) &&
                component.begin[axis] == 0;
            extend[2 * axis + 1] = (*open_faces)(axis, true) &&
                component.end[axis] == components.input_shape[axis];
        }
        if (std::any_of(extend.begin(), extend.end(), [](const bool value) {
                return value;
            })) {
            prepared.distance_mask = prepared.padded_mask;
            const auto z_size = prepared.padded_shape[0];
            const auto y_size = prepared.padded_shape[1];
            const auto x_size = prepared.padded_shape[2];
            for (std::ptrdiff_t z = 0; z < z_size; ++z) {
                for (std::ptrdiff_t y = 0; y < y_size; ++y) {
                    for (std::ptrdiff_t x = 0; x < x_size; ++x) {
                        const std::array<std::ptrdiff_t, 3> coordinate{z, y, x};
                        std::array<std::ptrdiff_t, 3> source = coordinate;
                        bool is_halo = false;
                        bool permitted = true;
                        for (std::size_t axis = 0; axis < 3; ++axis) {
                            if (coordinate[axis] == 0) {
                                is_halo = true;
                                permitted = permitted && extend[2 * axis];
                                source[axis] = 1;
                            } else if (
                                coordinate[axis] == prepared.padded_shape[axis] - 1
                            ) {
                                is_halo = true;
                                permitted = permitted && extend[2 * axis + 1];
                                source[axis] = prepared.padded_shape[axis] - 2;
                            }
                        }
                        if (!is_halo || !permitted) {
                            continue;
                        }
                        const auto destination = static_cast<std::size_t>(
                            z * static_cast<std::ptrdiff_t>(sz) +
                            y * static_cast<std::ptrdiff_t>(sy) + x
                        );
                        const auto source_index = static_cast<std::size_t>(
                            source[0] * static_cast<std::ptrdiff_t>(sz) +
                            source[1] * static_cast<std::ptrdiff_t>(sy) + source[2]
                        );
                        prepared.distance_mask[destination] =
                            prepared.padded_mask[source_index];
                    }
                }
            }
        }
    }

    prepared.required_target_voxels.reserve(required_targets.size());
    std::array<std::ptrdiff_t, 3> required_root_coordinate{};
    bool have_required_root = false;
    for (const auto &target : required_targets) {
        const auto z = static_cast<std::size_t>(
            target[0] - component.begin[0] + 1
        );
        const auto y = static_cast<std::size_t>(
            target[1] - component.begin[1] + 1
        );
        const auto x = static_cast<std::size_t>(
            target[2] - component.begin[2] + 1
        );
        const auto target_voxel = z * sz + y * sy + x;
        prepared.required_target_voxels.push_back(target_voxel);
        bool on_open_face = false;
        if (open_faces != nullptr) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                on_open_face = on_open_face ||
                    (target[axis] == 0 && (*open_faces)(axis, false)) ||
                    (target[axis] == components.input_shape[axis] - 1 &&
                     (*open_faces)(axis, true));
            }
        }
        if (
            on_open_face &&
            (!have_required_root || target > required_root_coordinate)
        ) {
            have_required_root = true;
            required_root_coordinate = target;
            prepared.required_root_voxel = target_voxel;
        }
    }
    return prepared;
}

template <class LabelT>
struct ComponentTarget {
    LabelT label{};
    std::array<std::ptrdiff_t, 3> coordinate{};
};

// Assign already validated foreground coordinates to the equality-connected
// run components discovered by extract_components. The O(number_of_runs)
// reverse lookup is allocated only for calls that actually carry targets, so
// ordinary TEASAR dispatch keeps its current memory profile.
template <class LabelT>
std::vector<std::vector<std::array<std::ptrdiff_t, 3>>>
assign_targets_to_components(
    const ComponentSet<LabelT> &components,
    const std::vector<ComponentTarget<LabelT>> &targets
) {
    std::vector<std::vector<std::array<std::ptrdiff_t, 3>>> assigned(
        components.components.size()
    );
    if (targets.empty()) {
        return assigned;
    }

    std::vector<std::size_t> component_of_run(
        components.runs.size(), std::numeric_limits<std::size_t>::max()
    );
    for (std::size_t component = 0;
         component < components.components.size(); ++component) {
        const auto &descriptor = components.components[component];
        for (std::size_t offset = 0; offset < descriptor.number_of_runs; ++offset) {
            const auto run_id = components.component_run_ids[
                descriptor.run_offset + offset
            ];
            component_of_run[run_id] = component;
        }
    }

    const auto y_size = static_cast<std::size_t>(components.input_shape[1]);
    for (const auto &target : targets) {
        const auto row = static_cast<std::size_t>(target.coordinate[0]) * y_size +
            static_cast<std::size_t>(target.coordinate[1]);
        const auto begin = components.row_offsets[row];
        const auto end = components.row_offsets[row + 1];
        auto run = std::lower_bound(
            components.runs.begin() + static_cast<std::ptrdiff_t>(begin),
            components.runs.begin() + static_cast<std::ptrdiff_t>(end),
            target.coordinate[2],
            [](const ComponentRun<LabelT> &candidate, const std::ptrdiff_t x) {
                return candidate.x_end < x;
            }
        );
        if (
            run == components.runs.begin() + static_cast<std::ptrdiff_t>(end) ||
            run->x_begin > target.coordinate[2] || run->label != target.label
        ) {
            throw std::runtime_error(
                "validated required target was not found in a component run"
            );
        }
        const auto run_id = static_cast<std::size_t>(
            std::distance(components.runs.begin(), run)
        );
        const auto component = component_of_run[run_id];
        if (component == std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("required target component lookup is inconsistent");
        }
        assigned[component].push_back(target.coordinate);
    }
    return assigned;
}

} // namespace bioimage_cpp::skeleton::detail
