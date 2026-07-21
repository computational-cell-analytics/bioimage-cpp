#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/skeleton/teasar.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace bioimage_cpp::skeleton {

template <class LabelT>
struct LabeledSkeleton {
    LabelT label{};
    SkeletonGraph skeleton;
};

template <class LabelT>
struct LabeledLatticeSkeleton {
    LabelT label{};
    LatticeSkeletonGraph skeleton;
};

template <class LabelT>
std::vector<LabeledSkeleton<LabelT>> teasar_labels(
    const ConstArrayView<LabelT> &labels,
    const LabelT background,
    const TeasarOptions &options = {}
) {
    static_assert(std::is_integral_v<LabelT> && !std::is_same_v<LabelT, bool>);
    ConstArrayView<std::uint8_t> shape_only{nullptr, labels.shape, {}};
    detail_teasar::validate_options(shape_only, options);
    BIOIMAGE_PROFILE_INIT(profile)

    auto components = detail::extract_labeled_components(
        labels, background, profile
    );
    std::vector<LatticeSkeletonGraph> results;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "component_teasar")
        results = detail_teasar::skeletonize_components(
            components, options, true
        );
    }
    std::vector<std::size_t> component_ids(results.size());
    for (std::size_t component = 0; component < component_ids.size(); ++component) {
        component_ids[component] = component;
    }
    std::sort(
        component_ids.begin(), component_ids.end(),
        [&](const std::size_t first, const std::size_t second) {
            const auto &first_component = components.components[first];
            const auto &second_component = components.components[second];
            if (first_component.label != second_component.label) {
                return first_component.label < second_component.label;
            }
            return first_component.first_flat_index <
                second_component.first_flat_index;
        }
    );

    std::vector<LabeledSkeleton<LabelT>> output;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "forest_assembly")
        std::size_t begin = 0;
        while (begin < component_ids.size()) {
            const auto label = components.components[component_ids[begin]].label;
            auto end = begin + 1;
            while (
                end < component_ids.size() &&
                components.components[component_ids[end]].label == label
            ) {
                ++end;
            }
            std::vector<std::size_t> label_components(
                component_ids.begin() + static_cast<std::ptrdiff_t>(begin),
                component_ids.begin() + static_cast<std::ptrdiff_t>(end)
            );
            output.push_back({
                label,
                detail_teasar::lattice_to_physical(
                    detail_teasar::assemble_skeleton_graphs(
                        results, label_components
                    ),
                    options.spacing
                ),
            });
            begin = end;
        }
    }
    BIOIMAGE_PROFILE_REPORT(profile)
    return output;
}

template <class LabelT>
std::vector<LabeledLatticeSkeleton<LabelT>> teasar_labels_block(
    const ConstArrayView<LabelT> &labels,
    const LabelT background,
    std::vector<LabeledVoxelTarget<LabelT>> required_targets,
    const detail::OpenBlockFaces &open_faces,
    const TeasarOptions &options = {}
) {
    static_assert(std::is_integral_v<LabelT> && !std::is_same_v<LabelT, bool>);
    ConstArrayView<std::uint8_t> shape_only{nullptr, labels.shape, {}};
    detail_teasar::validate_options(shape_only, options);

    std::vector<detail::ComponentTarget<LabelT>> local_targets;
    local_targets.reserve(required_targets.size());
    for (std::size_t row = 0; row < required_targets.size(); ++row) {
        const auto &target = required_targets[row];
        std::array<std::ptrdiff_t, 3> coordinate{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (
                target.coordinate[axis] < 0 ||
                static_cast<std::uint64_t>(target.coordinate[axis]) >=
                    static_cast<std::uint64_t>(labels.shape[axis])
            ) {
                throw std::invalid_argument(
                    "required_targets row " + std::to_string(row) +
                    " is out of bounds at axis " + std::to_string(axis)
                );
            }
            coordinate[axis] = static_cast<std::ptrdiff_t>(
                target.coordinate[axis]
            );
        }
        const auto flat = static_cast<std::size_t>(
            (coordinate[0] * labels.shape[1] + coordinate[1]) * labels.shape[2] +
            coordinate[2]
        );
        if (target.label == background || labels.data[flat] != target.label) {
            throw std::invalid_argument(
                "required_targets row " + std::to_string(row) +
                " does not match its semantic label"
            );
        }
        local_targets.push_back({target.label, coordinate});
    }
    std::sort(
        local_targets.begin(), local_targets.end(),
        [](const auto &first, const auto &second) {
            if (first.label != second.label) {
                return first.label < second.label;
            }
            return first.coordinate < second.coordinate;
        }
    );
    local_targets.erase(
        std::unique(
            local_targets.begin(), local_targets.end(),
            [](const auto &first, const auto &second) {
                return first.label == second.label &&
                    first.coordinate == second.coordinate;
            }
        ),
        local_targets.end()
    );

    BIOIMAGE_PROFILE_INIT(profile)
    auto components = detail::extract_labeled_components(
        labels, background, profile
    );
    auto component_targets = detail::assign_targets_to_components(
        components, local_targets
    );
    std::vector<LatticeSkeletonGraph> results;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "component_teasar")
        results = detail_teasar::skeletonize_components(
            components, options, true, &component_targets, &open_faces
        );
    }

    std::vector<std::size_t> component_ids(results.size());
    std::iota(component_ids.begin(), component_ids.end(), std::size_t{0});
    std::sort(
        component_ids.begin(), component_ids.end(),
        [&](const std::size_t first, const std::size_t second) {
            const auto &first_component = components.components[first];
            const auto &second_component = components.components[second];
            if (first_component.label != second_component.label) {
                return first_component.label < second_component.label;
            }
            return first_component.first_flat_index <
                second_component.first_flat_index;
        }
    );

    std::vector<LabeledLatticeSkeleton<LabelT>> output;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "forest_assembly")
        std::size_t begin = 0;
        while (begin < component_ids.size()) {
            const auto label = components.components[component_ids[begin]].label;
            auto end = begin + 1;
            while (
                end < component_ids.size() &&
                components.components[component_ids[end]].label == label
            ) {
                ++end;
            }
            std::vector<std::size_t> label_components(
                component_ids.begin() + static_cast<std::ptrdiff_t>(begin),
                component_ids.begin() + static_cast<std::ptrdiff_t>(end)
            );
            output.push_back({
                label,
                detail_teasar::assemble_skeleton_graphs(results, label_components),
            });
            begin = end;
        }
    }
    BIOIMAGE_PROFILE_REPORT(profile)
    return output;
}

} // namespace bioimage_cpp::skeleton
