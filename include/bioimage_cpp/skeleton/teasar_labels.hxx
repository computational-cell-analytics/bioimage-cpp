#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/skeleton/teasar.hxx"

#include <algorithm>
#include <cstddef>
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
    std::vector<SkeletonGraph> results;
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
                detail_teasar::assemble_skeleton_graphs(results, label_components),
            });
            begin = end;
        }
    }
    BIOIMAGE_PROFILE_REPORT(profile)
    return output;
}

} // namespace bioimage_cpp::skeleton
