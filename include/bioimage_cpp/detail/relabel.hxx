#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace bioimage_cpp::detail {

// Map an arbitrary labeling to the dense range [0, k) preserving the first
// occurrence order of each distinct input label.
inline std::vector<std::uint64_t> dense_relabel(const std::vector<std::uint64_t> &labels) {
    std::unordered_map<std::uint64_t, std::uint64_t> relabeling;
    std::vector<std::uint64_t> result(labels.size());
    for (std::size_t index = 0; index < labels.size(); ++index) {
        auto found = relabeling.find(labels[index]);
        if (found == relabeling.end()) {
            found = relabeling.emplace(labels[index], static_cast<std::uint64_t>(relabeling.size())).first;
        }
        result[index] = found->second;
    }
    return result;
}

// Mark each current cluster whose node set differs from the previous
// partition. An unchanged cluster contains all nodes from exactly one previous
// cluster. `labels` must use the dense range [0, number_of_clusters).
inline std::vector<std::uint8_t> changed_clusters(
    const std::vector<std::uint64_t> &labels,
    const std::vector<std::uint64_t> &previous_labels,
    const std::uint64_t number_of_clusters
) {
    std::vector<std::uint8_t> changed(
        static_cast<std::size_t>(number_of_clusters), 0
    );
    if (number_of_clusters == 0) {
        return changed;
    }
    if (previous_labels.size() != labels.size()) {
        std::fill(changed.begin(), changed.end(), 1);
        return changed;
    }

    const auto max_previous =
        *std::max_element(previous_labels.begin(), previous_labels.end());
    std::vector<std::uint64_t> previous_sizes(
        static_cast<std::size_t>(max_previous) + 1, 0
    );
    for (const auto label : previous_labels) {
        ++previous_sizes[static_cast<std::size_t>(label)];
    }

    constexpr auto unmapped = std::numeric_limits<std::uint64_t>::max();
    std::vector<std::uint64_t> previous_of_current(
        static_cast<std::size_t>(number_of_clusters), unmapped
    );
    std::vector<std::uint64_t> current_sizes(
        static_cast<std::size_t>(number_of_clusters), 0
    );

    for (std::size_t node = 0; node < labels.size(); ++node) {
        const auto current = static_cast<std::size_t>(labels[node]);
        const auto previous = previous_labels[node];
        ++current_sizes[current];
        if (changed[current]) {
            continue;
        }
        if (previous_of_current[current] == unmapped) {
            previous_of_current[current] = previous;
        } else if (previous_of_current[current] != previous) {
            changed[current] = 1;
        }
    }

    for (std::size_t current = 0; current < changed.size(); ++current) {
        if (changed[current]) {
            continue;
        }
        const auto previous = previous_of_current[current];
        if (previous == unmapped
            || previous_sizes[static_cast<std::size_t>(previous)]
                != current_sizes[current]) {
            changed[current] = 1;
        }
    }
    return changed;
}

} // namespace bioimage_cpp::detail
