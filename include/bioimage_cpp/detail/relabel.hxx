#pragma once

#include <cstddef>
#include <cstdint>
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

} // namespace bioimage_cpp::detail
