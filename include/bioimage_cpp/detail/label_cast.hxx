#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace bioimage_cpp::detail {

// Convert a label value to a graph node id (`std::uint64_t`), rejecting negative
// values for signed label dtypes. Shared by the region-adjacency-graph scan,
// edge-feature accumulation, and the distributed block-extraction primitives so
// they all treat labels identically.
template <class T>
std::uint64_t checked_label_to_node(const T value) {
    if constexpr (std::is_signed_v<T>) {
        if (value < 0) {
            throw std::invalid_argument("labels must not contain negative values");
        }
    }
    return static_cast<std::uint64_t>(value);
}

} // namespace bioimage_cpp::detail
