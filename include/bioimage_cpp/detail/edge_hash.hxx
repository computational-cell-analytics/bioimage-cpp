#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace bioimage_cpp::detail {

using NodeId = std::uint64_t;
using Edge = std::pair<NodeId, NodeId>;

inline Edge edge_key(NodeId u, NodeId v) {
    if (v < u) {
        std::swap(u, v);
    }
    return {u, v};
}

struct EdgeHash {
    std::size_t operator()(const Edge &edge) const {
        const auto first = static_cast<std::size_t>(edge.first);
        const auto second = static_cast<std::size_t>(edge.second);
        return first ^ (second + 0x9e3779b97f4a7c15ULL + (first << 6U) + (first >> 2U));
    }
};

} // namespace bioimage_cpp::detail
