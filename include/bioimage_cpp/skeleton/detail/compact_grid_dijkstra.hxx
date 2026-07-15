#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace bioimage_cpp::skeleton::detail {

inline constexpr std::uint32_t kNoCompactNode =
    std::numeric_limits<std::uint32_t>::max();

enum class CompactAdjacency {
    OnTheFly,
    Csr,
};

struct CompactNeighbor {
    std::ptrdiff_t delta = 0;
    double physical_length = 0.0;
};

// A deterministic foreground-only view of a zero-padded 3D mask. Compact IDs
// follow ascending full C-order indices, so the compact and dense Dijkstra
// heaps use the same tie-breaking order.
struct CompactGridDomain {
    std::vector<std::ptrdiff_t> shape;
    std::vector<std::ptrdiff_t> strides;
    std::vector<std::uint32_t> compact_to_full;
    std::vector<std::size_t> offsets;
    std::vector<std::uint32_t> targets;
    std::vector<std::uint8_t> neighbor_codes;
    std::vector<std::uint32_t> full_to_compact;
    std::array<CompactNeighbor, 26> neighbors{};

    [[nodiscard]] std::size_t size() const noexcept {
        return compact_to_full.size();
    }

    [[nodiscard]] bool has_csr() const noexcept {
        return !offsets.empty();
    }

    [[nodiscard]] bool has_full_lookup() const noexcept {
        return !full_to_compact.empty();
    }
};

inline void build_compact_neighbors(
    CompactGridDomain &domain,
    const std::array<double, 3> &spacing
) {
    std::size_t code = 0;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dz == 0 && dy == 0 && dx == 0) {
                    continue;
                }
                const double pz = static_cast<double>(dz) * spacing[0];
                const double py = static_cast<double>(dy) * spacing[1];
                const double px = static_cast<double>(dx) * spacing[2];
                domain.neighbors[code++] = {
                    static_cast<std::ptrdiff_t>(dz) * domain.strides[0] +
                        static_cast<std::ptrdiff_t>(dy) * domain.strides[1] + dx,
                    std::sqrt(pz * pz + py * py + px * px),
                };
            }
        }
    }
}

// Returns false when the foreground cannot be represented with uint32 IDs.
// The mask must have a one-voxel zero halo, which permits bounds-free neighbor
// lookup during construction and on-the-fly traversal.
inline bool build_compact_grid_domain(
    const ConstArrayView<std::uint8_t> &mask,
    const std::array<double, 3> &spacing,
    const CompactAdjacency adjacency,
    CompactGridDomain &domain
) {
    if (mask.shape.size() != 3) {
        throw std::invalid_argument("compact grid domain requires a 3D mask");
    }
    domain = {};
    domain.shape = mask.shape;
    domain.strides = bioimage_cpp::detail::c_order_strides(mask.shape);
    build_compact_neighbors(domain, spacing);

    const auto n = bioimage_cpp::detail::number_of_elements(mask.shape);
    if (n > static_cast<std::size_t>(kNoCompactNode)) {
        return false;
    }
    const auto foreground_count = static_cast<std::size_t>(
        std::count_if(mask.data, mask.data + n, [](const std::uint8_t value) {
            return value != 0;
        })
    );
    if (foreground_count > static_cast<std::size_t>(kNoCompactNode)) {
        return false;
    }

    domain.compact_to_full.reserve(foreground_count);
    domain.full_to_compact.assign(n, kNoCompactNode);
    for (std::size_t full = 0; full < n; ++full) {
        if (mask.data[full] == 0) {
            continue;
        }
        const auto compact = static_cast<std::uint32_t>(domain.compact_to_full.size());
        domain.full_to_compact[full] = compact;
        domain.compact_to_full.push_back(static_cast<std::uint32_t>(full));
    }

    if (adjacency == CompactAdjacency::OnTheFly) {
        return true;
    }

    domain.offsets.resize(foreground_count + 1, 0);
    for (std::size_t compact = 0; compact < foreground_count; ++compact) {
        const auto full = static_cast<std::ptrdiff_t>(
            domain.compact_to_full[compact]
        );
        std::size_t degree = 0;
        for (const auto &neighbor : domain.neighbors) {
            const auto target_full = static_cast<std::size_t>(full + neighbor.delta);
            degree += domain.full_to_compact[target_full] != kNoCompactNode ? 1 : 0;
        }
        domain.offsets[compact + 1] = domain.offsets[compact] + degree;
    }

    domain.targets.resize(domain.offsets.back());
    domain.neighbor_codes.resize(domain.offsets.back());
    for (std::size_t compact = 0; compact < foreground_count; ++compact) {
        const auto full = static_cast<std::ptrdiff_t>(
            domain.compact_to_full[compact]
        );
        auto edge = domain.offsets[compact];
        for (std::uint8_t code = 0; code < 26; ++code) {
            const auto target_full = static_cast<std::size_t>(
                full + domain.neighbors[code].delta
            );
            const auto target = domain.full_to_compact[target_full];
            if (target == kNoCompactNode) {
                continue;
            }
            domain.targets[edge] = target;
            domain.neighbor_codes[edge] = code;
            ++edge;
        }
    }
    std::vector<std::uint32_t>().swap(domain.full_to_compact);
    return true;
}

template <class Distance>
struct CompactHeapEntry {
    Distance distance = Distance{0};
    std::uint32_t node = 0;
};

template <class Distance>
struct CompactHeapGreater {
    bool operator()(
        const CompactHeapEntry<Distance> &a,
        const CompactHeapEntry<Distance> &b
    ) const noexcept {
        return a.distance > b.distance ||
            (a.distance == b.distance && a.node > b.node);
    }
};

struct CompactDijkstraStats {
    std::size_t pushes = 0;
    std::size_t pops = 0;
    std::size_t peak_heap = 0;

    void reset() noexcept {
        pushes = 0;
        pops = 0;
        peak_heap = 0;
    }
};

template <class Distance>
struct CompactDijkstraWorkspace {
    static_assert(std::is_same_v<Distance, float> || std::is_same_v<Distance, double>);

    std::vector<std::uint8_t> state;
    std::vector<std::uint32_t> predecessors;
    std::vector<CompactHeapEntry<Distance>> heap;
};

inline constexpr std::uint8_t kCompactDiscovered = 1;
inline constexpr std::uint8_t kCompactSettled = 2;
inline constexpr std::uint8_t kCompactTarget = 4;

template <class Distance>
inline void compact_heap_push(
    CompactDijkstraWorkspace<Distance> &workspace,
    const CompactHeapEntry<Distance> entry,
    CompactDijkstraStats *stats
) {
    workspace.heap.push_back(entry);
    std::push_heap(
        workspace.heap.begin(), workspace.heap.end(), CompactHeapGreater<Distance>{}
    );
    if (stats != nullptr) {
        ++stats->pushes;
        stats->peak_heap = std::max(stats->peak_heap, workspace.heap.size());
    }
}

template <class Distance>
inline CompactHeapEntry<Distance> compact_heap_pop(
    CompactDijkstraWorkspace<Distance> &workspace,
    CompactDijkstraStats *stats
) {
    std::pop_heap(
        workspace.heap.begin(), workspace.heap.end(), CompactHeapGreater<Distance>{}
    );
    const auto entry = workspace.heap.back();
    workspace.heap.pop_back();
    if (stats != nullptr) {
        ++stats->pops;
    }
    return entry;
}

template <CompactAdjacency Adjacency, class Body>
inline void for_each_compact_neighbor(
    const CompactGridDomain &domain,
    const std::uint32_t node,
    const Body &body
) {
    if constexpr (Adjacency == CompactAdjacency::Csr) {
        for (auto edge = domain.offsets[node]; edge < domain.offsets[node + 1]; ++edge) {
            body(
                domain.targets[edge],
                domain.neighbors[domain.neighbor_codes[edge]].physical_length
            );
        }
    } else {
        const auto full = static_cast<std::ptrdiff_t>(domain.compact_to_full[node]);
        for (const auto &neighbor : domain.neighbors) {
            const auto target_full = static_cast<std::size_t>(full + neighbor.delta);
            const auto target = domain.full_to_compact[target_full];
            if (target != kNoCompactNode) {
                body(target, neighbor.physical_length);
            }
        }
    }
}

template <CompactAdjacency Adjacency, class Distance>
inline void compact_physical_distance_field(
    const CompactGridDomain &domain,
    const std::uint32_t source,
    CompactDijkstraWorkspace<Distance> &workspace,
    std::vector<Distance> &distances,
    CompactDijkstraStats *stats = nullptr
) {
    const auto n = domain.size();
    if (source >= n) {
        throw std::invalid_argument("compact Dijkstra source is out of bounds");
    }
    if constexpr (Adjacency == CompactAdjacency::Csr) {
        if (!domain.has_csr()) {
            throw std::invalid_argument("compact CSR adjacency is not available");
        }
    } else if (!domain.has_full_lookup()) {
        throw std::invalid_argument("compact full-index lookup is not available");
    }

    distances.assign(n, std::numeric_limits<Distance>::infinity());
    workspace.state.assign(n, 0);
    workspace.heap.clear();
    if (stats != nullptr) {
        stats->reset();
    }
    distances[source] = Distance{0};
    workspace.state[source] = kCompactDiscovered;
    compact_heap_push(workspace, {Distance{0}, source}, stats);

    while (!workspace.heap.empty()) {
        const auto entry = compact_heap_pop(workspace, stats);
        const auto node = entry.node;
        if ((workspace.state[node] & kCompactSettled) != 0) {
            continue;
        }
        workspace.state[node] |= kCompactSettled;
        for_each_compact_neighbor<Adjacency>(
            domain, node,
            [&](const std::uint32_t target, const double physical_length) {
                if ((workspace.state[target] & kCompactSettled) != 0) {
                    return;
                }
                const auto candidate = static_cast<Distance>(
                    entry.distance + static_cast<Distance>(physical_length)
                );
                const bool discovered =
                    (workspace.state[target] & kCompactDiscovered) != 0;
                if (discovered && !(candidate < distances[target])) {
                    return;
                }
                workspace.state[target] |= kCompactDiscovered;
                distances[target] = candidate;
                compact_heap_push(workspace, {candidate, target}, stats);
            }
        );
    }
}

template <CompactAdjacency Adjacency, class Distance>
inline void compact_node_cost_path(
    const CompactGridDomain &domain,
    const std::uint32_t source,
    const std::vector<std::uint32_t> &targets,
    const std::vector<Distance> &costs,
    CompactDijkstraWorkspace<Distance> &workspace,
    std::vector<std::uint32_t> &path,
    CompactDijkstraStats *stats = nullptr
) {
    const auto n = domain.size();
    if (source >= n || targets.empty() || costs.size() != n) {
        throw std::invalid_argument("invalid compact node-cost path inputs");
    }
    for (const auto target : targets) {
        if (target >= n) {
            throw std::invalid_argument("compact Dijkstra target is out of bounds");
        }
    }
    if constexpr (Adjacency == CompactAdjacency::Csr) {
        if (!domain.has_csr()) {
            throw std::invalid_argument("compact CSR adjacency is not available");
        }
    } else if (!domain.has_full_lookup()) {
        throw std::invalid_argument("compact full-index lookup is not available");
    }

    workspace.state.assign(n, 0);
    workspace.predecessors.resize(n);
    workspace.heap.clear();
    if (stats != nullptr) {
        stats->reset();
    }
    for (const auto target : targets) {
        workspace.state[target] |= kCompactTarget;
    }
    workspace.state[source] |= kCompactDiscovered;
    workspace.predecessors[source] = source;
    compact_heap_push(workspace, {Distance{0}, source}, stats);

    auto reached = kNoCompactNode;
    while (!workspace.heap.empty()) {
        const auto entry = compact_heap_pop(workspace, stats);
        const auto node = entry.node;
        if ((workspace.state[node] & kCompactSettled) != 0) {
            continue;
        }
        workspace.state[node] |= kCompactSettled;
        if ((workspace.state[node] & kCompactTarget) != 0) {
            reached = node;
            break;
        }
        for_each_compact_neighbor<Adjacency>(
            domain, node,
            [&](const std::uint32_t target, const double) {
                if ((workspace.state[target] &
                     (kCompactDiscovered | kCompactSettled)) != 0) {
                    return;
                }
                const auto candidate = static_cast<Distance>(
                    entry.distance + costs[target]
                );
                workspace.state[target] |= kCompactDiscovered;
                workspace.predecessors[target] = node;
                compact_heap_push(workspace, {candidate, target}, stats);
            }
        );
    }
    if (reached == kNoCompactNode) {
        throw std::runtime_error("no compact Dijkstra target is reachable from source");
    }

    path.clear();
    auto node = reached;
    while (true) {
        path.push_back(node);
        const auto parent = workspace.predecessors[node];
        if (parent == node) {
            break;
        }
        node = parent;
        if (path.size() > n) {
            throw std::runtime_error("cycle in compact Dijkstra predecessor chain");
        }
    }
    std::reverse(path.begin(), path.end());
}

} // namespace bioimage_cpp::skeleton::detail
