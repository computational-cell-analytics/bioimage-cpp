#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/skeleton/teasar.hxx"
#include "bioimage_cpp/util/union_find.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bioimage_cpp::skeleton::distributed {

using LatticeSkeletonGraph = bioimage_cpp::skeleton::LatticeSkeletonGraph;

struct LatticeSkeletonView {
    ConstArrayView<std::int64_t> vertices;
    ConstArrayView<std::uint64_t> edges;
    ConstArrayView<float> radii;
};

inline void validate_lattice_skeleton(
    const LatticeSkeletonGraph &graph,
    const std::size_t fragment_index = 0
) {
    if (graph.radii.size() != graph.vertices.size()) {
        throw std::invalid_argument(
            "fragment " + std::to_string(fragment_index) +
            " radii length must match vertices"
        );
    }
    for (std::size_t vertex = 0; vertex < graph.radii.size(); ++vertex) {
        if (!(std::isfinite(graph.radii[vertex]) && graph.radii[vertex] >= 0.0f)) {
            throw std::invalid_argument(
                "fragment " + std::to_string(fragment_index) +
                " radii must be finite and non-negative"
            );
        }
    }
    const auto number_of_vertices = static_cast<std::uint64_t>(
        graph.vertices.size()
    );
    for (const auto &edge : graph.edges) {
        if (edge[0] >= number_of_vertices || edge[1] >= number_of_vertices) {
            throw std::invalid_argument(
                "fragment " + std::to_string(fragment_index) +
                " contains an edge endpoint outside its vertex range"
            );
        }
    }
}

inline void validate_lattice_skeleton(
    const LatticeSkeletonView &graph,
    const std::size_t fragment_index = 0
) {
    if (graph.vertices.ndim() != 2 || graph.vertices.shape[1] != 3) {
        throw std::invalid_argument(
            "fragment " + std::to_string(fragment_index) +
            " vertices must have shape (n, 3)"
        );
    }
    if (graph.edges.ndim() != 2 || graph.edges.shape[1] != 2) {
        throw std::invalid_argument(
            "fragment " + std::to_string(fragment_index) +
            " edges must have shape (n, 2)"
        );
    }
    if (graph.radii.ndim() != 1 || graph.radii.shape[0] != graph.vertices.shape[0]) {
        throw std::invalid_argument(
            "fragment " + std::to_string(fragment_index) +
            " radii length must match vertices"
        );
    }
    const auto number_of_vertices = static_cast<std::uint64_t>(
        graph.vertices.shape[0]
    );
    for (std::ptrdiff_t vertex = 0; vertex < graph.radii.shape[0]; ++vertex) {
        const auto radius = graph.radii.data[vertex];
        if (!(std::isfinite(radius) && radius >= 0.0f)) {
            throw std::invalid_argument(
                "fragment " + std::to_string(fragment_index) +
                " radii must be finite and non-negative"
            );
        }
    }
    for (std::ptrdiff_t edge = 0; edge < graph.edges.shape[0]; ++edge) {
        if (
            graph.edges.data[edge * 2] >= number_of_vertices ||
            graph.edges.data[edge * 2 + 1] >= number_of_vertices
        ) {
            throw std::invalid_argument(
                "fragment " + std::to_string(fragment_index) +
                " contains an edge endpoint outside its vertex range"
            );
        }
    }
}

namespace detail_merge {

inline std::size_t number_of_vertices(const LatticeSkeletonGraph &graph) {
    return graph.vertices.size();
}

inline std::size_t number_of_vertices(const LatticeSkeletonView &graph) {
    return static_cast<std::size_t>(graph.vertices.shape[0]);
}

inline std::size_t number_of_edges(const LatticeSkeletonGraph &graph) {
    return graph.edges.size();
}

inline std::size_t number_of_edges(const LatticeSkeletonView &graph) {
    return static_cast<std::size_t>(graph.edges.shape[0]);
}

inline VoxelCoordinate vertex(
    const LatticeSkeletonGraph &graph,
    const std::size_t index
) {
    return graph.vertices[index];
}

inline VoxelCoordinate vertex(
    const LatticeSkeletonView &graph,
    const std::size_t index
) {
    return {
        graph.vertices.data[index * 3],
        graph.vertices.data[index * 3 + 1],
        graph.vertices.data[index * 3 + 2],
    };
}

inline float radius(const LatticeSkeletonGraph &graph, const std::size_t index) {
    return graph.radii[index];
}

inline float radius(const LatticeSkeletonView &graph, const std::size_t index) {
    return graph.radii.data[index];
}

inline std::array<std::uint64_t, 2> edge(
    const LatticeSkeletonGraph &graph,
    const std::size_t index
) {
    return graph.edges[index];
}

inline std::array<std::uint64_t, 2> edge(
    const LatticeSkeletonView &graph,
    const std::size_t index
) {
    return {graph.edges.data[index * 2], graph.edges.data[index * 2 + 1]};
}

template <class Fragment>
LatticeSkeletonGraph merge_block_skeletons_impl(
    const std::vector<Fragment> &fragments
) {
    struct VertexRecord {
        VoxelCoordinate coordinate{};
        std::size_t fragment = 0;
        std::size_t old_vertex = 0;
        float radius = 0.0f;
    };

    std::size_t total_vertices = 0;
    std::size_t total_edges = 0;
    for (std::size_t fragment = 0; fragment < fragments.size(); ++fragment) {
        validate_lattice_skeleton(fragments[fragment], fragment);
        if (
            number_of_vertices(fragments[fragment]) >
            std::numeric_limits<std::size_t>::max() - total_vertices
        ) {
            throw std::overflow_error("merged vertex count overflows size_t");
        }
        total_vertices += number_of_vertices(fragments[fragment]);
        if (
            number_of_edges(fragments[fragment]) >
            std::numeric_limits<std::size_t>::max() - total_edges
        ) {
            throw std::overflow_error("merged edge count overflows size_t");
        }
        total_edges += number_of_edges(fragments[fragment]);
    }

    std::vector<VertexRecord> records;
    records.reserve(total_vertices);
    for (std::size_t fragment = 0; fragment < fragments.size(); ++fragment) {
        const auto &part = fragments[fragment];
        for (std::size_t vertex_id = 0;
             vertex_id < number_of_vertices(part); ++vertex_id) {
            records.push_back({
                vertex(part, vertex_id), fragment, vertex_id,
                radius(part, vertex_id)
            });
        }
    }
    std::sort(
        records.begin(), records.end(),
        [](const VertexRecord &first, const VertexRecord &second) {
            if (first.coordinate != second.coordinate) {
                return first.coordinate < second.coordinate;
            }
            if (first.fragment != second.fragment) {
                return first.fragment < second.fragment;
            }
            return first.old_vertex < second.old_vertex;
        }
    );

    std::vector<std::vector<std::uint64_t>> old_to_new(fragments.size());
    for (std::size_t fragment = 0; fragment < fragments.size(); ++fragment) {
        old_to_new[fragment].resize(number_of_vertices(fragments[fragment]));
    }
    LatticeSkeletonGraph output;
    output.vertices.reserve(records.size());
    output.radii.reserve(records.size());
    std::size_t begin = 0;
    while (begin < records.size()) {
        auto end = begin + 1;
        float radius = records[begin].radius;
        while (
            end < records.size() &&
            records[end].coordinate == records[begin].coordinate
        ) {
            radius = std::max(radius, records[end].radius);
            ++end;
        }
        if (output.vertices.size() >= std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("merged vertex id exceeds uint64 range");
        }
        const auto new_vertex = static_cast<std::uint64_t>(
            output.vertices.size()
        );
        output.vertices.push_back(records[begin].coordinate);
        output.radii.push_back(radius);
        for (auto record = begin; record < end; ++record) {
            old_to_new[records[record].fragment][records[record].old_vertex] =
                new_vertex;
        }
        begin = end;
    }

    output.edges.reserve(total_edges);
    for (std::size_t fragment = 0; fragment < fragments.size(); ++fragment) {
        for (std::size_t edge_id = 0;
             edge_id < number_of_edges(fragments[fragment]); ++edge_id) {
            const auto old_edge = edge(fragments[fragment], edge_id);
            auto first = old_to_new[fragment][
                static_cast<std::size_t>(old_edge[0])
            ];
            auto second = old_to_new[fragment][
                static_cast<std::size_t>(old_edge[1])
            ];
            if (first == second) {
                continue;
            }
            if (second < first) {
                std::swap(first, second);
            }
            output.edges.push_back({first, second});
        }
    }
    std::sort(output.edges.begin(), output.edges.end());
    output.edges.erase(
        std::unique(output.edges.begin(), output.edges.end()),
        output.edges.end()
    );
    return output;
}

} // namespace detail_merge

inline LatticeSkeletonGraph merge_block_skeletons(
    const std::vector<LatticeSkeletonGraph> &fragments
) {
    return detail_merge::merge_block_skeletons_impl(fragments);
}

inline LatticeSkeletonGraph merge_block_skeletons(
    const std::vector<LatticeSkeletonView> &fragments
) {
    return detail_merge::merge_block_skeletons_impl(fragments);
}

inline LatticeSkeletonGraph minimum_spanning_forest(
    const LatticeSkeletonGraph &graph,
    const std::array<double, 3> &spacing
) {
    validate_lattice_skeleton(graph);
    for (const auto value : spacing) {
        if (!(std::isfinite(value) && value > 0.0)) {
            throw std::invalid_argument("spacing values must be positive and finite");
        }
    }
    struct WeightedEdge {
        long double weight = 0.0L;
        VoxelCoordinate first_coordinate{};
        VoxelCoordinate second_coordinate{};
        std::uint64_t first = 0;
        std::uint64_t second = 0;
    };
    std::vector<WeightedEdge> weighted;
    weighted.reserve(graph.edges.size());
    for (const auto &edge : graph.edges) {
        if (edge[0] == edge[1]) {
            continue;
        }
        auto first = edge[0];
        auto second = edge[1];
        auto first_coordinate = graph.vertices[static_cast<std::size_t>(first)];
        auto second_coordinate = graph.vertices[static_cast<std::size_t>(second)];
        if (
            second_coordinate < first_coordinate ||
            (second_coordinate == first_coordinate && second < first)
        ) {
            std::swap(first, second);
            std::swap(first_coordinate, second_coordinate);
        }
        long double weight = 0.0L;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto delta =
                (static_cast<long double>(first_coordinate[axis]) -
                 static_cast<long double>(second_coordinate[axis])) *
                static_cast<long double>(spacing[axis]);
            weight += delta * delta;
        }
        if (!std::isfinite(weight)) {
            throw std::overflow_error("physical edge length overflowed");
        }
        weighted.push_back({
            weight, first_coordinate, second_coordinate, first, second
        });
    }
    std::sort(
        weighted.begin(), weighted.end(),
        [](const WeightedEdge &first, const WeightedEdge &second) {
            if (first.weight != second.weight) {
                return first.weight < second.weight;
            }
            if (first.first_coordinate != second.first_coordinate) {
                return first.first_coordinate < second.first_coordinate;
            }
            if (first.second_coordinate != second.second_coordinate) {
                return first.second_coordinate < second.second_coordinate;
            }
            if (first.first != second.first) {
                return first.first < second.first;
            }
            return first.second < second.second;
        }
    );

    util::UnionFind union_find(graph.vertices.size());
    LatticeSkeletonGraph output;
    output.vertices = graph.vertices;
    output.radii = graph.radii;
    output.edges.reserve(std::min(graph.edges.size(), graph.vertices.size()));
    for (const auto &edge : weighted) {
        const auto first_root = union_find.find(edge.first);
        const auto second_root = union_find.find(edge.second);
        if (first_root == second_root) {
            continue;
        }
        union_find.unite_roots(first_root, second_root);
        auto first = edge.first;
        auto second = edge.second;
        if (second < first) {
            std::swap(first, second);
        }
        output.edges.push_back({first, second});
    }
    std::sort(output.edges.begin(), output.edges.end());
    return output;
}

inline SkeletonGraph lattice_to_physical(
    const LatticeSkeletonGraph &graph,
    const std::array<double, 3> &spacing
) {
    validate_lattice_skeleton(graph);
    for (const auto value : spacing) {
        if (!(std::isfinite(value) && value > 0.0)) {
            throw std::invalid_argument("spacing values must be positive and finite");
        }
    }
    SkeletonGraph output;
    output.vertices.reserve(graph.vertices.size());
    output.radii = graph.radii;
    output.edges = graph.edges;
    for (const auto &coordinate : graph.vertices) {
        std::array<double, 3> physical{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            physical[axis] = static_cast<double>(coordinate[axis]) * spacing[axis];
            if (!std::isfinite(physical[axis])) {
                throw std::overflow_error("physical skeleton coordinate overflowed");
            }
        }
        output.vertices.push_back(physical);
    }
    return output;
}

} // namespace bioimage_cpp::skeleton::distributed
