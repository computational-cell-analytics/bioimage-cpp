#include "skeleton.hxx"
#include "ndarray.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/skeleton/distributed/border_targets.hxx"
#include "bioimage_cpp/skeleton/distributed/merge.hxx"
#include "bioimage_cpp/skeleton/teasar.hxx"
#include "bioimage_cpp/skeleton/teasar_labels.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

using UInt8Input = nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig>;
using ConstInt64Array = nb::ndarray<nb::numpy, const std::int64_t, nb::c_contig>;
using ConstUInt64Array = nb::ndarray<nb::numpy, const std::uint64_t, nb::c_contig>;
using ConstFloatArray = nb::ndarray<nb::numpy, const float, nb::c_contig>;

std::vector<std::ptrdiff_t> array_shape(const auto &array) {
    std::vector<std::ptrdiff_t> shape(array.ndim());
    for (std::size_t axis = 0; axis < array.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(array.shape(axis));
    }
    return shape;
}

std::array<std::int64_t, 3> origin_array(
    const std::vector<std::int64_t> &origin
) {
    if (origin.size() != 3) {
        throw std::invalid_argument("origin must contain exactly three values");
    }
    return {origin[0], origin[1], origin[2]};
}

std::array<double, 3> spacing_array(const std::vector<double> &spacing) {
    if (spacing.size() != 3) {
        throw std::invalid_argument("spacing must contain exactly three values");
    }
    return {spacing[0], spacing[1], spacing[2]};
}

std::vector<skeleton::distributed::BlockFace> face_vector(
    const std::vector<std::size_t> &axes,
    const std::vector<std::uint8_t> &high
) {
    if (axes.size() != high.size()) {
        throw std::invalid_argument("face axes and sides must have equal length");
    }
    std::vector<skeleton::distributed::BlockFace> faces;
    faces.reserve(axes.size());
    for (std::size_t index = 0; index < axes.size(); ++index) {
        if (high[index] > 1) {
            throw std::invalid_argument("face side flags must be zero or one");
        }
        faces.push_back({axes[index], high[index] != 0});
    }
    return faces;
}

skeleton::detail::OpenBlockFaces open_face_policy(
    const std::vector<std::size_t> &axes,
    const std::vector<std::uint8_t> &high
) {
    skeleton::detail::OpenBlockFaces output;
    for (const auto &face : face_vector(axes, high)) {
        output.values[2 * face.axis + static_cast<std::size_t>(face.high)] = true;
    }
    return output;
}

auto coordinate_array(const std::vector<skeleton::VoxelCoordinate> &coordinates) {
    auto output = detail::make_array_for_overwrite<std::int64_t>(
        {coordinates.size(), 3}
    );
    for (std::size_t row = 0; row < coordinates.size(); ++row) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            output.data()[row * 3 + axis] = coordinates[row][axis];
        }
    }
    return output;
}

auto edge_array(
    const std::vector<std::array<std::uint64_t, 2>> &edges
) {
    auto output = detail::make_array_for_overwrite<std::uint64_t>(
        {edges.size(), 2}
    );
    for (std::size_t row = 0; row < edges.size(); ++row) {
        output.data()[row * 2] = edges[row][0];
        output.data()[row * 2 + 1] = edges[row][1];
    }
    return output;
}

nb::tuple lattice_graph_to_tuple(
    const skeleton::distributed::LatticeSkeletonGraph &graph
) {
    return nb::make_tuple(
        coordinate_array(graph.vertices),
        edge_array(graph.edges),
        detail::copy_vector_to_array(graph.radii)
    );
}

nb::tuple physical_graph_to_tuple(const skeleton::SkeletonGraph &graph) {
    auto vertices = detail::make_array_for_overwrite<double>(
        {graph.vertices.size(), 3}
    );
    for (std::size_t row = 0; row < graph.vertices.size(); ++row) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            vertices.data()[row * 3 + axis] = graph.vertices[row][axis];
        }
    }
    return nb::make_tuple(
        vertices, edge_array(graph.edges), detail::copy_vector_to_array(graph.radii)
    );
}

std::vector<skeleton::VoxelCoordinate> global_targets_to_local(
    ConstInt64Array targets,
    const std::array<std::int64_t, 3> &origin,
    const std::vector<std::ptrdiff_t> &shape,
    const std::string &argument_name
) {
    if (targets.ndim() != 2 || targets.shape(1) != 3) {
        throw std::invalid_argument(argument_name + " must have shape (n, 3)");
    }
    std::vector<skeleton::VoxelCoordinate> local(targets.shape(0));
    for (std::size_t row = 0; row < targets.shape(0); ++row) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto global = targets.data()[row * 3 + axis];
            if (global < origin[axis]) {
                throw std::invalid_argument(
                    argument_name + " row " + std::to_string(row) +
                    " lies below the block origin"
                );
            }
            const auto difference = static_cast<std::uint64_t>(global) -
                static_cast<std::uint64_t>(origin[axis]);
            if (difference >= static_cast<std::uint64_t>(shape[axis])) {
                throw std::invalid_argument(
                    argument_name + " row " + std::to_string(row) +
                    " lies outside the block"
                );
            }
            local[row][axis] = static_cast<std::int64_t>(difference);
        }
    }
    return local;
}

std::int64_t checked_globalize(
    const std::int64_t local,
    const std::int64_t origin
) {
    if (local < 0 || origin > std::numeric_limits<std::int64_t>::max() - local) {
        throw std::overflow_error("global skeleton coordinate overflows int64");
    }
    return origin + local;
}

skeleton::distributed::LatticeSkeletonGraph globalize_graph(
    const skeleton::LatticeSkeletonGraph &graph,
    const std::array<std::int64_t, 3> &origin
) {
    skeleton::distributed::LatticeSkeletonGraph output;
    output.vertices.reserve(graph.vertices.size());
    output.edges = graph.edges;
    output.radii = graph.radii;
    for (const auto &local : graph.vertices) {
        skeleton::VoxelCoordinate global{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            global[axis] = checked_globalize(local[axis], origin[axis]);
        }
        output.vertices.push_back(global);
    }
    return output;
}

skeleton::TeasarOptions teasar_options(
    const std::vector<double> &spacing,
    const double scale,
    const double constant,
    const double pdrf_scale,
    const double pdrf_exponent,
    const std::size_t number_of_threads
) {
    const auto values = spacing_array(spacing);
    return {
        values, scale, constant, pdrf_scale, pdrf_exponent,
        number_of_threads
    };
}

auto block_border_targets_uint8(
    UInt8Input mask,
    const std::vector<std::size_t> &axes,
    const std::vector<std::uint8_t> &high,
    const std::vector<std::int64_t> &origin,
    const std::vector<double> &spacing,
    const std::size_t number_of_threads
) {
    ConstArrayView<std::uint8_t> view{mask.data(), array_shape(mask), {}};
    std::vector<skeleton::VoxelCoordinate> targets;
    {
        nb::gil_scoped_release release;
        targets = skeleton::distributed::block_border_targets(
            view, face_vector(axes, high), origin_array(origin),
            spacing_array(spacing), number_of_threads
        );
    }
    return coordinate_array(targets);
}

template <class LabelT>
nb::dict block_border_targets_labels_t(
    nb::ndarray<nb::numpy, const LabelT, nb::c_contig> labels,
    const LabelT background,
    const std::vector<std::size_t> &axes,
    const std::vector<std::uint8_t> &high,
    const std::vector<std::int64_t> &origin,
    const std::vector<double> &spacing,
    const std::size_t number_of_threads
) {
    ConstArrayView<LabelT> view{labels.data(), array_shape(labels), {}};
    std::vector<skeleton::LabeledVoxelTarget<LabelT>> targets;
    {
        nb::gil_scoped_release release;
        targets = skeleton::distributed::block_border_targets_labels(
            view, background, face_vector(axes, high), origin_array(origin),
            spacing_array(spacing), number_of_threads
        );
    }
    nb::dict output;
    std::size_t begin = 0;
    while (begin < targets.size()) {
        const auto label = targets[begin].label;
        auto end = begin + 1;
        while (end < targets.size() && targets[end].label == label) {
            ++end;
        }
        std::vector<skeleton::VoxelCoordinate> coordinates;
        coordinates.reserve(end - begin);
        for (auto index = begin; index < end; ++index) {
            coordinates.push_back(targets[index].coordinate);
        }
        if constexpr (std::is_signed_v<LabelT>) {
            output[nb::int_(static_cast<long long>(label))] =
                coordinate_array(coordinates);
        } else {
            output[nb::int_(static_cast<unsigned long long>(label))] =
                coordinate_array(coordinates);
        }
        begin = end;
    }
    return output;
}

nb::tuple block_teasar_uint8(
    UInt8Input mask,
    ConstInt64Array targets,
    const std::vector<std::size_t> &open_axes,
    const std::vector<std::uint8_t> &open_high,
    const std::vector<std::int64_t> &origin,
    const std::vector<double> &spacing,
    const double scale,
    const double constant,
    const double pdrf_scale,
    const double pdrf_exponent,
    const std::size_t number_of_threads
) {
    const auto shape = array_shape(mask);
    const auto origin_values = origin_array(origin);
    auto local_targets = global_targets_to_local(
        targets, origin_values, shape, "required_targets"
    );
    ConstArrayView<std::uint8_t> view{mask.data(), shape, {}};
    skeleton::LatticeSkeletonGraph result;
    {
        nb::gil_scoped_release release;
        result = skeleton::teasar_block(
            view, std::move(local_targets),
            open_face_policy(open_axes, open_high),
            teasar_options(
                spacing, scale, constant, pdrf_scale, pdrf_exponent,
                number_of_threads
            )
        );
    }
    return lattice_graph_to_tuple(globalize_graph(result, origin_values));
}

template <class LabelT>
nb::dict block_teasar_labels_t(
    nb::ndarray<nb::numpy, const LabelT, nb::c_contig> labels,
    const LabelT background,
    nb::dict target_map,
    const std::vector<std::size_t> &open_axes,
    const std::vector<std::uint8_t> &open_high,
    const std::vector<std::int64_t> &origin,
    const std::vector<double> &spacing,
    const double scale,
    const double constant,
    const double pdrf_scale,
    const double pdrf_exponent,
    const std::size_t number_of_threads
) {
    const auto shape = array_shape(labels);
    const auto origin_values = origin_array(origin);
    std::vector<skeleton::LabeledVoxelTarget<LabelT>> local_targets;
    for (auto [key, value] : target_map) {
        const auto label = nb::cast<LabelT>(key);
        const auto array = nb::cast<ConstInt64Array>(value);
        auto coordinates = global_targets_to_local(
            array, origin_values, shape, "required_targets"
        );
        for (const auto &coordinate : coordinates) {
            local_targets.push_back({label, coordinate});
        }
    }
    ConstArrayView<LabelT> view{labels.data(), shape, {}};
    std::vector<skeleton::LabeledLatticeSkeleton<LabelT>> results;
    {
        nb::gil_scoped_release release;
        results = skeleton::teasar_labels_block(
            view, background, std::move(local_targets),
            open_face_policy(open_axes, open_high),
            teasar_options(
                spacing, scale, constant, pdrf_scale, pdrf_exponent,
                number_of_threads
            )
        );
    }
    nb::dict output;
    for (const auto &result : results) {
        auto graph = globalize_graph(result.skeleton, origin_values);
        if constexpr (std::is_signed_v<LabelT>) {
            output[nb::int_(static_cast<long long>(result.label))] =
                lattice_graph_to_tuple(graph);
        } else {
            output[nb::int_(static_cast<unsigned long long>(result.label))] =
                lattice_graph_to_tuple(graph);
        }
    }
    return output;
}

skeleton::distributed::LatticeSkeletonGraph fragment_from_handle(
    nb::handle handle,
    const std::size_t fragment_index
) {
    const auto tuple = nb::cast<nb::tuple>(handle);
    if (tuple.size() != 3) {
        throw std::invalid_argument("each fragment must be a 3-tuple");
    }
    const auto vertices = nb::cast<ConstInt64Array>(tuple[0]);
    const auto edges = nb::cast<ConstUInt64Array>(tuple[1]);
    const auto radii = nb::cast<ConstFloatArray>(tuple[2]);
    if (vertices.ndim() != 2 || vertices.shape(1) != 3) {
        throw std::invalid_argument("fragment vertices must have shape (n, 3)");
    }
    if (edges.ndim() != 2 || edges.shape(1) != 2) {
        throw std::invalid_argument("fragment edges must have shape (n, 2)");
    }
    if (radii.ndim() != 1 || radii.shape(0) != vertices.shape(0)) {
        throw std::invalid_argument("fragment radii must have shape (n_vertices,)");
    }
    skeleton::distributed::LatticeSkeletonGraph graph;
    graph.vertices.resize(vertices.shape(0));
    graph.radii.assign(radii.data(), radii.data() + radii.shape(0));
    graph.edges.resize(edges.shape(0));
    for (std::size_t row = 0; row < vertices.shape(0); ++row) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            graph.vertices[row][axis] = vertices.data()[row * 3 + axis];
        }
    }
    for (std::size_t row = 0; row < edges.shape(0); ++row) {
        graph.edges[row] = {edges.data()[row * 2], edges.data()[row * 2 + 1]};
    }
    skeleton::distributed::validate_lattice_skeleton(graph, fragment_index);
    return graph;
}

struct LatticeFragmentArrays {
    ConstInt64Array vertices;
    ConstUInt64Array edges;
    ConstFloatArray radii;
};

LatticeFragmentArrays fragment_arrays_from_handle(nb::handle handle) {
    const auto tuple = nb::cast<nb::tuple>(handle);
    if (tuple.size() != 3) {
        throw std::invalid_argument("each fragment must be a 3-tuple");
    }
    return {
        nb::cast<ConstInt64Array>(tuple[0]),
        nb::cast<ConstUInt64Array>(tuple[1]),
        nb::cast<ConstFloatArray>(tuple[2]),
    };
}

skeleton::distributed::LatticeSkeletonView fragment_view(
    const LatticeFragmentArrays &arrays
) {
    return {
        {arrays.vertices.data(), array_shape(arrays.vertices), {}},
        {arrays.edges.data(), array_shape(arrays.edges), {}},
        {arrays.radii.data(), array_shape(arrays.radii), {}},
    };
}

nb::tuple merge_block_skeletons_binding(nb::list fragments) {
    std::vector<LatticeFragmentArrays> arrays;
    arrays.reserve(fragments.size());
    for (std::size_t index = 0; index < fragments.size(); ++index) {
        arrays.push_back(fragment_arrays_from_handle(fragments[index]));
    }
    std::vector<skeleton::distributed::LatticeSkeletonView> views;
    views.reserve(arrays.size());
    for (const auto &fragment : arrays) {
        views.push_back(fragment_view(fragment));
    }
    skeleton::distributed::LatticeSkeletonGraph result;
    {
        nb::gil_scoped_release release;
        result = skeleton::distributed::merge_block_skeletons(views);
    }
    return lattice_graph_to_tuple(result);
}

nb::tuple minimum_spanning_forest_binding(
    nb::tuple fragment,
    const std::vector<double> &spacing
) {
    auto graph = fragment_from_handle(fragment, 0);
    skeleton::distributed::LatticeSkeletonGraph result;
    {
        nb::gil_scoped_release release;
        result = skeleton::distributed::minimum_spanning_forest(
            graph, spacing_array(spacing)
        );
    }
    return lattice_graph_to_tuple(result);
}

nb::tuple lattice_to_physical_binding(
    nb::tuple fragment,
    const std::vector<double> &spacing
) {
    auto graph = fragment_from_handle(fragment, 0);
    skeleton::SkeletonGraph result;
    {
        nb::gil_scoped_release release;
        result = skeleton::distributed::lattice_to_physical(
            graph, spacing_array(spacing)
        );
    }
    return physical_graph_to_tuple(result);
}

} // namespace

void bind_skeleton_distributed(nb::module_ &m) {
    m.def(
        "_block_border_targets_uint8", &block_border_targets_uint8,
        nb::arg("mask"), nb::arg("axes"), nb::arg("high"),
        nb::arg("origin"), nb::arg("spacing"), nb::arg("n_threads")
    );
    m.def(
        "_block_teasar_uint8", &block_teasar_uint8,
        nb::arg("mask"), nb::arg("required_targets"),
        nb::arg("open_axes"), nb::arg("open_high"), nb::arg("origin"),
        nb::arg("spacing"), nb::arg("scale"), nb::arg("constant"),
        nb::arg("pdrf_scale"), nb::arg("pdrf_exponent"), nb::arg("n_threads")
    );

#define BIC_BIND_BLOCK_LABELS(name, type)                                      \
    m.def(                                                                      \
        "_block_border_targets_labels_" name,                                 \
        &block_border_targets_labels_t<type>,                                   \
        nb::arg("labels"), nb::arg("background"), nb::arg("axes"),         \
        nb::arg("high"), nb::arg("origin"), nb::arg("spacing"),            \
        nb::arg("n_threads")                                                   \
    );                                                                          \
    m.def(                                                                      \
        "_block_teasar_labels_" name,                                         \
        &block_teasar_labels_t<type>,                                           \
        nb::arg("labels"), nb::arg("background"),                            \
        nb::arg("required_targets"), nb::arg("open_axes"),                   \
        nb::arg("open_high"), nb::arg("origin"),                             \
        nb::arg("spacing"), nb::arg("scale"), nb::arg("constant"),         \
        nb::arg("pdrf_scale"), nb::arg("pdrf_exponent"),                    \
        nb::arg("n_threads")                                                   \
    )

    BIC_BIND_BLOCK_LABELS("uint8", std::uint8_t);
    BIC_BIND_BLOCK_LABELS("uint16", std::uint16_t);
    BIC_BIND_BLOCK_LABELS("uint32", std::uint32_t);
    BIC_BIND_BLOCK_LABELS("uint64", std::uint64_t);
    BIC_BIND_BLOCK_LABELS("int32", std::int32_t);
    BIC_BIND_BLOCK_LABELS("int64", std::int64_t);

#undef BIC_BIND_BLOCK_LABELS

    m.def(
        "_merge_block_skeletons", &merge_block_skeletons_binding,
        nb::arg("fragments")
    );
    m.def(
        "_minimum_spanning_forest", &minimum_spanning_forest_binding,
        nb::arg("fragment"), nb::arg("spacing")
    );
    m.def(
        "_lattice_to_physical", &lattice_to_physical_binding,
        nb::arg("fragment"), nb::arg("spacing")
    );
}

} // namespace bioimage_cpp::bindings
