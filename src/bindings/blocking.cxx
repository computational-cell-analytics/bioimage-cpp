#include "blocking.hxx"

#include "bioimage_cpp/blocking.hxx"

#include <nanobind/make_iterator.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <tuple>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

BlockWithHalo get_block_with_symmetric_halo(
    const Blocking &blocking,
    const std::uint64_t block_id,
    const CoordinateVector &halo
) {
    return blocking.get_block_with_halo(block_id, halo);
}

BlockWithHalo get_block_with_asymmetric_halo(
    const Blocking &blocking,
    const std::uint64_t block_id,
    const CoordinateVector &halo_begin,
    const CoordinateVector &halo_end
) {
    return blocking.get_block_with_halo(block_id, halo_begin, halo_end);
}

BlockWithHalo add_symmetric_halo(
    const Blocking &blocking,
    const Block &inner_block,
    const CoordinateVector &halo
) {
    return blocking.add_halo(inner_block, halo);
}

BlockWithHalo add_asymmetric_halo(
    const Blocking &blocking,
    const Block &inner_block,
    const CoordinateVector &halo_begin,
    const CoordinateVector &halo_end
) {
    return blocking.add_halo(inner_block, halo_begin, halo_end);
}

nb::object get_local_overlaps(
    const Blocking &blocking,
    const std::uint64_t block_a_id,
    const std::uint64_t block_b_id,
    const CoordinateVector &block_halo
) {
    const auto overlaps = blocking.get_local_overlaps(block_a_id, block_b_id, block_halo);
    if (!overlaps.has_value()) {
        return nb::none();
    }
    return nb::cast(std::make_tuple(
        overlaps->overlap_begin_a,
        overlaps->overlap_end_a,
        overlaps->overlap_begin_b,
        overlaps->overlap_end_b
    ));
}

} // namespace

void bind_blocking(nb::module_ &m) {
    nb::class_<Block>(m, "Block")
        .def(
            nb::init<CoordinateVector, CoordinateVector>(),
            nb::arg("begin"),
            nb::arg("end")
        )
        .def_prop_ro("begin", &Block::begin)
        .def_prop_ro("end", &Block::end)
        .def_prop_ro("shape", &Block::shape)
        .def_prop_ro("ndim", &Block::ndim);

    nb::class_<BlockWithHalo>(m, "BlockWithHalo")
        .def(
            nb::init<Block, Block>(),
            nb::arg("outer_block"),
            nb::arg("inner_block")
        )
        .def_prop_ro("outer_block", &BlockWithHalo::outer_block)
        .def_prop_ro("inner_block", &BlockWithHalo::inner_block)
        .def_prop_ro("inner_block_local", &BlockWithHalo::inner_block_local);

    nb::class_<Blocking>(m, "Blocking")
        .def(
            nb::init<CoordinateVector, CoordinateVector, CoordinateVector, CoordinateVector>(),
            nb::arg("roi_begin"),
            nb::arg("roi_end"),
            nb::arg("block_shape"),
            nb::arg("block_shift") = CoordinateVector{}
        )
        .def_prop_ro("roi_begin", &Blocking::roi_begin)
        .def_prop_ro("roi_end", &Blocking::roi_end)
        .def_prop_ro("block_shape", &Blocking::block_shape)
        .def_prop_ro("block_shift", &Blocking::block_shift)
        .def_prop_ro("blocks_per_axis", &Blocking::blocks_per_axis)
        .def_prop_ro("number_of_blocks", &Blocking::number_of_blocks)
        .def_prop_ro("ndim", &Blocking::ndim)
        .def("block_grid_position", &Blocking::block_grid_position, nb::arg("block_id"))
        .def(
            "get_neighbor_id",
            &Blocking::get_neighbor_id,
            nb::arg("block_id"),
            nb::arg("axis"),
            nb::arg("lower")
        )
        .def("get_block", &Blocking::get_block, nb::arg("block_id"))
        .def(
            "get_block_with_halo",
            &get_block_with_symmetric_halo,
            nb::arg("block_id"),
            nb::arg("halo")
        )
        .def(
            "get_block_with_halo",
            &get_block_with_asymmetric_halo,
            nb::arg("block_id"),
            nb::arg("halo_begin"),
            nb::arg("halo_end")
        )
        .def(
            "add_halo",
            &add_symmetric_halo,
            nb::arg("inner_block"),
            nb::arg("halo")
        )
        .def(
            "add_halo",
            &add_asymmetric_halo,
            nb::arg("inner_block"),
            nb::arg("halo_begin"),
            nb::arg("halo_end")
        )
        .def(
            "coordinates_to_block_id",
            &Blocking::coordinates_to_block_id,
            nb::arg("coordinates")
        )
        .def(
            "get_block_ids_in_bounding_box",
            &Blocking::get_block_ids_in_bounding_box,
            nb::arg("box_begin"),
            nb::arg("box_end")
        )
        .def(
            "get_block_ids_overlapping_bounding_box",
            &Blocking::get_block_ids_overlapping_bounding_box,
            nb::arg("box_begin"),
            nb::arg("box_end")
        )
        .def(
            "get_local_overlaps",
            &get_local_overlaps,
            nb::arg("block_a_id"),
            nb::arg("block_b_id"),
            nb::arg("block_halo")
        )
        .def(
            "get_block_ids_in_slice",
            &Blocking::get_block_ids_in_slice,
            nb::arg("z"),
            nb::arg("block_halo")
        );
}

} // namespace bioimage_cpp::bindings
