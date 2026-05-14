#include "ground_truth.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/overlap.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

using LabelArray = nb::ndarray<nb::numpy, const std::uint64_t, nb::c_contig>;
using SegmentationOverlap = ground_truth::SegmentationOverlap;

std::vector<std::ptrdiff_t> ndarray_shape(LabelArray array) {
    std::vector<std::ptrdiff_t> shape(array.ndim());
    for (std::size_t axis = 0; axis < array.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(array.shape(axis));
    }
    return shape;
}

SegmentationOverlap segmentation_overlap(LabelArray labels_a, LabelArray labels_b) {
    ConstArrayView<std::uint64_t> labels_a_view{
        labels_a.data(),
        ndarray_shape(labels_a),
        {},
    };
    ConstArrayView<std::uint64_t> labels_b_view{
        labels_b.data(),
        ndarray_shape(labels_b),
        {},
    };

    nb::gil_scoped_release release;
    return ground_truth::segmentation_overlap(labels_a_view, labels_b_view);
}

} // namespace

void bind_ground_truth(nb::module_ &m) {
    nb::class_<ground_truth::OverlapPair>(m, "_OverlapPair")
        .def_ro("label_a", &ground_truth::OverlapPair::label_a)
        .def_ro("label_b", &ground_truth::OverlapPair::label_b)
        .def_ro("count", &ground_truth::OverlapPair::count);

    nb::class_<SegmentationOverlap>(m, "_SegmentationOverlap")
        .def_prop_ro("total_count", &SegmentationOverlap::total_count)
        .def("labels_a", &SegmentationOverlap::labels_a)
        .def("labels_b", &SegmentationOverlap::labels_b)
        .def("counts_a", &SegmentationOverlap::counts_a)
        .def("counts_b", &SegmentationOverlap::counts_b)
        .def("count_a", &SegmentationOverlap::count_a, nb::arg("label"))
        .def("count_b", &SegmentationOverlap::count_b, nb::arg("label"))
        .def(
            "overlap_count",
            &SegmentationOverlap::overlap_count,
            nb::arg("label_a"),
            nb::arg("label_b")
        )
        .def("overlap_pairs", &SegmentationOverlap::overlap_pairs)
        .def(
            "overlaps_for_label_a",
            &SegmentationOverlap::overlaps_for_label_a,
            nb::arg("label")
        )
        .def(
            "overlaps_for_label_b",
            &SegmentationOverlap::overlaps_for_label_b,
            nb::arg("label")
        )
        .def(
            "best_overlap_for_label_a",
            &SegmentationOverlap::best_overlap_for_label_a,
            nb::arg("label"),
            nb::arg("ignore_zero") = false
        )
        .def(
            "best_overlap_for_label_b",
            &SegmentationOverlap::best_overlap_for_label_b,
            nb::arg("label"),
            nb::arg("ignore_zero") = false
        )
        .def(
            "is_label_a_overlapping_with_zero",
            &SegmentationOverlap::is_label_a_overlapping_with_zero,
            nb::arg("label")
        )
        .def(
            "is_label_b_overlapping_with_zero",
            &SegmentationOverlap::is_label_b_overlapping_with_zero,
            nb::arg("label")
        )
        .def(
            "different_overlap",
            &SegmentationOverlap::different_overlap,
            nb::arg("label_a_u"),
            nb::arg("label_a_v")
        );

    m.def(
        "_segmentation_overlap_uint64",
        &segmentation_overlap,
        nb::arg("labels_a"),
        nb::arg("labels_b"),
        "Compute sparse overlap counts between two uint64 label arrays."
    );
}

} // namespace bioimage_cpp::bindings
