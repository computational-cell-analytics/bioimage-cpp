#include "affinities.hxx"
#include "blocking.hxx"
#include "distance.hxx"
#include "filters.hxx"
#include "flow.hxx"
#include "graph.hxx"
#include "ground_truth.hxx"
#include "label_multiset.hxx"
#include "mesh.hxx"
#include "segmentation.hxx"
#include "skeleton.hxx"
#include "transformation.hxx"
#include "util.hxx"
#include "utils.hxx"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_core, m) {
    m.doc() = "C++ extension module for bioimage_cpp.";
    bioimage_cpp::bindings::bind_affinities(m);
    bioimage_cpp::bindings::bind_blocking(m);
    bioimage_cpp::bindings::bind_distance(m);
    bioimage_cpp::bindings::bind_filters(m);
    bioimage_cpp::bindings::bind_flow(m);
    bioimage_cpp::bindings::bind_graph(m);
    bioimage_cpp::bindings::bind_ground_truth(m);
    bioimage_cpp::bindings::bind_label_multiset(m);
    bioimage_cpp::bindings::bind_mesh(m);
    bioimage_cpp::bindings::bind_segmentation(m);
    bioimage_cpp::bindings::bind_skeleton(m);
    bioimage_cpp::bindings::bind_transformation(m);
    bioimage_cpp::bindings::bind_util(m);
    bioimage_cpp::bindings::bind_utils(m);
}
