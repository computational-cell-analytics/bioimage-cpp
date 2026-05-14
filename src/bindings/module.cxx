#include "blocking.hxx"
#include "graph.hxx"
#include "segmentation.hxx"
#include "utils.hxx"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_core, m) {
    m.doc() = "C++ extension module for bioimage_cpp.";
    bioimage_cpp::bindings::bind_blocking(m);
    bioimage_cpp::bindings::bind_graph(m);
    bioimage_cpp::bindings::bind_segmentation(m);
    bioimage_cpp::bindings::bind_utils(m);
}
