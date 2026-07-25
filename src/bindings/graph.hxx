#pragma once

#include <nanobind/nanobind.h>

namespace bioimage_cpp::bindings {

void bind_graph(nanobind::module_ &m);
void bind_graph_contraction(nanobind::module_ &m);

} // namespace bioimage_cpp::bindings
