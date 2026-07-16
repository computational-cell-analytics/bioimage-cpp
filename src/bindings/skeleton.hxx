#pragma once

#include <nanobind/nanobind.h>

namespace bioimage_cpp::bindings {

void bind_skeleton(nanobind::module_ &m);
void bind_skeleton_distributed(nanobind::module_ &m);

} // namespace bioimage_cpp::bindings
