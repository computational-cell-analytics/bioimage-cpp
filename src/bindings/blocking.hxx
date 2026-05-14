#pragma once

#include <nanobind/nanobind.h>

namespace bioimage_cpp::bindings {

void bind_blocking(nanobind::module_ &m);

} // namespace bioimage_cpp::bindings
