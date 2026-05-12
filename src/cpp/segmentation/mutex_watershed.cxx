#include "bioimage_cpp/segmentation/mutex_watershed.hxx"

#include <cstdint>

namespace bioimage_cpp {

template void mutex_watershed_grid<float>(
    const ConstArrayView<float> &,
    const std::vector<std::vector<std::ptrdiff_t>> &,
    std::size_t,
    const ArrayView<std::uint64_t> &
);

template void mutex_watershed_grid<double>(
    const ConstArrayView<double> &,
    const std::vector<std::vector<std::ptrdiff_t>> &,
    std::size_t,
    const ArrayView<std::uint64_t> &
);

} // namespace bioimage_cpp
