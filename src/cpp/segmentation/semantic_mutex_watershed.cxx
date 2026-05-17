#include "bioimage_cpp/segmentation/semantic_mutex_watershed.hxx"

#include <cstdint>

namespace bioimage_cpp {

template void semantic_mutex_watershed_grid<float>(
    const ConstArrayView<float> &,
    const ConstArrayView<std::uint8_t> &,
    const std::vector<std::vector<std::ptrdiff_t>> &,
    std::size_t,
    std::size_t,
    const ArrayView<std::uint64_t> &,
    const ArrayView<std::int64_t> &
);

template void semantic_mutex_watershed_grid<double>(
    const ConstArrayView<double> &,
    const ConstArrayView<std::uint8_t> &,
    const std::vector<std::vector<std::ptrdiff_t>> &,
    std::size_t,
    std::size_t,
    const ArrayView<std::uint64_t> &,
    const ArrayView<std::int64_t> &
);

} // namespace bioimage_cpp
