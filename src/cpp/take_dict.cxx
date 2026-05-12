#include "bioimage_cpp/take_dict.hxx"

#include <cstdint>
#include <unordered_map>

namespace bioimage_cpp {

template void take_dict<std::uint32_t>(
    const std::unordered_map<std::uint32_t, std::uint32_t> &,
    const ConstArrayView<std::uint32_t> &,
    const ArrayView<std::uint32_t> &
);

template void take_dict<std::uint64_t>(
    const std::unordered_map<std::uint64_t, std::uint64_t> &,
    const ConstArrayView<std::uint64_t> &,
    const ArrayView<std::uint64_t> &
);

template void take_dict<std::int32_t>(
    const std::unordered_map<std::int32_t, std::int32_t> &,
    const ConstArrayView<std::int32_t> &,
    const ArrayView<std::int32_t> &
);

template void take_dict<std::int64_t>(
    const std::unordered_map<std::int64_t, std::int64_t> &,
    const ConstArrayView<std::int64_t> &,
    const ArrayView<std::int64_t> &
);

} // namespace bioimage_cpp
