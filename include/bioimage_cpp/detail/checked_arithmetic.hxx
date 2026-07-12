#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace bioimage_cpp::detail {

inline std::size_t checked_size_cast(const std::uint64_t value, const char *name) {
    if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
        if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error(std::string(name) + " exceeds size_t range");
        }
    }
    return static_cast<std::size_t>(value);
}

inline std::size_t checked_size_add(
    const std::size_t a,
    const std::size_t b,
    const char *name
) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error(std::string(name) + " exceeds size_t range");
    }
    return a + b;
}

inline std::size_t checked_size_multiply(
    const std::size_t a,
    const std::size_t b,
    const char *name
) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error(std::string(name) + " exceeds size_t range");
    }
    return a * b;
}

inline std::uint64_t checked_u64_add(
    const std::uint64_t a,
    const std::uint64_t b,
    const char *name
) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error(std::string(name) + " exceeds uint64 range");
    }
    return a + b;
}

} // namespace bioimage_cpp::detail
