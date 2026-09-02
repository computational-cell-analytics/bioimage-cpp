#pragma once

#include "bioimage_cpp/detail/threading.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace bioimage_cpp::detail {

// True iff every float in [begin, end) is finite. The inner loop is
// branch-free (an OR-reduction over the exponent-all-ones test on the bit
// pattern) so the compiler vectorizes it on the SSE2 baseline; the block
// granularity gives an early exit once a non-finite value has been seen.
inline bool all_finite_range(
    const float *data, const std::size_t begin, const std::size_t end
) noexcept {
    constexpr std::size_t block = 4096;
    for (std::size_t start = begin; start < end; start += block) {
        const std::size_t stop = std::min(end, start + block);
        std::uint32_t any_special = 0;
        for (std::size_t i = start; i < stop; ++i) {
            std::uint32_t bits;
            std::memcpy(&bits, data + i, sizeof(bits));
            any_special |= static_cast<std::uint32_t>((bits & 0x7f800000u) == 0x7f800000u);
        }
        if (any_special != 0) {
            return false;
        }
    }
    return true;
}

// Parallel finiteness check over `n` floats. Worker threads are only used
// when every worker gets at least ~1M elements, so small arrays are checked
// inline without spawning threads.
inline bool all_finite(
    const float *data, const std::size_t n, const std::size_t number_of_threads
) {
    constexpr std::size_t min_elements_per_thread = std::size_t(1) << 20;
    const std::size_t max_useful = std::max<std::size_t>(1, n / min_elements_per_thread);
    const std::size_t n_threads =
        normalize_thread_count(std::min(number_of_threads, max_useful), n);
    if (n_threads <= 1) {
        return all_finite_range(data, 0, n);
    }
    std::vector<std::uint8_t> chunk_ok(n_threads, 1);
    parallel_for_chunks(
        n_threads, n,
        [&](const std::size_t thread_id, const std::size_t begin, const std::size_t end) {
            chunk_ok[thread_id] = all_finite_range(data, begin, end) ? 1 : 0;
        }
    );
    return std::all_of(chunk_ok.begin(), chunk_ok.end(), [](std::uint8_t ok) { return ok != 0; });
}

} // namespace bioimage_cpp::detail
