#pragma once

#include <algorithm>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

namespace bioimage_cpp::detail {

inline std::size_t normalize_thread_count(
    const std::size_t requested,
    const std::size_t number_of_work_items
) {
    if (number_of_work_items == 0) {
        return 1;
    }
    std::size_t n_threads = requested;
    if (n_threads == 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) {
            n_threads = 1;
        }
    }
    return std::max<std::size_t>(1, std::min(n_threads, number_of_work_items));
}

// Split [0, number_of_work_items) into n_threads contiguous chunks and invoke
// `chunk(thread_id, begin, end)` on each chunk. Thread 0 runs on the calling
// thread; threads 1..n_threads-1 run in parallel std::threads. The caller is
// responsible for picking n_threads via normalize_thread_count and for the
// thread safety of `chunk`.
template <class Chunk>
void parallel_for_chunks(
    const std::size_t n_threads,
    const std::size_t number_of_work_items,
    Chunk &&chunk
) {
    const auto bounds = [&](const std::size_t thread_id) {
        const auto begin = thread_id * number_of_work_items / n_threads;
        const auto end = (thread_id + 1) * number_of_work_items / n_threads;
        return std::pair<std::size_t, std::size_t>{begin, end};
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads > 0 ? n_threads - 1 : 0);
    for (std::size_t thread_id = 1; thread_id < n_threads; ++thread_id) {
        const auto [begin, end] = bounds(thread_id);
        threads.emplace_back([thread_id, begin, end, &chunk]() {
            chunk(thread_id, begin, end);
        });
    }
    const auto [begin, end] = bounds(0);
    chunk(0, begin, end);
    for (auto &thread : threads) {
        thread.join();
    }
}

} // namespace bioimage_cpp::detail
