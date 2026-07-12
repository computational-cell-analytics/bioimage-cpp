#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <mutex>
#include <stdexcept>
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
// thread; threads 1..n_threads-1 run in parallel std::jthreads. The caller is
// responsible for picking n_threads via normalize_thread_count and for the
// thread safety of `chunk`.
//
// Exceptions thrown by `chunk` are captured rather than allowed to escape a
// worker thread (which would call std::terminate). The first exception on any
// thread is stored and rethrown on the calling thread after every worker has
// been joined, so nanobind can translate it into a Python exception.
template <class Chunk>
void parallel_for_chunks(
    const std::size_t n_threads,
    const std::size_t number_of_work_items,
    Chunk &&chunk
) {
    if (n_threads == 0) {
        throw std::invalid_argument("parallel_for_chunks requires n_threads >= 1");
    }
    const auto bounds = [&](const std::size_t thread_id) {
        const auto begin = thread_id * number_of_work_items / n_threads;
        const auto end = (thread_id + 1) * number_of_work_items / n_threads;
        return std::pair<std::size_t, std::size_t>{begin, end};
    };

    std::exception_ptr first_exception;
    std::mutex exception_mutex;
    const auto guarded_chunk = [&] (
        const std::size_t thread_id,
        const std::size_t begin,
        const std::size_t end
    ) {
        try {
            chunk(thread_id, begin, end);
        } catch (...) {
            const std::lock_guard<std::mutex> lock(exception_mutex);
            if (!first_exception) {
                first_exception = std::current_exception();
            }
        }
    };

    // jthread's destructor joins, so a failure while creating a later worker
    // cannot destroy an earlier still-joinable thread and terminate the
    // process. We still join explicitly below to rethrow only after all work
    // has completed.
    std::vector<std::jthread> threads;
    threads.reserve(n_threads > 0 ? n_threads - 1 : 0);
    for (std::size_t thread_id = 1; thread_id < n_threads; ++thread_id) {
        const auto [begin, end] = bounds(thread_id);
        threads.emplace_back([thread_id, begin, end, &guarded_chunk]() {
            guarded_chunk(thread_id, begin, end);
        });
    }
    const auto [begin, end] = bounds(0);
    guarded_chunk(0, begin, end);
    for (auto &thread : threads) {
        thread.join();
    }
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
}

} // namespace bioimage_cpp::detail
