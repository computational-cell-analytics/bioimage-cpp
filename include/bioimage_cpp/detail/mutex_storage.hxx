#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace bioimage_cpp {

using MutexStorage = std::vector<std::unordered_set<std::uint64_t>>;

inline bool check_mutex(
    const std::uint64_t first,
    const std::uint64_t second,
    const MutexStorage &mutexes
) {
    const auto &first_mutexes = mutexes[first];
    const auto &second_mutexes = mutexes[second];
    if (first_mutexes.size() < second_mutexes.size()) {
        return first_mutexes.find(second) != first_mutexes.end();
    }
    return second_mutexes.find(first) != second_mutexes.end();
}

inline void insert_mutex(
    const std::uint64_t first,
    const std::uint64_t second,
    MutexStorage &mutexes
) {
    mutexes[first].insert(second);
    mutexes[second].insert(first);
}

inline void merge_mutexes(
    const std::uint64_t root_from,
    const std::uint64_t root_to,
    MutexStorage &mutexes
) {
    auto &mutexes_from = mutexes[root_from];
    auto &mutexes_to = mutexes[root_to];

    for (const auto other_root : mutexes_from) {
        auto &other_mutexes = mutexes[other_root];
        other_mutexes.erase(root_from);
        if (other_root != root_to) {
            other_mutexes.insert(root_to);
            mutexes_to.insert(other_root);
        }
    }
    mutexes_to.erase(root_from);
    mutexes_to.erase(root_to);
    mutexes_from.clear();
}

} // namespace bioimage_cpp
