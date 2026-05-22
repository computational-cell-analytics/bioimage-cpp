#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

namespace bioimage_cpp::label_multiset {

template <class V>
inline void reorder_inplace(V &v, const std::vector<std::size_t> &idx) {
    V tmp(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        tmp[i] = v[idx[i]];
    }
    v.swap(tmp);
}

template <class V1, class V2>
inline void argsort_by_first(V1 &v1, V2 &v2, const bool ascending = true) {
    std::vector<std::size_t> idx(v1.size());
    std::iota(idx.begin(), idx.end(), 0);
    if (ascending) {
        std::sort(idx.begin(), idx.end(),
                  [&v1](std::size_t a, std::size_t b) { return v1[a] < v1[b]; });
    } else {
        std::sort(idx.begin(), idx.end(),
                  [&v1](std::size_t a, std::size_t b) { return v1[a] > v1[b]; });
    }
    reorder_inplace(v1, idx);
    reorder_inplace(v2, idx);
}

template <class It1, class It2>
inline bool ranges_equal(It1 a_begin, It1 a_end, It2 b_begin, It2 b_end) {
    if (std::distance(a_begin, a_end) != std::distance(b_begin, b_end)) {
        return false;
    }
    return std::equal(a_begin, a_end, b_begin);
}

template <class IdType, class CountType>
struct HashKey {
    IdType id;
    CountType count;
    bool operator==(const HashKey &other) const noexcept {
        return id == other.id && count == other.count;
    }
};

struct HashKeyHash {
    template <class IdType, class CountType>
    std::size_t operator()(const HashKey<IdType, CountType> &key) const noexcept {
        std::size_t h1 = std::hash<IdType>{}(key.id);
        std::size_t h2 = std::hash<CountType>{}(key.count);
        // boost::hash_combine
        h1 ^= h2 + 0x9e3779b9ULL + (h1 << 6) + (h1 >> 2);
        return h1;
    }
};

} // namespace bioimage_cpp::label_multiset
