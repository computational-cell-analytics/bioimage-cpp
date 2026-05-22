#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/blocking.hxx"
#include "bioimage_cpp/label_multiset/multiset.hxx"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace bioimage_cpp::label_multiset {

template <class OffsetT, class IdT, class CountT>
inline void read_subset(
    const ConstArrayView<OffsetT> &offsets,
    const ConstArrayView<OffsetT> &sizes,
    const ConstArrayView<IdT> &ids,
    const ConstArrayView<CountT> &counts,
    std::vector<IdT> &ids_out,
    std::vector<CountT> &counts_out,
    const bool argsort = true
) {
    std::unordered_map<IdT, CountT> count_dict;

    const std::size_t n_offsets = static_cast<std::size_t>(offsets.shape[0]);
    for (std::size_t off_id = 0; off_id < n_offsets; ++off_id) {
        const std::size_t offset = static_cast<std::size_t>(offsets.data[off_id]);
        const std::size_t size = static_cast<std::size_t>(sizes.data[off_id]);
        for (std::size_t pos = offset; pos < offset + size; ++pos) {
            const IdT id = ids.data[pos];
            const CountT count = counts.data[pos];
            auto it = count_dict.find(id);
            if (it == count_dict.end()) {
                count_dict.emplace(id, count);
            } else {
                it->second += count;
            }
        }
    }

    ids_out.resize(count_dict.size());
    counts_out.resize(count_dict.size());
    std::size_t i = 0;
    for (const auto &elem : count_dict) {
        ids_out[i] = elem.first;
        counts_out[i] = elem.second;
        ++i;
    }
    if (argsort) {
        argsort_by_first(ids_out, counts_out);
    }
}

// Block-aware variant: collect (offset, size) pairs for every spatial position
// in the block, then call the flat overload. C-order strides over the *full*
// spatial domain.
template <class OffsetT, class IdT, class CountT>
inline void read_subset_block(
    const Block &block,
    const std::vector<std::size_t> &strides,
    const ConstArrayView<OffsetT> &offsets,
    const ConstArrayView<OffsetT> &entry_sizes,
    const ConstArrayView<OffsetT> &entry_offsets,
    const ConstArrayView<IdT> &ids,
    const ConstArrayView<CountT> &counts,
    std::vector<IdT> &ids_out,
    std::vector<CountT> &counts_out,
    const bool argsort = true
) {
    std::unordered_map<IdT, CountT> count_dict;
    const auto &begin = block.begin();
    const auto &end = block.end();
    const std::size_t ndim = begin.size();

    // Generic N-D iteration via incrementing coordinate vector.
    std::vector<Coordinate> coord = begin;
    while (true) {
        std::size_t index = 0;
        for (std::size_t d = 0; d < ndim; ++d) {
            index += static_cast<std::size_t>(coord[d]) * strides[d];
        }
        const std::size_t off = static_cast<std::size_t>(offsets.data[index]);
        const std::size_t entry_idx = static_cast<std::size_t>(entry_offsets.data[index]);
        const std::size_t size = static_cast<std::size_t>(entry_sizes.data[entry_idx]);
        for (std::size_t pos = off; pos < off + size; ++pos) {
            const IdT id = ids.data[pos];
            const CountT count = counts.data[pos];
            auto it = count_dict.find(id);
            if (it == count_dict.end()) {
                count_dict.emplace(id, count);
            } else {
                it->second += count;
            }
        }

        // Increment N-D coord (last axis fastest).
        std::ptrdiff_t axis = static_cast<std::ptrdiff_t>(ndim) - 1;
        for (; axis >= 0; --axis) {
            ++coord[axis];
            if (coord[axis] < end[axis]) {
                break;
            }
            coord[axis] = begin[axis];
        }
        if (axis < 0) {
            break;
        }
    }

    ids_out.resize(count_dict.size());
    counts_out.resize(count_dict.size());
    std::size_t i = 0;
    for (const auto &elem : count_dict) {
        ids_out[i] = elem.first;
        counts_out[i] = elem.second;
        ++i;
    }
    if (argsort) {
        argsort_by_first(ids_out, counts_out);
    }
}

inline std::vector<std::size_t> c_order_strides_for_shape(const CoordinateVector &shape) {
    const std::size_t ndim = shape.size();
    std::vector<std::size_t> strides(ndim, 1);
    if (ndim == 0) {
        return strides;
    }
    for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
        strides[d] = strides[d + 1] * static_cast<std::size_t>(shape[d + 1]);
    }
    return strides;
}

} // namespace bioimage_cpp::label_multiset
