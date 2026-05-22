#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/blocking.hxx"
#include "bioimage_cpp/label_multiset/multiset.hxx"
#include "bioimage_cpp/label_multiset/read_subset.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace bioimage_cpp::label_multiset {

// Build a deduplicated level-0 multiset from a (C-contiguous) label volume,
// where each spatial position corresponds to one block of the given blocking
// (the blocking's roi spans the full label volume, block_shape determines the
// aggregation footprint of each "voxel" in the resulting multiset).
//
// For block_shape == (1,1,...,1) this gives the trivial "one label, count 1
// per pixel" multiset.
//
// Outputs:
//   argmax            — length n_blocks
//   out_offsets       — length n_blocks
//   out_entry_offsets — length n_blocks
//   out_entry_sizes   — length n_unique
//   out_ids, out_counts — concatenated
template <class LabelT, class OffsetT, class IdT, class CountT>
inline void multiset_from_labels(
    const ConstArrayView<LabelT> &labels,
    const Blocking &blocking,
    ArrayView<IdT> &argmax,
    ArrayView<OffsetT> &out_offsets,
    ArrayView<OffsetT> &out_entry_offsets,
    std::vector<OffsetT> &out_entry_sizes,
    std::vector<IdT> &out_ids,
    std::vector<CountT> &out_counts
) {
    using Key = HashKey<IdT, CountT>;
    std::unordered_map<Key, std::vector<std::size_t>, HashKeyHash> candidate_dict;

    const std::size_t ndim = blocking.ndim();
    const auto &shape = blocking.roi_end();
    const auto strides = c_order_strides_for_shape(shape);

    const std::size_t n_blocks = static_cast<std::size_t>(blocking.number_of_blocks());
    std::vector<std::size_t> unique_entry_offsets;
    std::size_t current_candidate_id = 0;

    std::vector<IdT> this_ids;
    std::vector<CountT> this_counts;

    for (std::size_t block_id = 0; block_id < n_blocks; ++block_id) {
        const auto block = blocking.get_block(static_cast<std::uint64_t>(block_id));
        const auto &begin = block.begin();
        const auto &end = block.end();

        std::unordered_map<IdT, CountT> count_dict;
        std::vector<Coordinate> coord = begin;
        while (true) {
            std::size_t index = 0;
            for (std::size_t d = 0; d < ndim; ++d) {
                index += static_cast<std::size_t>(coord[d]) * strides[d];
            }
            const IdT id = static_cast<IdT>(labels.data[index]);
            auto it = count_dict.find(id);
            if (it == count_dict.end()) {
                count_dict.emplace(id, CountT{1});
            } else {
                it->second += CountT{1};
            }

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

        this_ids.assign(count_dict.size(), IdT{});
        this_counts.assign(count_dict.size(), CountT{});
        std::size_t i = 0;
        for (const auto &elem : count_dict) {
            this_ids[i] = elem.first;
            this_counts[i] = elem.second;
            ++i;
        }
        argsort_by_first(this_ids, this_counts);

        auto max_it = std::max_element(this_counts.begin(), this_counts.end());
        const IdT max_label = this_ids[static_cast<std::size_t>(std::distance(this_counts.begin(), max_it))];
        const CountT max_count = *max_it;
        argmax.data[block_id] = max_label;

        Key key{max_label, max_count};
        bool add_entry = true;
        auto cand_it = candidate_dict.find(key);
        if (cand_it != candidate_dict.end()) {
            for (const std::size_t c_id : cand_it->second) {
                const std::size_t c_offset = unique_entry_offsets[c_id];
                const std::size_t c_size = static_cast<std::size_t>(out_entry_sizes[c_id]);
                bool match = ranges_equal(
                    this_ids.begin(), this_ids.end(),
                    out_ids.begin() + c_offset, out_ids.begin() + c_offset + c_size
                );
                if (match) {
                    match = ranges_equal(
                        this_counts.begin(), this_counts.end(),
                        out_counts.begin() + c_offset, out_counts.begin() + c_offset + c_size
                    );
                }
                if (match) {
                    out_offsets.data[block_id] = static_cast<OffsetT>(c_offset);
                    out_entry_offsets.data[block_id] = static_cast<OffsetT>(c_id);
                    add_entry = false;
                    break;
                }
            }
        }

        if (add_entry) {
            const std::size_t this_offset = out_ids.size();
            out_offsets.data[block_id] = static_cast<OffsetT>(this_offset);
            out_entry_offsets.data[block_id] = static_cast<OffsetT>(current_candidate_id);
            unique_entry_offsets.emplace_back(this_offset);
            out_entry_sizes.emplace_back(static_cast<OffsetT>(this_ids.size()));
            out_ids.insert(out_ids.end(), this_ids.begin(), this_ids.end());
            out_counts.insert(out_counts.end(), this_counts.begin(), this_counts.end());
            if (cand_it == candidate_dict.end()) {
                candidate_dict.emplace(key, std::vector<std::size_t>{current_candidate_id});
            } else {
                cand_it->second.emplace_back(current_candidate_id);
            }
            ++current_candidate_id;
        }
    }
}

} // namespace bioimage_cpp::label_multiset
