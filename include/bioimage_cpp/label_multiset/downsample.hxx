#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/blocking.hxx"
#include "bioimage_cpp/label_multiset/multiset.hxx"
#include "bioimage_cpp/label_multiset/read_subset.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <unordered_map>
#include <vector>

namespace bioimage_cpp::label_multiset {

// Direct port of nifty::tools::downsampleMultiset. Reads a level-N multiset
// described by (offsets, entry_sizes, entry_offsets, ids, counts) defined over
// the *full* spatial grid (blocking.roi_end()), aggregates into the blocks of
// the given blocking, and writes a deduplicated level-(N+1) multiset.
//
// Outputs:
//   new_argmax        — length blocking.number_of_blocks()
//   new_offsets       — length blocking.number_of_blocks(); offset into new_ids
//   new_entry_offsets — length blocking.number_of_blocks(); maps block → unique entry idx
//   new_entry_sizes   — length n_unique; size of each unique entry
//   new_ids, new_counts — concatenated, deduplicated
template <class OffsetT, class IdT, class CountT>
inline void downsample_multiset(
    const Blocking &blocking,
    const ConstArrayView<OffsetT> &offsets,
    const ConstArrayView<OffsetT> &entry_sizes,
    const ConstArrayView<OffsetT> &entry_offsets,
    const ConstArrayView<IdT> &ids,
    const ConstArrayView<CountT> &counts,
    const int restrict_set,
    ArrayView<IdT> &new_argmax,
    ArrayView<OffsetT> &new_offsets,
    ArrayView<OffsetT> &new_entry_offsets,
    std::vector<OffsetT> &new_entry_sizes,
    std::vector<IdT> &new_ids,
    std::vector<CountT> &new_counts
) {
    using Key = HashKey<IdT, CountT>;
    std::unordered_map<Key, std::vector<std::size_t>, HashKeyHash> candidate_dict;

    const std::size_t n_blocks = static_cast<std::size_t>(blocking.number_of_blocks());
    const auto &shape = blocking.roi_end();
    const auto strides = c_order_strides_for_shape(shape);

    std::size_t current_candidate_id = 0;
    // Stored per unique entry. Indexed by candidate id.
    std::vector<std::size_t> unique_entry_offsets;

    for (std::size_t block_id = 0; block_id < n_blocks; ++block_id) {
        std::vector<IdT> this_ids;
        std::vector<CountT> this_counts;
        const auto block = blocking.get_block(static_cast<std::uint64_t>(block_id));
        read_subset_block(block, strides,
                          offsets, entry_sizes, entry_offsets, ids, counts,
                          this_ids, this_counts, /*argsort=*/true);

        IdT max_label{};
        CountT max_count{};
        if (restrict_set > 0 && static_cast<int>(this_ids.size()) > restrict_set) {
            // sort by count descending
            argsort_by_first(this_counts, this_ids, /*ascending=*/false);
            max_label = this_ids[0];
            max_count = this_counts[0];
            this_ids.resize(static_cast<std::size_t>(restrict_set));
            this_counts.resize(static_cast<std::size_t>(restrict_set));
            argsort_by_first(this_ids, this_counts);
        } else {
            auto max_it = std::max_element(this_counts.begin(), this_counts.end());
            max_label = this_ids[std::distance(this_counts.begin(), max_it)];
            max_count = *max_it;
        }
        new_argmax.data[block_id] = max_label;

        Key key{max_label, max_count};
        bool add_entry = true;
        auto cand_it = candidate_dict.find(key);
        if (cand_it != candidate_dict.end()) {
            for (const std::size_t c_id : cand_it->second) {
                const std::size_t c_offset = unique_entry_offsets[c_id];
                const std::size_t c_size = static_cast<std::size_t>(new_entry_sizes[c_id]);
                bool match = ranges_equal(
                    this_ids.begin(), this_ids.end(),
                    new_ids.begin() + c_offset, new_ids.begin() + c_offset + c_size
                );
                if (match) {
                    match = ranges_equal(
                        this_counts.begin(), this_counts.end(),
                        new_counts.begin() + c_offset, new_counts.begin() + c_offset + c_size
                    );
                }
                if (match) {
                    new_offsets.data[block_id] = static_cast<OffsetT>(c_offset);
                    new_entry_offsets.data[block_id] = static_cast<OffsetT>(c_id);
                    add_entry = false;
                    break;
                }
            }
        }

        if (add_entry) {
            const std::size_t this_offset = new_ids.size();
            new_offsets.data[block_id] = static_cast<OffsetT>(this_offset);
            new_entry_offsets.data[block_id] = static_cast<OffsetT>(current_candidate_id);
            unique_entry_offsets.emplace_back(this_offset);
            new_entry_sizes.emplace_back(static_cast<OffsetT>(this_ids.size()));
            new_ids.insert(new_ids.end(), this_ids.begin(), this_ids.end());
            new_counts.insert(new_counts.end(), this_counts.begin(), this_counts.end());
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
