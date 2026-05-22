#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/label_multiset/multiset.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <unordered_map>
#include <vector>

namespace bioimage_cpp::label_multiset {

template <class OffsetT, class IdT, class CountT>
class MultisetMerger {
public:
    using OffsetType = OffsetT;
    using IdType = IdT;
    using CountType = CountT;

    MultisetMerger(
        const ConstArrayView<OffsetT> &offsets,
        const ConstArrayView<OffsetT> &entry_sizes,
        const ConstArrayView<IdT> &ids,
        const ConstArrayView<CountT> &counts
    )
        : offsets_(offsets.data, offsets.data + offsets.shape[0]),
          entry_sizes_(entry_sizes.data, entry_sizes.data + entry_sizes.shape[0]),
          ids_(ids.data, ids.data + ids.shape[0]),
          counts_(counts.data, counts.data + counts.shape[0]) {
        init_hashed();
    }

    const std::vector<IdT> &ids() const { return ids_; }
    const std::vector<CountT> &counts() const { return counts_; }
    const std::vector<OffsetT> &offsets() const { return offsets_; }
    const std::vector<OffsetT> &entry_sizes() const { return entry_sizes_; }

    // Ingest a batch of *unique* entries described by (unique_offsets,
    // entry_sizes, ids, counts), then rewrite `offsets` so that each element
    // (which was indexed by "entry id within the input batch") becomes the
    // absolute byte offset into the deduplicated ids_/counts_ arrays.
    void update(
        const ConstArrayView<OffsetT> &unique_offsets,
        const ConstArrayView<OffsetT> &batch_entry_sizes,
        const ConstArrayView<IdT> &batch_ids,
        const ConstArrayView<CountT> &batch_counts,
        ArrayView<OffsetT> &offsets
    ) {
        const std::size_t n_entries = static_cast<std::size_t>(unique_offsets.shape[0]);
        // Maps batch entry id → absolute byte offset in ids_/counts_.
        std::unordered_map<OffsetT, OffsetT> new_offset_dict;

        for (std::size_t entry = 0; entry < n_entries; ++entry) {
            const OffsetT off = unique_offsets.data[entry];
            const OffsetT size = batch_entry_sizes.data[entry];

            const IdT *ids_begin = batch_ids.data + off;
            const IdT *ids_end = ids_begin + size;
            const CountT *counts_begin = batch_counts.data + off;
            const CountT *counts_end = counts_begin + size;

            auto max_it = std::max_element(counts_begin, counts_end);
            const IdT max_label = *(ids_begin + std::distance(counts_begin, max_it));
            const CountT max_count = *max_it;

            HashKey<IdT, CountT> key{max_label, max_count};
            auto hash_it = hashed_.find(key);
            bool new_entry = true;
            if (hash_it != hashed_.end()) {
                for (const std::size_t c_id : hash_it->second) {
                    const std::size_t c_offset = static_cast<std::size_t>(offsets_[c_id]);
                    const std::size_t c_size = static_cast<std::size_t>(entry_sizes_[c_id]);
                    bool match = ranges_equal(
                        ids_begin, ids_end,
                        ids_.begin() + c_offset, ids_.begin() + c_offset + c_size
                    );
                    if (match) {
                        match = ranges_equal(
                            counts_begin, counts_end,
                            counts_.begin() + c_offset, counts_.begin() + c_offset + c_size
                        );
                    }
                    if (match) {
                        new_entry = false;
                        new_offset_dict[static_cast<OffsetT>(entry)] = static_cast<OffsetT>(c_offset);
                        break;
                    }
                }
            }

            if (new_entry) {
                const std::size_t this_size = static_cast<std::size_t>(std::distance(ids_begin, ids_end));
                const std::size_t this_offset = ids_.size();
                offsets_.emplace_back(static_cast<OffsetT>(this_offset));
                entry_sizes_.emplace_back(static_cast<OffsetT>(this_size));
                new_offset_dict[static_cast<OffsetT>(entry)] = static_cast<OffsetT>(this_offset);
                ids_.insert(ids_.end(), ids_begin, ids_end);
                counts_.insert(counts_.end(), counts_begin, counts_end);
                const std::size_t this_id = offsets_.size() - 1;
                if (hash_it == hashed_.end()) {
                    hashed_.emplace(key, std::vector<std::size_t>{this_id});
                    // hash_it is invalidated by emplace; we don't reuse it.
                } else {
                    hash_it->second.emplace_back(this_id);
                }
            }
        }

        const std::size_t n_off = static_cast<std::size_t>(offsets.shape[0]);
        for (std::size_t i = 0; i < n_off; ++i) {
            offsets.data[i] = new_offset_dict[offsets.data[i]];
        }
    }

private:
    void init_hashed() {
        const std::size_t n_entries = offsets_.size();
        for (std::size_t entry = 0; entry < n_entries; ++entry) {
            const std::size_t off = static_cast<std::size_t>(offsets_[entry]);
            const std::size_t size = static_cast<std::size_t>(entry_sizes_[entry]);
            auto max_it = std::max_element(counts_.begin() + off, counts_.begin() + off + size);
            const IdT max_label = ids_[static_cast<std::size_t>(std::distance(counts_.begin(), max_it))];
            const CountT max_count = *max_it;
            HashKey<IdT, CountT> key{max_label, max_count};
            auto it = hashed_.find(key);
            if (it == hashed_.end()) {
                hashed_.emplace(key, std::vector<std::size_t>{entry});
            } else {
                it->second.emplace_back(entry);
            }
        }
    }

    std::vector<OffsetT> offsets_;
    std::vector<OffsetT> entry_sizes_;
    std::vector<IdT> ids_;
    std::vector<CountT> counts_;
    std::unordered_map<HashKey<IdT, CountT>, std::vector<std::size_t>, HashKeyHash> hashed_;
};

} // namespace bioimage_cpp::label_multiset
