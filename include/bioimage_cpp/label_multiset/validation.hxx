#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/blocking.hxx"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace bioimage_cpp::label_multiset {

namespace validation_detail {

template <class T>
inline std::size_t view_length(
    const ConstArrayView<T> &view,
    const char *name
) {
    if (view.ndim() != 1) {
        throw std::invalid_argument(
            std::string(name) + " must have ndim 1, got ndim=" +
            std::to_string(view.ndim())
        );
    }
    if (view.shape[0] < 0) {
        throw std::invalid_argument(std::string(name) + " has a negative length");
    }
    return static_cast<std::size_t>(view.shape[0]);
}

template <class T>
inline std::size_t view_length(
    const ArrayView<T> &view,
    const char *name
) {
    if (view.ndim() != 1) {
        throw std::invalid_argument(
            std::string(name) + " must have ndim 1, got ndim=" +
            std::to_string(view.ndim())
        );
    }
    if (view.shape[0] < 0) {
        throw std::invalid_argument(std::string(name) + " has a negative length");
    }
    return static_cast<std::size_t>(view.shape[0]);
}

template <class Integer>
inline std::size_t as_size(
    const Integer value,
    const char *name,
    const std::size_t index
) {
    static_assert(std::is_integral_v<Integer>);
    if constexpr (std::is_signed_v<Integer>) {
        if (value < 0) {
            throw std::invalid_argument(
                std::string(name) + "[" + std::to_string(index) +
                "] must be non-negative"
            );
        }
    }
    if (!std::in_range<std::size_t>(value)) {
        throw std::invalid_argument(
            std::string(name) + "[" + std::to_string(index) +
            "] does not fit size_t"
        );
    }
    return static_cast<std::size_t>(value);
}

inline std::size_t full_grid_size(const Blocking &blocking) {
    std::size_t result = 1;
    const auto &shape = blocking.roi_end();
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        const auto extent = as_size(shape[axis], "blocking.roi_end", axis);
        if (extent != 0 && result > std::numeric_limits<std::size_t>::max() / extent) {
            throw std::invalid_argument(
                "product(blocking.roi_end) does not fit size_t"
            );
        }
        result *= extent;
    }
    return result;
}

} // namespace validation_detail

template <class IdT, class CountT>
inline std::size_t validate_flat_storage(
    const ConstArrayView<IdT> &ids,
    const ConstArrayView<CountT> &counts
) {
    const auto n_ids = validation_detail::view_length(ids, "ids");
    const auto n_counts = validation_detail::view_length(counts, "counts");
    if (n_ids != n_counts) {
        throw std::invalid_argument(
            "ids and counts must have the same length, got ids length=" +
            std::to_string(n_ids) + ", counts length=" + std::to_string(n_counts)
        );
    }
    return n_ids;
}

template <class OffsetT>
inline void validate_flat_ranges(
    const ConstArrayView<OffsetT> &offsets,
    const ConstArrayView<OffsetT> &sizes,
    const std::size_t storage_length,
    const bool require_nonempty
) {
    const auto n_offsets = validation_detail::view_length(offsets, "offsets");
    const auto n_sizes = validation_detail::view_length(sizes, "sizes");
    if (n_offsets != n_sizes) {
        throw std::invalid_argument(
            "offsets and sizes must have the same length, got offsets length=" +
            std::to_string(n_offsets) + ", sizes length=" + std::to_string(n_sizes)
        );
    }
    for (std::size_t index = 0; index < n_offsets; ++index) {
        const auto offset =
            validation_detail::as_size(offsets.data[index], "offsets", index);
        const auto size =
            validation_detail::as_size(sizes.data[index], "sizes", index);
        if (require_nonempty && size == 0) {
            throw std::invalid_argument(
                "sizes[" + std::to_string(index) + "] must be greater than zero"
            );
        }
        if (offset > storage_length) {
            throw std::invalid_argument(
                "offsets[" + std::to_string(index) +
                "] exceeds flat storage length " + std::to_string(storage_length)
            );
        }
        if (size > storage_length - offset) {
            throw std::invalid_argument(
                "range at index " + std::to_string(index) +
                " exceeds flat storage length " + std::to_string(storage_length)
            );
        }
    }
}

template <class OffsetT, class IdT, class CountT>
inline void validate_read_subset(
    const ConstArrayView<OffsetT> &offsets,
    const ConstArrayView<OffsetT> &sizes,
    const ConstArrayView<IdT> &ids,
    const ConstArrayView<CountT> &counts
) {
    const auto storage_length = validate_flat_storage(ids, counts);
    validate_flat_ranges(offsets, sizes, storage_length, false);
}

template <class OffsetT, class IdT, class CountT>
inline void validate_flat_multiset(
    const ConstArrayView<OffsetT> &offsets,
    const ConstArrayView<OffsetT> &entry_sizes,
    const ConstArrayView<OffsetT> &entry_offsets,
    const ConstArrayView<IdT> &ids,
    const ConstArrayView<CountT> &counts
) {
    const auto n_offsets = validation_detail::view_length(offsets, "offsets");
    const auto n_entry_offsets =
        validation_detail::view_length(entry_offsets, "entry_offsets");
    const auto n_entries =
        validation_detail::view_length(entry_sizes, "entry_sizes");
    const auto storage_length = validate_flat_storage(ids, counts);

    if (n_offsets != n_entry_offsets) {
        throw std::invalid_argument(
            "offsets and entry_offsets must have the same length, got offsets length=" +
            std::to_string(n_offsets) + ", entry_offsets length=" +
            std::to_string(n_entry_offsets)
        );
    }
    if (n_offsets == 0) {
        if (n_entries != 0 || storage_length != 0) {
            throw std::invalid_argument(
                "an empty multiset must have empty entry_sizes, ids, and counts"
            );
        }
        return;
    }
    if (n_entries == 0) {
        throw std::invalid_argument(
            "entry_sizes must not be empty when spatial entries are present"
        );
    }

    std::vector<bool> referenced(n_entries, false);
    for (std::size_t entry = 0; entry < n_entries; ++entry) {
        const auto size = validation_detail::as_size(
            entry_sizes.data[entry], "entry_sizes", entry
        );
        if (size == 0) {
            throw std::invalid_argument(
                "entry_sizes[" + std::to_string(entry) +
                "] must be greater than zero"
            );
        }
    }
    for (std::size_t spatial = 0; spatial < n_offsets; ++spatial) {
        const auto entry = validation_detail::as_size(
            entry_offsets.data[spatial], "entry_offsets", spatial
        );
        if (entry >= n_entries) {
            throw std::invalid_argument(
                "entry_offsets[" + std::to_string(spatial) +
                "] must be less than entry_sizes length " +
                std::to_string(n_entries)
            );
        }
        referenced[entry] = true;
        const auto offset =
            validation_detail::as_size(offsets.data[spatial], "offsets", spatial);
        const auto size = validation_detail::as_size(
            entry_sizes.data[entry], "entry_sizes", entry
        );
        if (offset > storage_length) {
            throw std::invalid_argument(
                "offsets[" + std::to_string(spatial) +
                "] exceeds flat storage length " + std::to_string(storage_length)
            );
        }
        if (size > storage_length - offset) {
            throw std::invalid_argument(
                "entry range at spatial index " + std::to_string(spatial) +
                " exceeds flat storage length " + std::to_string(storage_length)
            );
        }
    }
    for (std::size_t entry = 0; entry < n_entries; ++entry) {
        if (!referenced[entry]) {
            throw std::invalid_argument(
                "entry_sizes[" + std::to_string(entry) +
                "] is not referenced by entry_offsets"
            );
        }
    }
}

template <class OffsetT, class IdT, class CountT>
inline void validate_downsample_input(
    const Blocking &blocking,
    const ConstArrayView<OffsetT> &offsets,
    const ConstArrayView<OffsetT> &entry_sizes,
    const ConstArrayView<OffsetT> &entry_offsets,
    const ConstArrayView<IdT> &ids,
    const ConstArrayView<CountT> &counts
) {
    validate_flat_multiset(offsets, entry_sizes, entry_offsets, ids, counts);
    const auto n_spatial = validation_detail::view_length(offsets, "offsets");
    const auto expected = validation_detail::full_grid_size(blocking);
    if (n_spatial != expected) {
        throw std::invalid_argument(
            "offsets length must equal product(blocking.roi_end), got offsets length=" +
            std::to_string(n_spatial) + ", expected length=" +
            std::to_string(expected)
        );
    }
}

template <class IdT, class OffsetT>
inline void validate_downsample_output(
    const Blocking &blocking,
    const ArrayView<IdT> &new_argmax,
    const ArrayView<OffsetT> &new_offsets,
    const ArrayView<OffsetT> &new_entry_offsets
) {
    const auto expected = validation_detail::as_size(
        blocking.number_of_blocks(), "blocking.number_of_blocks", 0
    );
    const auto n_argmax =
        validation_detail::view_length(new_argmax, "new_argmax");
    const auto n_offsets =
        validation_detail::view_length(new_offsets, "new_offsets");
    const auto n_entry_offsets =
        validation_detail::view_length(new_entry_offsets, "new_entry_offsets");
    if (n_argmax != expected || n_offsets != expected || n_entry_offsets != expected) {
        throw std::invalid_argument(
            "downsample output lengths must equal blocking.number_of_blocks"
        );
    }
}

template <class OffsetT, class IdT, class CountT>
inline void validate_merger_entries(
    const ConstArrayView<OffsetT> &unique_offsets,
    const ConstArrayView<OffsetT> &entry_sizes,
    const ConstArrayView<IdT> &ids,
    const ConstArrayView<CountT> &counts
) {
    const auto storage_length = validate_flat_storage(ids, counts);
    validate_flat_ranges(unique_offsets, entry_sizes, storage_length, true);
}

template <class OffsetT>
inline void validate_update_entry_indices(
    const ArrayView<OffsetT> &entry_indices,
    const std::size_t number_of_entries
) {
    const auto n_indices =
        validation_detail::view_length(entry_indices, "offsets");
    for (std::size_t index = 0; index < n_indices; ++index) {
        const auto entry = validation_detail::as_size(
            entry_indices.data[index], "offsets", index
        );
        if (entry >= number_of_entries) {
            throw std::invalid_argument(
                "offsets[" + std::to_string(index) +
                "] must be less than the number of batch entries " +
                std::to_string(number_of_entries)
            );
        }
    }
}

} // namespace bioimage_cpp::label_multiset
