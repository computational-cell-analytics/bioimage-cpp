#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bioimage_cpp::skeleton::detail {

// Persistent union of inclusive integer intervals, stored independently for
// each row. Inserting an interval invokes `newly_covered(begin, end)` only for
// sub-intervals that were not covered by earlier insertions. Stored intervals
// remain sorted, disjoint, and non-adjacent.
class RowIntervalUnion {
public:
    RowIntervalUnion(
        const std::size_t number_of_rows,
        const std::ptrdiff_t row_length
    )
        : rows_(number_of_rows), row_length_(row_length) {
        if (row_length < 0) {
            throw std::invalid_argument("row interval length must be non-negative");
        }
    }

    template <class Body>
    void insert(
        const std::size_t row,
        const std::ptrdiff_t begin,
        const std::ptrdiff_t end,
        Body &&newly_covered
    ) {
        if (row >= rows_.size()) {
            throw std::invalid_argument("row interval index is out of bounds");
        }
        if (begin < 0 || end < begin || end >= row_length_) {
            throw std::invalid_argument("row interval bounds are invalid");
        }

        auto &intervals = rows_[row];
        std::size_t first = 0;
        while (first < intervals.size() && intervals[first].second + 1 < begin) {
            ++first;
        }

        auto merged_begin = begin;
        auto merged_end = end;
        auto cursor = begin;
        std::size_t last = first;
        while (last < intervals.size() && intervals[last].first <= merged_end + 1) {
            const auto [covered_begin, covered_end] = intervals[last];
            if (covered_begin > cursor && cursor <= end) {
                newly_covered(cursor, std::min(end, covered_begin - 1));
            }
            cursor = std::max(cursor, covered_end + 1);
            merged_begin = std::min(merged_begin, covered_begin);
            merged_end = std::max(merged_end, covered_end);
            ++last;
        }
        if (cursor <= end) {
            newly_covered(cursor, end);
        }

        const auto merged = std::pair<std::ptrdiff_t, std::ptrdiff_t>{
            merged_begin, merged_end
        };
        if (first == last) {
            intervals.insert(
                intervals.begin() + static_cast<std::ptrdiff_t>(first), merged
            );
            return;
        }
        intervals[first] = merged;
        intervals.erase(
            intervals.begin() + static_cast<std::ptrdiff_t>(first + 1),
            intervals.begin() + static_cast<std::ptrdiff_t>(last)
        );
    }

private:
    std::vector<std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>>> rows_;
    std::ptrdiff_t row_length_ = 0;
};

} // namespace bioimage_cpp::skeleton::detail
