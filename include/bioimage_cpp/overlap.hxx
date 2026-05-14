#pragma once

#include "bioimage_cpp/array_view.hxx"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bioimage_cpp::ground_truth {

struct OverlapPair {
    std::uint64_t label_a = 0;
    std::uint64_t label_b = 0;
    std::uint64_t count = 0;
};

class SegmentationOverlap {
public:
    void add(const std::uint64_t label_a, const std::uint64_t label_b) {
        ++overlaps_a_[label_a][label_b];
        ++overlaps_b_[label_b][label_a];
        ++counts_a_[label_a];
        ++counts_b_[label_b];
        ++total_count_;
    }

    std::uint64_t total_count() const {
        return total_count_;
    }

    std::uint64_t count_a(const std::uint64_t label) const {
        return map_value_or_zero(counts_a_, label);
    }

    std::uint64_t count_b(const std::uint64_t label) const {
        return map_value_or_zero(counts_b_, label);
    }

    std::uint64_t overlap_count(
        const std::uint64_t label_a,
        const std::uint64_t label_b
    ) const {
        const auto found_a = overlaps_a_.find(label_a);
        if (found_a == overlaps_a_.end()) {
            return 0;
        }
        return map_value_or_zero(found_a->second, label_b);
    }

    std::vector<std::uint64_t> labels_a() const {
        return sorted_keys(counts_a_);
    }

    std::vector<std::uint64_t> labels_b() const {
        return sorted_keys(counts_b_);
    }

    std::vector<std::pair<std::uint64_t, std::uint64_t>> counts_a() const {
        return sorted_label_counts(counts_a_);
    }

    std::vector<std::pair<std::uint64_t, std::uint64_t>> counts_b() const {
        return sorted_label_counts(counts_b_);
    }

    std::vector<OverlapPair> overlap_pairs() const {
        std::vector<OverlapPair> result;
        for (const auto &label_overlaps : overlaps_a_) {
            for (const auto &overlap : label_overlaps.second) {
                result.push_back(OverlapPair{
                    label_overlaps.first,
                    overlap.first,
                    overlap.second,
                });
            }
        }
        sort_overlap_pairs(result);
        return result;
    }

    std::vector<std::pair<std::uint64_t, std::uint64_t>> overlaps_for_label_a(
        const std::uint64_t label_a
    ) const {
        return overlaps_for_label(overlaps_a_, label_a);
    }

    std::vector<std::pair<std::uint64_t, std::uint64_t>> overlaps_for_label_b(
        const std::uint64_t label_b
    ) const {
        return overlaps_for_label(overlaps_b_, label_b);
    }

    std::pair<std::uint64_t, std::uint64_t> best_overlap_for_label_a(
        const std::uint64_t label_a,
        const bool ignore_zero = false
    ) const {
        return best_overlap(overlaps_a_, label_a, ignore_zero);
    }

    std::pair<std::uint64_t, std::uint64_t> best_overlap_for_label_b(
        const std::uint64_t label_b,
        const bool ignore_zero = false
    ) const {
        return best_overlap(overlaps_b_, label_b, ignore_zero);
    }

    bool is_label_a_overlapping_with_zero(const std::uint64_t label_a) const {
        return overlap_count(label_a, 0) != 0;
    }

    bool is_label_b_overlapping_with_zero(const std::uint64_t label_b) const {
        const auto found_b = overlaps_b_.find(label_b);
        if (found_b == overlaps_b_.end()) {
            return false;
        }
        return found_b->second.find(0) != found_b->second.end();
    }

    double different_overlap(const std::uint64_t label_a_u, const std::uint64_t label_a_v) const {
        const auto found_u = overlaps_a_.find(label_a_u);
        const auto found_v = overlaps_a_.find(label_a_v);
        if (found_u == overlaps_a_.end() || found_v == overlaps_a_.end()) {
            throw std::out_of_range("labels must exist in segmentation A");
        }

        const auto size_u = static_cast<double>(count_a(label_a_u));
        const auto size_v = static_cast<double>(count_a(label_a_v));
        double result = 0.0;
        for (const auto &overlap_u : found_u->second) {
            for (const auto &overlap_v : found_v->second) {
                if (overlap_u.first != overlap_v.first) {
                    result +=
                        (static_cast<double>(overlap_u.second) / size_u) *
                        (static_cast<double>(overlap_v.second) / size_v);
                }
            }
        }
        return result;
    }

private:
    using CountMap = std::unordered_map<std::uint64_t, std::uint64_t>;
    using OverlapMap = std::unordered_map<std::uint64_t, CountMap>;

    static std::uint64_t map_value_or_zero(
        const CountMap &map,
        const std::uint64_t label
    ) {
        const auto found = map.find(label);
        return found == map.end() ? 0 : found->second;
    }

    static std::vector<std::uint64_t> sorted_keys(const CountMap &map) {
        std::vector<std::uint64_t> result;
        result.reserve(map.size());
        for (const auto &entry : map) {
            result.push_back(entry.first);
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    static std::vector<std::pair<std::uint64_t, std::uint64_t>> sorted_label_counts(
        const CountMap &map
    ) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> result;
        result.reserve(map.size());
        for (const auto &entry : map) {
            result.push_back(entry);
        }
        std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
            return a.first < b.first;
        });
        return result;
    }

    static void sort_overlap_pairs(std::vector<OverlapPair> &pairs) {
        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
            if (a.label_a != b.label_a) {
                return a.label_a < b.label_a;
            }
            return a.label_b < b.label_b;
        });
    }

    static std::vector<std::pair<std::uint64_t, std::uint64_t>> overlaps_for_label(
        const OverlapMap &overlaps,
        const std::uint64_t label
    ) {
        const auto found = overlaps.find(label);
        if (found == overlaps.end()) {
            return {};
        }
        return sorted_label_counts(found->second);
    }

    static std::pair<std::uint64_t, std::uint64_t> best_overlap(
        const OverlapMap &overlaps,
        const std::uint64_t label,
        const bool ignore_zero
    ) {
        const auto found = overlaps.find(label);
        if (found == overlaps.end()) {
            return {0, 0};
        }

        std::uint64_t best_label = 0;
        std::uint64_t best_count = 0;
        for (const auto &overlap : found->second) {
            if (ignore_zero && overlap.first == 0) {
                continue;
            }
            if (
                overlap.second > best_count ||
                (overlap.second == best_count && overlap.first < best_label)
            ) {
                best_label = overlap.first;
                best_count = overlap.second;
            }
        }
        return {best_label, best_count};
    }

    CountMap counts_a_;
    CountMap counts_b_;
    OverlapMap overlaps_a_;
    OverlapMap overlaps_b_;
    std::uint64_t total_count_ = 0;
};

inline std::uint64_t array_size(const std::vector<std::ptrdiff_t> &shape) {
    std::uint64_t size = 1;
    for (const auto axis_size : shape) {
        if (axis_size < 0) {
            throw std::invalid_argument("shape entries must be non-negative");
        }
        size *= static_cast<std::uint64_t>(axis_size);
    }
    return size;
}

inline SegmentationOverlap segmentation_overlap(
    const ConstArrayView<std::uint64_t> &labels_a,
    const ConstArrayView<std::uint64_t> &labels_b
) {
    if (labels_a.shape != labels_b.shape) {
        throw std::invalid_argument("labels_a and labels_b must have the same shape");
    }

    SegmentationOverlap result;
    const auto size = array_size(labels_a.shape);
    for (std::uint64_t index = 0; index < size; ++index) {
        result.add(labels_a.data[index], labels_b.data[index]);
    }
    return result;
}

} // namespace bioimage_cpp::ground_truth
