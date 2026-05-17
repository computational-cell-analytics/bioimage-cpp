#pragma once

#include <cstdint>
#include <vector>

namespace bioimage_cpp {

using SemanticLabeling = std::vector<std::int64_t>;

// Returns true if the two roots already carry different non-negative semantic
// labels. Merging two such roots would conflict the semantic assignments, so
// the caller must skip that merge.
inline bool check_semantic_constraint(
    const std::uint64_t first,
    const std::uint64_t second,
    const SemanticLabeling &semantic_labels
) {
    const auto label_first = semantic_labels[first];
    const auto label_second = semantic_labels[second];
    if (label_first >= 0 && label_second >= 0) {
        return label_first != label_second;
    }
    return false;
}

// Assigns `class_id` to `root` only if `root` has no semantic label yet
// (-1 == unassigned). Already-assigned roots keep their first assignment;
// because semantic edges are processed in descending-weight order this means
// each root keeps the strongest class assignment seen for it.
inline void assign_semantic_label(
    const std::uint64_t root,
    const std::int64_t class_id,
    SemanticLabeling &semantic_labels
) {
    if (semantic_labels[root] < 0) {
        semantic_labels[root] = class_id;
    }
}

// After two roots merge, propagate a non-negative semantic label from one to
// the other if exactly one was assigned. If both are assigned the caller is
// expected to have rejected the merge via `check_semantic_constraint`.
inline void merge_semantic_labels(
    const std::uint64_t first,
    const std::uint64_t second,
    SemanticLabeling &semantic_labels
) {
    const auto label_first = semantic_labels[first];
    const auto label_second = semantic_labels[second];
    if (label_first >= 0 && label_second < 0) {
        semantic_labels[second] = label_first;
    } else if (label_first < 0 && label_second >= 0) {
        semantic_labels[first] = label_second;
    }
}

} // namespace bioimage_cpp
