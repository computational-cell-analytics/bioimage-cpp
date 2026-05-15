#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace bioimage_cpp::detail {

// Disjoint-set / union-find with path compression and union-by-rank.
//
// Three merge variants are provided so both rank-driven and caller-driven
// merge directions are expressible:
//   * merge(u, v):        find both, then union by rank.
//   * merge_to(stable, removed): find both, then force `stable` to be the
//                                parent regardless of rank. The caller picks
//                                which root survives.
//   * unite_roots(a, b):  preconditions: a and b are roots and a != b. Union
//                         by rank without an additional find().
class UnionFind {
public:
    explicit UnionFind(const std::size_t size) : parents_(size), ranks_(size, 0) {
        std::iota(parents_.begin(), parents_.end(), std::uint64_t{0});
    }

    std::uint64_t find(const std::uint64_t node) {
        auto current = static_cast<std::size_t>(node);
        while (parents_[current] != current) {
            parents_[current] = parents_[parents_[current]];
            current = static_cast<std::size_t>(parents_[current]);
        }
        return static_cast<std::uint64_t>(current);
    }

    std::uint64_t merge(std::uint64_t first, std::uint64_t second) {
        first = find(first);
        second = find(second);
        if (first == second) {
            return first;
        }
        return unite_roots(first, second);
    }

    std::uint64_t merge_to(std::uint64_t stable, std::uint64_t removed) {
        stable = find(stable);
        removed = find(removed);
        if (stable == removed) {
            return stable;
        }
        parents_[static_cast<std::size_t>(removed)] = stable;
        if (ranks_[static_cast<std::size_t>(stable)] <= ranks_[static_cast<std::size_t>(removed)]) {
            ranks_[static_cast<std::size_t>(stable)] = ranks_[static_cast<std::size_t>(removed)] + 1;
        }
        return stable;
    }

    std::uint64_t unite_roots(std::uint64_t first, std::uint64_t second) {
        if (ranks_[static_cast<std::size_t>(first)] < ranks_[static_cast<std::size_t>(second)]) {
            std::swap(first, second);
        }
        parents_[static_cast<std::size_t>(second)] = first;
        if (ranks_[static_cast<std::size_t>(first)] == ranks_[static_cast<std::size_t>(second)]) {
            ++ranks_[static_cast<std::size_t>(first)];
        }
        return first;
    }

    [[nodiscard]] std::size_t size() const {
        return parents_.size();
    }

    // Re-initialise the union-find to `n` singletons. Reuses the existing
    // vectors' capacity when possible, so the workspace pattern (reusing a
    // single `UnionFind` across many solver invocations on graphs of varying
    // size) avoids reallocations.
    void reset(const std::size_t n) {
        parents_.resize(n);
        ranks_.assign(n, 0);
        std::iota(parents_.begin(), parents_.end(), std::uint64_t{0});
    }

private:
    std::vector<std::uint64_t> parents_;
    std::vector<std::uint64_t> ranks_;
};

} // namespace bioimage_cpp::detail
