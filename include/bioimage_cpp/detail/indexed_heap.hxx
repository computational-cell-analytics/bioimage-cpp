#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bioimage_cpp::detail {

// Sentinel used by both locators below to mark "this key is not in the heap".
inline constexpr std::size_t indexed_heap_not_in_heap =
    std::numeric_limits<std::size_t>::max();

// Locator concept used by IndexedHeap:
//   std::size_t at(key) const         — returns position in heap, or `not_in_heap`.
//   void set(key, std::size_t pos)    — records a key's position. `not_in_heap` clears it.

// Locator for keys drawn from [0, capacity). Backed by a flat vector so the
// position lookup is a single load.
class DenseLocator {
public:
    static constexpr std::size_t not_in_heap = indexed_heap_not_in_heap;

    explicit DenseLocator(const std::size_t capacity = 0)
        : positions_(capacity, not_in_heap) {}

    [[nodiscard]] std::size_t at(const std::size_t key) const {
        return positions_[key];
    }

    void set(const std::size_t key, const std::size_t pos) {
        positions_[key] = pos;
    }

    // Resize and reset every position to `not_in_heap`. Use between solver
    // invocations on different graphs.
    void reset_capacity(const std::size_t capacity) {
        positions_.assign(capacity, not_in_heap);
    }

private:
    std::vector<std::size_t> positions_;
};

// Locator for keys with sparse / arbitrary identity. Position lookup is a
// hash-map probe.
template <class KeyT, class Hash = std::hash<KeyT>>
class SparseLocator {
public:
    static constexpr std::size_t not_in_heap = indexed_heap_not_in_heap;

    [[nodiscard]] std::size_t at(const KeyT &key) const {
        const auto it = positions_.find(key);
        return it == positions_.end() ? not_in_heap : it->second;
    }

    void set(const KeyT &key, const std::size_t pos) {
        if (pos == not_in_heap) {
            positions_.erase(key);
        } else {
            positions_.insert_or_assign(key, pos);
        }
    }

    void reserve(const std::size_t expected_keys) {
        positions_.reserve(expected_keys);
    }

    void clear() {
        positions_.clear();
    }

private:
    std::unordered_map<KeyT, std::size_t, Hash> positions_;
};

// Addressable max-heap with mutable priorities.
//
// Each key maps to exactly one entry in the heap, tracked via the Locator.
// `push`, `change`, `push_or_change`, `erase`, and `pop` are O(log n) and keep
// the locator in sync with the heap permutation. `top`, `contains`, and
// `priority_of` are O(1). Compared to a `std::priority_queue` with lazy
// invalidation, the heap never carries stale entries — its size equals the
// number of currently active keys.
//
// The comparator `Compare` follows the standard "less-than → max-heap"
// convention used by `std::priority_queue`. Specialize with `std::greater<>`
// for a min-heap.
template <
    class KeyT,
    class PriorityT,
    class Locator,
    class Compare = std::less<PriorityT>
>
class IndexedHeap {
public:
    using key_type = KeyT;
    using priority_type = PriorityT;

    struct Entry {
        KeyT key;
        PriorityT priority;
    };

    static constexpr std::size_t not_in_heap = Locator::not_in_heap;

    IndexedHeap() = default;
    explicit IndexedHeap(Locator locator) : locator_(std::move(locator)) {}

    [[nodiscard]] std::size_t size() const { return heap_.size(); }
    [[nodiscard]] bool empty() const { return heap_.empty(); }

    [[nodiscard]] bool contains(const KeyT &key) const {
        return locator_.at(key) != not_in_heap;
    }

    [[nodiscard]] const PriorityT &priority_of(const KeyT &key) const {
        return heap_[locator_.at(key)].priority;
    }

    [[nodiscard]] const Entry &top() const {
        return heap_.front();
    }

    // Precondition: `key` is not currently in the heap.
    void push(KeyT key, PriorityT priority) {
        const auto pos = heap_.size();
        heap_.push_back({std::move(key), std::move(priority)});
        locator_.set(heap_.back().key, pos);
        sift_up(pos);
    }

    // Precondition: `key` is currently in the heap.
    void change(const KeyT &key, PriorityT priority) {
        const auto pos = locator_.at(key);
        apply_change(pos, std::move(priority));
    }

    void push_or_change(KeyT key, PriorityT priority) {
        const auto pos = locator_.at(key);
        if (pos == not_in_heap) {
            push(std::move(key), std::move(priority));
        } else {
            apply_change(pos, std::move(priority));
        }
    }

    // No-op when the key is not in the heap. This is the "remove if exists"
    // semantic every caller needs.
    void erase(const KeyT &key) {
        const auto pos = locator_.at(key);
        if (pos != not_in_heap) {
            erase_at(pos);
        }
    }

    Entry pop() {
        Entry top = heap_.front();
        erase_at(0);
        return top;
    }

    void clear() {
        for (const auto &entry : heap_) {
            locator_.set(entry.key, not_in_heap);
        }
        heap_.clear();
    }

    Locator &locator() { return locator_; }
    const Locator &locator() const { return locator_; }

private:
    std::vector<Entry> heap_;
    Locator locator_;
    Compare compare_;

    void swap_positions(const std::size_t a, const std::size_t b) {
        std::swap(heap_[a], heap_[b]);
        locator_.set(heap_[a].key, a);
        locator_.set(heap_[b].key, b);
    }

    void sift_up(std::size_t pos) {
        while (pos > 0) {
            const auto parent = (pos - 1) / 2;
            if (!compare_(heap_[parent].priority, heap_[pos].priority)) {
                break;
            }
            swap_positions(parent, pos);
            pos = parent;
        }
    }

    void sift_down(std::size_t pos) {
        const auto n = heap_.size();
        while (true) {
            const auto left = 2 * pos + 1;
            if (left >= n) {
                break;
            }
            auto target = left;
            const auto right = left + 1;
            if (right < n && compare_(heap_[left].priority, heap_[right].priority)) {
                target = right;
            }
            if (!compare_(heap_[pos].priority, heap_[target].priority)) {
                break;
            }
            swap_positions(pos, target);
            pos = target;
        }
    }

    void apply_change(const std::size_t pos, PriorityT new_priority) {
        const bool increased = compare_(heap_[pos].priority, new_priority);
        heap_[pos].priority = std::move(new_priority);
        if (increased) {
            sift_up(pos);
        } else {
            sift_down(pos);
        }
    }

    void erase_at(const std::size_t pos) {
        const auto last = heap_.size() - 1;
        locator_.set(heap_[pos].key, not_in_heap);
        if (pos != last) {
            heap_[pos] = std::move(heap_[last]);
            locator_.set(heap_[pos].key, pos);
            heap_.pop_back();
            // The replacement could need sifting in either direction.
            sift_up(pos);
            sift_down(pos);
        } else {
            heap_.pop_back();
        }
    }
};

template <class PriorityT, class Compare = std::less<PriorityT>>
class DenseIndexedHeap final
    : public IndexedHeap<std::size_t, PriorityT, DenseLocator, Compare> {
    using base = IndexedHeap<std::size_t, PriorityT, DenseLocator, Compare>;

public:
    DenseIndexedHeap() = default;
    explicit DenseIndexedHeap(const std::size_t capacity)
        : base(DenseLocator(capacity)) {}

    void reset_capacity(const std::size_t capacity) {
        this->clear();
        this->locator().reset_capacity(capacity);
    }
};

template <
    class KeyT,
    class PriorityT,
    class Hash = std::hash<KeyT>,
    class Compare = std::less<PriorityT>
>
class SparseIndexedHeap final
    : public IndexedHeap<KeyT, PriorityT, SparseLocator<KeyT, Hash>, Compare> {
    using base = IndexedHeap<KeyT, PriorityT, SparseLocator<KeyT, Hash>, Compare>;

public:
    SparseIndexedHeap() = default;

    void reserve(const std::size_t expected_keys) {
        this->locator().reserve(expected_keys);
    }
};

} // namespace bioimage_cpp::detail
