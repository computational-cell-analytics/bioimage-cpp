#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bioimage_cpp {

using Coordinate = std::int64_t;
using CoordinateVector = std::vector<Coordinate>;

class Block {
public:
    Block() = default;

    Block(CoordinateVector begin, CoordinateVector end)
        : begin_(std::move(begin)), end_(std::move(end)) {
        if (begin_.size() != end_.size()) {
            throw std::invalid_argument("block begin and end must have the same length");
        }
        for (std::size_t axis = 0; axis < begin_.size(); ++axis) {
            if (end_[axis] < begin_[axis]) {
                throw std::invalid_argument("block end must be >= begin for every axis");
            }
        }
    }

    const CoordinateVector &begin() const {
        return begin_;
    }

    const CoordinateVector &end() const {
        return end_;
    }

    CoordinateVector shape() const {
        CoordinateVector result(begin_.size());
        for (std::size_t axis = 0; axis < begin_.size(); ++axis) {
            result[axis] = end_[axis] - begin_[axis];
        }
        return result;
    }

    std::size_t ndim() const {
        return begin_.size();
    }

private:
    CoordinateVector begin_;
    CoordinateVector end_;
};

class BlockWithHalo {
public:
    BlockWithHalo() = default;

    BlockWithHalo(Block outer_block, Block inner_block)
        : outer_block_(std::move(outer_block)),
          inner_block_(std::move(inner_block)),
          inner_block_local_() {
        if (outer_block_.ndim() != inner_block_.ndim()) {
            throw std::invalid_argument("outer_block and inner_block must have the same ndim");
        }

        CoordinateVector local_begin(inner_block_.ndim());
        CoordinateVector local_end(inner_block_.ndim());
        const auto inner_shape = inner_block_.shape();
        for (std::size_t axis = 0; axis < inner_block_.ndim(); ++axis) {
            local_begin[axis] = inner_block_.begin()[axis] - outer_block_.begin()[axis];
            local_end[axis] = local_begin[axis] + inner_shape[axis];
        }
        inner_block_local_ = Block(std::move(local_begin), std::move(local_end));
    }

    const Block &outer_block() const {
        return outer_block_;
    }

    const Block &inner_block() const {
        return inner_block_;
    }

    const Block &inner_block_local() const {
        return inner_block_local_;
    }

private:
    Block outer_block_;
    Block inner_block_;
    Block inner_block_local_;
};

struct LocalOverlaps {
    CoordinateVector overlap_begin_a;
    CoordinateVector overlap_end_a;
    CoordinateVector overlap_begin_b;
    CoordinateVector overlap_end_b;
};

class Blocking {
public:
    Blocking(
        CoordinateVector roi_begin,
        CoordinateVector roi_end,
        CoordinateVector block_shape,
        CoordinateVector block_shift = {}
    )
        : roi_begin_(std::move(roi_begin)),
          roi_end_(std::move(roi_end)),
          block_shape_(std::move(block_shape)),
          block_shift_(std::move(block_shift)),
          blocks_per_axis_(),
          blocks_per_axis_strides_(),
          number_of_blocks_(1) {
        validate_constructor_arguments();

        const auto ndim = roi_begin_.size();
        blocks_per_axis_.assign(ndim, 0);
        blocks_per_axis_strides_.assign(ndim, 1);

        for (std::size_t axis = 0; axis < ndim; ++axis) {
            const auto dim_size = roi_end_[axis] - (roi_begin_[axis] - block_shift_[axis]);
            blocks_per_axis_[axis] = ceil_div(dim_size, block_shape_[axis]);
            number_of_blocks_ *= static_cast<std::uint64_t>(blocks_per_axis_[axis]);
        }

        for (std::ptrdiff_t axis = static_cast<std::ptrdiff_t>(ndim) - 2; axis >= 0; --axis) {
            const auto index = static_cast<std::size_t>(axis);
            blocks_per_axis_strides_[index] =
                blocks_per_axis_strides_[index + 1] * blocks_per_axis_[index + 1];
        }
    }

    const CoordinateVector &roi_begin() const {
        return roi_begin_;
    }

    const CoordinateVector &roi_end() const {
        return roi_end_;
    }

    const CoordinateVector &block_shape() const {
        return block_shape_;
    }

    const CoordinateVector &block_shift() const {
        return block_shift_;
    }

    const CoordinateVector &blocks_per_axis() const {
        return blocks_per_axis_;
    }

    std::uint64_t number_of_blocks() const {
        return number_of_blocks_;
    }

    CoordinateVector block_grid_position(const std::uint64_t block_id) const {
        require_valid_block_id(block_id);
        CoordinateVector result(ndim());
        for (std::size_t axis = 0; axis < ndim(); ++axis) {
            result[axis] = block_axis_position(block_id, axis);
        }
        return result;
    }

    std::int64_t get_neighbor_id(
        const std::uint64_t block_id,
        const std::size_t axis,
        const bool lower
    ) const {
        require_valid_block_id(block_id);
        require_valid_axis(axis);

        const auto position = block_axis_position(block_id, axis);
        if ((lower && position == 0) || (!lower && position == blocks_per_axis_[axis] - 1)) {
            return -1;
        }

        const auto stride = blocks_per_axis_strides_[axis];
        return static_cast<std::int64_t>(block_id) + (lower ? -stride : stride);
    }

    Block get_block(const std::uint64_t block_id) const {
        require_valid_block_id(block_id);

        CoordinateVector begin_coord(ndim());
        CoordinateVector end_coord(ndim());
        auto index = block_id;
        for (std::size_t axis = 0; axis < ndim(); ++axis) {
            const auto block_coord = static_cast<Coordinate>(index / blocks_per_axis_strides_[axis]);
            index -= static_cast<std::uint64_t>(block_coord) * blocks_per_axis_strides_[axis];

            const auto begin_at_axis =
                (roi_begin_[axis] - block_shift_[axis]) + block_coord * block_shape_[axis];
            begin_coord[axis] = std::max(begin_at_axis, roi_begin_[axis]);
            end_coord[axis] = std::min(begin_at_axis + block_shape_[axis], roi_end_[axis]);
        }
        return Block(std::move(begin_coord), std::move(end_coord));
    }

    BlockWithHalo get_block_with_halo(
        const std::uint64_t block_id,
        const CoordinateVector &halo_begin,
        const CoordinateVector &halo_end
    ) const {
        const auto inner_block = get_block(block_id);
        return add_halo(inner_block, halo_begin, halo_end);
    }

    BlockWithHalo get_block_with_halo(
        const std::uint64_t block_id,
        const CoordinateVector &halo
    ) const {
        return get_block_with_halo(block_id, halo, halo);
    }

    BlockWithHalo add_halo(
        const Block &inner_block,
        const CoordinateVector &halo_begin,
        const CoordinateVector &halo_end
    ) const {
        require_vector_length(halo_begin, "halo_begin");
        require_vector_length(halo_end, "halo_end");
        if (inner_block.ndim() != ndim()) {
            throw std::invalid_argument("inner_block ndim must match blocking ndim");
        }

        CoordinateVector outer_begin(ndim());
        CoordinateVector outer_end(ndim());
        for (std::size_t axis = 0; axis < ndim(); ++axis) {
            if (halo_begin[axis] < 0 || halo_end[axis] < 0) {
                throw std::invalid_argument("halo values must be non-negative");
            }
            outer_begin[axis] = std::max(inner_block.begin()[axis] - halo_begin[axis], roi_begin_[axis]);
            outer_end[axis] = std::min(inner_block.end()[axis] + halo_end[axis], roi_end_[axis]);
        }
        return BlockWithHalo(Block(std::move(outer_begin), std::move(outer_end)), inner_block);
    }

    BlockWithHalo add_halo(const Block &inner_block, const CoordinateVector &halo) const {
        return add_halo(inner_block, halo, halo);
    }

    std::uint64_t coordinates_to_block_id(const CoordinateVector &coordinates) const {
        require_vector_length(coordinates, "coordinates");

        std::uint64_t block_id = 0;
        for (std::size_t axis = 0; axis < ndim(); ++axis) {
            if (coordinates[axis] < roi_begin_[axis] || coordinates[axis] >= roi_end_[axis]) {
                throw std::out_of_range("coordinates must lie inside the half-open ROI");
            }
            const auto shifted = coordinates[axis] - (roi_begin_[axis] - block_shift_[axis]);
            const auto block_coord = shifted / block_shape_[axis];
            block_id += static_cast<std::uint64_t>(block_coord) * blocks_per_axis_strides_[axis];
        }
        return block_id;
    }

    std::vector<std::uint64_t> get_block_ids_in_bounding_box(
        const CoordinateVector &box_begin,
        const CoordinateVector &box_end
    ) const {
        require_box(box_begin, box_end);

        std::vector<std::uint64_t> result;
        for (std::uint64_t block_id = 0; block_id < number_of_blocks_; ++block_id) {
            const auto block = get_block(block_id);
            bool enclosed = true;
            for (std::size_t axis = 0; axis < ndim(); ++axis) {
                if (block.begin()[axis] < box_begin[axis] || block.end()[axis] > box_end[axis]) {
                    enclosed = false;
                    break;
                }
            }
            if (enclosed) {
                result.push_back(block_id);
            }
        }
        return result;
    }

    std::vector<std::uint64_t> get_block_ids_overlapping_bounding_box(
        const CoordinateVector &box_begin,
        const CoordinateVector &box_end
    ) const {
        require_box(box_begin, box_end);

        CoordinateVector first(ndim());
        CoordinateVector last(ndim());
        for (std::size_t axis = 0; axis < ndim(); ++axis) {
            const auto overlap_begin = std::max(box_begin[axis], roi_begin_[axis]);
            const auto overlap_end = std::min(box_end[axis], roi_end_[axis]);
            if (overlap_begin >= overlap_end) {
                return {};
            }
            first[axis] = block_axis_position_for_coordinate(overlap_begin, axis);
            last[axis] = block_axis_position_for_coordinate(overlap_end - 1, axis);
        }

        std::vector<std::uint64_t> result;
        CoordinateVector position = first;
        while (true) {
            std::uint64_t block_id = 0;
            for (std::size_t axis = 0; axis < ndim(); ++axis) {
                block_id += static_cast<std::uint64_t>(position[axis]) * blocks_per_axis_strides_[axis];
            }
            result.push_back(block_id);

            std::ptrdiff_t axis = static_cast<std::ptrdiff_t>(ndim()) - 1;
            for (; axis >= 0; --axis) {
                const auto index = static_cast<std::size_t>(axis);
                if (position[index] < last[index]) {
                    ++position[index];
                    for (std::size_t reset_axis = index + 1; reset_axis < ndim(); ++reset_axis) {
                        position[reset_axis] = first[reset_axis];
                    }
                    break;
                }
            }
            if (axis < 0) {
                break;
            }
        }
        return result;
    }

    std::optional<LocalOverlaps> get_local_overlaps(
        const std::uint64_t block_a_id,
        const std::uint64_t block_b_id,
        const CoordinateVector &block_halo
    ) const {
        require_vector_length(block_halo, "block_halo");

        const auto block_a = get_block_with_halo(block_a_id, block_halo).outer_block();
        const auto block_b = get_block_with_halo(block_b_id, block_halo).outer_block();

        CoordinateVector global_begin(ndim());
        CoordinateVector global_end(ndim());
        for (std::size_t axis = 0; axis < ndim(); ++axis) {
            global_begin[axis] = std::max(block_a.begin()[axis], block_b.begin()[axis]);
            global_end[axis] = std::min(block_a.end()[axis], block_b.end()[axis]);
            if (global_begin[axis] >= global_end[axis]) {
                return std::nullopt;
            }
        }

        LocalOverlaps overlaps;
        overlaps.overlap_begin_a.resize(ndim());
        overlaps.overlap_end_a.resize(ndim());
        overlaps.overlap_begin_b.resize(ndim());
        overlaps.overlap_end_b.resize(ndim());
        for (std::size_t axis = 0; axis < ndim(); ++axis) {
            overlaps.overlap_begin_a[axis] = global_begin[axis] - block_a.begin()[axis];
            overlaps.overlap_end_a[axis] = global_end[axis] - block_a.begin()[axis];
            overlaps.overlap_begin_b[axis] = global_begin[axis] - block_b.begin()[axis];
            overlaps.overlap_end_b[axis] = global_end[axis] - block_b.begin()[axis];
        }
        return overlaps;
    }

    std::vector<std::uint64_t> get_block_ids_in_slice(
        const Coordinate z,
        const CoordinateVector &block_halo
    ) const {
        require_vector_length(block_halo, "block_halo");
        if (ndim() == 0) {
            return {};
        }

        std::vector<std::uint64_t> result;
        for (std::uint64_t block_id = 0; block_id < number_of_blocks_; ++block_id) {
            const auto block = get_block_with_halo(block_id, block_halo).outer_block();
            if (z >= block.begin()[0] && z < block.end()[0]) {
                result.push_back(block_id);
            }
        }
        return result;
    }

    std::size_t ndim() const {
        return roi_begin_.size();
    }

private:
    static Coordinate ceil_div(const Coordinate numerator, const Coordinate denominator) {
        return numerator == 0 ? 0 : (numerator - 1) / denominator + 1;
    }

    void validate_constructor_arguments() {
        if (roi_begin_.empty()) {
            throw std::invalid_argument("roi_begin must not be empty");
        }
        if (roi_end_.size() != roi_begin_.size()) {
            throw std::invalid_argument("roi_end length must match roi_begin length");
        }
        if (block_shape_.size() != roi_begin_.size()) {
            throw std::invalid_argument("block_shape length must match roi_begin length");
        }
        if (block_shift_.empty()) {
            block_shift_.assign(roi_begin_.size(), 0);
        }
        if (block_shift_.size() != roi_begin_.size()) {
            throw std::invalid_argument("block_shift length must match roi_begin length");
        }

        for (std::size_t axis = 0; axis < roi_begin_.size(); ++axis) {
            if (roi_end_[axis] < roi_begin_[axis]) {
                throw std::invalid_argument("roi_end must be >= roi_begin for every axis");
            }
            if (block_shape_[axis] <= 0) {
                throw std::invalid_argument("block_shape values must be positive");
            }
            if (block_shift_[axis] < 0) {
                throw std::invalid_argument("block_shift values must be non-negative");
            }
        }
    }

    void require_vector_length(const CoordinateVector &coordinates, const char *name) const {
        if (coordinates.size() != ndim()) {
            throw std::invalid_argument(
                std::string(name) + " length must match blocking ndim"
            );
        }
    }

    void require_box(const CoordinateVector &box_begin, const CoordinateVector &box_end) const {
        require_vector_length(box_begin, "box_begin");
        require_vector_length(box_end, "box_end");
        for (std::size_t axis = 0; axis < ndim(); ++axis) {
            if (box_end[axis] < box_begin[axis]) {
                throw std::invalid_argument("box_end must be >= box_begin for every axis");
            }
        }
    }

    void require_valid_axis(const std::size_t axis) const {
        if (axis >= ndim()) {
            throw std::out_of_range("axis is out of range");
        }
    }

    void require_valid_block_id(const std::uint64_t block_id) const {
        if (block_id >= number_of_blocks_) {
            throw std::out_of_range("block_id is out of range");
        }
    }

    Coordinate block_axis_position(const std::uint64_t block_id, const std::size_t axis) const {
        require_valid_axis(axis);
        auto index = block_id;
        Coordinate position = 0;
        for (std::size_t current_axis = 0; current_axis <= axis; ++current_axis) {
            position = static_cast<Coordinate>(index / blocks_per_axis_strides_[current_axis]);
            index -= static_cast<std::uint64_t>(position) * blocks_per_axis_strides_[current_axis];
        }
        return position;
    }

    Coordinate block_axis_position_for_coordinate(
        const Coordinate coordinate,
        const std::size_t axis
    ) const {
        const auto shifted = coordinate - (roi_begin_[axis] - block_shift_[axis]);
        auto position = shifted / block_shape_[axis];
        position = std::max<Coordinate>(0, position);
        position = std::min(position, blocks_per_axis_[axis] - 1);
        return position;
    }

    CoordinateVector roi_begin_;
    CoordinateVector roi_end_;
    CoordinateVector block_shape_;
    CoordinateVector block_shift_;
    CoordinateVector blocks_per_axis_;
    std::vector<std::uint64_t> blocks_per_axis_strides_;
    std::uint64_t number_of_blocks_;
};

} // namespace bioimage_cpp
