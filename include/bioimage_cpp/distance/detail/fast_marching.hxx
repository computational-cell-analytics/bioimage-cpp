#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/indexed_heap.hxx"
#include "bioimage_cpp/detail/profile.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

// First-order Godunov fast marching method (Eikonal solver) on a regular grid.
// This is the mask backend for the geodesic-distance functions; it matches the
// scheme used by scikit-fmm (order=1). See geodesic_mask.hxx for the public API.

namespace bioimage_cpp::distance::detail {

inline constexpr double kFmmInfinity = std::numeric_limits<double>::infinity();

enum class FmmState : std::uint8_t { Far = 0, Narrow = 1, Frozen = 2 };

// Reusable workspace for repeated single-/multi-source solves on the same grid.
// `dist`/`state` and the narrow-band heap are allocated once and reset between
// solves so the pairwise driver can reuse one workspace per thread.
class GridFastMarching {
public:
    GridFastMarching(
        const ConstArrayView<std::uint8_t> &mask,
        const std::vector<double> &sampling,
        const ConstArrayView<double> *speed
    )
        : shape_(mask.shape),
          strides_(bioimage_cpp::detail::c_order_strides(mask.shape)),
          n_(bioimage_cpp::detail::number_of_elements(mask.shape)),
          ndim_(mask.shape.size()),
          mask_(mask.data),
          speed_(speed != nullptr ? speed->data : nullptr),
          dist_(n_, kFmmInfinity),
          state_(n_, FmmState::Far),
          heap_(n_) {
        h_ = sampling;
        inv_h2_.resize(ndim_);
        for (std::size_t d = 0; d < ndim_; ++d) {
            const double h = sampling[d];
            inv_h2_[d] = 1.0 / (h * h);
        }
        // Axis neighbours are exactly +-1 along one axis, so the flat-index
        // delta to a neighbour is +-strides_[d] (a single add), and its bounds
        // reduce to a per-axis compare against the popped voxel's coordinate.
        // We therefore decode the popped voxel once (coords_) instead of calling
        // valid_offset_target (div + mod per axis) ~2*ndim + (2*ndim)^2 times.
        coords_.resize(ndim_);
        nb_coords_.resize(ndim_);
        terms_.reserve(ndim_);
    }

    [[nodiscard]] std::size_t size() const { return n_; }
    [[nodiscard]] double distance(std::size_t index) const { return dist_[index]; }
    [[nodiscard]] const std::vector<double> &distances() const { return dist_; }

    // Run the marching front from the given in-mask source voxels (linear
    // indices). Sources outside the mask are ignored. Results are left in
    // `dist_`; unreachable / out-of-domain voxels keep +inf.
    void solve(const std::vector<std::size_t> &sources) {
        reset();
        BIOIMAGE_PROFILE_INIT(prof);
        for (const auto source : sources) {
            if (mask_[source] == 0) {
                continue;  // a source outside the domain cannot seed the front
            }
            dist_[source] = 0.0;
            state_[source] = FmmState::Narrow;
            heap_.push_or_change(source, 0.0);
        }

        while (!heap_.empty()) {
            std::size_t p = 0;
            {
                BIOIMAGE_PROFILE_SCOPE(prof, "pop");
                const auto top = heap_.pop();
                p = top.key;
            }
            state_[p] = FmmState::Frozen;

            BIOIMAGE_PROFILE_SCOPE(prof, "relax");
            // Decode p once; a neighbour's coords differ from p's in one axis
            // only, so nb_coords_ is p's coords with that single entry adjusted.
            bioimage_cpp::detail::coords_from_index(p, strides_, ndim_, coords_.data());
            for (std::size_t d = 0; d < ndim_; ++d) {
                nb_coords_[d] = coords_[d];
            }
            // Visit +axis then -axis for each axis in turn: the same neighbour
            // order (and thus heap-insertion order) as the previous offsets_ loop.
            for (std::size_t d = 0; d < ndim_; ++d) {
                const std::ptrdiff_t cd = coords_[d];
                for (int side = 0; side < 2; ++side) {
                    const std::ptrdiff_t ncd = (side == 0) ? cd + 1 : cd - 1;
                    if (ncd < 0 || ncd >= shape_[d]) {
                        continue;
                    }
                    const std::uint64_t nb = (side == 0)
                        ? p + static_cast<std::uint64_t>(strides_[d])
                        : p - static_cast<std::uint64_t>(strides_[d]);
                    if (mask_[nb] == 0 || state_[nb] == FmmState::Frozen) {
                        continue;
                    }
                    nb_coords_[d] = ncd;
                    const double t = solve_eikonal(nb, nb_coords_.data());
                    nb_coords_[d] = cd;
                    if (t < dist_[nb]) {
                        dist_[nb] = t;
                        state_[nb] = FmmState::Narrow;
                        heap_.push_or_change(nb, t);
                    }
                }
            }
        }
        BIOIMAGE_PROFILE_REPORT(prof);
    }

    // Write the first-order upwind gradient of the solved field into `grad`
    // (shape (*shape, ndim), row-major float32). Component i is dT/dx_i, pointing
    // toward increasing distance (away from the nearest source); it is 0 at
    // sources / local minima, background, and unreachable voxels. Following
    // -grad traces the geodesic back toward the source; ||grad|| ~= slowness.
    void write_gradient(ArrayView<float> &grad) const {
        std::vector<std::ptrdiff_t> coords(ndim_);
        for (std::size_t p = 0; p < n_; ++p) {
            float *out = grad.data + p * ndim_;
            const double tp = dist_[p];
            if (mask_[p] == 0 || !std::isfinite(tp)) {
                for (std::size_t d = 0; d < ndim_; ++d) {
                    out[d] = 0.0f;
                }
                continue;
            }
            bioimage_cpp::detail::coords_from_index(p, strides_, ndim_, coords.data());
            for (std::size_t d = 0; d < ndim_; ++d) {
                // +strides_[d] is the plus side, -strides_[d] the minus side.
                // Keep only strictly-upwind (smaller-valued, in-mask) neighbours.
                double t_plus = kFmmInfinity;
                double t_minus = kFmmInfinity;
                if (coords[d] + 1 < shape_[d]) {
                    const std::uint64_t nb = p + static_cast<std::uint64_t>(strides_[d]);
                    if (mask_[nb] != 0 && dist_[nb] < tp) {
                        t_plus = dist_[nb];
                    }
                }
                if (coords[d] > 0) {
                    const std::uint64_t nb = p - static_cast<std::uint64_t>(strides_[d]);
                    if (mask_[nb] != 0 && dist_[nb] < tp) {
                        t_minus = dist_[nb];
                    }
                }

                double g = 0.0;
                if (t_minus == kFmmInfinity && t_plus == kFmmInfinity) {
                    g = 0.0;  // source / local minimum: no upwind neighbour
                } else if (t_minus <= t_plus) {
                    g = (tp - t_minus) / h_[d];  // front from -i => dT/dx_i >= 0
                } else {
                    g = (t_plus - tp) / h_[d];    // front from +i => dT/dx_i <= 0
                }
                out[d] = static_cast<float>(g);
            }
        }
    }

private:
    struct Term {
        double u;       // smaller frozen neighbour value along an axis
        double inv_h2;  // 1 / spacing^2 for that axis
    };

    void reset() {
        std::fill(dist_.begin(), dist_.end(), kFmmInfinity);
        std::fill(state_.begin(), state_.end(), FmmState::Far);
        heap_.clear();
    }

    // First-order Godunov update at voxel p (assumed in-mask, not frozen).
    // `coords` are p's per-axis coordinates (p differs from the popped voxel in
    // one axis only, so the caller supplies them without a fresh decode).
    double solve_eikonal(std::uint64_t p, const std::ptrdiff_t *coords) {
        const double f = (speed_ != nullptr) ? speed_[p] : 1.0;
        if (!(f > 0.0)) {
            return kFmmInfinity;  // zero/negative speed is impassable
        }
        const double slowness = 1.0 / f;

        terms_.clear();
        for (std::size_t d = 0; d < ndim_; ++d) {
            double best = kFmmInfinity;
            const std::ptrdiff_t cd = coords[d];
            if (cd + 1 < shape_[d]) {
                const std::uint64_t nb = p + static_cast<std::uint64_t>(strides_[d]);
                if (state_[nb] == FmmState::Frozen) {
                    best = std::min(best, dist_[nb]);
                }
            }
            if (cd > 0) {
                const std::uint64_t nb = p - static_cast<std::uint64_t>(strides_[d]);
                if (state_[nb] == FmmState::Frozen) {
                    best = std::min(best, dist_[nb]);
                }
            }
            if (best != kFmmInfinity) {
                terms_.push_back({best, inv_h2_[d]});
            }
        }
        if (terms_.empty()) {
            return kFmmInfinity;
        }

        std::sort(terms_.begin(), terms_.end(), [](const Term &a, const Term &b) {
            return a.u < b.u;
        });

        // Incrementally include axis terms (ascending u) and solve the quadratic
        //   sum_w t^2 - 2 sum_wu t + (sum_wu2 - slowness^2) = 0
        // stopping once the candidate no longer exceeds the next unused u.
        const double s2 = slowness * slowness;
        double sum_w = 0.0, sum_wu = 0.0, sum_wu2 = 0.0;
        double t = kFmmInfinity;
        for (std::size_t k = 0; k < terms_.size(); ++k) {
            sum_w += terms_[k].inv_h2;
            sum_wu += terms_[k].inv_h2 * terms_[k].u;
            sum_wu2 += terms_[k].inv_h2 * terms_[k].u * terms_[k].u;

            const double disc = sum_wu * sum_wu - sum_w * (sum_wu2 - s2);
            if (disc < 0.0) {
                break;  // keep the candidate from fewer terms
            }
            const double candidate = (sum_wu + std::sqrt(disc)) / sum_w;
            t = candidate;
            if (k + 1 == terms_.size() || candidate <= terms_[k + 1].u) {
                break;
            }
        }
        return t;
    }

    std::vector<std::ptrdiff_t> shape_;
    std::vector<std::ptrdiff_t> strides_;
    std::size_t n_;
    std::size_t ndim_;
    const std::uint8_t *mask_;
    const double *speed_;
    std::vector<double> inv_h2_;
    std::vector<double> h_;
    std::vector<std::ptrdiff_t> coords_;      // scratch: popped voxel's coords
    std::vector<std::ptrdiff_t> nb_coords_;   // scratch: current neighbour's coords

    std::vector<double> dist_;
    std::vector<FmmState> state_;
    bioimage_cpp::detail::DenseIndexedHeap<double, std::greater<double>> heap_;
    std::vector<Term> terms_;
};

} // namespace bioimage_cpp::distance::detail
