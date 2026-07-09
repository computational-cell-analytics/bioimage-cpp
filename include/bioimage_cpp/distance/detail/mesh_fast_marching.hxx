#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/indexed_heap.hxx"
#include "bioimage_cpp/detail/profile.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// First-order fast marching method on a triangle mesh (Kimmel & Sethian, PNAS
// 1998): the mesh backend for the geodesic-distance functions. See
// geodesic_mesh.hxx for the public API and development/distance/ for the exact
// (pygeodesic) reference oracle.
//
// The per-triangle update solves for a linear arrival-time field with gradient
// magnitude equal to the local slowness, and accepts it only when the
// characteristic falls inside the triangle wedge (otherwise it falls back to
// the two edge updates). Obtuse-angle "unfolding" is not done yet, so accuracy
// is first-order and degrades on very obtuse triangulations.

namespace bioimage_cpp::distance::detail {

inline constexpr double kMeshFmmInfinity = std::numeric_limits<double>::infinity();

class MeshFastMarching {
public:
    MeshFastMarching(
        const ConstArrayView<double> &vertices,
        const ConstArrayView<std::int64_t> &faces,
        const ConstArrayView<double> *speed
    )
        : vertices_(vertices.data),
          n_vertices_(static_cast<std::size_t>(vertices.shape[0])),
          faces_(faces.data),
          n_faces_(static_cast<std::size_t>(faces.shape[0])),
          speed_(speed != nullptr ? speed->data : nullptr),
          dist_(n_vertices_, kMeshFmmInfinity),
          state_(n_vertices_, MeshState::Far),
          heap_(n_vertices_) {
        build_vertex_face_incidence();
    }

    [[nodiscard]] std::size_t size() const { return n_vertices_; }
    [[nodiscard]] double distance(std::size_t vertex) const { return dist_[vertex]; }
    [[nodiscard]] const std::vector<double> &distances() const { return dist_; }

    // Run the marching front from the given source vertex indices. Results are
    // left in `dist_`; vertices in a different connected component keep +inf.
    void solve(const std::vector<std::size_t> &sources) {
        reset();
        BIOIMAGE_PROFILE_INIT(prof);
        for (const auto source : sources) {
            dist_[source] = 0.0;
            state_[source] = MeshState::Narrow;
            heap_.push_or_change(source, 0.0);
        }

        while (!heap_.empty()) {
            std::size_t v = 0;
            {
                BIOIMAGE_PROFILE_SCOPE(prof, "pop");
                const auto top = heap_.pop();
                v = top.key;
            }
            state_[v] = MeshState::Frozen;

            BIOIMAGE_PROFILE_SCOPE(prof, "relax");
            // Only the faces incident to the just-frozen v carry new frozen
            // information, so relax each of their other corners from that face
            // alone (using v, now frozen, as the/one frozen source) rather than
            // rescanning every incident face of w. This is exactly equivalent to
            // the previous update_vertex rescan: dist_[w] is a running minimum
            // and already reflects every other incident face's contribution from
            // the earlier freeze of that face's corner, so min(dist_[w], full
            // rescan) collapses to min(dist_[w], this-face candidate).
            for (std::size_t k = face_offsets_[v]; k < face_offsets_[v + 1]; ++k) {
                const std::size_t f = face_ids_[k];
                for (std::size_t corner = 0; corner < 3; ++corner) {
                    const std::size_t w = static_cast<std::size_t>(faces_[f * 3 + corner]);
                    if (w == v || state_[w] == MeshState::Frozen) {
                        continue;
                    }
                    // The two corners of f other than w, in face-storage order,
                    // matching the previous update_vertex so triangle_update sees
                    // identical operands (its acute update is only symmetric up
                    // to rounding, so operand order is load-bearing for
                    // bit-identical output).
                    std::array<std::size_t, 2> others{};
                    std::size_t count = 0;
                    for (std::size_t c2 = 0; c2 < 3; ++c2) {
                        const auto u = static_cast<std::size_t>(faces_[f * 3 + c2]);
                        if (u != w) {
                            others[count++] = u;
                        }
                    }
                    const std::size_t a = others[0];
                    const std::size_t b = others[1];
                    const bool fa = state_[a] == MeshState::Frozen;
                    const bool fb = state_[b] == MeshState::Frozen;
                    double t = kMeshFmmInfinity;
                    if (fa && fb) {
                        t = triangle_update(w, a, b);
                    } else if (fa) {
                        t = dist_[a] + slowness(w) * edge_length(w, a);
                    } else if (fb) {
                        t = dist_[b] + slowness(w) * edge_length(w, b);
                    }
                    if (t < dist_[w]) {
                        dist_[w] = t;
                        state_[w] = MeshState::Narrow;
                        heap_.push_or_change(w, t);
                    }
                }
            }
        }
        BIOIMAGE_PROFILE_REPORT(prof);
    }

private:
    enum class MeshState : std::uint8_t { Far = 0, Narrow = 1, Frozen = 2 };

    void reset() {
        std::fill(dist_.begin(), dist_.end(), kMeshFmmInfinity);
        std::fill(state_.begin(), state_.end(), MeshState::Far);
        heap_.clear();
    }

    // Build the vertex -> incident-faces CSR (offsets + face ids), mirroring the
    // count-then-scatter pattern in detail::mesh_smoothing::build_adjacency.
    void build_vertex_face_incidence() {
        face_offsets_.assign(n_vertices_ + 1, 0);
        for (std::size_t f = 0; f < n_faces_; ++f) {
            for (std::size_t corner = 0; corner < 3; ++corner) {
                const std::int64_t v = faces_[f * 3 + corner];
                if (v < 0 || static_cast<std::size_t>(v) >= n_vertices_) {
                    throw std::invalid_argument(
                        "faces contains vertex index " + std::to_string(v) +
                        " out of range [0, " + std::to_string(n_vertices_) + ")"
                    );
                }
                ++face_offsets_[static_cast<std::size_t>(v) + 1];
            }
        }
        for (std::size_t i = 1; i < face_offsets_.size(); ++i) {
            face_offsets_[i] += face_offsets_[i - 1];
        }
        face_ids_.resize(face_offsets_.back());
        std::vector<std::size_t> insert_pos(face_offsets_.begin(), face_offsets_.end() - 1);
        for (std::size_t f = 0; f < n_faces_; ++f) {
            for (std::size_t corner = 0; corner < 3; ++corner) {
                const auto v = static_cast<std::size_t>(faces_[f * 3 + corner]);
                face_ids_[insert_pos[v]++] = f;
            }
        }
    }

    [[nodiscard]] double slowness(std::size_t vertex) const {
        if (speed_ == nullptr) {
            return 1.0;
        }
        const double f = speed_[vertex];
        return (f > 0.0) ? 1.0 / f : kMeshFmmInfinity;
    }

    [[nodiscard]] double edge_length(std::size_t u, std::size_t v) const {
        const double *pu = vertices_ + u * 3;
        const double *pv = vertices_ + v * 3;
        const double dx = pu[0] - pv[0];
        const double dy = pu[1] - pv[1];
        const double dz = pu[2] - pv[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    // Kimmel-Sethian planar-wavefront update of vertex c from the triangle
    // (c, a, b) with a, b frozen. Returns the accepted arrival time, or the
    // better of the two edge updates when the acute update is not causal.
    double triangle_update(std::size_t c, std::size_t a, std::size_t b) {
        const double s = slowness(c);
        const double len_a = edge_length(c, b);  // |CB| pairs with T_B
        const double len_b = edge_length(c, a);  // |CA| pairs with T_A
        const double edge_cand =
            std::min(dist_[a] + s * len_b, dist_[b] + s * len_a);
        if (!(len_a > 0.0) || !(len_b > 0.0) || std::isinf(s)) {
            return edge_cand;
        }

        const double *pc = vertices_ + c * 3;
        const double *pa = vertices_ + a * 3;
        const double *pb = vertices_ + b * 3;
        double dot = 0.0;
        for (std::size_t d = 0; d < 3; ++d) {
            dot += (pa[d] - pc[d]) * (pb[d] - pc[d]);
        }
        const double cos_theta = dot / (len_a * len_b);
        const double sin2 = std::max(0.0, 1.0 - cos_theta * cos_theta);

        const double ta = dist_[a];
        const double tb = dist_[b];
        const double aa = len_a;  // |CB|
        const double bb = len_b;  // |CA|

        // F t^2 - 2 K t + L = 0 for the linear arrival-time field over the
        // triangle with |grad T| = s (see header note for the derivation).
        const double F = aa * aa + bb * bb - 2.0 * aa * bb * cos_theta;
        const double K = aa * aa * ta - aa * bb * cos_theta * (ta + tb) + bb * bb * tb;
        const double L = aa * aa * ta * ta - 2.0 * aa * bb * cos_theta * ta * tb +
                         bb * bb * tb * tb - s * s * aa * aa * bb * bb * sin2;
        const double disc = K * K - F * L;
        if (F > 0.0 && disc >= 0.0) {
            const double t = (K + std::sqrt(disc)) / F;
            if (t >= ta && t >= tb) {
                // Causality: the characteristic -grad T must point into the
                // wedge, i.e. both barycentric weights are non-negative.
                const double lam = aa * aa * (t - ta) - aa * bb * cos_theta * (t - tb);
                const double mu = bb * bb * (t - tb) - aa * bb * cos_theta * (t - ta);
                if (lam >= 0.0 && mu >= 0.0) {
                    return std::min(edge_cand, t);
                }
            }
        }
        return edge_cand;
    }

    const double *vertices_;
    std::size_t n_vertices_;
    const std::int64_t *faces_;
    std::size_t n_faces_;
    const double *speed_;

    std::vector<std::size_t> face_offsets_;
    std::vector<std::size_t> face_ids_;

    std::vector<double> dist_;
    std::vector<MeshState> state_;
    bioimage_cpp::detail::DenseIndexedHeap<double, std::greater<double>> heap_;
};

} // namespace bioimage_cpp::distance::detail
