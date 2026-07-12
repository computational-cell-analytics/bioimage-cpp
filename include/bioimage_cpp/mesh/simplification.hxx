#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/edge_hash.hxx"
#include "bioimage_cpp/detail/indexed_heap.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bioimage_cpp::mesh {

template <class V, class S>
struct SimplifyMeshResult {
    std::vector<V> vertices;
    std::vector<std::int64_t> faces;
    std::vector<V> normals;
    std::optional<std::vector<S>> values;
};

namespace detail::simplification {

using NodeId = std::uint64_t;
using FaceId = std::size_t;
using Edge = bioimage_cpp::detail::Edge;
using EdgeHash = bioimage_cpp::detail::EdgeHash;
using Vec3 = std::array<double, 3>;
using Triangle = std::array<NodeId, 3>;

inline Vec3 subtract(const Vec3 &a, const Vec3 &b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline Vec3 add_scaled(const Vec3 &a, const Vec3 &b, const double t) {
    return {a[0] + t * b[0], a[1] + t * b[1], a[2] + t * b[2]};
}

inline double dot(const Vec3 &a, const Vec3 &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

inline double norm(const Vec3 &v) {
    return std::sqrt(dot(v, v));
}

struct Quadric {
    std::array<double, 16> matrix{};

    Quadric &operator+=(const Quadric &other) {
        for (std::size_t i = 0; i < matrix.size(); ++i) {
            matrix[i] += other.matrix[i];
        }
        return *this;
    }

    void add_scaled(const Quadric &other, const double scale) {
        for (std::size_t i = 0; i < matrix.size(); ++i) {
            matrix[i] += scale * other.matrix[i];
        }
    }

    [[nodiscard]] double evaluate(const Vec3 &point) const {
        const std::array<double, 4> h{point[0], point[1], point[2], 1.0};
        double value = 0.0;
        for (std::size_t row = 0; row < 4; ++row) {
            for (std::size_t col = 0; col < 4; ++col) {
                value += h[row] * matrix[row * 4 + col] * h[col];
            }
        }
        return value;
    }

    [[nodiscard]] double bilinear(
        const std::array<double, 4> &a,
        const std::array<double, 4> &b
    ) const {
        double value = 0.0;
        for (std::size_t row = 0; row < 4; ++row) {
            for (std::size_t col = 0; col < 4; ++col) {
                value += a[row] * matrix[row * 4 + col] * b[col];
            }
        }
        return value;
    }
};

inline Quadric operator+(Quadric lhs, const Quadric &rhs) {
    lhs += rhs;
    return lhs;
}

inline Quadric plane_quadric(const Vec3 &unit_normal, const double d, const double weight) {
    const std::array<double, 4> plane{unit_normal[0], unit_normal[1], unit_normal[2], d};
    Quadric quadric;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            quadric.matrix[row * 4 + col] = weight * plane[row] * plane[col];
        }
    }
    return quadric;
}

inline Triangle triangle_key(Triangle triangle) {
    std::sort(triangle.begin(), triangle.end());
    return triangle;
}

struct TriangleHash {
    std::size_t operator()(const Triangle &triangle) const {
        std::size_t seed = 0;
        for (const auto value : triangle) {
            const auto item = static_cast<std::size_t>(value);
            seed ^= item + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        }
        return seed;
    }
};

struct Face {
    Triangle vertices{};
    bool alive = true;
};

struct Vertex {
    Vec3 position{};
    Quadric quadric;
    double value = 0.0;
    bool alive = true;
    std::unordered_set<FaceId> faces;
    std::unordered_set<NodeId> neighbours;
};

struct EdgeRecord {
    std::array<FaceId, 2> faces{};
    std::size_t count = 0;
};

struct FaceGeometry {
    Vec3 normal{};
    double area = 0.0;
    Quadric quadric;
};

struct Candidate {
    Vec3 position{};
    double cost = 0.0;
    double interpolation = 0.0;
};

inline bool solve_3x3(const Quadric &quadric, Vec3 &solution) {
    double augmented[3][4] = {
        {quadric.matrix[0], quadric.matrix[1], quadric.matrix[2], -quadric.matrix[3]},
        {quadric.matrix[4], quadric.matrix[5], quadric.matrix[6], -quadric.matrix[7]},
        {quadric.matrix[8], quadric.matrix[9], quadric.matrix[10], -quadric.matrix[11]},
    };
    double max_entry = 0.0;
    for (const auto &row : augmented) {
        for (std::size_t col = 0; col < 3; ++col) {
            max_entry = std::max(max_entry, std::abs(row[col]));
        }
    }
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() * max_entry;
    if (max_entry == 0.0) {
        return false;
    }

    for (std::size_t col = 0; col < 3; ++col) {
        std::size_t pivot = col;
        for (std::size_t row = col + 1; row < 3; ++row) {
            if (std::abs(augmented[row][col]) > std::abs(augmented[pivot][col])) {
                pivot = row;
            }
        }
        if (std::abs(augmented[pivot][col]) <= tolerance) {
            return false;
        }
        if (pivot != col) {
            for (std::size_t entry = col; entry < 4; ++entry) {
                std::swap(augmented[col][entry], augmented[pivot][entry]);
            }
        }
        const double divisor = augmented[col][col];
        for (std::size_t entry = col; entry < 4; ++entry) {
            augmented[col][entry] /= divisor;
        }
        for (std::size_t row = 0; row < 3; ++row) {
            if (row == col) continue;
            const double factor = augmented[row][col];
            for (std::size_t entry = col; entry < 4; ++entry) {
                augmented[row][entry] -= factor * augmented[col][entry];
            }
        }
    }
    solution = {augmented[0][3], augmented[1][3], augmented[2][3]};
    return std::isfinite(solution[0]) && std::isfinite(solution[1]) && std::isfinite(solution[2]);
}

template <class V, class I, class S>
class Simplifier {
public:
    Simplifier(
        const ConstArrayView<V> &vertices,
        const ConstArrayView<I> &faces,
        const ConstArrayView<S> *values,
        const double feature_angle,
        const double feature_weight
    ) : has_values_(values != nullptr), feature_angle_(feature_angle), feature_weight_(feature_weight) {
        initialize(vertices, faces, values);
    }

    SimplifyMeshResult<V, S> run(const std::size_t target_faces) {
        build_heap();
        while (alive_faces_ > target_faces && !heap_.empty()) {
            const Edge edge = heap_.pop().key;
            const auto candidate = make_candidate(edge);
            if (!candidate.has_value() || !collapse_is_valid(edge, candidate->position)) {
                continue;
            }
            collapse(edge, *candidate);
        }
        return finalize();
    }

private:
    using Priority = std::tuple<double, NodeId, NodeId>;
    using Heap = bioimage_cpp::detail::SparseIndexedHeap<
        Edge, Priority, EdgeHash, std::greater<Priority>
    >;

    std::vector<Vertex> vertices_;
    std::vector<Face> faces_;
    std::vector<FaceGeometry> face_geometry_;
    std::unordered_map<Edge, EdgeRecord, EdgeHash> edges_;
    std::unordered_map<Triangle, FaceId, TriangleHash> face_lookup_;
    Heap heap_;
    std::size_t alive_faces_ = 0;
    bool has_values_ = false;
    double feature_angle_ = 45.0;
    double feature_weight_ = 10.0;
    double area_tolerance_ = 0.0;

    void initialize(
        const ConstArrayView<V> &vertices,
        const ConstArrayView<I> &faces,
        const ConstArrayView<S> *values
    ) {
        if (vertices.shape.size() != 2 || vertices.shape[1] != 3) {
            throw std::invalid_argument("vertices must have shape (n_vertices, 3)");
        }
        if (faces.shape.size() != 2 || faces.shape[1] != 3) {
            throw std::invalid_argument("faces must have shape (n_faces, 3)");
        }
        const std::size_t n_vertices = static_cast<std::size_t>(vertices.shape[0]);
        const std::size_t n_faces = static_cast<std::size_t>(faces.shape[0]);
        if (n_vertices == 0 || n_faces == 0) {
            throw std::invalid_argument("vertices and faces must describe a non-empty mesh");
        }
        if (values != nullptr
            && (values->shape.size() != 1
                || static_cast<std::size_t>(values->shape[0]) != n_vertices)) {
            throw std::invalid_argument("values must have shape (n_vertices,)");
        }

        vertices_.resize(n_vertices);
        Vec3 lower{
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
        };
        Vec3 upper{
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
        };
        for (std::size_t vertex = 0; vertex < n_vertices; ++vertex) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const double coordinate = static_cast<double>(vertices.data[vertex * 3 + axis]);
                if (!std::isfinite(coordinate)) {
                    throw std::invalid_argument("vertices must contain only finite values");
                }
                vertices_[vertex].position[axis] = coordinate;
                lower[axis] = std::min(lower[axis], coordinate);
                upper[axis] = std::max(upper[axis], coordinate);
            }
            if (values != nullptr) {
                const double value = static_cast<double>(values->data[vertex]);
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("values must contain only finite values");
                }
                vertices_[vertex].value = value;
            }
        }
        const Vec3 diagonal = subtract(upper, lower);
        const double scale_squared = dot(diagonal, diagonal);
        if (scale_squared == 0.0) {
            throw std::invalid_argument("vertices must span a non-zero bounding box");
        }
        area_tolerance_ = 64.0 * std::numeric_limits<double>::epsilon() * scale_squared;

        faces_.resize(n_faces);
        face_geometry_.resize(n_faces);
        std::vector<bool> referenced(n_vertices, false);
        edges_.reserve(n_faces * 3 / 2 + 1);
        face_lookup_.reserve(n_faces);
        for (std::size_t face_id = 0; face_id < n_faces; ++face_id) {
            Triangle triangle{};
            for (std::size_t corner = 0; corner < 3; ++corner) {
                const auto raw = faces.data[face_id * 3 + corner];
                if constexpr (std::is_signed_v<I>) {
                    if (raw < 0) {
                        throw std::invalid_argument("faces contains a negative vertex index");
                    }
                }
                const auto vertex = static_cast<NodeId>(raw);
                if (vertex >= n_vertices) {
                    throw std::invalid_argument("faces contains a vertex index outside [0, n_vertices)");
                }
                triangle[corner] = vertex;
                referenced[static_cast<std::size_t>(vertex)] = true;
            }
            if (triangle[0] == triangle[1] || triangle[0] == triangle[2]
                || triangle[1] == triangle[2]) {
                throw std::invalid_argument("faces must not contain repeated vertex indices");
            }
            const Triangle key = triangle_key(triangle);
            if (!face_lookup_.emplace(key, face_id).second) {
                throw std::invalid_argument("faces must not contain duplicate triangles");
            }
            faces_[face_id].vertices = triangle;
            vertices_[static_cast<std::size_t>(triangle[0])].faces.insert(face_id);
            vertices_[static_cast<std::size_t>(triangle[1])].faces.insert(face_id);
            vertices_[static_cast<std::size_t>(triangle[2])].faces.insert(face_id);
            face_geometry_[face_id] = geometry_of(triangle);
            if (face_geometry_[face_id].area * 2.0 <= area_tolerance_) {
                throw std::invalid_argument("faces must not contain zero-area triangles");
            }
            for (const auto vertex : triangle) {
                vertices_[static_cast<std::size_t>(vertex)].quadric += face_geometry_[face_id].quadric;
            }
            add_edge(triangle[0], triangle[1], face_id);
            add_edge(triangle[1], triangle[2], face_id);
            add_edge(triangle[2], triangle[0], face_id);
        }
        if (std::find(referenced.begin(), referenced.end(), false) != referenced.end()) {
            throw std::invalid_argument("every vertex must be referenced by at least one face");
        }
        alive_faces_ = n_faces;
        validate_edge_orientations();
        validate_vertex_links();
        add_feature_quadrics();
    }

    FaceGeometry geometry_of(const Triangle &triangle) const {
        const Vec3 &a = vertices_[static_cast<std::size_t>(triangle[0])].position;
        const Vec3 &b = vertices_[static_cast<std::size_t>(triangle[1])].position;
        const Vec3 &c = vertices_[static_cast<std::size_t>(triangle[2])].position;
        const Vec3 raw_normal = cross(subtract(b, a), subtract(c, a));
        const double length = norm(raw_normal);
        FaceGeometry geometry;
        geometry.area = 0.5 * length;
        if (length > 0.0) {
            geometry.normal = {
                raw_normal[0] / length,
                raw_normal[1] / length,
                raw_normal[2] / length,
            };
            const double d = -dot(geometry.normal, a);
            geometry.quadric = plane_quadric(geometry.normal, d, geometry.area);
        }
        return geometry;
    }

    void add_edge(const NodeId first, const NodeId second, const FaceId face_id) {
        const Edge key = bioimage_cpp::detail::edge_key(first, second);
        auto &record = edges_[key];
        if (record.count == 2) {
            throw std::invalid_argument("faces must form a 2-manifold: an edge has more than two faces");
        }
        record.faces[record.count++] = face_id;
        vertices_[static_cast<std::size_t>(first)].neighbours.insert(second);
        vertices_[static_cast<std::size_t>(second)].neighbours.insert(first);
    }

    void remove_edge(const NodeId first, const NodeId second, const FaceId face_id) {
        const Edge key = bioimage_cpp::detail::edge_key(first, second);
        auto edge_it = edges_.find(key);
        if (edge_it == edges_.end()) return;
        auto &record = edge_it->second;
        std::size_t position = record.count;
        for (std::size_t index = 0; index < record.count; ++index) {
            if (record.faces[index] == face_id) position = index;
        }
        if (position == record.count) return;
        record.faces[position] = record.faces[record.count - 1];
        --record.count;
        if (record.count == 0) {
            edges_.erase(edge_it);
            vertices_[static_cast<std::size_t>(first)].neighbours.erase(second);
            vertices_[static_cast<std::size_t>(second)].neighbours.erase(first);
        }
    }

    bool face_traverses_edge_forward(const Face &face, const Edge &edge) const {
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const NodeId first = face.vertices[corner];
            const NodeId second = face.vertices[(corner + 1) % 3];
            if (first == edge.first && second == edge.second) return true;
            if (first == edge.second && second == edge.first) return false;
        }
        return false;
    }

    void validate_edge_orientations() const {
        for (const auto &[edge, record] : edges_) {
            if (record.count == 2) {
                const bool first = face_traverses_edge_forward(faces_[record.faces[0]], edge);
                const bool second = face_traverses_edge_forward(faces_[record.faces[1]], edge);
                if (first == second) {
                    throw std::invalid_argument(
                        "faces must have consistent winding across shared edges"
                    );
                }
            }
        }
    }

    bool is_boundary_vertex(const NodeId vertex) const {
        for (const NodeId neighbour : vertices_[static_cast<std::size_t>(vertex)].neighbours) {
            const auto it = edges_.find(bioimage_cpp::detail::edge_key(vertex, neighbour));
            if (it != edges_.end() && it->second.count == 1) return true;
        }
        return false;
    }

    void validate_vertex_links() const {
        for (NodeId vertex = 0; vertex < vertices_.size(); ++vertex) {
            std::unordered_map<NodeId, std::vector<NodeId>> link;
            for (const FaceId face_id : vertices_[static_cast<std::size_t>(vertex)].faces) {
                const auto &triangle = faces_[face_id].vertices;
                NodeId first = 0;
                NodeId second = 0;
                bool found_first = false;
                for (const NodeId item : triangle) {
                    if (item == vertex) continue;
                    if (!found_first) {
                        first = item;
                        found_first = true;
                    } else {
                        second = item;
                    }
                }
                link[first].push_back(second);
                link[second].push_back(first);
            }
            std::size_t degree_one = 0;
            for (const auto &[_, adjacent] : link) {
                if (adjacent.size() == 1) ++degree_one;
                else if (adjacent.size() != 2) {
                    throw std::invalid_argument("faces must form a manifold around every vertex");
                }
            }
            const bool boundary = is_boundary_vertex(vertex);
            if ((!boundary && degree_one != 0) || (boundary && degree_one != 2)) {
                throw std::invalid_argument("faces must form one manifold fan around every vertex");
            }
            std::unordered_set<NodeId> visited;
            std::vector<NodeId> pending{link.begin()->first};
            while (!pending.empty()) {
                const NodeId current = pending.back();
                pending.pop_back();
                if (!visited.insert(current).second) continue;
                for (const NodeId adjacent : link.at(current)) pending.push_back(adjacent);
            }
            if (visited.size() != link.size()) {
                throw std::invalid_argument("faces must form one manifold fan around every vertex");
            }
        }
    }

    void add_feature_quadrics() {
        if (feature_weight_ == 0.0) return;
        const double radians = feature_angle_ * std::acos(-1.0) / 180.0;
        const double threshold = std::cos(radians);
        for (const auto &[edge, record] : edges_) {
            if (record.count != 2) continue;
            const auto &first = face_geometry_[record.faces[0]];
            const auto &second = face_geometry_[record.faces[1]];
            const double cosine = std::clamp(dot(first.normal, second.normal), -1.0, 1.0);
            if (cosine <= threshold) {
                Quadric penalty;
                penalty.add_scaled(first.quadric, feature_weight_);
                penalty.add_scaled(second.quadric, feature_weight_);
                vertices_[static_cast<std::size_t>(edge.first)].quadric += penalty;
                vertices_[static_cast<std::size_t>(edge.second)].quadric += penalty;
            }
        }
    }

    bool boundary_policy_allows(const Edge &edge) const {
        const auto edge_it = edges_.find(edge);
        if (edge_it == edges_.end()) return false;
        const bool first_boundary = is_boundary_vertex(edge.first);
        const bool second_boundary = is_boundary_vertex(edge.second);
        if (first_boundary != second_boundary) return false;
        if (first_boundary && edge_it->second.count != 1) return false;
        return true;
    }

    bool link_condition_holds(const Edge &edge) const {
        const auto edge_it = edges_.find(edge);
        if (edge_it == edges_.end()) return false;
        std::unordered_set<NodeId> expected;
        for (std::size_t index = 0; index < edge_it->second.count; ++index) {
            const auto &triangle = faces_[edge_it->second.faces[index]].vertices;
            for (const NodeId vertex : triangle) {
                if (vertex != edge.first && vertex != edge.second) expected.insert(vertex);
            }
        }
        std::unordered_set<NodeId> common;
        const auto &first_neighbours = vertices_[static_cast<std::size_t>(edge.first)].neighbours;
        const auto &second_neighbours = vertices_[static_cast<std::size_t>(edge.second)].neighbours;
        for (const NodeId neighbour : first_neighbours) {
            if (neighbour != edge.second && second_neighbours.contains(neighbour)) {
                common.insert(neighbour);
            }
        }
        return common == expected;
    }

    Candidate best_of_samples(const Quadric &quadric, const Edge &edge) const {
        const Vec3 &first = vertices_[static_cast<std::size_t>(edge.first)].position;
        const Vec3 &second = vertices_[static_cast<std::size_t>(edge.second)].position;
        const Vec3 midpoint = add_scaled(first, subtract(second, first), 0.5);
        const std::array<Vec3, 3> points{first, second, midpoint};
        const std::array<double, 3> parameters{0.0, 1.0, 0.5};
        std::size_t best = 0;
        double best_cost = quadric.evaluate(points[0]);
        for (std::size_t index = 1; index < points.size(); ++index) {
            const double cost = quadric.evaluate(points[index]);
            if (cost < best_cost) {
                best = index;
                best_cost = cost;
            }
        }
        return {points[best], std::max(0.0, best_cost), parameters[best]};
    }

    std::optional<Candidate> make_candidate(const Edge &edge) const {
        const auto edge_it = edges_.find(edge);
        if (edge_it == edges_.end() || !boundary_policy_allows(edge) || !link_condition_holds(edge)) {
            return std::nullopt;
        }
        const auto &first_vertex = vertices_[static_cast<std::size_t>(edge.first)];
        const auto &second_vertex = vertices_[static_cast<std::size_t>(edge.second)];
        const Quadric quadric = first_vertex.quadric + second_vertex.quadric;
        Candidate candidate;
        if (edge_it->second.count == 1) {
            const Vec3 direction = subtract(second_vertex.position, first_vertex.position);
            const std::array<double, 4> origin{
                first_vertex.position[0], first_vertex.position[1], first_vertex.position[2], 1.0,
            };
            const std::array<double, 4> delta{direction[0], direction[1], direction[2], 0.0};
            const double a = quadric.bilinear(delta, delta);
            const double b = 2.0 * quadric.bilinear(origin, delta);
            if (a > 64.0 * std::numeric_limits<double>::epsilon()) {
                candidate.interpolation = std::clamp(-b / (2.0 * a), 0.0, 1.0);
                candidate.position = add_scaled(
                    first_vertex.position, direction, candidate.interpolation
                );
                candidate.cost = std::max(0.0, quadric.evaluate(candidate.position));
            } else {
                candidate = best_of_samples(quadric, edge);
            }
        } else if (!solve_3x3(quadric, candidate.position)) {
            candidate = best_of_samples(quadric, edge);
        } else {
            const Vec3 direction = subtract(second_vertex.position, first_vertex.position);
            candidate.interpolation = std::clamp(
                dot(subtract(candidate.position, first_vertex.position), direction) / dot(direction, direction),
                0.0,
                1.0
            );
            candidate.cost = std::max(0.0, quadric.evaluate(candidate.position));
        }
        if (!std::isfinite(candidate.cost)) return std::nullopt;
        return candidate;
    }

    std::vector<FaceId> affected_faces(const Edge &edge) const {
        std::vector<FaceId> result;
        const auto &first = vertices_[static_cast<std::size_t>(edge.first)].faces;
        const auto &second = vertices_[static_cast<std::size_t>(edge.second)].faces;
        result.reserve(first.size() + second.size());
        result.insert(result.end(), first.begin(), first.end());
        result.insert(result.end(), second.begin(), second.end());
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }

    bool collapse_is_valid(const Edge &edge, const Vec3 &position) const {
        if (!boundary_policy_allows(edge) || !link_condition_holds(edge)) return false;
        const auto affected = affected_faces(edge);
        const std::unordered_set<FaceId> affected_set(affected.begin(), affected.end());
        std::unordered_set<Triangle, TriangleHash> new_faces;
        for (const FaceId face_id : affected) {
            const auto &face = faces_[face_id];
            const bool has_first = std::find(
                face.vertices.begin(), face.vertices.end(), edge.first
            ) != face.vertices.end();
            const bool has_second = std::find(
                face.vertices.begin(), face.vertices.end(), edge.second
            ) != face.vertices.end();
            if (has_first && has_second) continue;

            Triangle updated = face.vertices;
            for (NodeId &vertex : updated) {
                if (vertex == edge.second) vertex = edge.first;
            }
            Vec3 positions[3];
            for (std::size_t corner = 0; corner < 3; ++corner) {
                positions[corner] = updated[corner] == edge.first
                    ? position
                    : vertices_[static_cast<std::size_t>(updated[corner])].position;
            }
            const Vec3 new_normal = cross(
                subtract(positions[1], positions[0]), subtract(positions[2], positions[0])
            );
            const auto &old_normal = face_geometry_[face_id].normal;
            if (norm(new_normal) <= area_tolerance_ || dot(new_normal, old_normal) <= 0.0) {
                return false;
            }
            const Triangle key = triangle_key(updated);
            if (!new_faces.insert(key).second) return false;
            const auto existing = face_lookup_.find(key);
            if (existing != face_lookup_.end() && !affected_set.contains(existing->second)) {
                return false;
            }
        }
        return true;
    }

    void build_heap() {
        heap_.reserve(edges_.size());
        std::vector<typename Heap::Entry> entries;
        entries.reserve(edges_.size());
        for (const auto &[edge, _] : edges_) {
            const auto candidate = make_candidate(edge);
            if (candidate.has_value() && collapse_is_valid(edge, candidate->position)) {
                entries.push_back({edge, {candidate->cost, edge.first, edge.second}});
            }
        }
        heap_.build_heap(std::move(entries));
    }

    void erase_face_adjacency(const FaceId face_id) {
        const auto triangle = faces_[face_id].vertices;
        face_lookup_.erase(triangle_key(triangle));
        for (const NodeId vertex : triangle) {
            vertices_[static_cast<std::size_t>(vertex)].faces.erase(face_id);
        }
        remove_edge(triangle[0], triangle[1], face_id);
        remove_edge(triangle[1], triangle[2], face_id);
        remove_edge(triangle[2], triangle[0], face_id);
    }

    void add_face_adjacency(const FaceId face_id) {
        const auto triangle = faces_[face_id].vertices;
        face_lookup_.emplace(triangle_key(triangle), face_id);
        for (const NodeId vertex : triangle) {
            vertices_[static_cast<std::size_t>(vertex)].faces.insert(face_id);
        }
        add_edge(triangle[0], triangle[1], face_id);
        add_edge(triangle[1], triangle[2], face_id);
        add_edge(triangle[2], triangle[0], face_id);
    }

    void collapse(const Edge &edge, const Candidate &candidate) {
        std::unordered_set<NodeId> dirty_vertices;
        dirty_vertices.insert(edge.first);
        dirty_vertices.insert(edge.second);
        for (const NodeId neighbour : vertices_[static_cast<std::size_t>(edge.first)].neighbours) {
            dirty_vertices.insert(neighbour);
        }
        for (const NodeId neighbour : vertices_[static_cast<std::size_t>(edge.second)].neighbours) {
            dirty_vertices.insert(neighbour);
        }
        for (const NodeId vertex : dirty_vertices) {
            if (!vertices_[static_cast<std::size_t>(vertex)].alive) continue;
            for (const NodeId neighbour : vertices_[static_cast<std::size_t>(vertex)].neighbours) {
                heap_.erase(bioimage_cpp::detail::edge_key(vertex, neighbour));
            }
        }

        const auto affected = affected_faces(edge);
        for (const FaceId face_id : affected) erase_face_adjacency(face_id);

        auto &kept = vertices_[static_cast<std::size_t>(edge.first)];
        auto &removed = vertices_[static_cast<std::size_t>(edge.second)];
        kept.position = candidate.position;
        kept.quadric += removed.quadric;
        if (has_values_) {
            kept.value = (1.0 - candidate.interpolation) * kept.value
                + candidate.interpolation * removed.value;
        }
        removed.alive = false;
        removed.faces.clear();
        removed.neighbours.clear();

        for (const FaceId face_id : affected) {
            auto &face = faces_[face_id];
            bool has_first = false;
            bool has_second = false;
            for (const NodeId vertex : face.vertices) {
                has_first = has_first || vertex == edge.first;
                has_second = has_second || vertex == edge.second;
            }
            if (has_first && has_second) {
                face.alive = false;
                --alive_faces_;
                continue;
            }
            for (NodeId &vertex : face.vertices) {
                if (vertex == edge.second) vertex = edge.first;
            }
            face_geometry_[face_id] = geometry_of(face.vertices);
            add_face_adjacency(face_id);
        }

        dirty_vertices.erase(edge.second);
        for (const NodeId neighbour : kept.neighbours) dirty_vertices.insert(neighbour);
        for (const NodeId vertex : dirty_vertices) {
            if (!vertices_[static_cast<std::size_t>(vertex)].alive) continue;
            std::vector<NodeId> neighbours(
                vertices_[static_cast<std::size_t>(vertex)].neighbours.begin(),
                vertices_[static_cast<std::size_t>(vertex)].neighbours.end()
            );
            std::sort(neighbours.begin(), neighbours.end());
            for (const NodeId neighbour : neighbours) {
                const Edge candidate_edge = bioimage_cpp::detail::edge_key(vertex, neighbour);
                const auto next = make_candidate(candidate_edge);
                if (next.has_value() && collapse_is_valid(candidate_edge, next->position)) {
                    heap_.push_or_change(
                        candidate_edge,
                        {next->cost, candidate_edge.first, candidate_edge.second}
                    );
                } else {
                    heap_.erase(candidate_edge);
                }
            }
        }
    }

    SimplifyMeshResult<V, S> finalize() const {
        SimplifyMeshResult<V, S> result;
        std::vector<std::int64_t> compact(vertices_.size(), -1);
        std::size_t output_vertices = 0;
        for (std::size_t vertex = 0; vertex < vertices_.size(); ++vertex) {
            if (vertices_[vertex].alive && !vertices_[vertex].faces.empty()) {
                compact[vertex] = static_cast<std::int64_t>(output_vertices++);
                for (const double coordinate : vertices_[vertex].position) {
                    result.vertices.push_back(static_cast<V>(coordinate));
                }
                if (has_values_) {
                    if (!result.values.has_value()) result.values.emplace();
                    result.values->push_back(static_cast<S>(vertices_[vertex].value));
                }
            }
        }
        result.faces.reserve(alive_faces_ * 3);
        for (const auto &face : faces_) {
            if (!face.alive) continue;
            for (const NodeId vertex : face.vertices) {
                result.faces.push_back(compact[static_cast<std::size_t>(vertex)]);
            }
        }

        std::vector<Vec3> accumulated_normals(output_vertices, Vec3{0.0, 0.0, 0.0});
        for (std::size_t face = 0; face < result.faces.size() / 3; ++face) {
            Vec3 positions[3];
            for (std::size_t corner = 0; corner < 3; ++corner) {
                const auto vertex = static_cast<std::size_t>(result.faces[face * 3 + corner]);
                positions[corner] = {
                    static_cast<double>(result.vertices[vertex * 3]),
                    static_cast<double>(result.vertices[vertex * 3 + 1]),
                    static_cast<double>(result.vertices[vertex * 3 + 2]),
                };
            }
            const Vec3 normal = cross(
                subtract(positions[1], positions[0]), subtract(positions[2], positions[0])
            );
            for (std::size_t corner = 0; corner < 3; ++corner) {
                const auto vertex = static_cast<std::size_t>(result.faces[face * 3 + corner]);
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    accumulated_normals[vertex][axis] += normal[axis];
                }
            }
        }
        result.normals.resize(output_vertices * 3);
        for (std::size_t vertex = 0; vertex < output_vertices; ++vertex) {
            const Vec3 &normal = accumulated_normals[vertex];
            const double length = norm(normal);
            for (std::size_t axis = 0; axis < 3; ++axis) {
                result.normals[vertex * 3 + axis] = static_cast<V>(normal[axis] / length);
            }
        }
        return result;
    }
};

} // namespace detail::simplification

// Simplify an oriented, manifold triangle mesh with constrained quadric-error
// edge collapses. Topology and mesh boundaries are preserved; sharp features
// receive a configurable soft penalty. Optional scalar values are interpolated
// along each collapsed edge and vertex normals are recomputed from final faces.
template <class V, class I, class S = V>
SimplifyMeshResult<V, S> simplify_mesh(
    const ConstArrayView<V> &vertices,
    const ConstArrayView<I> &faces,
    const double reduction,
    const ConstArrayView<S> *values = nullptr,
    const double feature_angle = 45.0,
    const double feature_weight = 10.0
) {
    if (!std::isfinite(reduction) || reduction < 0.0 || reduction >= 1.0) {
        throw std::invalid_argument("reduction must be finite and in [0, 1)");
    }
    if (!std::isfinite(feature_angle) || feature_angle < 0.0 || feature_angle > 180.0) {
        throw std::invalid_argument("feature_angle must be finite and in [0, 180]");
    }
    if (!std::isfinite(feature_weight) || feature_weight < 0.0) {
        throw std::invalid_argument("feature_weight must be finite and non-negative");
    }
    if (faces.shape.size() != 2) {
        throw std::invalid_argument("faces must have shape (n_faces, 3)");
    }
    const auto n_faces = static_cast<std::size_t>(faces.shape[0]);
    const auto target = static_cast<std::size_t>(
        std::ceil((1.0 - reduction) * static_cast<double>(n_faces))
    );
    detail::simplification::Simplifier<V, I, S> simplifier(
        vertices, faces, values, feature_angle, feature_weight
    );
    return simplifier.run(target);
}

} // namespace bioimage_cpp::mesh
