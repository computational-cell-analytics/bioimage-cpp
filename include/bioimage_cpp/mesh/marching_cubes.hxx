#pragma once

// Marching Cubes 33 / Lewiner implementation.
//
// The MC33 lookup tables and the case-selection structure below are derived
// from scikit-image 0.26.0 (BSD-3-Clause), whose implementation credits the
// original algorithm by Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and
// Geovan Tavares, "Efficient implementation of Marching Cubes' cases with
// topological guarantees", Journal of Graphics Tools 8(2), 2003. The
// scikit-image BSD-3-Clause notice must accompany redistributions of this
// derived table data.

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/mesh/detail/mc33_luts.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bioimage_cpp::mesh {

enum class MarchingCubesMethod {
    Lewiner,
    Lorensen,
};

struct MarchingCubesResult {
    // All vector-valued arrays are flat C-order arrays with a trailing size-3
    // component axis. Vertices/normals use the reference kernel's x/y/z order;
    // the binding converts them to NumPy z/y/x order before returning them.
    std::vector<float> vertices;
    std::vector<std::int32_t> faces;
    std::vector<float> normals;
    std::vector<float> values;
};

namespace detail::marching_cubes {

constexpr double kEpsilon = std::numeric_limits<double>::epsilon();

class Lut {
public:
    explicit Lut(const detail::EncodedLut &encoded)
        : shape_(encoded.shape), values_(decode(encoded.data)) {}

    [[nodiscard]] int get1(const int i0) const {
        return values_[static_cast<std::size_t>(i0)];
    }

    [[nodiscard]] int get2(const int i0, const int i1) const {
        return values_[static_cast<std::size_t>(i0 * shape_[1] + i1)];
    }

    [[nodiscard]] int get3(const int i0, const int i1, const int i2) const {
        return values_[static_cast<std::size_t>((i0 * shape_[1] + i1) * shape_[2] + i2)];
    }

private:
    static int base64_value(const char value) {
        if (value >= 'A' && value <= 'Z') return value - 'A';
        if (value >= 'a' && value <= 'z') return value - 'a' + 26;
        if (value >= '0' && value <= '9') return value - '0' + 52;
        if (value == '+') return 62;
        if (value == '/') return 63;
        return -1;
    }

    static std::vector<std::int8_t> decode(const std::string_view encoded) {
        std::vector<std::int8_t> decoded;
        decoded.reserve(encoded.size() * 3 / 4);
        int accumulator = 0;
        int bits = -8;
        for (const char value : encoded) {
            if (value == '=') break;
            const int digit = base64_value(value);
            if (digit < 0) continue;
            accumulator = (accumulator << 6) | digit;
            bits += 6;
            if (bits >= 0) {
                decoded.push_back(static_cast<std::int8_t>((accumulator >> bits) & 0xff));
                bits -= 8;
            }
        }
        return decoded;
    }

    std::array<int, 3> shape_{};
    std::vector<std::int8_t> values_;
};

struct Luts {
    Lut cases_classic{detail::kCASESCLASSIC};
    Lut cases{detail::kCASES};
    Lut tiling1{detail::kTILING1};
    Lut tiling2{detail::kTILING2};
    Lut tiling3_1{detail::kTILING3_1};
    Lut tiling3_2{detail::kTILING3_2};
    Lut tiling4_1{detail::kTILING4_1};
    Lut tiling4_2{detail::kTILING4_2};
    Lut tiling5{detail::kTILING5};
    Lut tiling6_1_1{detail::kTILING6_1_1};
    Lut tiling6_1_2{detail::kTILING6_1_2};
    Lut tiling6_2{detail::kTILING6_2};
    Lut tiling7_1{detail::kTILING7_1};
    Lut tiling7_2{detail::kTILING7_2};
    Lut tiling7_3{detail::kTILING7_3};
    Lut tiling7_4_1{detail::kTILING7_4_1};
    Lut tiling7_4_2{detail::kTILING7_4_2};
    Lut tiling8{detail::kTILING8};
    Lut tiling9{detail::kTILING9};
    Lut tiling10_1_1{detail::kTILING10_1_1};
    Lut tiling10_1_1_alt{detail::kTILING10_1_1_};
    Lut tiling10_1_2{detail::kTILING10_1_2};
    Lut tiling10_2{detail::kTILING10_2};
    Lut tiling10_2_alt{detail::kTILING10_2_};
    Lut tiling11{detail::kTILING11};
    Lut tiling12_1_1{detail::kTILING12_1_1};
    Lut tiling12_1_1_alt{detail::kTILING12_1_1_};
    Lut tiling12_1_2{detail::kTILING12_1_2};
    Lut tiling12_2{detail::kTILING12_2};
    Lut tiling12_2_alt{detail::kTILING12_2_};
    Lut tiling13_1{detail::kTILING13_1};
    Lut tiling13_1_alt{detail::kTILING13_1_};
    Lut tiling13_2{detail::kTILING13_2};
    Lut tiling13_2_alt{detail::kTILING13_2_};
    Lut tiling13_3{detail::kTILING13_3};
    Lut tiling13_3_alt{detail::kTILING13_3_};
    Lut tiling13_4{detail::kTILING13_4};
    Lut tiling13_5_1{detail::kTILING13_5_1};
    Lut tiling13_5_2{detail::kTILING13_5_2};
    Lut tiling14{detail::kTILING14};
    Lut test3{detail::kTEST3};
    Lut test4{detail::kTEST4};
    Lut test6{detail::kTEST6};
    Lut test7{detail::kTEST7};
    Lut test10{detail::kTEST10};
    Lut test12{detail::kTEST12};
    Lut test13{detail::kTEST13};
    Lut subconfig13{detail::kSUBCONFIG13};
};

inline const Luts &luts() {
    static const Luts instance;
    return instance;
}

constexpr std::array<std::array<int, 2>, 12> kEdgeRelativeX{{
    {{0, 1}}, {{1, 1}}, {{1, 0}}, {{0, 0}}, {{0, 1}}, {{1, 1}},
    {{1, 0}}, {{0, 0}}, {{0, 0}}, {{1, 1}}, {{1, 1}}, {{0, 0}},
}};
constexpr std::array<std::array<int, 2>, 12> kEdgeRelativeY{{
    {{0, 0}}, {{0, 1}}, {{1, 1}}, {{1, 0}}, {{0, 0}}, {{0, 1}},
    {{1, 1}}, {{1, 0}}, {{0, 0}}, {{0, 0}}, {{1, 1}}, {{1, 1}},
}};
constexpr std::array<std::array<int, 2>, 12> kEdgeRelativeZ{{
    {{0, 0}}, {{0, 0}}, {{0, 0}}, {{0, 0}}, {{1, 1}}, {{1, 1}},
    {{1, 1}}, {{1, 1}}, {{0, 1}}, {{0, 1}}, {{0, 1}}, {{0, 1}},
}};

class Cell {
public:
    Cell(const int nx, const int ny)
        : nx_(nx), ny_(ny),
          face_layer1_(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * 4, -1),
          face_layer2_(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * 4, -1) {}

    void new_z_value() {
        face_layer1_.swap(face_layer2_);
        std::fill(face_layer2_.begin(), face_layer2_.end(), -1);
    }

    void set_cube(
        const double isovalue, const int x, const int y, const int z, const int step,
        const float v0, const float v1, const float v2, const float v3,
        const float v4, const float v5, const float v6, const float v7
    ) {
        x_ = x;
        y_ = y;
        z_ = z;
        step_ = step;
        v0_ = static_cast<double>(v0) - isovalue;
        v1_ = static_cast<double>(v1) - isovalue;
        v2_ = static_cast<double>(v2) - isovalue;
        v3_ = static_cast<double>(v3) - isovalue;
        v4_ = static_cast<double>(v4) - isovalue;
        v5_ = static_cast<double>(v5) - isovalue;
        v6_ = static_cast<double>(v6) - isovalue;
        v7_ = static_cast<double>(v7) - isovalue;
        index_ = (v0_ > 0.0 ? 1 : 0) | (v1_ > 0.0 ? 2 : 0) | (v2_ > 0.0 ? 4 : 0)
            | (v3_ > 0.0 ? 8 : 0) | (v4_ > 0.0 ? 16 : 0) | (v5_ > 0.0 ? 32 : 0)
            | (v6_ > 0.0 ? 64 : 0) | (v7_ > 0.0 ? 128 : 0);
        center_calculated_ = false;
        center_vertex_ = -1;
    }

    [[nodiscard]] int index() const { return index_; }
    [[nodiscard]] double v0() const { return v0_; }
    [[nodiscard]] double v1() const { return v1_; }
    [[nodiscard]] double v2() const { return v2_; }
    [[nodiscard]] double v3() const { return v3_; }
    [[nodiscard]] double v4() const { return v4_; }
    [[nodiscard]] double v5() const { return v5_; }
    [[nodiscard]] double v6() const { return v6_; }
    [[nodiscard]] double v7() const { return v7_; }

    void add_triangles(const Lut &lut, const int lut_index, const int n_triangles) {
        prepare_for_triangles();
        for (int triangle = 0; triangle < n_triangles; ++triangle) {
            for (int corner = 0; corner < 3; ++corner) {
                add_face_from_edge(lut.get2(lut_index, triangle * 3 + corner));
            }
        }
    }

    void add_triangles2(const Lut &lut, const int lut_index, const int lut_index2, const int n_triangles) {
        prepare_for_triangles();
        for (int triangle = 0; triangle < n_triangles; ++triangle) {
            for (int corner = 0; corner < 3; ++corner) {
                add_face_from_edge(lut.get3(lut_index, lut_index2, triangle * 3 + corner));
            }
        }
    }

    [[nodiscard]] MarchingCubesResult take_result() {
        for (std::size_t vertex = 0; vertex < values_.size(); ++vertex) {
            const std::size_t base = vertex * 3;
            const double length = std::sqrt(
                static_cast<double>(normals_[base]) * normals_[base]
                + static_cast<double>(normals_[base + 1]) * normals_[base + 1]
                + static_cast<double>(normals_[base + 2]) * normals_[base + 2]
            );
            const double scale = length == 0.0 ? 0.0 : 1.0 / length;
            normals_[base] = static_cast<float>(normals_[base] * scale);
            normals_[base + 1] = static_cast<float>(normals_[base + 1] * scale);
            normals_[base + 2] = static_cast<float>(normals_[base + 2] * scale);
        }
        return {
            std::move(vertices_),
            std::move(faces_),
            std::move(normals_),
            std::move(values_),
        };
    }

private:
    int add_vertex(const double x, const double y, const double z) {
        if (values_.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("marching cubes produced too many vertices for int32 faces");
        }
        const int index = static_cast<int>(values_.size());
        vertices_.push_back(static_cast<float>(x));
        vertices_.push_back(static_cast<float>(y));
        vertices_.push_back(static_cast<float>(z));
        normals_.insert(normals_.end(), {0.0F, 0.0F, 0.0F});
        values_.push_back(0.0F);
        return index;
    }

    void add_face(const int index) {
        faces_.push_back(static_cast<std::int32_t>(index));
        values_[static_cast<std::size_t>(index)] = std::max(
            values_[static_cast<std::size_t>(index)], static_cast<float>(vmax_)
        );
    }

    void add_gradient(const int vertex, const double x, const double y, const double z) {
        const std::size_t base = static_cast<std::size_t>(vertex) * 3;
        normals_[base] += static_cast<float>(x);
        normals_[base + 1] += static_cast<float>(y);
        normals_[base + 2] += static_cast<float>(z);
    }

    void add_gradient_from_corner(const int vertex, const int corner, const double strength) {
        const std::size_t base = static_cast<std::size_t>(corner) * 3;
        add_gradient(vertex, gradient_[base] * strength, gradient_[base + 1] * strength,
                     gradient_[base + 2] * strength);
    }

    std::pair<std::vector<std::int32_t> *, std::size_t> face_layer_index(int edge) {
        std::size_t cube = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(y_)
            + static_cast<std::size_t>(x_);
        std::size_t slot = 0;
        std::vector<std::int32_t> *layer = nullptr;
        if (edge < 8) {
            if (edge < 4) {
                layer = &face_layer1_;
            } else {
                edge -= 4;
                layer = &face_layer2_;
            }
            if (edge == 1) {
                cube += static_cast<std::size_t>(step_);
                slot = 1;
            } else if (edge == 2) {
                cube += static_cast<std::size_t>(nx_) * static_cast<std::size_t>(step_);
            } else if (edge == 3) {
                slot = 1;
            }
        } else {
            layer = &face_layer1_;
            slot = 2;
            if (edge == 9) cube += static_cast<std::size_t>(step_);
            else if (edge == 10) {
                cube += (static_cast<std::size_t>(nx_) + 1)
                    * static_cast<std::size_t>(step_);
            } else if (edge == 11) {
                cube += static_cast<std::size_t>(nx_) * static_cast<std::size_t>(step_);
            }
        }
        return {layer, 4 * cube + slot};
    }

    void add_face_from_edge(const int edge) {
        if (edge == 12) {
            if (!center_calculated_) calculate_center_vertex();
            if (center_vertex_ < 0) {
                center_vertex_ = add_vertex(center_x_, center_y_, center_z_);
            }
            add_face(center_vertex_);
            add_gradient(center_vertex_, center_gx_, center_gy_, center_gz_);
            return;
        }

        const int dx1 = kEdgeRelativeX[static_cast<std::size_t>(edge)][0];
        const int dx2 = kEdgeRelativeX[static_cast<std::size_t>(edge)][1];
        const int dy1 = kEdgeRelativeY[static_cast<std::size_t>(edge)][0];
        const int dy2 = kEdgeRelativeY[static_cast<std::size_t>(edge)][1];
        const int dz1 = kEdgeRelativeZ[static_cast<std::size_t>(edge)][0];
        const int dz2 = kEdgeRelativeZ[static_cast<std::size_t>(edge)][1];
        const int corner1 = dz1 * 4 + dy1 * 2 + dx1;
        const int corner2 = dz2 * 4 + dy2 * 2 + dx2;
        const double strength1 = 1.0 / (kEpsilon + std::abs(values_by_corner_[corner1]));
        const double strength2 = 1.0 / (kEpsilon + std::abs(values_by_corner_[corner2]));
        const auto [layer, layer_index] = face_layer_index(edge);
        int vertex = (*layer)[layer_index];
        if (vertex < 0) {
            const double weight = strength1 + strength2;
            vertex = add_vertex(
                x_ + step_ * (dx1 * strength1 + dx2 * strength2) / weight,
                y_ + step_ * (dy1 * strength1 + dy2 * strength2) / weight,
                z_ + step_ * (dz1 * strength1 + dz2 * strength2) / weight
            );
            (*layer)[layer_index] = vertex;
        }
        add_face(vertex);
        add_gradient_from_corner(vertex, corner1, strength1);
        add_gradient_from_corner(vertex, corner2, strength2);
    }

    void prepare_for_triangles() {
        values_by_corner_ = {v0_, v1_, v3_, v2_, v4_, v5_, v7_, v6_};
        double minimum = 0.0;
        double maximum = 0.0;
        for (const double value : values_by_corner_) {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        vmax_ = maximum - minimum;
        auto set_gradient = [this](const int corner, const double x, const double y, const double z) {
            const std::size_t base = static_cast<std::size_t>(corner) * 3;
            gradient_[base] = x;
            gradient_[base + 1] = y;
            gradient_[base + 2] = z;
        };
        set_gradient(0, v0_ - v1_, v0_ - v3_, v0_ - v4_);
        set_gradient(1, v0_ - v1_, v1_ - v2_, v1_ - v5_);
        set_gradient(2, v3_ - v2_, v1_ - v2_, v2_ - v6_);
        set_gradient(3, v3_ - v2_, v0_ - v3_, v3_ - v7_);
        set_gradient(4, v4_ - v5_, v4_ - v7_, v0_ - v4_);
        set_gradient(5, v4_ - v5_, v5_ - v6_, v1_ - v5_);
        set_gradient(6, v7_ - v6_, v5_ - v6_, v2_ - v6_);
        set_gradient(7, v7_ - v6_, v4_ - v7_, v3_ - v7_);
    }

    void calculate_center_vertex() {
        const std::array<double, 8> strengths{{
            1.0 / (kEpsilon + std::abs(v0_)), 1.0 / (kEpsilon + std::abs(v1_)),
            1.0 / (kEpsilon + std::abs(v2_)), 1.0 / (kEpsilon + std::abs(v3_)),
            1.0 / (kEpsilon + std::abs(v4_)), 1.0 / (kEpsilon + std::abs(v5_)),
            1.0 / (kEpsilon + std::abs(v6_)), 1.0 / (kEpsilon + std::abs(v7_)),
        }};
        const std::array<int, 8> xs{{0, 1, 1, 0, 0, 1, 1, 0}};
        const std::array<int, 8> ys{{0, 0, 1, 1, 0, 0, 1, 1}};
        const std::array<int, 8> zs{{0, 0, 0, 0, 1, 1, 1, 1}};
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double sum = 0.0;
        center_gx_ = 0.0;
        center_gy_ = 0.0;
        center_gz_ = 0.0;
        for (int corner = 0; corner < 8; ++corner) {
            const double strength = strengths[static_cast<std::size_t>(corner)];
            x += xs[static_cast<std::size_t>(corner)] * strength;
            y += ys[static_cast<std::size_t>(corner)] * strength;
            z += zs[static_cast<std::size_t>(corner)] * strength;
            sum += strength;
            const std::size_t base = static_cast<std::size_t>(corner) * 3;
            center_gx_ += strength * gradient_[base];
            center_gy_ += strength * gradient_[base + 1];
            center_gz_ += strength * gradient_[base + 2];
        }
        center_x_ = x_ + step_ * x / sum;
        center_y_ = y_ + step_ * y / sum;
        center_z_ = z_ + step_ * z / sum;
        center_calculated_ = true;
    }

    int nx_ = 0;
    int ny_ = 0;
    int x_ = 0;
    int y_ = 0;
    int z_ = 0;
    int step_ = 1;
    int index_ = 0;
    double v0_ = 0.0;
    double v1_ = 0.0;
    double v2_ = 0.0;
    double v3_ = 0.0;
    double v4_ = 0.0;
    double v5_ = 0.0;
    double v6_ = 0.0;
    double v7_ = 0.0;
    double vmax_ = 0.0;
    std::array<double, 8> values_by_corner_{};
    std::array<double, 24> gradient_{};
    bool center_calculated_ = false;
    int center_vertex_ = -1;
    double center_x_ = 0.0;
    double center_y_ = 0.0;
    double center_z_ = 0.0;
    double center_gx_ = 0.0;
    double center_gy_ = 0.0;
    double center_gz_ = 0.0;
    std::vector<std::int32_t> face_layer1_;
    std::vector<std::int32_t> face_layer2_;
    std::vector<float> vertices_;
    std::vector<std::int32_t> faces_;
    std::vector<float> normals_;
    std::vector<float> values_;
};

inline bool test_face(const Cell &cell, const int face) {
    const int absolute_face = std::abs(face);
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;
    if (absolute_face == 1) { a = cell.v0(); b = cell.v4(); c = cell.v5(); d = cell.v1(); }
    else if (absolute_face == 2) { a = cell.v1(); b = cell.v5(); c = cell.v6(); d = cell.v2(); }
    else if (absolute_face == 3) { a = cell.v2(); b = cell.v6(); c = cell.v7(); d = cell.v3(); }
    else if (absolute_face == 4) { a = cell.v3(); b = cell.v7(); c = cell.v4(); d = cell.v0(); }
    else if (absolute_face == 5) { a = cell.v0(); b = cell.v3(); c = cell.v2(); d = cell.v1(); }
    else if (absolute_face == 6) { a = cell.v4(); b = cell.v7(); c = cell.v6(); d = cell.v5(); }
    const double determinant = a * c - b * d;
    if (determinant > -kEpsilon && determinant < kEpsilon) return face >= 0;
    return face * a * determinant >= 0.0;
}

inline bool test_internal(const Cell &cell, const Luts &luts, const int case_, const int config,
                          const int subconfig, const int sign);
inline void select_mc33_tiling(const Luts &luts, Cell &cell, int case_, int config);

inline void remove_degenerate_faces(MarchingCubesResult &result) {
    const std::size_t n_vertices = result.values.size();
    std::vector<std::int32_t> map(n_vertices);
    for (std::size_t i = 0; i < n_vertices; ++i) map[i] = static_cast<std::int32_t>(i);
    std::vector<bool> keep_face(result.faces.size() / 3, true);
    const auto equal_vertex = [&result](const std::int32_t a, const std::int32_t b) {
        const std::size_t first = static_cast<std::size_t>(a) * 3;
        const std::size_t second = static_cast<std::size_t>(b) * 3;
        return result.vertices[first] == result.vertices[second]
            && result.vertices[first + 1] == result.vertices[second + 1]
            && result.vertices[first + 2] == result.vertices[second + 2];
    };
    for (std::size_t face = 0; face < keep_face.size(); ++face) {
        const std::int32_t a = result.faces[face * 3];
        const std::int32_t b = result.faces[face * 3 + 1];
        const std::int32_t c = result.faces[face * 3 + 2];
        if (equal_vertex(a, b)) { map[static_cast<std::size_t>(a)] = map[static_cast<std::size_t>(b)] = std::min(map[static_cast<std::size_t>(a)], map[static_cast<std::size_t>(b)]); keep_face[face] = false; }
        if (equal_vertex(a, c)) { map[static_cast<std::size_t>(a)] = map[static_cast<std::size_t>(c)] = std::min(map[static_cast<std::size_t>(a)], map[static_cast<std::size_t>(c)]); keep_face[face] = false; }
        if (equal_vertex(b, c)) { map[static_cast<std::size_t>(b)] = map[static_cast<std::size_t>(c)] = std::min(map[static_cast<std::size_t>(b)], map[static_cast<std::size_t>(c)]); keep_face[face] = false; }
    }
    std::vector<std::int32_t> compact(n_vertices, -1);
    MarchingCubesResult output;
    for (std::size_t vertex = 0; vertex < n_vertices; ++vertex) {
        if (map[vertex] == static_cast<std::int32_t>(vertex)) {
            compact[vertex] = static_cast<std::int32_t>(output.values.size());
            output.vertices.insert(output.vertices.end(), result.vertices.begin() + static_cast<std::ptrdiff_t>(vertex * 3), result.vertices.begin() + static_cast<std::ptrdiff_t>(vertex * 3 + 3));
            output.normals.insert(output.normals.end(), result.normals.begin() + static_cast<std::ptrdiff_t>(vertex * 3), result.normals.begin() + static_cast<std::ptrdiff_t>(vertex * 3 + 3));
            output.values.push_back(result.values[vertex]);
        }
    }
    for (std::size_t face = 0; face < keep_face.size(); ++face) {
        if (!keep_face[face]) continue;
        for (int corner = 0; corner < 3; ++corner) {
            const auto old = static_cast<std::size_t>(result.faces[face * 3 + static_cast<std::size_t>(corner)]);
            output.faces.push_back(compact[static_cast<std::size_t>(map[old])]);
        }
    }
    result = std::move(output);
}

inline bool test_internal(
    const Cell &cell,
    const Luts &luts,
    const int case_,
    const int config,
    const int subconfig,
    const int sign
) {
    double at = 0.0;
    double bt = 0.0;
    double ct = 0.0;
    double dt = 0.0;
    if (case_ == 4 || case_ == 10) {
        const double a = (cell.v4() - cell.v0()) * (cell.v6() - cell.v2())
            - (cell.v7() - cell.v3()) * (cell.v5() - cell.v1());
        const double b = cell.v2() * (cell.v4() - cell.v0())
            + cell.v0() * (cell.v6() - cell.v2()) - cell.v1() * (cell.v7() - cell.v3())
            - cell.v3() * (cell.v5() - cell.v1());
        const double t = -b / (2.0 * a + kEpsilon);
        if (t < 0.0 || t > 1.0) return sign > 0;
        at = cell.v0() + (cell.v4() - cell.v0()) * t;
        bt = cell.v3() + (cell.v7() - cell.v3()) * t;
        ct = cell.v2() + (cell.v6() - cell.v2()) * t;
        dt = cell.v1() + (cell.v5() - cell.v1()) * t;
    } else {
        int edge = -1;
        if (case_ == 6) edge = luts.test6.get2(config, 2);
        else if (case_ == 7) edge = luts.test7.get2(config, 4);
        else if (case_ == 12) edge = luts.test12.get2(config, 3);
        else if (case_ == 13) edge = luts.tiling13_5_1.get3(config, subconfig, 0);
        else return false;
        double t = 0.0;
        switch (edge) {
        case 0:
            t = cell.v0() / (cell.v0() - cell.v1() + kEpsilon);
            bt = cell.v3() + (cell.v2() - cell.v3()) * t;
            ct = cell.v7() + (cell.v6() - cell.v7()) * t;
            dt = cell.v4() + (cell.v5() - cell.v4()) * t;
            break;
        case 1:
            t = cell.v1() / (cell.v1() - cell.v2() + kEpsilon);
            bt = cell.v0() + (cell.v3() - cell.v0()) * t;
            ct = cell.v4() + (cell.v7() - cell.v4()) * t;
            dt = cell.v5() + (cell.v6() - cell.v5()) * t;
            break;
        case 2:
            t = cell.v2() / (cell.v2() - cell.v3() + kEpsilon);
            bt = cell.v1() + (cell.v0() - cell.v1()) * t;
            ct = cell.v5() + (cell.v4() - cell.v5()) * t;
            dt = cell.v6() + (cell.v7() - cell.v6()) * t;
            break;
        case 3:
            t = cell.v3() / (cell.v3() - cell.v0() + kEpsilon);
            bt = cell.v2() + (cell.v1() - cell.v2()) * t;
            ct = cell.v6() + (cell.v5() - cell.v6()) * t;
            dt = cell.v7() + (cell.v4() - cell.v7()) * t;
            break;
        case 4:
            t = cell.v4() / (cell.v4() - cell.v5() + kEpsilon);
            bt = cell.v7() + (cell.v6() - cell.v7()) * t;
            ct = cell.v3() + (cell.v2() - cell.v3()) * t;
            dt = cell.v0() + (cell.v1() - cell.v0()) * t;
            break;
        case 5:
            t = cell.v5() / (cell.v5() - cell.v6() + kEpsilon);
            bt = cell.v4() + (cell.v7() - cell.v4()) * t;
            ct = cell.v0() + (cell.v3() - cell.v0()) * t;
            dt = cell.v1() + (cell.v2() - cell.v1()) * t;
            break;
        case 6:
            t = cell.v6() / (cell.v6() - cell.v7() + kEpsilon);
            bt = cell.v5() + (cell.v4() - cell.v5()) * t;
            ct = cell.v1() + (cell.v0() - cell.v1()) * t;
            dt = cell.v2() + (cell.v3() - cell.v2()) * t;
            break;
        case 7:
            t = cell.v7() / (cell.v7() - cell.v4() + kEpsilon);
            bt = cell.v6() + (cell.v5() - cell.v6()) * t;
            ct = cell.v2() + (cell.v1() - cell.v2()) * t;
            dt = cell.v3() + (cell.v0() - cell.v3()) * t;
            break;
        case 8:
            t = cell.v0() / (cell.v0() - cell.v4() + kEpsilon);
            bt = cell.v3() + (cell.v7() - cell.v3()) * t;
            ct = cell.v2() + (cell.v6() - cell.v2()) * t;
            dt = cell.v1() + (cell.v5() - cell.v1()) * t;
            break;
        case 9:
            t = cell.v1() / (cell.v1() - cell.v5() + kEpsilon);
            bt = cell.v0() + (cell.v4() - cell.v0()) * t;
            ct = cell.v3() + (cell.v7() - cell.v3()) * t;
            dt = cell.v2() + (cell.v6() - cell.v2()) * t;
            break;
        case 10:
            t = cell.v2() / (cell.v2() - cell.v6() + kEpsilon);
            bt = cell.v1() + (cell.v5() - cell.v1()) * t;
            ct = cell.v0() + (cell.v4() - cell.v0()) * t;
            dt = cell.v3() + (cell.v7() - cell.v3()) * t;
            break;
        case 11:
            t = cell.v3() / (cell.v3() - cell.v7() + kEpsilon);
            bt = cell.v2() + (cell.v6() - cell.v2()) * t;
            ct = cell.v1() + (cell.v5() - cell.v1()) * t;
            dt = cell.v0() + (cell.v4() - cell.v0()) * t;
            break;
        default:
            return false;
        }
    }
    int test = 0;
    if (at >= 0.0) test += 1;
    if (bt >= 0.0) test += 2;
    if (ct >= 0.0) test += 4;
    if (dt >= 0.0) test += 8;
    switch (test) {
    case 0: case 1: case 2: case 3: case 4: case 6: case 8: case 9:
        return sign > 0;
    case 5:
        return at * ct - bt * dt < kEpsilon ? sign > 0 : false;
    case 7: case 11: case 13: case 14: case 15:
        return sign < 0;
    case 12:
        return sign > 0;
    case 10:
        return at * ct - bt * dt >= kEpsilon ? sign > 0 : false;
    default:
        return sign < 0;
    }
}

inline void select_mc33_tiling(const Luts &luts, Cell &cell, const int case_, const int config) {
    int subconfig = 0;
    switch (case_) {
    case 1: cell.add_triangles(luts.tiling1, config, 1); break;
    case 2: cell.add_triangles(luts.tiling2, config, 2); break;
    case 3: {
        const bool split = test_face(cell, luts.test3.get1(config));
        cell.add_triangles(split ? luts.tiling3_2 : luts.tiling3_1, config, split ? 4 : 2);
        break;
    }
    case 4: {
        const bool connected = test_internal(cell, luts, case_, config, subconfig, luts.test4.get1(config));
        cell.add_triangles(connected ? luts.tiling4_1 : luts.tiling4_2, config, connected ? 2 : 6);
        break;
    }
    case 5: cell.add_triangles(luts.tiling5, config, 3); break;
    case 6:
        if (test_face(cell, luts.test6.get2(config, 0))) cell.add_triangles(luts.tiling6_2, config, 5);
        else if (test_internal(cell, luts, case_, config, subconfig, luts.test6.get2(config, 1))) cell.add_triangles(luts.tiling6_1_1, config, 3);
        else cell.add_triangles(luts.tiling6_1_2, config, 9);
        break;
    case 7:
        if (test_face(cell, luts.test7.get2(config, 0))) subconfig += 1;
        if (test_face(cell, luts.test7.get2(config, 1))) subconfig += 2;
        if (test_face(cell, luts.test7.get2(config, 2))) subconfig += 4;
        if (subconfig == 0) cell.add_triangles(luts.tiling7_1, config, 3);
        else if (subconfig == 1 || subconfig == 2 || subconfig == 4) cell.add_triangles2(luts.tiling7_2, config, subconfig == 1 ? 0 : subconfig == 2 ? 1 : 2, 5);
        else if (subconfig == 3 || subconfig == 5 || subconfig == 6) cell.add_triangles2(luts.tiling7_3, config, subconfig == 3 ? 0 : subconfig == 5 ? 1 : 2, 9);
        else if (test_internal(cell, luts, case_, config, subconfig, luts.test7.get2(config, 3))) cell.add_triangles(luts.tiling7_4_2, config, 9);
        else cell.add_triangles(luts.tiling7_4_1, config, 5);
        break;
    case 8: cell.add_triangles(luts.tiling8, config, 2); break;
    case 9: cell.add_triangles(luts.tiling9, config, 4); break;
    case 10:
        if (test_face(cell, luts.test10.get2(config, 0))) {
            if (test_face(cell, luts.test10.get2(config, 1))) cell.add_triangles(luts.tiling10_1_1_alt, config, 4);
            else cell.add_triangles(luts.tiling10_2, config, 8);
        } else if (test_face(cell, luts.test10.get2(config, 1))) cell.add_triangles(luts.tiling10_2_alt, config, 8);
        else if (test_internal(cell, luts, case_, config, subconfig, luts.test10.get2(config, 2))) cell.add_triangles(luts.tiling10_1_1, config, 4);
        else cell.add_triangles(luts.tiling10_1_2, config, 8);
        break;
    case 11: cell.add_triangles(luts.tiling11, config, 4); break;
    case 12:
        if (test_face(cell, luts.test12.get2(config, 0))) {
            if (test_face(cell, luts.test12.get2(config, 1))) cell.add_triangles(luts.tiling12_1_1_alt, config, 4);
            else cell.add_triangles(luts.tiling12_2, config, 8);
        } else if (test_face(cell, luts.test12.get2(config, 1))) cell.add_triangles(luts.tiling12_2_alt, config, 8);
        else if (test_internal(cell, luts, case_, config, subconfig, luts.test12.get2(config, 2))) cell.add_triangles(luts.tiling12_1_1, config, 4);
        else cell.add_triangles(luts.tiling12_1_2, config, 8);
        break;
    case 13:
        for (int face = 0; face < 6; ++face) {
            if (test_face(cell, luts.test13.get2(config, face))) subconfig |= 1 << face;
        }
        subconfig = luts.subconfig13.get1(subconfig);
        if (subconfig == 0) cell.add_triangles(luts.tiling13_1, config, 4);
        else if (subconfig >= 1 && subconfig <= 6) cell.add_triangles2(luts.tiling13_2, config, subconfig - 1, 6);
        else if (subconfig >= 7 && subconfig <= 18) cell.add_triangles2(luts.tiling13_3, config, subconfig - 7, 10);
        else if (subconfig >= 19 && subconfig <= 22) cell.add_triangles2(luts.tiling13_4, config, subconfig - 19, 12);
        else if (subconfig >= 23 && subconfig <= 26) {
            const int local = subconfig - 23;
            if (test_internal(cell, luts, case_, config, local, luts.test13.get2(config, 6))) cell.add_triangles2(luts.tiling13_5_1, config, local, 6);
            else cell.add_triangles2(luts.tiling13_5_2, config, local, 10);
        } else if (subconfig >= 27 && subconfig <= 38) cell.add_triangles2(luts.tiling13_3_alt, config, subconfig - 27, 10);
        else if (subconfig >= 39 && subconfig <= 44) cell.add_triangles2(luts.tiling13_2_alt, config, subconfig - 39, 6);
        else if (subconfig == 45) cell.add_triangles(luts.tiling13_1_alt, config, 4);
        break;
    case 14: cell.add_triangles(luts.tiling14, config, 4); break;
    default: break;
    }
}

template <bool UseMask>
inline void traverse_cells(
    const ConstArrayView<float> &volume,
    const double level,
    const int step_size,
    const MarchingCubesMethod method,
    const ConstArrayView<std::uint8_t> *mask,
    const Luts &tables,
    Cell &cell
) {
    const int nz = static_cast<int>(volume.shape[0]);
    const int ny = static_cast<int>(volume.shape[1]);
    const int nx = static_cast<int>(volume.shape[2]);
    const auto at = [nx, ny, &volume](const int z, const int y, const int x) {
        return volume.data[(static_cast<std::size_t>(z) * ny + y) * nx + x];
    };
    const int max_x = nx - 2 * step_size;
    const int max_y = ny - 2 * step_size;
    const int max_z = nz - 2 * step_size;
    for (int z = -step_size; z < max_z;) {
        z += step_size;
        cell.new_z_value();
        const int z_next = z + step_size;
        for (int y = -step_size; y < max_y;) {
            y += step_size;
            const int y_next = y + step_size;
            for (int x = -step_size; x < max_x;) {
                x += step_size;
                const int x_next = x + step_size;
                if constexpr (UseMask) {
                    const auto mask_index =
                        (static_cast<std::size_t>(z_next) * ny + y_next) * nx + x_next;
                    if (mask->data[mask_index] == 0) continue;
                }
                cell.set_cube(
                    level, x, y, z, step_size,
                    at(z, y, x), at(z, y, x_next),
                    at(z, y_next, x_next), at(z, y_next, x),
                    at(z_next, y, x), at(z_next, y, x_next),
                    at(z_next, y_next, x_next), at(z_next, y_next, x)
                );
                if (method == MarchingCubesMethod::Lorensen) {
                    int triangles = 0;
                    while (triangles < 5 && tables.cases_classic.get2(cell.index(), 3 * triangles) != -1) ++triangles;
                    if (triangles != 0) cell.add_triangles(tables.cases_classic, cell.index(), triangles);
                } else {
                    const int case_ = tables.cases.get2(cell.index(), 0);
                    if (case_ > 0) select_mc33_tiling(tables, cell, case_, tables.cases.get2(cell.index(), 1));
                }
            }
        }
    }
}

} // namespace detail::marching_cubes

// Extract an isosurface from a 3-D float32 image. The core deliberately does
// not know about Python layout conventions beyond the C-order ArrayView; the
// binding handles argument validation, output axis order, and spacing.
inline MarchingCubesResult marching_cubes(
    const ConstArrayView<float> &volume,
    const double level,
    const int step_size,
    const MarchingCubesMethod method,
    const ConstArrayView<std::uint8_t> *mask,
    const bool allow_degenerate
) {
    BIOIMAGE_PROFILE_INIT(profiler);
    if (volume.shape.size() != 3) {
        throw std::invalid_argument("volume must have ndim=3");
    }
    for (const auto axis : volume.shape) {
        if (axis < 2) throw std::invalid_argument("volume dimensions must all be at least 2");
        if (axis > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("volume dimensions exceed marching cubes index range");
        }
    }
    if (step_size < 1) throw std::invalid_argument("step_size must be at least 1");
    if (mask != nullptr && mask->shape != volume.shape) {
        throw std::invalid_argument("mask must have the same shape as volume");
    }
    const int ny = static_cast<int>(volume.shape[1]);
    const int nx = static_cast<int>(volume.shape[2]);
    const detail::marching_cubes::Luts *tables_ptr = nullptr;
    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "lookup_tables");
        tables_ptr = &detail::marching_cubes::luts();
    }
    const auto &tables = *tables_ptr;
    std::optional<detail::marching_cubes::Cell> cell;
    cell.emplace(nx, ny);
    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "cell_traversal");
        if (mask == nullptr) {
            detail::marching_cubes::traverse_cells<false>(
                volume, level, step_size, method, nullptr, tables, *cell
            );
        } else {
            detail::marching_cubes::traverse_cells<true>(
                volume, level, step_size, method, mask, tables, *cell
            );
        }
    }
    MarchingCubesResult result;
    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "finalize_normals");
        result = cell->take_result();
    }
    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "cell_cleanup");
        cell.reset();
    }
    if (!allow_degenerate) {
        BIOIMAGE_PROFILE_SCOPE(profiler, "remove_degenerate_faces");
        detail::marching_cubes::remove_degenerate_faces(result);
    }
    BIOIMAGE_PROFILE_REPORT(profiler);
    return result;
}

} // namespace bioimage_cpp::mesh
