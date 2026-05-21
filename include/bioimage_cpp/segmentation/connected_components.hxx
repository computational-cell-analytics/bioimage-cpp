#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/profile.hxx"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bioimage_cpp::segmentation {

namespace detail_cc {

// Label-space union-find for connected-components.
//
// Backed by a single `std::vector<std::uint32_t>` of parents that grows as
// new provisional labels are allocated. Union-by-min (smaller label becomes
// the parent) keeps the structure flat enough that path-halving in `find`
// gives near-constant amortised cost, and it removes the need for a separate
// rank array. Label 0 is reserved for background.
struct LabelUnionFind {
    std::vector<std::uint32_t> parents;

    LabelUnionFind() {
        parents.push_back(0);
    }

    std::uint32_t new_label() {
        const auto lbl = static_cast<std::uint32_t>(parents.size());
        parents.push_back(lbl);
        return lbl;
    }

    std::uint32_t find(std::uint32_t a) {
        while (parents[a] != a) {
            parents[a] = parents[parents[a]];
            a = parents[a];
        }
        return a;
    }

    std::uint32_t unite(std::uint32_t a, std::uint32_t b) {
        a = find(a);
        b = find(b);
        if (a == b) {
            return a;
        }
        if (a > b) {
            std::swap(a, b);
        }
        parents[b] = a;
        return a;
    }
};

// 2D 4-connectivity scan. Backward neighbours: W (i-1), N (i-X).
template <class InT, bool Binary>
void label_pass1_2d_4(
    const InT *image, const InT background,
    const std::ptrdiff_t Y, const std::ptrdiff_t X,
    std::uint32_t *prov, LabelUnionFind &uf
) {
    auto merge_into = [&](const std::ptrdiff_t neighbour, std::uint32_t &lbl, const InT v) {
        bool same;
        if constexpr (Binary) {
            (void)v;
            same = (image[neighbour] != background);
        } else {
            same = (image[neighbour] == v);
        }
        if (same) {
            const std::uint32_t n_lbl = prov[neighbour];
            lbl = (lbl == 0) ? n_lbl : uf.unite(lbl, n_lbl);
        }
    };

    for (std::ptrdiff_t y = 0; y < Y; ++y) {
        const std::ptrdiff_t row = y * X;
        const bool yv = y > 0;
        for (std::ptrdiff_t x = 0; x < X; ++x) {
            const std::ptrdiff_t i = row + x;
            if (image[i] == background) {
                prov[i] = 0;
                continue;
            }
            const InT v = image[i];
            std::uint32_t lbl = 0;
            if (yv) {
                merge_into(i - X, lbl, v);
            }
            if (x > 0) {
                merge_into(i - 1, lbl, v);
            }
            prov[i] = (lbl != 0) ? lbl : uf.new_label();
        }
    }
}

// 2D 8-connectivity scan. Backward neighbours: NW, N, NE, W.
template <class InT, bool Binary>
void label_pass1_2d_8(
    const InT *image, const InT background,
    const std::ptrdiff_t Y, const std::ptrdiff_t X,
    std::uint32_t *prov, LabelUnionFind &uf
) {
    auto merge_into = [&](const std::ptrdiff_t neighbour, std::uint32_t &lbl, const InT v) {
        bool same;
        if constexpr (Binary) {
            (void)v;
            same = (image[neighbour] != background);
        } else {
            same = (image[neighbour] == v);
        }
        if (same) {
            const std::uint32_t n_lbl = prov[neighbour];
            lbl = (lbl == 0) ? n_lbl : uf.unite(lbl, n_lbl);
        }
    };

    for (std::ptrdiff_t y = 0; y < Y; ++y) {
        const std::ptrdiff_t row = y * X;
        const bool yv = y > 0;
        for (std::ptrdiff_t x = 0; x < X; ++x) {
            const std::ptrdiff_t i = row + x;
            if (image[i] == background) {
                prov[i] = 0;
                continue;
            }
            const InT v = image[i];
            std::uint32_t lbl = 0;
            if (yv) {
                if (x > 0) {
                    merge_into(i - X - 1, lbl, v);
                }
                merge_into(i - X, lbl, v);
                if (x < X - 1) {
                    merge_into(i - X + 1, lbl, v);
                }
            }
            if (x > 0) {
                merge_into(i - 1, lbl, v);
            }
            prov[i] = (lbl != 0) ? lbl : uf.new_label();
        }
    }
}

// 3D 6-connectivity scan. Backward neighbours: U (i-SZ), N (i-SY), W (i-1).
template <class InT, bool Binary>
void label_pass1_3d_6(
    const InT *image, const InT background,
    const std::ptrdiff_t Z, const std::ptrdiff_t Y, const std::ptrdiff_t X,
    std::uint32_t *prov, LabelUnionFind &uf
) {
    const std::ptrdiff_t SY = X;
    const std::ptrdiff_t SZ = Y * X;
    auto merge_into = [&](const std::ptrdiff_t neighbour, std::uint32_t &lbl, const InT v) {
        bool same;
        if constexpr (Binary) {
            (void)v;
            same = (image[neighbour] != background);
        } else {
            same = (image[neighbour] == v);
        }
        if (same) {
            const std::uint32_t n_lbl = prov[neighbour];
            lbl = (lbl == 0) ? n_lbl : uf.unite(lbl, n_lbl);
        }
    };

    for (std::ptrdiff_t z = 0; z < Z; ++z) {
        const bool zv = z > 0;
        for (std::ptrdiff_t y = 0; y < Y; ++y) {
            const bool yv = y > 0;
            for (std::ptrdiff_t x = 0; x < X; ++x) {
                const std::ptrdiff_t i = z * SZ + y * SY + x;
                if (image[i] == background) {
                    prov[i] = 0;
                    continue;
                }
                const InT v = image[i];
                std::uint32_t lbl = 0;
                if (zv) {
                    merge_into(i - SZ, lbl, v);
                }
                if (yv) {
                    merge_into(i - SY, lbl, v);
                }
                if (x > 0) {
                    merge_into(i - 1, lbl, v);
                }
                prov[i] = (lbl != 0) ? lbl : uf.new_label();
            }
        }
    }
}

// 3D 18-connectivity scan. Backward neighbours: 9 offsets — every offset in
// {-1,0,1}^3 whose first non-zero entry is -1 and whose non-zero count is at
// most 2.
template <class InT, bool Binary>
void label_pass1_3d_18(
    const InT *image, const InT background,
    const std::ptrdiff_t Z, const std::ptrdiff_t Y, const std::ptrdiff_t X,
    std::uint32_t *prov, LabelUnionFind &uf
) {
    const std::ptrdiff_t SY = X;
    const std::ptrdiff_t SZ = Y * X;
    auto merge_into = [&](const std::ptrdiff_t neighbour, std::uint32_t &lbl, const InT v) {
        bool same;
        if constexpr (Binary) {
            (void)v;
            same = (image[neighbour] != background);
        } else {
            same = (image[neighbour] == v);
        }
        if (same) {
            const std::uint32_t n_lbl = prov[neighbour];
            lbl = (lbl == 0) ? n_lbl : uf.unite(lbl, n_lbl);
        }
    };

    for (std::ptrdiff_t z = 0; z < Z; ++z) {
        const bool zv = z > 0;
        for (std::ptrdiff_t y = 0; y < Y; ++y) {
            const bool yv = y > 0;
            const bool yu = y < Y - 1;
            for (std::ptrdiff_t x = 0; x < X; ++x) {
                const std::ptrdiff_t i = z * SZ + y * SY + x;
                if (image[i] == background) {
                    prov[i] = 0;
                    continue;
                }
                const InT v = image[i];
                std::uint32_t lbl = 0;
                const bool xv = x > 0;
                const bool xu = x < X - 1;
                if (zv) {
                    if (yv) {
                        merge_into(i - SZ - SY, lbl, v);
                    }
                    if (xv) {
                        merge_into(i - SZ - 1, lbl, v);
                    }
                    merge_into(i - SZ, lbl, v);
                    if (xu) {
                        merge_into(i - SZ + 1, lbl, v);
                    }
                    if (yu) {
                        merge_into(i - SZ + SY, lbl, v);
                    }
                }
                if (yv) {
                    if (xv) {
                        merge_into(i - SY - 1, lbl, v);
                    }
                    merge_into(i - SY, lbl, v);
                    if (xu) {
                        merge_into(i - SY + 1, lbl, v);
                    }
                }
                if (xv) {
                    merge_into(i - 1, lbl, v);
                }
                prov[i] = (lbl != 0) ? lbl : uf.new_label();
            }
        }
    }
}

// 3D 26-connectivity scan. Backward neighbours: 13 offsets — every offset in
// {-1,0,1}^3 whose first non-zero entry is -1.
template <class InT, bool Binary>
void label_pass1_3d_26(
    const InT *image, const InT background,
    const std::ptrdiff_t Z, const std::ptrdiff_t Y, const std::ptrdiff_t X,
    std::uint32_t *prov, LabelUnionFind &uf
) {
    const std::ptrdiff_t SY = X;
    const std::ptrdiff_t SZ = Y * X;
    auto merge_into = [&](const std::ptrdiff_t neighbour, std::uint32_t &lbl, const InT v) {
        bool same;
        if constexpr (Binary) {
            (void)v;
            same = (image[neighbour] != background);
        } else {
            same = (image[neighbour] == v);
        }
        if (same) {
            const std::uint32_t n_lbl = prov[neighbour];
            lbl = (lbl == 0) ? n_lbl : uf.unite(lbl, n_lbl);
        }
    };

    for (std::ptrdiff_t z = 0; z < Z; ++z) {
        const bool zv = z > 0;
        for (std::ptrdiff_t y = 0; y < Y; ++y) {
            const bool yv = y > 0;
            const bool yu = y < Y - 1;
            for (std::ptrdiff_t x = 0; x < X; ++x) {
                const std::ptrdiff_t i = z * SZ + y * SY + x;
                if (image[i] == background) {
                    prov[i] = 0;
                    continue;
                }
                const InT v = image[i];
                std::uint32_t lbl = 0;
                const bool xv = x > 0;
                const bool xu = x < X - 1;
                if (zv) {
                    if (yv) {
                        if (xv) {
                            merge_into(i - SZ - SY - 1, lbl, v);
                        }
                        merge_into(i - SZ - SY, lbl, v);
                        if (xu) {
                            merge_into(i - SZ - SY + 1, lbl, v);
                        }
                    }
                    if (xv) {
                        merge_into(i - SZ - 1, lbl, v);
                    }
                    merge_into(i - SZ, lbl, v);
                    if (xu) {
                        merge_into(i - SZ + 1, lbl, v);
                    }
                    if (yu) {
                        if (xv) {
                            merge_into(i - SZ + SY - 1, lbl, v);
                        }
                        merge_into(i - SZ + SY, lbl, v);
                        if (xu) {
                            merge_into(i - SZ + SY + 1, lbl, v);
                        }
                    }
                }
                if (yv) {
                    if (xv) {
                        merge_into(i - SY - 1, lbl, v);
                    }
                    merge_into(i - SY, lbl, v);
                    if (xu) {
                        merge_into(i - SY + 1, lbl, v);
                    }
                }
                if (xv) {
                    merge_into(i - 1, lbl, v);
                }
                prov[i] = (lbl != 0) ? lbl : uf.new_label();
            }
        }
    }
}

// Walk the provisional buffer once, resolve every non-zero label to its
// union-find root, and assign dense first-occurrence labels into `out`.
template <class InT>
void dense_relabel(
    const std::uint32_t *prov, LabelUnionFind &uf,
    const InT *image, const InT background,
    const std::uint64_t N, std::uint64_t *out
) {
    std::vector<std::uint64_t> root_to_dense(uf.parents.size(), 0);
    std::uint64_t next_dense = 1;
    for (std::uint64_t i = 0; i < N; ++i) {
        if (image[i] == background) {
            out[i] = 0;
            continue;
        }
        const std::uint32_t root = uf.find(prov[i]);
        std::uint64_t dense = root_to_dense[root];
        if (dense == 0) {
            dense = next_dense++;
            root_to_dense[root] = dense;
        }
        out[i] = dense;
    }
}

} // namespace detail_cc

// Two-pass SAUF connected-components labeling.
//
// Pixels with `image[i] == background` are background and written as 0. Two
// non-background pixels share an output label iff there is a path between
// them through `connectivity`-neighbour steps along which the input value is
// constant. Output labels are dense, start at 1, and are assigned in
// row-major first-occurrence order. Supports 2D and 3D arrays.
//
// `connectivity` is in [1, ndim]: 1 = orthogonal (4/6), 2 = + edge diagonals
// (8 in 2D, 18 in 3D), 3 = + corner diagonals (26 in 3D).
//
// When `binary` is true the kernel skips per-pixel value-equality compares
// and treats every non-background pixel as a member of one foreground class.
// The Python wrapper sets this whenever the input dtype is `bool`.
template <class InT>
void label(
    const ConstArrayView<InT> &image,
    const InT background,
    const int connectivity,
    const bool binary,
    const ArrayView<std::uint64_t> &out
) {
    BIOIMAGE_PROFILE_INIT(profile);
    const auto &shape = image.shape;
    const int ndim = static_cast<int>(shape.size());
    if (ndim != 2 && ndim != 3) {
        throw std::invalid_argument(
            "image must have ndim 2 or 3, got ndim=" + std::to_string(ndim)
        );
    }
    if (connectivity < 1 || connectivity > ndim) {
        throw std::invalid_argument(
            "connectivity must be in [1, ndim], got connectivity=" +
            std::to_string(connectivity) + " for ndim=" + std::to_string(ndim)
        );
    }

    std::uint64_t N = 1;
    for (const auto extent : shape) {
        N *= static_cast<std::uint64_t>(extent);
    }

    for (std::uint64_t i = 0; i < N; ++i) {
        out.data[i] = 0;
    }
    if (N == 0) {
        return;
    }

    std::vector<std::uint32_t> prov(static_cast<std::size_t>(N), 0);
    detail_cc::LabelUnionFind uf;

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "pass1");
        if (ndim == 2) {
            const std::ptrdiff_t Y = shape[0];
            const std::ptrdiff_t X = shape[1];
            if (binary) {
                if (connectivity == 1) {
                    detail_cc::label_pass1_2d_4<InT, true>(
                        image.data, background, Y, X, prov.data(), uf
                    );
                } else {
                    detail_cc::label_pass1_2d_8<InT, true>(
                        image.data, background, Y, X, prov.data(), uf
                    );
                }
            } else {
                if (connectivity == 1) {
                    detail_cc::label_pass1_2d_4<InT, false>(
                        image.data, background, Y, X, prov.data(), uf
                    );
                } else {
                    detail_cc::label_pass1_2d_8<InT, false>(
                        image.data, background, Y, X, prov.data(), uf
                    );
                }
            }
        } else {
            const std::ptrdiff_t Z = shape[0];
            const std::ptrdiff_t Y = shape[1];
            const std::ptrdiff_t X = shape[2];
            if (binary) {
                if (connectivity == 1) {
                    detail_cc::label_pass1_3d_6<InT, true>(
                        image.data, background, Z, Y, X, prov.data(), uf
                    );
                } else if (connectivity == 2) {
                    detail_cc::label_pass1_3d_18<InT, true>(
                        image.data, background, Z, Y, X, prov.data(), uf
                    );
                } else {
                    detail_cc::label_pass1_3d_26<InT, true>(
                        image.data, background, Z, Y, X, prov.data(), uf
                    );
                }
            } else {
                if (connectivity == 1) {
                    detail_cc::label_pass1_3d_6<InT, false>(
                        image.data, background, Z, Y, X, prov.data(), uf
                    );
                } else if (connectivity == 2) {
                    detail_cc::label_pass1_3d_18<InT, false>(
                        image.data, background, Z, Y, X, prov.data(), uf
                    );
                } else {
                    detail_cc::label_pass1_3d_26<InT, false>(
                        image.data, background, Z, Y, X, prov.data(), uf
                    );
                }
            }
        }
    }

    {
        BIOIMAGE_PROFILE_SCOPE(profile, "pass2");
        detail_cc::dense_relabel<InT>(
            prov.data(), uf, image.data, background, N, out.data
        );
    }

    BIOIMAGE_PROFILE_REPORT(profile);
}

} // namespace bioimage_cpp::segmentation
