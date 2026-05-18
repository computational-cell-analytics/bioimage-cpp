#include "util.hxx"

#include "bioimage_cpp/util/union_find.hxx"

#include <nanobind/ndarray.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

using EdgeArray = nb::ndarray<nb::numpy, const std::uint64_t, nb::c_contig>;
using NodeArray = nb::ndarray<nb::numpy, const std::uint64_t, nb::c_contig, nb::ndim<1>>;
using OutputArray = nb::ndarray<nb::numpy, std::uint64_t, nb::c_contig>;

void merge_edges(
    util::UnionFind &uf,
    EdgeArray edges
) {
    if (edges.ndim() != 2 || edges.shape(1) != 2) {
        std::string shape = "(";
        for (std::size_t axis = 0; axis < edges.ndim(); ++axis) {
            if (axis > 0) {
                shape += ", ";
            }
            shape += std::to_string(edges.shape(axis));
        }
        shape += ")";
        throw std::invalid_argument(
            "edges must have shape (N, 2), got shape " + shape
        );
    }

    const auto n_edges = edges.shape(0);
    const auto *data = edges.data();

    {
        nb::gil_scoped_release release;
        for (std::size_t i = 0; i < n_edges; ++i) {
            uf.merge(data[2 * i], data[2 * i + 1]);
        }
    }
}

OutputArray find_nodes(util::UnionFind &uf, NodeArray nodes) {
    const auto n = nodes.shape(0);
    auto *out = new std::uint64_t[n]();
    nb::capsule owner(out, [](void *p) noexcept { delete[] static_cast<std::uint64_t *>(p); });

    const auto *input = nodes.data();
    {
        nb::gil_scoped_release release;
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = uf.find(input[i]);
        }
    }

    std::size_t shape[1] = {n};
    return OutputArray(out, 1, shape, owner);
}

OutputArray element_labeling(util::UnionFind &uf) {
    const auto n = uf.size();
    auto *out = new std::uint64_t[n]();
    nb::capsule owner(out, [](void *p) noexcept { delete[] static_cast<std::uint64_t *>(p); });

    {
        nb::gil_scoped_release release;
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = uf.find(static_cast<std::uint64_t>(i));
        }
    }

    std::size_t shape[1] = {n};
    return OutputArray(out, 1, shape, owner);
}

} // namespace

void bind_util(nb::module_ &m) {
    nb::module_ util_module = m.def_submodule("util", "Utility data structures.");

    nb::class_<util::UnionFind>(util_module, "UnionFind")
        .def(
            nb::init<std::size_t>(),
            nb::arg("size"),
            "Create a union-find over `size` singleton elements."
        )
        .def_prop_ro(
            "size",
            &util::UnionFind::size,
            "Number of elements in the union-find."
        )
        .def(
            "find",
            &util::UnionFind::find,
            nb::arg("node"),
            "Return the (path-compressed) root of `node`."
        )
        .def(
            "find",
            &find_nodes,
            nb::arg("nodes"),
            "Return the roots for a 1-D uint64 array of node indices."
        )
        .def(
            "merge",
            &util::UnionFind::merge,
            nb::arg("first"),
            nb::arg("second"),
            "Union the sets containing `first` and `second`. Returns the new root."
        )
        .def(
            "merge",
            &merge_edges,
            nb::arg("edges"),
            "Bulk-merge from an (N, 2) uint64 array of node-pair edges."
        )
        .def(
            "merge_to",
            &util::UnionFind::merge_to,
            nb::arg("stable"),
            nb::arg("removed"),
            "Union the sets containing `stable` and `removed`, forcing "
            "`stable`'s root to survive. Returns the new root."
        )
        .def(
            "element_labeling",
            &element_labeling,
            "Return a uint64 array of length `size` where entry i is the "
            "(path-compressed) root of element i."
        )
        .def(
            "reset",
            &util::UnionFind::reset,
            nb::arg("size"),
            "Reinitialise the union-find to `size` singleton elements."
        );
}

} // namespace bioimage_cpp::bindings
