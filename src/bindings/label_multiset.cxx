#include "label_multiset.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/blocking.hxx"
#include "bioimage_cpp/label_multiset/downsample.hxx"
#include "bioimage_cpp/label_multiset/from_labels.hxx"
#include "bioimage_cpp/label_multiset/merger.hxx"
#include "bioimage_cpp/label_multiset/read_subset.hxx"

#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

using IdT = std::uint64_t;
using CountT = std::uint32_t;
using OffsetT = std::uint64_t;
using LabelT = std::uint64_t;

using OffsetArray = nb::ndarray<nb::numpy, const OffsetT, nb::c_contig>;
using IdArray = nb::ndarray<nb::numpy, const IdT, nb::c_contig>;
using CountArray = nb::ndarray<nb::numpy, const CountT, nb::c_contig>;
using OffsetOutArray = nb::ndarray<nb::numpy, OffsetT, nb::c_contig>;
using IdOutArray = nb::ndarray<nb::numpy, IdT, nb::c_contig>;
using CountOutArray = nb::ndarray<nb::numpy, CountT, nb::c_contig>;

template <class T>
ConstArrayView<T> view_1d(const nb::ndarray<nb::numpy, const T, nb::c_contig> &arr,
                          const char *name) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(
            std::string(name) + " must have ndim 1, got ndim=" +
            std::to_string(arr.ndim())
        );
    }
    return ConstArrayView<T>{
        arr.data(),
        {static_cast<std::ptrdiff_t>(arr.shape(0))},
        {},
    };
}

template <class T>
nb::ndarray<nb::numpy, T, nb::c_contig> alloc_1d(std::size_t n) {
    auto *data = new T[n]();
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] static_cast<T *>(p);
    });
    std::size_t shape[1] = {n};
    return nb::ndarray<nb::numpy, T, nb::c_contig>(data, 1, shape, owner);
}

template <class T>
nb::ndarray<nb::numpy, T, nb::c_contig> vector_to_array(std::vector<T> &&v) {
    auto *data = new T[v.size()];
    std::copy(v.begin(), v.end(), data);
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] static_cast<T *>(p);
    });
    std::size_t shape[1] = {v.size()};
    return nb::ndarray<nb::numpy, T, nb::c_contig>(data, 1, shape, owner);
}

// ----- read_subset -----

std::pair<nb::ndarray<nb::numpy, IdT, nb::c_contig>,
          nb::ndarray<nb::numpy, CountT, nb::c_contig>>
bind_read_subset_flat(OffsetArray offsets, OffsetArray sizes,
                      IdArray ids, CountArray counts,
                      bool argsort) {
    auto offsets_v = view_1d(offsets, "offsets");
    auto sizes_v = view_1d(sizes, "sizes");
    auto ids_v = view_1d(ids, "ids");
    auto counts_v = view_1d(counts, "counts");
    if (offsets_v.shape[0] != sizes_v.shape[0]) {
        throw std::invalid_argument("offsets and sizes must have the same length");
    }
    if (ids_v.shape[0] != counts_v.shape[0]) {
        throw std::invalid_argument("ids and counts must have the same length");
    }

    std::vector<IdT> ids_out;
    std::vector<CountT> counts_out;
    {
        nb::gil_scoped_release release;
        label_multiset::read_subset(offsets_v, sizes_v, ids_v, counts_v,
                                    ids_out, counts_out, argsort);
    }
    return {vector_to_array(std::move(ids_out)),
            vector_to_array(std::move(counts_out))};
}

// ----- downsample_multiset -----

std::tuple<nb::ndarray<nb::numpy, IdT, nb::c_contig>,
           nb::ndarray<nb::numpy, OffsetT, nb::c_contig>,
           nb::ndarray<nb::numpy, OffsetT, nb::c_contig>,
           nb::ndarray<nb::numpy, OffsetT, nb::c_contig>,
           nb::ndarray<nb::numpy, IdT, nb::c_contig>,
           nb::ndarray<nb::numpy, CountT, nb::c_contig>>
bind_downsample_multiset(const Blocking &blocking,
                         OffsetArray offsets,
                         OffsetArray entry_sizes,
                         OffsetArray entry_offsets,
                         IdArray ids,
                         CountArray counts,
                         int restrict_set) {
    auto offsets_v = view_1d(offsets, "offsets");
    auto entry_sizes_v = view_1d(entry_sizes, "entry_sizes");
    auto entry_offsets_v = view_1d(entry_offsets, "entry_offsets");
    auto ids_v = view_1d(ids, "ids");
    auto counts_v = view_1d(counts, "counts");
    if (ids_v.shape[0] != counts_v.shape[0]) {
        throw std::invalid_argument("ids and counts must have the same length");
    }
    if (offsets_v.shape[0] != entry_offsets_v.shape[0]) {
        throw std::invalid_argument(
            "offsets and entry_offsets must have the same length"
        );
    }

    const std::size_t n_blocks = static_cast<std::size_t>(blocking.number_of_blocks());

    auto new_argmax = alloc_1d<IdT>(n_blocks);
    auto new_offsets = alloc_1d<OffsetT>(n_blocks);
    auto new_entry_offsets = alloc_1d<OffsetT>(n_blocks);

    std::vector<OffsetT> new_entry_sizes;
    std::vector<IdT> new_ids;
    std::vector<CountT> new_counts;

    ArrayView<IdT> argmax_view{
        new_argmax.data(),
        {static_cast<std::ptrdiff_t>(n_blocks)}, {}};
    ArrayView<OffsetT> off_view{
        new_offsets.data(),
        {static_cast<std::ptrdiff_t>(n_blocks)}, {}};
    ArrayView<OffsetT> eoff_view{
        new_entry_offsets.data(),
        {static_cast<std::ptrdiff_t>(n_blocks)}, {}};

    {
        nb::gil_scoped_release release;
        label_multiset::downsample_multiset(
            blocking, offsets_v, entry_sizes_v, entry_offsets_v, ids_v, counts_v,
            restrict_set,
            argmax_view, off_view, eoff_view,
            new_entry_sizes, new_ids, new_counts
        );
    }

    return std::make_tuple(
        std::move(new_argmax),
        std::move(new_offsets),
        std::move(new_entry_offsets),
        vector_to_array(std::move(new_entry_sizes)),
        vector_to_array(std::move(new_ids)),
        vector_to_array(std::move(new_counts))
    );
}

// ----- multiset_from_labels -----

template <class L>
std::tuple<nb::ndarray<nb::numpy, IdT, nb::c_contig>,
           nb::ndarray<nb::numpy, OffsetT, nb::c_contig>,
           nb::ndarray<nb::numpy, OffsetT, nb::c_contig>,
           nb::ndarray<nb::numpy, OffsetT, nb::c_contig>,
           nb::ndarray<nb::numpy, IdT, nb::c_contig>,
           nb::ndarray<nb::numpy, CountT, nb::c_contig>>
bind_multiset_from_labels_t(
    nb::ndarray<nb::numpy, const L, nb::c_contig> labels,
    const Blocking &blocking
) {
    if (labels.ndim() != blocking.ndim()) {
        throw std::invalid_argument(
            "labels ndim must match blocking ndim, got labels ndim=" +
            std::to_string(labels.ndim())
        );
    }
    std::vector<std::ptrdiff_t> shape(labels.ndim());
    for (std::size_t a = 0; a < labels.ndim(); ++a) {
        shape[a] = static_cast<std::ptrdiff_t>(labels.shape(a));
        if (shape[a] != blocking.roi_end()[a]) {
            throw std::invalid_argument(
                "labels shape must match blocking.roi_end()"
            );
        }
    }
    ConstArrayView<L> labels_v{labels.data(), shape, {}};

    const std::size_t n_blocks = static_cast<std::size_t>(blocking.number_of_blocks());
    auto argmax = alloc_1d<IdT>(n_blocks);
    auto offsets = alloc_1d<OffsetT>(n_blocks);
    auto entry_offsets = alloc_1d<OffsetT>(n_blocks);

    std::vector<OffsetT> entry_sizes;
    std::vector<IdT> ids;
    std::vector<CountT> counts;

    ArrayView<IdT> argmax_view{argmax.data(),
        {static_cast<std::ptrdiff_t>(n_blocks)}, {}};
    ArrayView<OffsetT> off_view{offsets.data(),
        {static_cast<std::ptrdiff_t>(n_blocks)}, {}};
    ArrayView<OffsetT> eoff_view{entry_offsets.data(),
        {static_cast<std::ptrdiff_t>(n_blocks)}, {}};

    {
        nb::gil_scoped_release release;
        label_multiset::multiset_from_labels<L, OffsetT, IdT, CountT>(
            labels_v, blocking,
            argmax_view, off_view, eoff_view,
            entry_sizes, ids, counts
        );
    }
    return std::make_tuple(
        std::move(argmax),
        std::move(offsets),
        std::move(entry_offsets),
        vector_to_array(std::move(entry_sizes)),
        vector_to_array(std::move(ids)),
        vector_to_array(std::move(counts))
    );
}

// ----- MultisetMerger -----

class PyMultisetMerger {
public:
    PyMultisetMerger(OffsetArray offsets, OffsetArray entry_sizes,
                     IdArray ids, CountArray counts)
        : merger_(
              view_1d(offsets, "offsets"),
              view_1d(entry_sizes, "entry_sizes"),
              view_1d(ids, "ids"),
              view_1d(counts, "counts")
          ) {}

    nb::ndarray<nb::numpy, OffsetT, nb::c_contig>
    update(OffsetArray unique_offsets, OffsetArray entry_sizes,
           IdArray ids, CountArray counts, OffsetOutArray offsets) {
        auto uo_v = view_1d(unique_offsets, "unique_offsets");
        auto es_v = view_1d(entry_sizes, "entry_sizes");
        auto ids_v = view_1d(ids, "ids");
        auto counts_v = view_1d(counts, "counts");
        if (offsets.ndim() != 1) {
            throw std::invalid_argument("offsets must have ndim 1");
        }
        ArrayView<OffsetT> off_view{
            offsets.data(),
            {static_cast<std::ptrdiff_t>(offsets.shape(0))},
            {}};
        {
            nb::gil_scoped_release release;
            merger_.update(uo_v, es_v, ids_v, counts_v, off_view);
        }
        return offsets;
    }

    nb::ndarray<nb::numpy, IdT, nb::c_contig> ids_array() const {
        return copy_to_array<IdT>(merger_.ids());
    }
    nb::ndarray<nb::numpy, CountT, nb::c_contig> counts_array() const {
        return copy_to_array<CountT>(merger_.counts());
    }
    nb::ndarray<nb::numpy, OffsetT, nb::c_contig> offsets_array() const {
        return copy_to_array<OffsetT>(merger_.offsets());
    }
    nb::ndarray<nb::numpy, OffsetT, nb::c_contig> entry_sizes_array() const {
        return copy_to_array<OffsetT>(merger_.entry_sizes());
    }

private:
    template <class T>
    static nb::ndarray<nb::numpy, T, nb::c_contig> copy_to_array(const std::vector<T> &v) {
        auto *data = new T[v.size()];
        std::copy(v.begin(), v.end(), data);
        nb::capsule owner(data, [](void *p) noexcept {
            delete[] static_cast<T *>(p);
        });
        std::size_t shape[1] = {v.size()};
        return nb::ndarray<nb::numpy, T, nb::c_contig>(data, 1, shape, owner);
    }

    label_multiset::MultisetMerger<OffsetT, IdT, CountT> merger_;
};

} // namespace

void bind_label_multiset(nb::module_ &m) {
    m.def("_read_subset", &bind_read_subset_flat,
          nb::arg("offsets"), nb::arg("sizes"),
          nb::arg("ids"), nb::arg("counts"),
          nb::arg("argsort") = true);

    m.def("_downsample_multiset", &bind_downsample_multiset,
          nb::arg("blocking"),
          nb::arg("offsets"), nb::arg("entry_sizes"), nb::arg("entry_offsets"),
          nb::arg("ids"), nb::arg("counts"),
          nb::arg("restrict_set") = -1);

    m.def("_multiset_from_labels_u32", &bind_multiset_from_labels_t<std::uint32_t>,
          nb::arg("labels"), nb::arg("blocking"));
    m.def("_multiset_from_labels_u64", &bind_multiset_from_labels_t<std::uint64_t>,
          nb::arg("labels"), nb::arg("blocking"));

    nb::class_<PyMultisetMerger>(m, "_MultisetMerger")
        .def(nb::init<OffsetArray, OffsetArray, IdArray, CountArray>(),
             nb::arg("offsets"), nb::arg("entry_sizes"),
             nb::arg("ids"), nb::arg("counts"))
        .def("update", &PyMultisetMerger::update,
             nb::arg("unique_offsets"), nb::arg("entry_sizes"),
             nb::arg("ids"), nb::arg("counts"), nb::arg("offsets"))
        .def("get_ids", &PyMultisetMerger::ids_array)
        .def("get_counts", &PyMultisetMerger::counts_array)
        .def("get_offsets", &PyMultisetMerger::offsets_array)
        .def("get_entry_sizes", &PyMultisetMerger::entry_sizes_array);
}

} // namespace bioimage_cpp::bindings
