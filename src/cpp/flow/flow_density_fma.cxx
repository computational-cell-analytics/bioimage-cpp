#include "bioimage_cpp/flow/flow_density.hxx"

#if !defined(BIOIMAGE_FLOW_FMA_DISPATCH)
#error "flow_density_fma.cxx must only be built with BIOIMAGE_FLOW_FMA_DISPATCH"
#endif

namespace bioimage_cpp::flow::detail {

void trace_all_fma_2d(
    std::vector<std::array<float, 2>> &positions,
    const std::array<const float *, 2> &channels,
    const GridLayout<2> &grid,
    const std::uint8_t *mask,
    const std::size_t n_threads,
    const std::size_t n_iter,
    const float dt,
    const float tol
) {
    trace_all<2, true, true, true, true>(
        positions, channels, grid, mask, n_threads, n_iter, dt, tol
    );
}

void trace_all_fma_3d(
    std::vector<std::array<float, 3>> &positions,
    const std::array<const float *, 3> &channels,
    const GridLayout<3> &grid,
    const std::uint8_t *mask,
    const std::size_t n_threads,
    const std::size_t n_iter,
    const float dt,
    const float tol
) {
    trace_all<3, true, true, true, true>(
        positions, channels, grid, mask, n_threads, n_iter, dt, tol
    );
}

} // namespace bioimage_cpp::flow::detail
