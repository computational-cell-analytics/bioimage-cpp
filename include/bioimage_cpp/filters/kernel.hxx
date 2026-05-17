#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace bioimage_cpp::filters {

// Half-Gaussian-derivative 1D kernel.
//
// Storage convention:
//   half_coefs has length `radius + 1`. half_coefs[0] is the centre tap.
//   For symmetric kernels (order 0/2), kernel[k] = kernel[-k] = half_coefs[|k|].
//   For antisymmetric kernels (order 1), kernel[k] = sign(k) * half_coefs[|k|],
//     with kernel[0] = 0 (so half_coefs[0] is always 0 for antisymmetric).
//
// Convolution convention (cross-correlation):
//   output[i] = sum_{k=-R..R} kernel[k] * input[i + k]
// For symmetric:    output[i] = h[0]*x[i] + sum_{k>=1} h[k]*(x[i+k] + x[i-k])
// For antisymmetric: output[i] = sum_{k>=1} h[k]*(x[i+k] - x[i-k])
//
// Normalisation targets are chosen so that convolving with the polynomial
// f(x) = x^order / order! yields a constant 1 at the centre, i.e. these
// kernels evaluate the derivative directly (not the derivative scaled by
// sigma^order, in contrast to some other conventions).
struct Kernel1D {
    std::vector<float> half_coefs;
    int radius = 0;
    int order = 0;
    bool is_symmetric = true;
};

inline int gaussian_kernel_radius(double sigma, int order, double window_ratio) {
    const double effective_ratio =
        (window_ratio > 0.0) ? window_ratio : (3.0 + 0.5 * static_cast<double>(order));
    const int r = static_cast<int>(std::ceil(effective_ratio * sigma));
    return std::max(1, r);
}

// Build a 1D Gaussian (derivative) kernel.
// sigma > 0; order in {0, 1, 2}; window_ratio = 0 selects the default radius
// (ceil((3 + 0.5*order)*sigma)).
inline Kernel1D gaussian_kernel(double sigma, int order, double window_ratio = 0.0) {
    if (!(sigma > 0.0)) {
        throw std::invalid_argument(
            "sigma must be positive, got sigma=" + std::to_string(sigma)
        );
    }
    if (order < 0 || order > 2) {
        throw std::invalid_argument(
            "order must be 0, 1 or 2, got order=" + std::to_string(order)
        );
    }

    const int radius = gaussian_kernel_radius(sigma, order, window_ratio);
    const double inv_sigma2 = 1.0 / (sigma * sigma);
    const double inv_2sigma2 = 0.5 * inv_sigma2;

    Kernel1D kernel;
    kernel.radius = radius;
    kernel.order = order;
    kernel.is_symmetric = (order != 1);
    kernel.half_coefs.assign(static_cast<std::size_t>(radius + 1), 0.0f);

    // Generate raw (unnormalised) half-kernel values in double precision.
    std::vector<double> raw(static_cast<std::size_t>(radius + 1));
    for (int i = 0; i <= radius; ++i) {
        const double x = static_cast<double>(i);
        const double g = std::exp(-x * x * inv_2sigma2);
        switch (order) {
            case 0:
                raw[i] = g;
                break;
            case 1:
                // Store +x/sigma^2 * g(x) for i>=0 so the antisymmetric loop
                // combines as (x[i+k] - x[i-k]) and produces the standard
                // positive d/dx convention.
                raw[i] = (x * inv_sigma2) * g;
                break;
            case 2:
                raw[i] = (x * x * inv_sigma2 * inv_sigma2 - inv_sigma2) * g;
                break;
        }
    }

    if (order == 0) {
        // Sum to 1.
        double sum = raw[0];
        for (int i = 1; i <= radius; ++i) sum += 2.0 * raw[i];
        const double scale = 1.0 / sum;
        for (int i = 0; i <= radius; ++i) {
            kernel.half_coefs[static_cast<std::size_t>(i)] = static_cast<float>(raw[i] * scale);
        }
    } else if (order == 1) {
        // Antisymmetric. Normalise so sum_{k>=1} k * h[k] = 0.5, i.e. the
        // discrete first derivative of f(x)=x evaluates to 1.
        raw[0] = 0.0;  // ensure centre tap is exactly zero
        double moment = 0.0;
        for (int i = 1; i <= radius; ++i) {
            moment += static_cast<double>(i) * raw[i];
        }
        if (moment == 0.0) {
            throw std::runtime_error("degenerate Gaussian first-derivative kernel");
        }
        const double scale = 0.5 / moment;
        for (int i = 0; i <= radius; ++i) {
            kernel.half_coefs[static_cast<std::size_t>(i)] = static_cast<float>(raw[i] * scale);
        }
    } else {
        // order == 2. DC-correct (kernel sums to 0), then normalise so
        // sum_{k>=1} k^2 * h[k] = 1, i.e. discrete second derivative of
        // f(x) = x^2/2 evaluates to 1.
        double sum = raw[0];
        for (int i = 1; i <= radius; ++i) sum += 2.0 * raw[i];
        const double dc = sum / static_cast<double>(2 * radius + 1);
        for (int i = 0; i <= radius; ++i) raw[i] -= dc;

        double moment = 0.0;
        for (int i = 1; i <= radius; ++i) {
            moment += static_cast<double>(i) * static_cast<double>(i) * raw[i];
        }
        if (moment == 0.0) {
            throw std::runtime_error("degenerate Gaussian second-derivative kernel");
        }
        const double scale = 1.0 / moment;
        for (int i = 0; i <= radius; ++i) {
            kernel.half_coefs[static_cast<std::size_t>(i)] = static_cast<float>(raw[i] * scale);
        }
    }

    return kernel;
}

} // namespace bioimage_cpp::filters
