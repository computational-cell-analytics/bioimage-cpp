#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace bioimage_cpp::filters {

// Eigenvalues of a 2x2 symmetric matrix
//   [xx  xy]
//   [xy  yy]
// computed via the closed form (tr/2) +- sqrt(((xx-yy)/2)^2 + xy^2), then
// sorted descending. SoA inputs/outputs of length n.
inline void ev2_symmetric_descending(
    const float *__restrict xx,
    const float *__restrict xy,
    const float *__restrict yy,
    float *__restrict e_large,
    float *__restrict e_small,
    std::ptrdiff_t n
) {
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const float half_tr = 0.5f * (xx[i] + yy[i]);
        const float half_diff = 0.5f * (xx[i] - yy[i]);
        const float disc = std::sqrt(half_diff * half_diff + xy[i] * xy[i]);
        e_large[i] = half_tr + disc;
        e_small[i] = half_tr - disc;
    }
}

namespace detail {

// Eigenvalues of a 3x3 symmetric matrix at a single pixel, sorted descending.
// Uses David Eberly's trigonometric closed-form solution (see VIGRA's
// mathutil.hxx). Steps: normalise by the maximum absolute element to guard
// against overflow; compute the deviatoric matrix B = A - (tr/3) I; the three
// eigenvalues are a + 2*p*cos(phi + k*2pi/3) where p = sqrt(tr(B^2)/6),
// phi = acos(clamp(det(B)/(2 p^3), -1, 1))/3.
inline void ev3_one_descending(
    float a00, float a01, float a02,
    float a11, float a12, float a22,
    float &e0, float &e1, float &e2
) {
    float m = std::abs(a00);
    m = std::max(m, std::abs(a01));
    m = std::max(m, std::abs(a02));
    m = std::max(m, std::abs(a11));
    m = std::max(m, std::abs(a12));
    m = std::max(m, std::abs(a22));

    if (m == 0.0f) {
        e0 = 0.0f;
        e1 = 0.0f;
        e2 = 0.0f;
        return;
    }

    const float inv_m = 1.0f / m;
    a00 *= inv_m;
    a01 *= inv_m;
    a02 *= inv_m;
    a11 *= inv_m;
    a12 *= inv_m;
    a22 *= inv_m;

    const float a = (a00 + a11 + a22) * (1.0f / 3.0f);
    const float b00 = a00 - a;
    const float b11 = a11 - a;
    const float b22 = a22 - a;

    // trace(B B^T) = b00^2 + b11^2 + b22^2 + 2*(a01^2 + a02^2 + a12^2)
    const float trace_b2 =
        b00 * b00 + b11 * b11 + b22 * b22 + 2.0f * (a01 * a01 + a02 * a02 + a12 * a12);
    const float p2 = trace_b2 * (1.0f / 6.0f);

    if (!(p2 > 0.0f)) {
        const float val = a * m;
        e0 = val;
        e1 = val;
        e2 = val;
        return;
    }
    const float p = std::sqrt(p2);

    const float det_b = b00 * (b11 * b22 - a12 * a12)
                        - a01 * (a01 * b22 - a12 * a02)
                        + a02 * (a01 * a12 - b11 * a02);

    float r = det_b / (2.0f * p2 * p);
    // Clamp to [-1, 1] to guard against floating-point excursions that would
    // otherwise produce NaN from acos.
    r = std::max(-1.0f, std::min(1.0f, r));

    constexpr float kTwoPiOver3 = 2.0943951023931953f;
    const float phi = std::acos(r) * (1.0f / 3.0f);
    const float two_p = 2.0f * p;

    const float root_large = a + two_p * std::cos(phi);
    const float root_small = a + two_p * std::cos(phi + kTwoPiOver3);
    const float root_mid = 3.0f * a - root_large - root_small;

    e0 = m * root_large;
    e1 = m * root_mid;
    e2 = m * root_small;
}

} // namespace detail

// Eigenvalues of a 3x3 symmetric matrix
//   [xx  xy  xz]
//   [xy  yy  yz]
//   [xz  yz  zz]
// sorted descending (e0 >= e1 >= e2). SoA inputs/outputs of length n.
inline void ev3_symmetric_descending(
    const float *__restrict xx,
    const float *__restrict xy,
    const float *__restrict xz,
    const float *__restrict yy,
    const float *__restrict yz,
    const float *__restrict zz,
    float *__restrict e0,
    float *__restrict e1,
    float *__restrict e2,
    std::ptrdiff_t n
) {
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        detail::ev3_one_descending(
            xx[i], xy[i], xz[i], yy[i], yz[i], zz[i],
            e0[i], e1[i], e2[i]
        );
    }
}

} // namespace bioimage_cpp::filters
