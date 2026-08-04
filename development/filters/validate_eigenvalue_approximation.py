"""Validate the internal 3 x 3 eigenvalue approximations.

Run this script after an editable build:

    python development/filters/validate_eigenvalue_approximation.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np


ACOS_COEFFICIENTS = np.array(
    [
        1.4866664409637451,
        -0.07770606875419617,
        0.005770874209702015,
        -0.0005768424598500133,
        0.00006643808592343703,
        -0.000008315640116052236,
        0.0000010878551393034286,
        -0.00000014935945102934056,
    ],
    dtype=np.float32,
)
COS_COEFFICIENTS = np.array(
    [
        1.0,
        -0.4999999701976776,
        0.041666433215141296,
        -0.0013882536441087723,
        0.000024095119442790747,
    ],
    dtype=np.float32,
)
SINC_COEFFICIENTS = np.array(
    [
        1.0,
        -0.1666666567325592,
        0.008333305828273296,
        -0.0001983467664103955,
        0.0000026878346943703946,
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the internal vector-math approximations."
    )
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=91)
    return parser.parse_args()


def approximate_acos(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    magnitude = np.abs(values)
    z = magnitude + magnitude - np.float32(1.0)
    two_z = z + z
    b1 = np.zeros_like(z)
    b2 = np.zeros_like(z)
    for coefficient in ACOS_COEFFICIENTS[:0:-1]:
        b0 = two_z * b1 + (coefficient - b2)
        b2 = b1
        b1 = b0
    q = z * b1 + (ACOS_COEFFICIENTS[0] - b2)
    base = np.sqrt(np.maximum(np.float32(0.0), np.float32(1.0) - magnitude)) * q
    return np.where(values < 0.0, np.float32(np.pi) - base, base)


def approximate_cos(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float32)
    u = phi * phi
    result = np.full_like(u, COS_COEFFICIENTS[-1])
    for coefficient in COS_COEFFICIENTS[-2::-1]:
        result = result * u + coefficient
    return result


def approximate_sin(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float32)
    u = phi * phi
    sinc = np.full_like(u, SINC_COEFFICIENTS[-1])
    for coefficient in SINC_COEFFICIENTS[-2::-1]:
        sinc = sinc * u + coefficient
    return phi * sinc


def components_from_matrices(matrices: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack(
            [
                matrices[:, 0, 0],
                matrices[:, 0, 1],
                matrices[:, 0, 2],
                matrices[:, 1, 1],
                matrices[:, 1, 2],
                matrices[:, 2, 2],
            ]
        ),
        dtype=np.float32,
    )


def direct_eigenvalues(components: np.ndarray) -> np.ndarray:
    from bioimage_cpp import _core

    out = np.empty((components.shape[1], 3), dtype=np.float32)
    _core._filters_ev3_symmetric_float32(components, out)
    return out


def maximum_scaled_error(got: np.ndarray, expected: np.ndarray) -> float:
    scale = np.maximum(
        np.max(np.abs(expected), axis=1), np.finfo(np.float32).tiny
    )
    return float(np.max(np.max(np.abs(got - expected), axis=1) / scale))


def random_matrices(samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    components = rng.normal(size=(6, samples)).astype(np.float32)
    scales = np.power(
        10.0, rng.uniform(-15.0, 15.0, size=samples)
    ).astype(np.float32)
    components *= scales

    matrices = np.empty((samples, 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = components[0]
    matrices[:, 0, 1] = matrices[:, 1, 0] = components[1]
    matrices[:, 0, 2] = matrices[:, 2, 0] = components[2]
    matrices[:, 1, 1] = components[3]
    matrices[:, 1, 2] = matrices[:, 2, 1] = components[4]
    matrices[:, 2, 2] = components[5]
    return matrices


def adversarial_matrices(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spectra = []
    smallest_subnormal = float(
        np.nextafter(np.float32(0.0), np.float32(1.0))
    )
    for scale in (
        smallest_subnormal,
        np.finfo(np.float32).tiny,
        1e-20,
        1e-8,
        1.0,
        1e8,
        1e20,
    ):
        spectra.extend(
            [
                [0.0, 0.0, 0.0],
                [scale, scale, scale],
                [3.0 * scale, 3.0 * scale, -2.0 * scale],
                [3.0 * scale, -2.0 * scale, -2.0 * scale],
                [scale, scale * (1.0 + 2e-6), -0.5 * scale],
                [scale, scale * (1.0 - 2e-6), -0.5 * scale],
            ]
        )

    matrices = []
    for eigenvalues in spectra:
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        matrices.append(
            (q @ np.diag(eigenvalues) @ q.T).astype(np.float32)
        )
    return np.stack(matrices)


def main() -> int:
    args = parse_args()
    if args.samples < 8:
        raise ValueError("--samples must be at least 8")

    domain = np.linspace(-1.0, 1.0, args.samples + 1, dtype=np.float32)
    acos_error = float(
        np.max(
            np.abs(
                approximate_acos(domain).astype(np.float64)
                - np.arccos(domain.astype(np.float64))
            )
        )
    )
    phi = np.linspace(0.0, np.pi / 3.0, args.samples + 1, dtype=np.float32)
    phi64 = phi.astype(np.float64)
    cos_error = float(
        np.max(np.abs(approximate_cos(phi).astype(np.float64) - np.cos(phi64)))
    )
    sin_error = float(
        np.max(np.abs(approximate_sin(phi).astype(np.float64) - np.sin(phi64)))
    )

    random = random_matrices(args.samples, args.seed)
    random_components = components_from_matrices(random)
    random_ref = np.linalg.eigvalsh(random)[..., ::-1].astype(np.float32)
    automatic = direct_eigenvalues(random_components)
    random_error = maximum_scaled_error(automatic, random_ref)

    adversarial = adversarial_matrices(args.seed + 1)
    adversarial_ref = np.linalg.eigvalsh(adversarial)[..., ::-1].astype(np.float32)
    adversarial_got = direct_eigenvalues(components_from_matrices(adversarial))
    adversarial_error = maximum_scaled_error(adversarial_got, adversarial_ref)

    from bioimage_cpp import _core

    scalar_error = 0.0
    original_force_scalar = os.environ.get("BIOIMAGE_CPP_FILTERS_FORCE_SCALAR")
    try:
        if _core._filters_eigenvalue_backend() == "avx2":
            os.environ["BIOIMAGE_CPP_FILTERS_FORCE_SCALAR"] = "1"
            scalar = direct_eigenvalues(random_components)
            scalar_error = maximum_scaled_error(automatic, scalar)
    finally:
        if original_force_scalar is None:
            os.environ.pop("BIOIMAGE_CPP_FILTERS_FORCE_SCALAR", None)
        else:
            os.environ["BIOIMAGE_CPP_FILTERS_FORCE_SCALAR"] = original_force_scalar

    print(f"acos max absolute error:       {acos_error:.3e}")
    print(f"cos max absolute error:        {cos_error:.3e}")
    print(f"sin max absolute error:        {sin_error:.3e}")
    print(f"random eigen max scaled error: {random_error:.3e}")
    print(f"tail eigen max scaled error:   {adversarial_error:.3e}")
    if scalar_error:
        print(f"AVX2/scalar max scaled error:  {scalar_error:.3e}")

    passed = (
        acos_error < 6e-7
        and cos_error < 1e-7
        and sin_error < 1e-7
        and random_error < 2e-5
        and adversarial_error < 3e-4
        and scalar_error < 3e-5
        and np.isfinite(automatic).all()
        and np.isfinite(adversarial_got).all()
    )
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
