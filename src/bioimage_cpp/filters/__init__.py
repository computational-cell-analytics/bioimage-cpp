"""Gaussian-family image filters and symmetric tensor operations."""

from ._filters import (
    gaussian_derivative,
    gaussian_gradient_magnitude,
    gaussian_smoothing,
    hessian_of_gaussian_eigenvalues,
    laplacian_of_gaussian,
    structure_tensor,
    structure_tensor_eigenvalues,
    symmetric_eigenvector,
)

__all__ = [
    "gaussian_smoothing",
    "gaussian_derivative",
    "gaussian_gradient_magnitude",
    "laplacian_of_gaussian",
    "hessian_of_gaussian_eigenvalues",
    "structure_tensor",
    "structure_tensor_eigenvalues",
    "symmetric_eigenvector",
]
