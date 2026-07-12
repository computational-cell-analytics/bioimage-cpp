"""Flow-field tracing utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import _core
from .._validation import strict_index


def _normalize_mask(fg_mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    mask = np.asarray(fg_mask)
    if mask.shape != shape:
        raise ValueError(f"fg_mask has shape {mask.shape}, expected {shape}")
    return np.ascontiguousarray(mask, dtype=np.uint8)


def _normalize_sigma(
    sigma: float | Sequence[float],
    ndim: int,
    spacing: Sequence[float] | None,
) -> float | tuple[float, ...]:
    if spacing is not None and ndim == 3:
        sig = np.asarray(sigma, dtype=np.float32)
        if sig.ndim == 0:
            sig = np.full(ndim, float(sig), dtype=np.float32)
        if sig.shape != (ndim,):
            raise ValueError(
                f"sigma must be a scalar or sequence of length {ndim}, got shape {sig.shape}"
            )
        sp = np.asarray(spacing, dtype=np.float32)
        if sp.shape != (ndim,):
            raise ValueError(
                f"spacing must be a sequence of length {ndim}, got shape {sp.shape}"
            )
        if not np.all(np.isfinite(sp)) or np.any(sp <= 0):
            raise ValueError("spacing values must be finite and positive")
        return tuple((sig / sp).tolist())
    return sigma


def compute_flow_density(
    flow: np.ndarray,
    fg_mask: np.ndarray,
    *,
    n_iter: int = 50,
    dt: float = 0.2,
    tol: float = 0.005,
    method: str = "rk2",
    restrict_to_mask: bool = True,
    sigma: float | Sequence[float] | None = None,
    spacing: Sequence[float] | None = None,
    number_of_threads: int = 1,
) -> np.ndarray:
    """Compute convergence density from tracing a flow field.

    Parameters
    ----------
    flow:
        Channel-first flow field with shape ``(ndim, *fg_mask.shape)``.
        The values must already point in the tracing direction. If starting
        from directed distances that point toward boundaries, pass ``-dist``.
    fg_mask:
        Foreground mask. Density is traced from, and retained only inside,
        non-zero mask pixels.
    n_iter:
        Maximum number of tracing iterations. Tracing stops earlier when all
        particles have converged (see ``tol``) or left the mask (see
        ``restrict_to_mask``).
    dt:
        Step size for every iteration. Must be finite and non-negative.
    tol:
        Per-particle convergence tolerance. A particle is frozen once its
        per-iteration displacement ``|dt * step|_inf`` drops below this value.
        Set to ``0`` to disable convergence-based early exit.
    method:
        Integration method. ``"euler"`` (one sample per step) or ``"rk2"``
        (midpoint method, two samples per step but converges in fewer
        iterations).
    restrict_to_mask:
        When ``True``, particles that step outside the foreground mask are
        frozen at their last in-mask position.
    sigma:
        Optional Gaussian smoothing sigma applied to the density after
        tracing.
    spacing:
        Optional physical spacing. For 3D data and scalar ``sigma``, smoothing
        uses ``sigma / spacing`` per axis, matching the reference convention.
    number_of_threads:
        Number of threads used for the particle-tracing iteration. The final
        density scatter and the (optional) Gaussian smoothing are not
        parallelized here. Results are deterministic regardless of the value.

    Returns
    -------
    numpy.ndarray
        ``float32`` density map with shape ``fg_mask.shape``.
    """
    array = np.asarray(flow)
    if array.ndim not in (3, 4):
        raise ValueError(
            "flow must have shape (ndim, *shape) for 2D or 3D data, "
            f"got ndim={array.ndim}"
        )

    ndim = array.ndim - 1
    if array.shape[0] != ndim:
        raise ValueError(
            f"flow first axis must match spatial ndim={ndim}, got {array.shape[0]}"
        )

    n_steps = strict_index(n_iter, "n_iter", minimum=0)
    step_size = float(dt)
    if not np.isfinite(step_size) or step_size < 0:
        raise ValueError("dt must be finite and >= 0")
    tolerance = float(tol)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tol must be finite and >= 0")
    integration_method = str(method)
    if integration_method not in ("euler", "rk2"):
        raise ValueError(f"method must be 'euler' or 'rk2', got '{integration_method}'")
    n_threads = strict_index(number_of_threads, "number_of_threads", minimum=1)

    # Finiteness of the flow tensor is validated once in the binding layer
    # (bindings/flow.cxx); it is deliberately not re-scanned here.
    contiguous_flow = np.ascontiguousarray(array, dtype=np.float32)
    mask = _normalize_mask(fg_mask, tuple(contiguous_flow.shape[1:]))

    if ndim == 2:
        density = _core._compute_flow_density_2d_float32(
            contiguous_flow,
            mask,
            n_steps,
            step_size,
            tolerance,
            integration_method,
            bool(restrict_to_mask),
            n_threads,
        )
    else:
        density = _core._compute_flow_density_3d_float32(
            contiguous_flow,
            mask,
            n_steps,
            step_size,
            tolerance,
            integration_method,
            bool(restrict_to_mask),
            n_threads,
        )

    if sigma is not None:
        from .. import filters

        sigma_for_filter = _normalize_sigma(sigma, ndim, spacing)
        density = filters.gaussian_smoothing(density, sigma_for_filter).astype(
            np.float32, copy=False
        )
        density *= mask

    return density
