"""Numerically stable geometry for support-restricted monitor Jacobians."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_RTOL = 1e-10


def as_matrix(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def as_vector(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector


def singular_tolerance(
    singular_values: NDArray[np.float64], shape: tuple[int, int], rtol: float
) -> float:
    if rtol <= 0:
        raise ValueError("rtol must be positive")
    scale = float(singular_values[0]) if singular_values.size else 0.0
    return rtol * max(shape) * max(scale, 1.0)


def numerical_rank(jacobian: ArrayLike, *, rtol: float = DEFAULT_RTOL) -> int:
    matrix = as_matrix(jacobian, name="jacobian")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = singular_tolerance(singular_values, matrix.shape, rtol)
    return int(np.count_nonzero(singular_values > tolerance))


def nullspace(
    jacobian: ArrayLike, *, rtol: float = DEFAULT_RTOL
) -> NDArray[np.float64]:
    """Return an orthonormal basis whose columns span `ker(jacobian)`."""

    matrix = as_matrix(jacobian, name="jacobian")
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    tolerance = singular_tolerance(singular_values, matrix.shape, rtol)
    rank = int(np.count_nonzero(singular_values > tolerance))
    return vh[rank:].T.copy()


def harmful_nullspace_projection(
    jacobian: ArrayLike,
    task_gradient: ArrayLike,
    *,
    rtol: float = DEFAULT_RTOL,
) -> NDArray[np.float64]:
    matrix = as_matrix(jacobian, name="jacobian")
    gradient = as_vector(task_gradient, name="task_gradient")
    if matrix.shape[1] != gradient.size:
        raise ValueError("jacobian columns must match task-gradient dimension")
    basis = nullspace(matrix, rtol=rtol)
    if basis.shape[1] == 0:
        return np.zeros_like(gradient)
    return basis @ (basis.T @ gradient)


def harmful_nullspace_score(
    jacobian: ArrayLike,
    task_gradient: ArrayLike,
    *,
    rtol: float = DEFAULT_RTOL,
) -> float:
    projection = harmful_nullspace_projection(
        jacobian, task_gradient, rtol=rtol
    )
    return float(np.linalg.norm(projection))

