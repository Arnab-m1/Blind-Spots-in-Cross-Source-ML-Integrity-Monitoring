"""Task-conditioned monitor visibility certificates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import (
    DEFAULT_RTOL,
    as_matrix,
    as_vector,
    harmful_nullspace_score,
    numerical_rank,
)


@dataclass(frozen=True)
class Certificate:
    gamma: float
    harmful_nullspace_score: float
    rank: int
    nullity: int
    information_matrix: NDArray[np.float64]
    inverse_covariance: NDArray[np.float64]
    pseudoinverse: NDArray[np.float64]
    denominator: float | None
    is_blind: bool


def _inverse_covariance(
    covariance: ArrayLike | None, output_dimension: int
) -> NDArray[np.float64]:
    if covariance is None:
        return np.eye(output_dimension, dtype=float)
    omega = as_matrix(covariance, name="covariance")
    if omega.shape != (output_dimension, output_dimension):
        raise ValueError("covariance shape must match monitor-output dimension")
    if not np.allclose(omega, omega.T, atol=1e-12, rtol=1e-10):
        raise ValueError("covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(omega)
    if eigenvalues.size and float(eigenvalues[0]) <= 0:
        raise ValueError("covariance must be positive definite")
    return np.linalg.inv(omega)


def task_visibility_certificate(
    jacobian: ArrayLike,
    task_gradient: ArrayLike,
    covariance: ArrayLike | None = None,
    *,
    rtol: float = DEFAULT_RTOL,
) -> Certificate:
    """Compute `Gamma_M` and its associated local geometry.

    `Gamma_M` is defined only for a nonzero task gradient. A harmful component
    in the restricted Jacobian's kernel yields an exact zero certificate.
    """

    matrix = as_matrix(jacobian, name="jacobian")
    gradient = as_vector(task_gradient, name="task_gradient")
    if matrix.shape[1] != gradient.size:
        raise ValueError("jacobian columns must match task-gradient dimension")
    if float(np.linalg.norm(gradient)) <= rtol:
        raise ValueError("task_gradient must be numerically nonzero")

    inverse_covariance = _inverse_covariance(covariance, matrix.shape[0])
    information = matrix.T @ inverse_covariance @ matrix
    rank = numerical_rank(matrix, rtol=rtol)
    nullity = matrix.shape[1] - rank
    harmful_score = harmful_nullspace_score(matrix, gradient, rtol=rtol)
    gradient_tolerance = rtol * max(float(np.linalg.norm(gradient)), 1.0)

    pseudoinverse = np.linalg.pinv(information, rtol=rtol)
    if harmful_score > gradient_tolerance:
        return Certificate(
            gamma=0.0,
            harmful_nullspace_score=harmful_score,
            rank=rank,
            nullity=nullity,
            information_matrix=information,
            inverse_covariance=inverse_covariance,
            pseudoinverse=pseudoinverse,
            denominator=None,
            is_blind=True,
        )

    denominator = float(gradient @ pseudoinverse @ gradient)
    if denominator <= 0 or not np.isfinite(denominator):
        raise ArithmeticError("positive-margin certificate has invalid denominator")
    return Certificate(
        gamma=float(1.0 / np.sqrt(denominator)),
        harmful_nullspace_score=harmful_score,
        rank=rank,
        nullity=nullity,
        information_matrix=information,
        inverse_covariance=inverse_covariance,
        pseudoinverse=pseudoinverse,
        denominator=denominator,
        is_blind=False,
    )

