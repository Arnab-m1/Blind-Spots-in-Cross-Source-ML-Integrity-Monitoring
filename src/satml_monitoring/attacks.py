"""Closed-form local attacks induced by task-conditioned visibility."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .certificates import task_visibility_certificate
from .geometry import DEFAULT_RTOL, as_matrix, as_vector, harmful_nullspace_projection


def _unit(vector: NDArray[np.float64], *, tolerance: float) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm <= tolerance:
        raise ValueError("cannot normalize a numerically zero direction")
    return vector / norm


def task_gradient_attack(task_gradient: ArrayLike, budget: float) -> NDArray[np.float64]:
    gradient = as_vector(task_gradient, name="task_gradient")
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    return budget * _unit(gradient, tolerance=DEFAULT_RTOL)


def harmful_tangent_attack(
    jacobian: ArrayLike,
    task_gradient: ArrayLike,
    budget: float,
    *,
    rtol: float = DEFAULT_RTOL,
) -> NDArray[np.float64]:
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    projection = harmful_nullspace_projection(
        jacobian, task_gradient, rtol=rtol
    )
    return budget * _unit(projection, tolerance=rtol)


def minimum_evidence_attack(
    jacobian: ArrayLike,
    task_gradient: ArrayLike,
    task_shift: float,
    covariance: ArrayLike | None = None,
    *,
    rtol: float = DEFAULT_RTOL,
) -> NDArray[np.float64]:
    """Return a representative minimum-evidence local attack.

    In the blind case, the representative is the minimum-norm harmful
    nullspace vector satisfying `u.T @ b = task_shift`. Otherwise it is the
    pseudoinverse solution from Theorem B.
    """

    matrix = as_matrix(jacobian, name="jacobian")
    gradient = as_vector(task_gradient, name="task_gradient")
    certificate = task_visibility_certificate(
        matrix, gradient, covariance, rtol=rtol
    )
    if certificate.is_blind:
        projection = harmful_nullspace_projection(matrix, gradient, rtol=rtol)
        squared_norm = float(projection @ projection)
        if squared_norm <= rtol:
            raise ArithmeticError("blind certificate lacks a harmful direction")
        return task_shift * projection / squared_norm

    assert certificate.denominator is not None
    return (
        task_shift
        * (certificate.pseudoinverse @ gradient)
        / certificate.denominator
    )


def monitor_evidence(
    jacobian: ArrayLike,
    attack: ArrayLike,
    covariance: ArrayLike | None = None,
) -> float:
    matrix = as_matrix(jacobian, name="jacobian")
    vector = as_vector(attack, name="attack")
    if matrix.shape[1] != vector.size:
        raise ValueError("jacobian columns must match attack dimension")
    displacement = matrix @ vector
    if covariance is None:
        return float(np.linalg.norm(displacement))
    omega = as_matrix(covariance, name="covariance")
    return float(np.sqrt(displacement @ np.linalg.solve(omega, displacement)))


def finite_difference_jacobian(
    function,
    point: ArrayLike,
    *,
    step: float = 1e-5,
) -> NDArray[np.float64]:
    """Central-difference Jacobian for differentiable model adapters."""

    x = as_vector(point, name="point")
    if step <= 0:
        raise ValueError("step must be positive")
    baseline = np.atleast_1d(np.asarray(function(x), dtype=float))
    jacobian = np.empty((baseline.size, x.size), dtype=float)
    for column in range(x.size):
        offset = np.zeros_like(x)
        offset[column] = step
        plus = np.atleast_1d(np.asarray(function(x + offset), dtype=float))
        minus = np.atleast_1d(np.asarray(function(x - offset), dtype=float))
        if plus.shape != baseline.shape or minus.shape != baseline.shape:
            raise ValueError("function output shape changed under perturbation")
        jacobian[:, column] = (plus - minus) / (2.0 * step)
    return jacobian


def continuation_attack(
    monitor,
    task,
    point: ArrayLike,
    support: ArrayLike,
    budget: float,
    *,
    steps: int = 20,
    correction_steps: int = 5,
    finite_difference_step: float = 1e-5,
    rtol: float = DEFAULT_RTOL,
) -> NDArray[np.float64]:
    """Trace a locally monitor-preserving, task-increasing attack path."""

    x0 = as_vector(point, name="point")
    indices = np.asarray(support, dtype=int)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("support must be a nonempty index vector")
    if np.any(indices < 0) or np.any(indices >= x0.size):
        raise ValueError("support index outside point dimension")
    if budget < 0 or steps <= 0 or correction_steps < 0:
        raise ValueError("invalid continuation parameters")

    target_monitor = np.atleast_1d(np.asarray(monitor(x0), dtype=float))
    attack = np.zeros(indices.size, dtype=float)
    step_size = budget / steps
    for _ in range(steps):
        current = x0.copy()
        current[indices] += attack
        monitor_jacobian = finite_difference_jacobian(
            monitor, current, step=finite_difference_step
        )[:, indices]
        task_gradient = finite_difference_jacobian(
            lambda value: np.asarray([task(value)]),
            current,
            step=finite_difference_step,
        )[0, indices]
        tangent = harmful_nullspace_projection(
            monitor_jacobian, task_gradient, rtol=rtol
        )
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= rtol:
            break
        attack += step_size * tangent / tangent_norm

        for _ in range(correction_steps):
            current = x0.copy()
            current[indices] += attack
            residual = np.atleast_1d(np.asarray(monitor(current), dtype=float)) - target_monitor
            if float(np.linalg.norm(residual)) <= 1e-9:
                break
            jacobian = finite_difference_jacobian(
                monitor, current, step=finite_difference_step
            )[:, indices]
            attack -= np.linalg.pinv(jacobian, rcond=rtol) @ residual

        attack_norm = float(np.linalg.norm(attack))
        if attack_norm > budget:
            attack *= budget / attack_norm
    return attack


def constrained_oracle_attack(
    monitor,
    task,
    point: ArrayLike,
    support: ArrayLike,
    budget: float,
    monitor_budget: float,
    *,
    max_iterations: int = 300,
) -> NDArray[np.float64]:
    """Numerically maximize task shift under norm and monitor constraints."""

    try:
        from scipy.optimize import minimize
    except ImportError as error:
        raise RuntimeError("scipy is required for the constrained oracle") from error

    x0 = as_vector(point, name="point")
    indices = np.asarray(support, dtype=int)
    clean_monitor = np.atleast_1d(np.asarray(monitor(x0), dtype=float))
    clean_task = float(task(x0))

    def attacked(local_attack: NDArray[np.float64]) -> NDArray[np.float64]:
        value = x0.copy()
        value[indices] += local_attack
        return value

    result = minimize(
        lambda value: -(float(task(attacked(value))) - clean_task),
        np.zeros(indices.size, dtype=float),
        method="SLSQP",
        constraints=[
            {"type": "ineq", "fun": lambda value: budget - np.linalg.norm(value)},
            {
                "type": "ineq",
                "fun": lambda value: monitor_budget
                - np.linalg.norm(
                    np.atleast_1d(np.asarray(monitor(attacked(value)), dtype=float))
                    - clean_monitor
                ),
            },
        ],
        options={"maxiter": max_iterations, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"constrained oracle failed: {result.message}")
    return np.asarray(result.x, dtype=float)
