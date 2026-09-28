"""Task-aware monitor hardening primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .certificates import task_visibility_certificate
from .geometry import DEFAULT_RTOL, as_matrix, as_vector


@dataclass(frozen=True)
class CandidateCheck:
    name: str
    jacobian_rows: NDArray[np.float64]

    @classmethod
    def from_array(cls, name: str, rows: ArrayLike) -> "CandidateCheck":
        if not name:
            raise ValueError("candidate name must be nonempty")
        return cls(name=name, jacobian_rows=as_matrix(rows, name="jacobian_rows"))


def augmented_jacobian(
    base: ArrayLike, candidates: Iterable[CandidateCheck]
) -> NDArray[np.float64]:
    matrix = as_matrix(base, name="base")
    rows = [matrix]
    for candidate in candidates:
        if candidate.jacobian_rows.shape[1] != matrix.shape[1]:
            raise ValueError("candidate width must match base Jacobian")
        rows.append(candidate.jacobian_rows)
    return np.vstack(rows)


def greedy_task_aware_hardening(
    base_jacobian: ArrayLike,
    task_gradients: Sequence[ArrayLike],
    candidates: Sequence[CandidateCheck],
    budget: int,
    *,
    rtol: float = DEFAULT_RTOL,
) -> list[CandidateCheck]:
    """Greedily maximize the worst task-conditioned visibility margin.

    This is a heuristic. No optimality or submodularity guarantee is claimed.
    """

    base = as_matrix(base_jacobian, name="base_jacobian")
    gradients = [as_vector(value, name="task_gradient") for value in task_gradients]
    if not gradients:
        raise ValueError("at least one task gradient is required")
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if any(gradient.size != base.shape[1] for gradient in gradients):
        raise ValueError("task-gradient dimensions must match the Jacobian")

    selected: list[CandidateCheck] = []
    remaining = list(candidates)
    for _ in range(min(budget, len(remaining))):
        best_index = -1
        best_margin = -np.inf
        for index, candidate in enumerate(remaining):
            augmented = augmented_jacobian(base, [*selected, candidate])
            margin = min(
                task_visibility_certificate(
                    augmented, gradient, rtol=rtol
                ).gamma
                for gradient in gradients
            )
            if margin > best_margin:
                best_index = index
                best_margin = margin
        selected.append(remaining.pop(best_index))
    return selected

