"""Differentiable adapters for the small scikit-learn models used here."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import as_vector


def _activation_derivative(name: str, preactivation: NDArray[np.float64]) -> NDArray[np.float64]:
    if name == "identity":
        return np.ones_like(preactivation)
    if name == "relu":
        return (preactivation > 0).astype(float)
    if name == "tanh":
        value = np.tanh(preactivation)
        return 1.0 - value**2
    if name == "logistic":
        value = 1.0 / (1.0 + np.exp(-preactivation))
        return value * (1.0 - value)
    raise ValueError(f"unsupported activation: {name}")


def _activate(name: str, value: NDArray[np.float64]) -> NDArray[np.float64]:
    if name == "identity":
        return value
    if name == "relu":
        return np.maximum(value, 0.0)
    if name == "tanh":
        return np.tanh(value)
    if name == "logistic":
        return 1.0 / (1.0 + np.exp(-value))
    raise ValueError(f"unsupported activation: {name}")


def mlp_regressor_value_and_jacobian(
    model, point: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return output and exact local Jacobian for an `MLPRegressor`."""

    activation = as_vector(point, name="point")
    jacobian = np.eye(activation.size)
    layer_count = len(model.coefs_)
    for index, (weights, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
        preactivation = activation @ weights + bias
        activation_name = (
            model.out_activation_ if index == layer_count - 1 else model.activation
        )
        derivative = _activation_derivative(activation_name, preactivation)
        jacobian = derivative[:, None] * (weights.T @ jacobian)
        activation = _activate(activation_name, preactivation)
    return np.atleast_1d(activation), np.atleast_2d(jacobian)

