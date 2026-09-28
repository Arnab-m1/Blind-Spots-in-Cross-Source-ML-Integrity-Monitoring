from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.attacks import finite_difference_jacobian
from satml_monitoring.model_adapters import mlp_regressor_value_and_jacobian


class FakeMLP:
    activation = "tanh"
    out_activation_ = "identity"
    coefs_ = [
        np.array([[0.5, -0.2], [0.3, 0.7]]),
        np.array([[1.2], [-0.4]]),
    ]
    intercepts_ = [np.array([0.1, -0.1]), np.array([0.2])]

    def predict_one(self, point: np.ndarray) -> np.ndarray:
        hidden = np.tanh(point @ self.coefs_[0] + self.intercepts_[0])
        return hidden @ self.coefs_[1] + self.intercepts_[1]


class ModelAdapterTests(unittest.TestCase):
    def test_mlp_jacobian_matches_finite_difference(self) -> None:
        model = FakeMLP()
        point = np.array([0.4, -0.7])
        value, jacobian = mlp_regressor_value_and_jacobian(model, point)
        numerical = finite_difference_jacobian(model.predict_one, point)
        np.testing.assert_allclose(value, model.predict_one(point), atol=1e-12)
        np.testing.assert_allclose(jacobian, numerical, atol=1e-8)


if __name__ == "__main__":
    unittest.main()

