from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.attacks import (
    constrained_oracle_attack,
    continuation_attack,
    finite_difference_jacobian,
)


class NonlinearAttackTests(unittest.TestCase):
    def test_finite_difference_jacobian(self) -> None:
        function = lambda x: np.array([x[0] ** 2 + x[1], x[0] * x[1]])
        point = np.array([2.0, 3.0])
        expected = np.array([[4.0, 1.0], [3.0, 2.0]])
        np.testing.assert_allclose(
            finite_difference_jacobian(function, point), expected, atol=1e-8
        )

    def test_continuation_preserves_linear_monitor(self) -> None:
        monitor = lambda x: np.array([x[0] - x[1]])
        task = lambda x: 3.0 * x[0] + 2.0 * x[1]
        point = np.zeros(2)
        attack = continuation_attack(monitor, task, point, [0, 1], 1.0)
        attacked = point + attack
        np.testing.assert_allclose(monitor(attacked), monitor(point), atol=1e-8)
        self.assertAlmostEqual(np.linalg.norm(attack), 1.0, places=7)
        self.assertGreater(task(attacked), task(point))

    def test_oracle_respects_constraints(self) -> None:
        monitor = lambda x: np.array([x[0] - x[1]])
        task = lambda x: 3.0 * x[0] + 2.0 * x[1]
        point = np.zeros(2)
        attack = constrained_oracle_attack(
            monitor, task, point, [0, 1], budget=1.0, monitor_budget=1e-7
        )
        self.assertLessEqual(np.linalg.norm(attack), 1.0 + 1e-7)
        self.assertLessEqual(abs(attack[0] - attack[1]), 2e-7)
        self.assertGreater(task(attack), 3.5)


if __name__ == "__main__":
    unittest.main()
