from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.certificates import task_visibility_certificate
from satml_monitoring.hardening import (
    CandidateCheck,
    augmented_jacobian,
    greedy_task_aware_hardening,
)


class HardeningTests(unittest.TestCase):
    def test_added_information_is_monotone(self) -> None:
        base = np.array([[1.0, -1.0]])
        gradient = np.array([3.0, 2.0])
        before = task_visibility_certificate(base, gradient).gamma
        anchor = CandidateCheck.from_array("sum", [[1.0, 1.0]])
        after_matrix = augmented_jacobian(base, [anchor])
        after = task_visibility_certificate(after_matrix, gradient).gamma
        self.assertEqual(before, 0.0)
        self.assertGreater(after, before)

    def test_greedy_selects_task_relevant_check(self) -> None:
        base = np.array([[1.0, 0.0, 0.0]])
        gradient = np.array([0.0, 1.0, 0.0])
        candidates = [
            CandidateCheck.from_array("irrelevant", [[0.0, 0.0, 1.0]]),
            CandidateCheck.from_array("task-relevant", [[0.0, 1.0, 0.0]]),
        ]
        selected = greedy_task_aware_hardening(
            base, [gradient], candidates, budget=1
        )
        self.assertEqual(selected[0].name, "task-relevant")


if __name__ == "__main__":
    unittest.main()

