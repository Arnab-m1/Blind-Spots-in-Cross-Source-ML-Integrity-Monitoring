from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.attacks import minimum_evidence_attack, monitor_evidence
from satml_monitoring.certificates import task_visibility_certificate
from satml_monitoring.geometry import harmful_nullspace_score, nullspace, numerical_rank


class GeometryTests(unittest.TestCase):
    def test_nullspace_and_rank(self) -> None:
        jacobian = np.array([[1.0, -1.0]])
        basis = nullspace(jacobian)
        self.assertEqual(numerical_rank(jacobian), 1)
        self.assertEqual(basis.shape, (2, 1))
        np.testing.assert_allclose(jacobian @ basis, 0.0, atol=1e-12)
        np.testing.assert_allclose(basis.T @ basis, np.eye(1), atol=1e-12)

    def test_harmful_projection(self) -> None:
        jacobian = np.array([[1.0, -1.0]])
        gradient = np.array([3.0, 2.0])
        self.assertAlmostEqual(
            harmful_nullspace_score(jacobian, gradient), 5.0 / np.sqrt(2.0)
        )


class CertificateTests(unittest.TestCase):
    def test_blind_certificate_and_attack(self) -> None:
        jacobian = np.array([[1.0, -1.0]])
        gradient = np.array([3.0, 2.0])
        certificate = task_visibility_certificate(jacobian, gradient)
        attack = minimum_evidence_attack(jacobian, gradient, task_shift=5.0)
        self.assertTrue(certificate.is_blind)
        self.assertEqual(certificate.gamma, 0.0)
        np.testing.assert_allclose(jacobian @ attack, 0.0, atol=1e-12)
        self.assertAlmostEqual(float(gradient @ attack), 5.0)

    def test_positive_certificate_closed_form(self) -> None:
        jacobian = np.diag([2.0, 1.0])
        gradient = np.array([1.0, 1.0])
        certificate = task_visibility_certificate(jacobian, gradient)
        expected_gamma = 1.0 / np.sqrt(1.0 / 4.0 + 1.0)
        self.assertFalse(certificate.is_blind)
        self.assertAlmostEqual(certificate.gamma, expected_gamma)

        attack = minimum_evidence_attack(jacobian, gradient, task_shift=2.0)
        self.assertAlmostEqual(float(gradient @ attack), 2.0)
        self.assertAlmostEqual(
            monitor_evidence(jacobian, attack), 2.0 * expected_gamma
        )

    def test_covariance_weighting(self) -> None:
        jacobian = np.eye(2)
        gradient = np.array([1.0, 0.0])
        covariance = np.diag([4.0, 1.0])
        certificate = task_visibility_certificate(
            jacobian, gradient, covariance
        )
        self.assertAlmostEqual(certificate.gamma, 0.5)

    def test_zero_gradient_rejected(self) -> None:
        with self.assertRaises(ValueError):
            task_visibility_certificate(np.eye(2), np.zeros(2))

    def test_invertible_coordinate_change_preserves_gamma(self) -> None:
        jacobian = np.array([[2.0, 0.5], [0.0, 1.0]])
        gradient = np.array([1.0, -0.25])
        change = np.array([[1.5, 0.2], [-0.4, 0.8]])
        original = task_visibility_certificate(jacobian, gradient)
        transformed = task_visibility_certificate(
            jacobian @ change, change.T @ gradient
        )
        self.assertAlmostEqual(original.gamma, transformed.gamma)

    def test_task_scaling_rescales_gamma(self) -> None:
        jacobian = np.diag([2.0, 1.0])
        gradient = np.array([1.0, 1.0])
        original = task_visibility_certificate(jacobian, gradient)
        scaled = task_visibility_certificate(jacobian, -3.0 * gradient)
        self.assertAlmostEqual(scaled.gamma, original.gamma / 3.0)


if __name__ == "__main__":
    unittest.main()
