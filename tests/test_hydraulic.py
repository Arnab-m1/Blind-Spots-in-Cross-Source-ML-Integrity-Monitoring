from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.datasets.hydraulic import (
    SENSOR_SPECS,
    grouped_splits,
    split_audit,
    summarize_cycles,
    summary_parameter_jacobian,
    transform_summary_features,
)


class HydraulicPreprocessingTests(unittest.TestCase):
    def test_summary_features(self) -> None:
        values = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
        summary = summarize_cycles(values)
        np.testing.assert_allclose(summary[:, 0], [2.0, 2.0])
        np.testing.assert_allclose(summary[:, 1], [np.sqrt(2.0 / 3.0), 0.0])
        np.testing.assert_allclose(summary[:, 2], [np.sqrt(14.0 / 3.0), 2.0])
        np.testing.assert_allclose(summary[:, 3:], [[1.0, 3.0], [2.0, 2.0]])

    def test_grouped_split_is_complete_and_disjoint(self) -> None:
        targets = np.column_stack([np.arange(50) % 2, np.arange(50) % 3])
        splits = grouped_splits(targets, seed=17, block_size=5)
        audit = split_audit(targets, splits)
        self.assertTrue(audit["disjoint_and_complete"])
        for indices in splits.values():
            represented_groups = set((indices // 5).tolist())
            for other_indices in splits.values():
                if indices is not other_indices:
                    self.assertFalse(represented_groups & set((other_indices // 5).tolist()))

    def test_bias_gain_transform_matches_raw_trace_summaries(self) -> None:
        raw = np.tile(np.array([[1.0, 2.0, 4.0]]), (len(SENSOR_SPECS), 1))
        summaries = np.hstack([summarize_cycles(raw[i : i + 1]) for i in range(len(SENSOR_SPECS))])
        gain = np.linspace(-0.1, 0.1, len(SENSOR_SPECS))
        bias = np.linspace(-0.3, 0.3, len(SENSOR_SPECS))
        transformed = transform_summary_features(summaries, gain, bias)
        expected = np.hstack([
            summarize_cycles(((1 + gain[i]) * raw[i] + bias[i])[None, :])
            for i in range(len(SENSOR_SPECS))
        ])
        np.testing.assert_allclose(transformed, expected)

    def test_parameter_jacobian_matches_finite_difference(self) -> None:
        block = np.array([2.0, 0.5, np.sqrt(4.25), 1.5, 2.5])
        features = np.tile(block, len(SENSOR_SPECS))[None, :]
        scales = np.linspace(0.5, 1.5, len(SENSOR_SPECS))
        jacobian = summary_parameter_jacobian(features, scales)[0]
        epsilon = 1e-6
        gain = np.zeros(len(SENSOR_SPECS)); gain[0] = epsilon
        numeric_gain = (transform_summary_features(features, gain, np.zeros_like(gain)) - features)[0] / epsilon
        bias = np.zeros(len(SENSOR_SPECS)); bias[0] = epsilon * scales[0]
        numeric_bias = (transform_summary_features(features, np.zeros_like(bias), bias) - features)[0] / epsilon
        np.testing.assert_allclose(jacobian[:, 0], numeric_gain, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(jacobian[:, 1], numeric_bias, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
