from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
    from satml_monitoring.temporal_monitors import GDN, TranAD
except ImportError:
    torch = None

from satml_monitoring.sequences import cycle_windows as windows


class TemporalPreprocessingTests(unittest.TestCase):
    def test_windows_do_not_cross_cycle_boundaries(self):
        sequences = np.stack([
            np.full((6, 2), 1.0, dtype=np.float32),
            np.full((6, 2), 2.0, dtype=np.float32),
        ])
        values, targets = windows(sequences, 3, predict=True)
        self.assertEqual(values.shape, (6, 3, 2))
        self.assertEqual(targets.shape, (6, 2))
        self.assertTrue(np.all(values[:3] == 1.0))
        self.assertTrue(np.all(values[3:] == 2.0))

    @unittest.skipIf(torch is None, "PyTorch optional dependency is unavailable")
    def test_model_output_shapes_and_gradients(self):
        values = torch.randn(4, 10, 17)
        gdn = GDN(17, 10, embedding_dim=8, hidden_dim=12, top_k=3)
        tranad = TranAD(17, 10, model_dim=16, heads=4, layers=1)
        self.assertEqual(tuple(gdn(values).shape), (4, 17))
        self.assertEqual(tuple(tranad(values).shape), (4, 10, 17))
        (gdn(values).square().mean() + tranad(values).square().mean()).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in gdn.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in tranad.parameters()))


if __name__ == "__main__":
    unittest.main()
