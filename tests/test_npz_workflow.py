from pathlib import Path
import tempfile
import unittest

import numpy as np

from brainfed.cli import evaluate_checkpoint
from brainfed.config import configuration
from brainfed.data import load_data
from brainfed.experiment import run_experiment


class NpzWorkflowTests(unittest.TestCase):
    def test_external_npz_training_and_rotation_evaluation(self):
        cfg = configuration(
            {
                "data": {
                    "task": "vision",
                    "input_dim": 1728,
                    "train_samples": 30,
                    "test_samples": 15,
                },
                "model": {"hidden_dim": 8, "timesteps": 6, "encoding": "latency"},
                "federation": {"rounds": 2},
            }
        )
        synthetic = load_data(cfg)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "prepared.npz"
            np.savez(
                data_path,
                x_train=synthetic.train.tensors[0].numpy(),
                y_train=synthetic.train.tensors[1].numpy(),
                x_test=synthetic.test.tensors[0].numpy(),
                y_test=synthetic.test.tensors[1].numpy(),
            )
            cfg["data"].update(source="npz", path=str(data_path))
            summary = run_experiment(cfg, root / "run")
            self.assertEqual(len(summary["data"]["sha256"]), 64)
            result = evaluate_checkpoint(root / "run" / "model.pt", str(data_path), 15)
            self.assertEqual(result["rotation_degrees"], 15)
            self.assertEqual(result["samples"], 15)
            self.assertTrue(0 <= result["accuracy"] <= 1)


if __name__ == "__main__":
    unittest.main()
