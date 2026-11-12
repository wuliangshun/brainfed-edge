import json
from pathlib import Path
import tempfile
import unittest

from brainfed.cli import compare, evaluate_checkpoint
from brainfed.config import configuration
from brainfed.experiment import run_experiment


class ExperimentTests(unittest.TestCase):
    def config(self, patch=None):
        cfg = configuration({"federation": {"rounds": 4}})
        if patch:
            for section, values in patch.items():
                cfg[section].update(values)
        return cfg

    def test_run_logs_checkpoint_and_evaluation_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            summary = run_experiment(self.config(), root)
            self.assertGreater(summary["aggregation_events"], 0)
            result = evaluate_checkpoint(root / "model.pt", None, 0)
            self.assertEqual(result["accuracy"], summary["final"]["accuracy"])
            self.assertAlmostEqual(result["mse"], summary["final"]["mse"])
            self.assertTrue((root / "metrics.csv").exists())
            self.assertEqual(
                json.loads((root / "data_provenance.json").read_text())["source"], "synthetic"
            )
            with self.assertRaises(FileExistsError):
                run_experiment(self.config(), root)

    def test_identical_seed_reproduces_model_metrics_and_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            a = run_experiment(self.config(), Path(directory) / "a")
            b = run_experiment(self.config(), Path(directory) / "b")
            self.assertEqual(a["final"], b["final"])
            self.assertEqual(a["uplink_payload_bytes"], b["uplink_payload_bytes"])

    def test_no_quorum_keeps_global_model_unchanged(self):
        cfg = self.config({"federation": {"offline": {str(i): [1, 2, 3, 4] for i in [2, 3, 4]}}})
        with tempfile.TemporaryDirectory() as directory:
            summary = run_experiment(cfg, Path(directory) / "no_quorum")
            self.assertEqual(summary["aggregation_events"], 0)
            self.assertEqual(summary["uplink_payload_bytes"], 0)
            self.assertEqual(summary["initial"], summary["final"])

    def test_periodic_dense_fedavg_aggregates_every_round(self):
        cfg = self.config({"aggregation": {"policy": "periodic", "period": 1, "keep_ratio": 1.0}})
        with tempfile.TemporaryDirectory() as directory:
            summary = run_experiment(cfg, Path(directory) / "periodic")
            self.assertEqual(summary["aggregation_events"], 4)
            self.assertEqual(
                summary["uplink_payload_bytes"], summary["dense_uplink_equivalent_bytes"]
            )

    def test_failure_and_recovery_logs_real_catchup(self):
        cfg = self.config(
            {
                "federation": {"rounds": 6, "offline": {"0": [3, 4]}},
                "aggregation": {"policy": "periodic", "keep_ratio": 1.0},
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "failure"
            run_experiment(cfg, root)
            events = json.loads((root / "events.json").read_text())
            self.assertTrue(
                any(e["event"] == "client_catchup" and e["client"] == 0 for e in events)
            )
            elections = json.loads((root / "elections.json").read_text())
            self.assertTrue(any(e["leader"] == 1 for e in elections))

    def test_comparison_runs_all_four_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "compare"
            result = compare(self.config({"federation": {"rounds": 2}}), root, [9])
            self.assertEqual(len(result["groups"]), 4)
            self.assertTrue((root / "comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
