import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from brainfed.aggregation import (
    AggregationClock,
    SaliencyTracker,
    SparseTensor,
    Update,
    aggregate,
    pack_update,
)
from brainfed.config import configuration, load_config
from brainfed.data import dirichlet_partition, load_data
from brainfed.encoding import encode
from brainfed.model import build_model, rate_targets
from brainfed.neurons import BoxSpike, SpikingCell


class ConfigurationTests(unittest.TestCase):
    def test_unknown_options_are_rejected(self):
        with self.assertRaises(ValueError):
            configuration({"model": {"hiddden_dim": 32}})

    def test_invalid_ranges_are_rejected(self):
        for patch in [
            {"aggregation": {"keep_ratio": 0}},
            {"model": {"dt": float("nan")}},
            {"aggregation": {"min_interval": 5, "max_interval": 2}},
            {"data": {"input_dim": 0}},
        ]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                configuration(patch)

    def test_npz_path_is_relative_to_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.yaml"
            path.write_text("data:\n  source: npz\n  path: ../test.npz\n")
            self.assertEqual(
                load_config(path)["data"]["path"], str(path.parent.parent / "test.npz")
            )

    def test_taskbook_networks_forward_backward_at_actual_dimensions(self):
        root = Path(__file__).resolve().parents[1]
        torch.set_num_threads(1)
        for task, dimensions in [
            ("audio", (128, 2000, 3)),
            ("vision", (1728, 2500, 3)),
            ("radar", (4800, 1000, 5)),
        ]:
            with self.subTest(task=task):
                cfg = load_config(root / "configs" / f"{task}.yaml")
                model = build_model(cfg)
                self.assertEqual(
                    (model.fc1.in_features, model.fc1.out_features, model.fc2.out_features),
                    dimensions,
                )
                spikes = encode(
                    torch.rand(2, dimensions[0]),
                    cfg["model"]["timesteps"],
                    cfg["model"]["encoding"],
                )
                rates = model(spikes)
                loss = (rates - rate_targets(torch.tensor([0, 1]), dimensions[2])).square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(model.fc1.weight.grad).all())


class EncodingAndNeuronTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_static_encoders_are_binary_and_silent_on_zero(self):
        x = torch.tensor([[0.0, 0.4, 1.0]])
        for method in ["rate", "latency", "phase", "burst"]:
            out = encode(x, 12, method, torch.Generator().manual_seed(3))
            self.assertEqual(tuple(out.shape), (12, 1, 3))
            self.assertTrue(torch.isin(out, torch.tensor([0.0, 1.0])).all())
            self.assertEqual(out[:, :, 0].sum().item(), 0)

    def test_latency_and_burst_ordering(self):
        x = torch.tensor([[0.2, 0.8]])
        latency = encode(x, 12, "latency")[:, 0]
        self.assertGreater(latency[:, 0].argmax().item(), latency[:, 1].argmax().item())
        burst = encode(x, 12, "burst")[:, 0]
        self.assertLess(burst[:, 0].sum().item(), burst[:, 1].sum().item())

    def test_rate_encoder_reproducible_and_unbiased(self):
        x = torch.full((100, 10), 0.3)
        a = encode(x, 200, "rate", torch.Generator().manual_seed(7))
        b = encode(x, 200, "rate", torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(a, b))
        self.assertAlmostEqual(a.mean().item(), 0.3, delta=0.01)

    def test_preencoded_shape(self):
        x = torch.zeros(3, 8, 4)
        self.assertEqual(tuple(encode(x, 8, "preencoded").shape), (8, 3, 4))
        with self.assertRaises(ValueError):
            encode(x, 7, "preencoded")

    def test_surrogate_derivative(self):
        x = torch.tensor([-2.0, -0.2, 0.2, 2.0], requires_grad=True)
        y = BoxSpike.apply(x, 1.0)
        y.sum().backward()
        torch.testing.assert_close(y, torch.tensor([0.0, 0.0, 1.0, 1.0]))
        torch.testing.assert_close(x.grad, torch.tensor([0.0, 1.0, 1.0, 0.0]))

    def test_bptt_reaches_input_and_no_state_leaks_between_batches(self):
        torch.manual_seed(8)
        cfg = configuration()
        model = build_model(cfg)
        x = torch.rand(12, 4, 16, requires_grad=True)
        a, b = model(x), model(x)
        torch.testing.assert_close(a, b)
        loss = (a - rate_targets(torch.tensor([0, 1, 2, 1]), 3)).square().mean()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad[0].abs().sum().item(), 0)

    def test_mdn_recovery_changes_state_and_lif_does_not(self):
        cfg = configuration()["model"]
        current = torch.ones(2, 3)
        _, state = SpikingCell(cfg)(current)
        self.assertGreater(state[1].abs().sum().item(), 0)
        cfg = {**cfg, "neuron": "lif"}
        _, state = SpikingCell(cfg)(current)
        self.assertEqual(state[1].abs().sum().item(), 0)

    def test_fixed_batch_training_reduces_rate_loss(self):
        torch.manual_seed(42)
        cfg = configuration()
        bundle = load_data(cfg)
        x, y = (v[:30] for v in bundle.train.tensors)
        spikes = encode(x, 12, "rate", torch.Generator().manual_seed(42))
        model = build_model(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        target = rate_targets(y, 3)
        initial = (model(spikes) - target).square().mean().item()
        for _ in range(35):
            optimizer.zero_grad()
            loss = (model(spikes) - target).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        final = (model(spikes) - target).square().mean().item()
        self.assertLess(final, initial * 0.9)


class DatasetTests(unittest.TestCase):
    def test_partition_complete_disjoint_and_repeatable(self):
        y = np.arange(150) % 3
        a = dirichlet_partition(y, 5, 0.5, 17)
        b = dirichlet_partition(y, 5, 0.5, 17)
        self.assertEqual(a, b)
        merged = sum(a, [])
        self.assertEqual(sorted(merged), list(range(len(y))))
        self.assertGreaterEqual(min(map(len, a)), 2)

    def test_partition_rejects_impossible_size(self):
        with self.assertRaises(ValueError):
            dirichlet_partition(np.arange(3), 5, 0.5, 42)

    def test_npz_adapter_rejects_noninteger_labels_and_nan_features(self):
        for bad_labels, bad_feature in [(True, False), (False, True)]:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "data.npz"
                x = np.zeros((8, 16), dtype=np.float32)
                if bad_feature:
                    x[0, 0] = np.nan
                y = np.zeros(8, dtype=np.float32 if bad_labels else np.int64)
                np.savez(path, x_train=x, y_train=y, x_test=x, y_test=y)
                cfg = configuration({"data": {"source": "npz", "path": str(path)}})
                with self.assertRaises(ValueError):
                    load_data(cfg)


class AggregationTests(unittest.TestCase):
    def test_dense_limit_equals_sample_weighted_fedavg(self):
        baseline = {"fc1.weight": torch.tensor([[2.0, 4.0]])}
        a = {"fc1.weight": torch.tensor([[4.0, 8.0]])}
        b = {"fc1.weight": torch.tensor([[8.0, 12.0]])}
        updates = [
            pack_update(0, 1, 3, a, baseline, SaliencyTracker(), 1),
            pack_update(1, 3, 3, b, baseline, SaliencyTracker(), 1),
        ]
        result = aggregate(baseline, updates, 3)
        torch.testing.assert_close(
            result["fc1.weight"], (a["fc1.weight"] + 3 * b["fc1.weight"]) / 4
        )

    def test_sparse_missing_coordinates_are_zero_weighted_deltas(self):
        baseline = {"fc1.bias": torch.tensor([2.0, 4.0])}
        update = Update(
            0, 1, 0, {"fc1.bias": SparseTensor(torch.tensor([0]), torch.tensor([8.0]), (2,))}
        )
        empty = Update(
            1,
            3,
            0,
            {"fc1.bias": SparseTensor(torch.tensor([], dtype=torch.long), torch.tensor([]), (2,))},
        )
        torch.testing.assert_close(
            aggregate(baseline, [update, empty], 0)["fc1.bias"], torch.tensor([4.0, 4.0])
        )

    def test_neuron_mask_couples_weight_row_and_bias_and_counts_indices(self):
        before = {"fc1.weight": torch.zeros(4, 8), "fc1.bias": torch.zeros(4)}
        after = {k: v.clone() for k, v in before.items()}
        after["fc1.weight"][2] = 3
        after["fc1.bias"][2] = 2
        tracker = SaliencyTracker()
        tracker.update(before, after)
        update = pack_update(0, 1, 0, after, before, tracker, 0.25)
        self.assertEqual(update.tensors["fc1.bias"].indices.tolist(), [2])
        self.assertEqual(update.tensors["fc1.weight"].indices.tolist(), list(range(16, 24)))
        self.assertEqual(update.nbytes, 9 * (8 + 4))

    def test_signed_updates_do_not_cancel_saliency(self):
        before = {"fc1.weight": torch.zeros(1, 2)}
        after = {"fc1.weight": torch.tensor([[2.0, -2.0]])}
        self.assertGreater(SaliencyTracker().update(before, after), 0)

    def test_stale_duplicate_or_bad_shape_update_rejected(self):
        baseline = {"fc1.bias": torch.zeros(2)}
        for updates in [
            [Update(0, 1, 1, baseline)],
            [Update(0, 1, 0, baseline), Update(0, 1, 0, baseline)],
            [Update(0, 1, 0, {"fc1.bias": torch.zeros(3)})],
        ]:
            with self.assertRaises(ValueError):
                aggregate(baseline, updates, 0)

    def test_event_threshold_bootstrap_and_forced_max_interval(self):
        cfg = configuration()["aggregation"]
        clock = AggregationClock({**cfg, "initial_threshold": 100, "max_interval": 3})
        self.assertFalse(clock.observe(0))
        self.assertFalse(clock.observe(0))
        self.assertTrue(clock.observe(0))
        self.assertEqual(clock.elapsed, 3)
        clock.committed()
        self.assertEqual(clock.elapsed, 0)
        self.assertGreater(clock.threshold, 0)

    def test_failed_commit_preserves_accumulation(self):
        clock = AggregationClock(configuration()["aggregation"])
        clock.observe(10)
        self.assertTrue(clock.observe(10))
        # Caller does not commit when no leader/quorum exists.
        self.assertTrue(clock.observe(0))
        self.assertEqual(clock.accumulated, 20)


if __name__ == "__main__":
    unittest.main()
