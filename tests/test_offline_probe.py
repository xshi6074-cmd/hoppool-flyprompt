"""Scientific contracts: observed-only data, recoverable drift, isolated fitting."""
import json
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from utils.offline_probe import (ObservedSamples, ProbeSettings, classification_metrics,
                                 fit_linear_probe, mutable_state, online_scores,
                                 paired_forgetting, run_probes, save_snapshot,
                                 split_validation)


class FeatureDataset(Dataset):
    def __init__(self, count):
        generator = torch.Generator().manual_seed(count)
        self.targets = [7, 3] * count + [99] * 3
        self.images = torch.tensor([[4., 0.], [0., 4.]] * count + [[-4., -4.]] * 3)
        self.images += torch.randn(self.images.shape, generator=generator) * .1
        self.loaded = []

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        self.loaded.append(index)
        return self.images[index], self.targets[index]


class FeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.fc = nn.Linear(2, 3)
        self.adapter = nn.Linear(2, 2, bias=False)
        self.num_classes = 3
        self.use_ema = False
        with torch.no_grad():
            self.adapter.weight.copy_(torch.eye(2))
            self.backbone.fc.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [0., 0.]]))
            self.backbone.fc.bias.zero_()

    def extract_features(self, inputs):
        return self.adapter(inputs)


class OfflineProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_observed_indices_are_deduplicated_not_expanded_by_class(self):
        observed = ObservedSamples()
        observed.add([0, 4, 0], [7, 3, 7])
        self.assertEqual(observed.labels, {0: 7, 4: 3})
        self.assertEqual(observed.visits, 3)
        with self.assertRaises(ValueError):
            observed.add([0], [3])

    def test_failed_extraction_restores_model_modes_and_all_random_states(self):
        model, dataset, observed = FeatureModel(), FeatureDataset(2), ObservedSamples()
        model.train()
        model.adapter.eval()
        observed.add([0, 1], [7, 3])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            save_snapshot(directory, model, dataset, observed, [7, 3], 0, 2,
                          dict(avg_acc=1., cls_acc=[1., 1., 0.]), {}, "mean")
            before = mutable_state(model)
            python_state, numpy_state = random.getstate(), np.random.get_state()
            torch_state = torch.get_rng_state().clone()

            def fail_extraction(inputs):
                random.random()
                np.random.random()
                torch.rand(1)
                with torch.no_grad():
                    model.adapter.weight.add_(2.)
                raise RuntimeError("feature extraction failed")

            with mock.patch.object(model, "extract_features", side_effect=fail_extraction):
                with self.assertRaisesRegex(RuntimeError, "feature extraction failed"):
                    run_probes(model, dataset, dataset, None, directory, "cpu", ProbeSettings())
            self.assertEqual(random.getstate(), python_state)
            current_numpy = np.random.get_state()
            self.assertEqual(current_numpy[0], numpy_state[0])
            np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
            self.assertEqual(current_numpy[2:], numpy_state[2:])
            torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
            for name, value in mutable_state(model).items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            self.assertTrue(model.training)
            self.assertFalse(model.adapter.training)

    def test_validation_keeps_singletons_and_is_reproducible(self):
        labels = torch.tensor([0] + [1] * 20 + [2] * 4)
        training, validation = split_validation(labels, 12)
        self.assertIn(0, training.tolist())
        self.assertNotIn(0, validation.tolist())
        self.assertFalse(set(training.tolist()) & set(validation.tolist()))
        self.assertEqual(len(training) + len(validation), len(labels))
        torch.testing.assert_close(validation, split_validation(labels, 12)[1])

    def test_f_last_uses_original_eligibility_for_both_heads(self):
        original = [[.8, .2, 0], [.4, .4, .8]]
        probe = [[.9, 0, 0], [.8, .5, .9]]
        eligible, _, _, original_f, probe_f = paired_forgetting(original, probe)
        np.testing.assert_array_equal(eligible, [True, True, False])
        self.assertAlmostEqual(original_f, .1)
        self.assertAlmostEqual(probe_f, -.2)  # Keep improvements; do not clip.

    def test_nonfinite_features_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_linear_probe(torch.tensor([[float("nan")], [0.]]), torch.tensor([0, 1]),
                             2, ProbeSettings(), 1, "cpu")

    def test_ema_scores_match_masked_original_ensemble(self):
        model = FeatureModel()
        model.use_ema = True
        model.ema_fc = nn.ModuleList([nn.Linear(2, 3)])
        x = torch.tensor([[1., 2.], [-1., 1.]])
        values = [model.backbone.fc(x), model.ema_fc[0](x)]
        expected = torch.stack([value[:, :2].softmax(1) for value in values], -1).max(-1).values
        actual = online_scores(model, x, 2, "cpu", 1, "softmax_max_prob")
        torch.testing.assert_close(actual, expected)

    def test_snapshot_refit_recovers_shift_without_touching_model_or_future_data(self):
        model = FeatureModel()
        train, test = FeatureDataset(40), FeatureDataset(10)
        observed = ObservedSamples()
        settings = ProbeSettings(epochs=30, patience=5, lr=.05, batchsize=16)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "tests") as temporary:
            directory = Path(temporary) / "seed_1_lp"

            def checkpoint(task, start, end):
                indices = list(range(start, end))
                observed.add(indices, [train.targets[i] for i in indices])
                with torch.no_grad():
                    features = model.extract_features(test.images[:20])
                    metrics = classification_metrics(model.backbone.fc(features)[:, :2],
                                                     torch.tensor([0, 1] * 10), 2)
                evaluation = dict(avg_acc=metrics["accuracy"],
                                  cls_acc=metrics["cls_accuracy"].tolist() + [0.])
                save_snapshot(directory, model, train, observed, [7, 3], task,
                              observed.visits, evaluation, {"rnd_seed": 1}, "softmax_max_prob")

            checkpoint(0, 0, 40)
            with torch.no_grad():
                model.adapter.weight.copy_(torch.tensor([[0., 1.], [1., 0.]]))
            checkpoint(1, 40, 60)
            model.train()
            model.adapter.eval()
            state = mutable_state(model)
            rng = torch.get_rng_state().clone()
            rows = run_probes(model, train, test, lambda x: x, directory,
                              "cpu", settings, seed=1, cache_features=True)
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            self.assertTrue(model.training)
            self.assertFalse(model.adapter.training)
            for key, value in mutable_state(model).items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
            self.assertEqual(rows[-1]["online"]["accuracy"], 0.)
            self.assertGreater(rows[-1]["probe"]["accuracy"], .95)
            self.assertEqual(set(train.loaded), set(range(60)))
            self.assertEqual(set(test.loaded), set(range(20)))
            fit = json.loads((directory / "t01_fit.json").read_text())
            self.assertEqual(fit["fitting_count"], 60)
            with np.load(directory / "history.npz") as history:
                self.assertAlmostEqual(float(history["online_f_last"]), 1., places=5)
                self.assertAlmostEqual(float(history["probe_f_last"]), 0., places=5)
            # The persisted-cache CLI exercises the actual rerun and plotting
            # path without importing hflayers or accessing any image dataset.
            result = subprocess.run(
                [sys.executable, "scripts/offline_linear_probe.py", str(directory),
                 "--from-cache", "--epochs", "30", "--patience", "5", "--lr", ".05"],
                cwd=root, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((directory / "refit/plots/paired.png").is_file())
            with np.load(directory / "refit/history.npz") as rerun:
                self.assertGreater(float(rerun["probe_accuracy"][-1]), .95)
            plots = Path(temporary) / "plots"
            result = subprocess.run(
                [sys.executable, "scripts/plot_offline_probe.py", temporary,
                 "--output-dir", str(plots), "--expected-files", "1"],
                cwd=root, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((plots / "f_last.png").is_file())


if __name__ == "__main__":
    unittest.main()
