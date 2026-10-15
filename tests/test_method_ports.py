"""Public method entrypoints and diagnostic time-axis contracts."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import test_method_split as method_support
from test_method_split import make_images, make_model, trainable_names


class MethodEntrypoints(unittest.TestCase):
    def parse(self, *arguments):
        return method_support.ConfigurationTest().parse(*arguments)

    def test_lora_cli_builds_output_only_adapter_with_its_trainer(self):
        from methods import METHODS
        from models import MODELS
        from utils.train_utils import STEP_AWARE_METHODS
        self.assertIn("hfpool_lora", MODELS)
        self.assertIn("hfpool_lora", METHODS)
        self.assertIn("hfpool_lora", STEP_AWARE_METHODS)
        args = self.parse("--method", "hfpool_lora", "--opt_name", "adam")
        self.assertEqual(args.hopfield_trainable, ["o"])
        self.assertEqual(args.hopfield_o_mode, "lora_merge_batch")
        model = make_model(MODELS[args.method], num_pooling_blocks=1)
        self.assertEqual(model.adapter.o_mode, "lora_merge_batch")
        self.assertEqual(model.adapter.trainable, {"o"})
        names = trainable_names(model)
        self.assertTrue(any("lora_A" in name for name in names))
        self.assertFalse(any("original" in name for name in names))
        self.assertEqual(model(make_images()).shape, (2, 6))

    def test_gelu_cli_has_no_attention_pool_and_supports_optional_lora(self):
        from models import MODELS
        self.assertIn("spatial_gelu", MODELS)
        args = self.parse("--method", "spatial_gelu")
        self.assertEqual(args.hopfield_pooling_mode, "spatial_gelu")
        self.assertEqual(args.hopfield_trainable, ["o"])
        self.assertEqual(args.hopfield_o_mode, "full")
        for output_mode in ("full", "lora_merge_batch"):
            model = make_model(MODELS[args.method], num_pooling_blocks=1,
                               hopfield_o_mode=output_mode)
            self.assertFalse(any("hopfield." in name for name, _ in model.named_modules()))
            self.assertEqual(model.adapter.pooling_mode, "spatial_gelu")
            logits = model(make_images())
            logits.square().mean().backward()
            self.assertEqual(logits.shape, (2, 6))
            self.assertTrue(any(parameter.grad is not None for parameter in model.adapter.parameters()
                                if parameter.requires_grad))

    def test_existing_hfpool_defaults_and_explicit_choices_remain(self):
        args = self.parse("--method", "hfpool")
        self.assertEqual(args.hopfield_trainable, ["q"])
        self.assertEqual(args.hopfield_o_mode, "full")
        self.assertEqual(args.hopfield_pooling_mode, "attention")
        args = self.parse("--method", "spatial_gelu", "--hopfield_o_mode", "lora")
        self.assertEqual(args.hopfield_o_mode, "lora")
        from models import MODELS
        with self.assertRaises(ValueError):
            make_model(MODELS["hfpool_lora"], hopfield_trainable=["k"])
        with self.assertRaises(ValueError):
            make_model(MODELS["spatial_gelu"], hopfield_pooling_mode="attention")

    def test_public_trainers_keep_complete_batch_update_and_merge_protocol(self):
        from methods import METHODS
        from models import MODELS
        for method in ("hfpool_lora", "spatial_gelu"):
            with self.subTest(method=method):
                model = make_model(MODELS[method], num_pooling_blocks=1)
                trainer = method_support.make_trainer(METHODS[method], model)
                head_before = model.backbone.fc.weight.detach().clone()
                with mock.patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as step:
                    loss, accuracy = trainer.online_step(make_images(), torch.tensor([0, 1]),
                                                         torch.tensor([0, 1]))
                self.assertEqual(step.call_count, 3)
                self.assertTrue(np.isfinite(loss))
                self.assertGreaterEqual(accuracy, 0.)
                self.assertFalse(torch.equal(head_before, model.backbone.fc.weight))
                if method == "hfpool_lora":
                    self.assertEqual(trainer.lora_seen_samples, 2)
                    self.assertEqual(trainer.lora_samples_since_merge, 0)
                    for lora in model.adapter.lora_output_weights():
                        self.assertEqual(int(torch.count_nonzero(lora.lora_B)), 0)
                        self.assertNotIn(lora.lora_A, trainer.optimizer.state)
                        self.assertNotIn(lora.lora_B, trainer.optimizer.state)


class GeometryTimelines(unittest.TestCase):
    def test_different_seed_visit_counts_keep_their_real_time_axes(self):
        from scripts.plot_cls_probe import metric_series
        items = [dict(samples=np.array([10, 20]), metadata=dict(seed=1),
                      metrics=[dict(score=.8), dict(score=.6)]),
                 dict(samples=np.array([12, 30]), metadata=dict(seed=2),
                      metrics=[dict(score=.7), dict(score=.5)])]
        series = metric_series(items, "score", "samples")
        self.assertEqual(len(series), 2)
        np.testing.assert_array_equal(series[0][0], [10, 20])
        np.testing.assert_array_equal(series[1][0], [12, 30])
        self.assertIsNone(series[0][2])
        checkpoints = metric_series(items, "score", "checkpoint")
        self.assertEqual(len(checkpoints), 1)
        np.testing.assert_array_equal(checkpoints[0][0], [0, 1])
        np.testing.assert_allclose(checkpoints[0][1], [.75, .55])

    def test_one_method_history_can_render_without_a_paired_reference(self):
        from scripts.plot_cls_probe import check_pairs, render
        from utils.cls_probe import summarize_history
        labels = np.array([0, 0, 1, 1])
        clean = np.array([[1., 0.], [.9, .1], [0., 1.], [.1, .9]])
        features = np.stack([clean, clean + .1])
        record = dict(dataset="toy", backbone="tiny", method="SpatialGELU",
                      metadata=dict(seed=1, dataset_signature="same"),
                      indices=np.arange(4), labels=labels, clean=clean,
                      samples=np.array([10, 20]),
                      metrics=summarize_history(features, clean, labels, [10, 20]))
        self.assertEqual(check_pairs([record]), 0)
        with tempfile.TemporaryDirectory() as temporary:
            render([record], temporary)
            self.assertTrue((Path(temporary) / "cls_metrics.csv").exists())
            self.assertEqual(len(list(Path(temporary).glob("*.png"))), 2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
