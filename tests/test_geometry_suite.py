"""Mechanism and scientific contracts for the next HFPool geometry queue."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from utils.cls_probe import geometry, run_cls_probe, summarize_history
from utils.ema_diagnostics import probability_loss
from utils.offline_probe import ObservedSamples, mutable_state, save_snapshot


class ToyDataset(Dataset):
    targets = [7, 3] * 4 + [99]

    def __init__(self):
        self.loaded = []

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        self.loaded.append(index)
        return torch.tensor([1., 0.] if self.targets[index] == 7 else [0., 1.]), self.targets[index]


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 3)

    def forward_features(self, images):
        return images[:, None]


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ToyBackbone()
        self.adapter = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.adapter.weight.copy_(torch.eye(2))

    def extract_features(self, images):
        return self.adapter(images)


class CLSContracts(unittest.TestCase):
    def test_observed_cohort_temporal_axis_and_state_isolation(self):
        model, dataset, observed = ToyModel(), ToyDataset(), ObservedSamples()
        observed.add(list(range(8)), dataset.targets[:8])
        arguments = dict(method="hfpool", dataset="toy", backbone="toy", batchsize=2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            directory = Path(temporary)
            save_snapshot(directory, model, dataset, observed, [7, 3], 0, 8, {}, arguments, "mean")
            with torch.no_grad():
                model.adapter.weight.copy_(torch.tensor([[0., -1.], [1., 0.]]))
            save_snapshot(directory, model, dataset, observed, [7, 3], 1, 8, {}, arguments, "mean")
            state, rng = mutable_state(model), torch.get_rng_state().clone()
            model.train()
            rows = run_cls_probe(model, dataset, None, directory, "cpu", seed=3, per_class=2, batchsize=2)
            self.assertEqual(len(set(dataset.loaded)), 4)
            self.assertNotIn(8, dataset.loaded)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(value, mutable_state(model)[key]) for key, value in state.items()))
            self.assertAlmostEqual(rows[1]["same_image_cos_first"], 0.)
            self.assertGreater(rows[1]["temporal_variance"], 0.)
            with np.load(directory / "history.npz") as data:
                self.assertEqual(data["features"].shape, (2, 4, 2))
                self.assertEqual(json.loads(str(data["metadata"]))["per_class"], 2)

    def test_variance_axes_and_cosine_diagonal_exclusion(self):
        labels = np.array([0, 0, 1, 1])
        features = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        values = geometry(features, labels)
        self.assertAlmostEqual(values["same_class_cos"], -1.)
        self.assertAlmostEqual(values["different_class_cos"], 0.)
        constant = np.tile([1., 5.], (4, 1))
        self.assertEqual(geometry(constant, labels)["sample_variance"], 0.)
        history = np.stack([constant, constant + 2.])
        rows = summarize_history(history, constant, labels, [1, 2])
        self.assertEqual(rows[-1]["temporal_variance"], 1.)

    def test_probability_ensemble_loss_is_nll_after_normalization(self):
        scores = torch.tensor([[.6, .8, 0.]])
        target = torch.tensor([1])
        self.assertAlmostEqual(float(probability_loss(scores, target)), -np.log(.8/1.4), places=6)


class ModelContracts(unittest.TestCase):
    def test_ema_bank_does_not_change_rng_and_only_averages_head(self):
        from test_method_split import make_model
        from models.hfpool import HFPool
        plain = make_model(HFPool, hopfield_trainable=["o"], num_pooling_blocks=1)
        rng = torch.get_rng_state().clone()
        ema = make_model(HFPool, hopfield_trainable=["o"], num_pooling_blocks=1, use_ema=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value, ema.state_dict()[key]) for key, value in plain.state_dict().items()))
        old = ema.ema_fc[0].weight.clone()
        with torch.no_grad():
            ema.backbone.fc.weight.add_(1.)
        ema.update_ema_fc()
        torch.testing.assert_close(ema.ema_fc[0].weight, old + .1)

    def test_spatial_compression_and_lora_merge_preserve_function(self):
        from models.hfpool import HopfieldPoolingPrompts, SpatialGELUPooling
        tokens = torch.arange(197*8, dtype=torch.float32).reshape(1, 197, 8) / 1000
        pool = SpatialGELUPooling(8)
        with torch.no_grad():
            pool.out_proj.weight.copy_(torch.eye(8))
            pool.out_proj.bias.zero_()
        expected = torch.nn.functional.adaptive_avg_pool2d(tokens[:, 1:].transpose(1, 2).reshape(1, 8, 14, 14), (3, 3))
        expected = torch.cat((tokens[:, :1], expected.flatten(2).transpose(1, 2)), dim=1)
        torch.testing.assert_close(pool(tokens), torch.nn.functional.gelu(expected))
        adapter = HopfieldPoolingPrompts(8, num_pooling_blocks=1, trainable=["o"],
            pooling_mode="spatial_gelu", o_mode="lora_merge_batch", lora_rank=2, lora_alpha=2)
        lora = adapter.lora_output_weights()[0]
        with torch.no_grad():
            lora.lora_B.fill_(.1)
        before = adapter.hopfield_pooling_layers[0](tokens)
        before.square().mean().backward()
        self.assertIsNotNone(lora.lora_A.grad)
        self.assertFalse(adapter.output_projections()[0].parametrizations.weight.original.requires_grad)
        adapter.merge_lora_output_weights()
        torch.testing.assert_close(adapter.hopfield_pooling_layers[0](tokens), before)
        self.assertEqual(int(torch.count_nonzero(lora.lora_B)), 0)
        self.assertTrue(all(p.dtype == torch.float32 for p in adapter.parameters()))

    def test_ema_diagnostic_keeps_online_primary_and_skips_overflow(self):
        from test_method_split import make_model, make_images, make_trainer
        from models.frozen_vit import Baseline
        from methods.frozen_vit import FrozenViTTrainer
        model = make_model(Baseline, ema_diagnostics=True)
        trainer = make_trainer(FrozenViTTrainer, model, exposed_classes=[0, 1])
        trainer.mask[:2] = 0
        images, labels = make_images(), torch.tensor([0, 1])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            trainer.log_dir, trainer.rnd_seed = directory, 1
            result = trainer.online_evaluate([(images, labels.clone())], task_id=0, end=True)
            metrics = json.loads((Path(directory) / "seed_1_ema.jsonl").read_text())["heads"]
            self.assertEqual(result["avg_acc"], metrics["online"]["accuracy"])
            self.assertEqual(len(metrics), 5)
        class OverflowScaler:
            scale_value = 2.
            def scale(self, loss): return loss
            def get_scale(self): return self.scale_value
            def step(self, optimizer): pass
            def update(self): self.scale_value = 1.
        trainer.scaler = OverflowScaler()
        before = model.ema_fc[0].weight.detach().clone()
        with torch.no_grad():
            model.backbone.fc.weight.add_(1.)
        trainer.online_train([images, labels])
        torch.testing.assert_close(model.ema_fc[0].weight, before)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
