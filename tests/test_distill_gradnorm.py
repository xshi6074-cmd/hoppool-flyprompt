import math
import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
import warnings
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn

from utils.distill_gradnorm import (
    DistillationDelayGate,
    GradNormLiteController,
    resolve_distill_weight_mode,
)


class DistillationDelayGateTest(unittest.TestCase):
    def test_negative_delay_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            DistillationDelayGate(-1)

    def test_two_complete_batches_are_delayed_after_snapshot(self):
        gate = DistillationDelayGate(100)
        teacher_calls = 0

        self.assertEqual(gate.state, "waiting_teacher")
        self.assertFalse(gate.active)
        self.assertTrue(gate.finish_batch(observed_snapshot_count=1, batch_samples=64))

        batch_states = []
        for _ in range(3):
            batch_states.append((gate.state, gate.samples_since_snapshot))
            if gate.active:
                teacher_calls += 1
            gate.finish_batch(observed_snapshot_count=1, batch_samples=64)

        self.assertEqual(
            batch_states,
            [("delay", 0), ("delay", 64), ("active", 128)],
        )
        self.assertEqual(teacher_calls, 1)

    def test_every_snapshot_resets_delay(self):
        gate = DistillationDelayGate(100)
        gate.finish_batch(1, 64)
        gate.finish_batch(1, 128)
        self.assertTrue(gate.active)

        self.assertTrue(gate.finish_batch(2, 64))
        self.assertEqual(gate.state, "delay")
        self.assertEqual(gate.samples_since_snapshot, 0)

    def test_delayed_training_path_never_enters_teacher_branch(self):
        fake_methods = types.ModuleType("methods")
        fake_methods.__path__ = []
        fake_frozen_vit_module = types.ModuleType("methods.frozen_vit")
        fake_frozen_vit_module.FrozenViTTrainer = object
        module_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "methods"
            / "mlp_generator.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_isolated_mlp_generator_method", module_path
        )
        method_module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
                sys.modules,
                {
                    "methods": fake_methods,
                    "methods.frozen_vit": fake_frozen_vit_module,
                }):
            spec.loader.exec_module(method_module)

        class PromptGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Sequential(nn.Linear(2, 2))

            def forward(self, inputs):
                return self.mlp(inputs)

        class CountingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.prompt_generator = PromptGenerator()
                self.distill_weight_mode = "gradnorm"
                self.teacher_calls = 0
                self.reset_calls = 0
                self.last_raw_distillation = 0.0
                self.last_effective_distill_weight = 0.0
                self.last_replay_class_count = 0

            def reset_distillation_observation(self):
                self.reset_calls += 1
                self.last_raw_distillation = 0.0
                self.last_effective_distill_weight = 0.0
                self.last_replay_class_count = 0

            def forward(
                    self,
                    _images,
                    clean_cls,
                    return_distill,
                    current_labels=None):
                del current_labels
                logits = self.prompt_generator(clean_cls)
                if not return_distill:
                    return logits
                self.teacher_calls += 1
                self.last_replay_class_count = 1
                raw = self.prompt_generator.mlp[-1].weight.square().mean()
                self.last_raw_distillation = float(raw.detach().item())
                return logits, raw

        trainer = method_module.MLPGenerator.__new__(method_module.MLPGenerator)
        model = CountingModel()
        trainer.model = model
        trainer.model_without_ddp = model
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
        trainer.criterion = nn.CrossEntropyLoss()
        trainer.use_amp = False
        trainer.topk = 1
        trainer.update_schedule = lambda: None
        trainer._gradnorm_controller = GradNormLiteController(
            target_ratio=0.25,
            log_ema=0.9,
            weight_min=1e-4,
            weight_max=1e4,
        )
        trainer._reset_loss_accumulators()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            loss, _accuracy = trainer._online_train_cached(
                images=torch.zeros(2, 1),
                labels=torch.tensor([0, 1]),
                clean_cls=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                logit_mask=torch.zeros(2),
                distillation_active=False,
            )

        self.assertEqual(model.teacher_calls, 0)
        self.assertEqual(model.reset_calls, 1)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(trainer.last_distillation_loss, 0.0)
        self.assertEqual(trainer.last_distillation_objective, 0.0)


class GradNormLiteControllerTest(unittest.TestCase):
    @staticmethod
    def _controller(**overrides):
        arguments = dict(
            target_ratio=0.25,
            log_ema=0.9,
            weight_min=1e-4,
            weight_max=1e4,
        )
        arguments.update(overrides)
        return GradNormLiteController(**arguments)

    def test_first_update_hits_target_ratio(self):
        parameter = torch.tensor([1.0, -1.0], requires_grad=True)
        classification_loss = 2.0 * parameter.sum()
        distillation_loss = 8.0 * parameter.sum()

        result = self._controller().compute(
            classification_loss, distillation_loss, parameter
        )

        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.target_weight, 0.0625, places=7)
        self.assertAlmostEqual(result.weight, 0.0625, places=7)
        self.assertAlmostEqual(result.actual_grad_ratio, 0.25, places=7)
        self.assertAlmostEqual(result.gradient_cosine, 1.0, places=7)

    def test_log_ema_and_reset(self):
        controller = self._controller()
        parameter = torch.tensor([1.0], requires_grad=True)
        first = controller.compute(
            2.0 * parameter.sum(), 8.0 * parameter.sum(), parameter
        )
        second = controller.compute(
            4.0 * parameter.sum(), 4.0 * parameter.sum(), parameter
        )
        expected = math.exp(0.9 * math.log(0.0625) + 0.1 * math.log(0.25))

        self.assertAlmostEqual(first.weight, 0.0625, places=7)
        self.assertAlmostEqual(second.target_weight, 0.25, places=7)
        self.assertAlmostEqual(second.weight, expected, places=7)

        controller.reset()
        after_reset = controller.compute(
            4.0 * parameter.sum(), 4.0 * parameter.sum(), parameter
        )
        self.assertAlmostEqual(after_reset.weight, 0.25, places=7)

    def test_bounds_are_reported(self):
        parameter = torch.tensor([1.0], requires_grad=True)
        lower = self._controller(weight_min=0.1).compute(
            parameter.sum(), 100.0 * parameter.sum(), parameter
        )
        upper = self._controller(weight_max=1.0).compute(
            100.0 * parameter.sum(), parameter.sum(), parameter
        )

        self.assertAlmostEqual(lower.weight, 0.1, places=7)
        self.assertTrue(lower.boundary_hit)
        self.assertAlmostEqual(upper.weight, 1.0, places=7)
        self.assertTrue(upper.boundary_hit)

    def test_invalid_controller_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            self._controller(target_ratio=0.0)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
            self._controller(log_ema=1.0)
        with self.assertRaisesRegex(ValueError, "no smaller"):
            self._controller(weight_min=2.0, weight_max=1.0)

    def test_zero_and_nonfinite_gradients_do_not_update_controller(self):
        controller = self._controller()
        parameter = torch.tensor([1.0], requires_grad=True)

        zero = controller.compute(
            parameter.sum(), (parameter * 0.0).sum(), parameter
        )
        nonfinite = controller.compute(
            parameter.sum(), (parameter * float("nan")).sum(), parameter
        )

        self.assertFalse(zero.valid)
        self.assertEqual(zero.reason, "zero_distillation_gradient")
        self.assertFalse(nonfinite.valid)
        self.assertEqual(nonfinite.reason, "nonfinite_gradient")

        valid = controller.compute(
            2.0 * parameter.sum(), 8.0 * parameter.sum(), parameter
        )
        self.assertAlmostEqual(valid.weight, 0.0625, places=7)

    def test_ddp_statistics_are_summed_before_norms(self):
        controller = self._controller()
        parameter = torch.tensor([1.0], requires_grad=True)

        def add_remote_rank(statistics, op):
            del op
            statistics.add_(statistics.new_tensor([12.0, 0.0, 0.0]))

        with mock.patch(
                "utils.distill_gradnorm.dist.is_initialized",
                return_value=True), mock.patch(
                    "utils.distill_gradnorm.dist.all_reduce",
                    side_effect=add_remote_rank) as all_reduce:
            result = controller.compute(
                2.0 * parameter.sum(), 8.0 * parameter.sum(), parameter
            )

        all_reduce.assert_called_once()
        self.assertAlmostEqual(result.classification_grad_norm, 4.0, places=7)
        self.assertAlmostEqual(result.distillation_grad_norm, 8.0, places=7)
        self.assertAlmostEqual(result.weight, 0.125, places=7)
        self.assertAlmostEqual(result.actual_grad_ratio, 0.25, places=7)

    @unittest.skipUnless(dist.is_available(), "torch.distributed is unavailable")
    def test_controller_probe_then_backward_is_ddp_safe(self):
        if dist.is_initialized():
            self.skipTest("test requires ownership of the process group")
        with tempfile.TemporaryDirectory() as directory:
            init_file = (pathlib.Path(directory) / "ddp_init").resolve()
            dist.init_process_group(
                backend="gloo",
                init_method=init_file.as_uri(),
                rank=0,
                world_size=1,
            )
            try:
                module = nn.parallel.DistributedDataParallel(nn.Linear(2, 1))
                module._set_static_graph()
                inputs = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
                output = module(inputs)
                classification_loss = output.square().mean()
                distillation_loss = (output - 1.0).square().mean()
                result = self._controller().compute(
                    classification_loss,
                    distillation_loss,
                    module.module.weight,
                )
                self.assertTrue(result.valid)
                total = (
                    classification_loss
                    + distillation_loss.new_tensor(result.weight)
                    * distillation_loss
                )
                total.backward()
                self.assertTrue(torch.isfinite(module.module.weight.grad).all())
            finally:
                dist.destroy_process_group()


class WeightModeCompatibilityTest(unittest.TestCase):
    @staticmethod
    def _load_model_class():
        class FakeBackbone(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.num_features = 4
                self.cls_token = nn.Parameter(torch.zeros(1, 1, 4))
                self.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
                self.fc = nn.Linear(4, num_classes)

        fake_models = types.ModuleType("models")
        fake_models.__path__ = []
        fake_backbone_module = types.ModuleType("models.backbone")
        fake_backbone_module.create_backbone = (
            lambda _name, num_classes, **_kwargs: FakeBackbone(num_classes)
        )
        module_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "models"
            / "mlp_generator.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_isolated_mlp_generator", module_path
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
                sys.modules,
                {"models": fake_models, "models.backbone": fake_backbone_module}):
            spec.loader.exec_module(module)
        return module.MLPGenerator

    def test_legacy_flag_resolves_to_uncertainty(self):
        self.assertEqual(
            resolve_distill_weight_mode("fixed", legacy_learnable=True),
            "uncertainty",
        )
        self.assertEqual(
            resolve_distill_weight_mode("uncertainty", legacy_learnable=False),
            "uncertainty",
        )

    def test_legacy_flag_conflicts_with_gradnorm(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_distill_weight_mode("gradnorm", legacy_learnable=True)

    def test_fixed_uncertainty_and_gradnorm_parameter_contracts(self):
        model_class = self._load_model_class()
        common = dict(
            task_num=2,
            num_classes=3,
            backbone_name="fake",
            prompt_blocks=(0,),
            teacher_lags=(1,),
            covariance_rank=2,
            pretrained=False,
        )
        fixed = model_class(
            **common, distill_weight_mode="fixed", distill_weight=2.0
        )
        uncertainty = model_class(
            **common, distill_weight=2.0, learnable_distill_weight=True
        )
        gradnorm = model_class(
            **common, distill_weight_mode="gradnorm", distill_weight=2.0
        )

        self.assertIsNone(fixed.log_distill_scale)
        self.assertIsInstance(uncertainty.log_distill_scale, nn.Parameter)
        self.assertIsNone(gradnorm.log_distill_scale)
        self.assertNotIn(
            "log_distill_scale", dict(gradnorm.named_parameters())
        )

        raw = torch.tensor(3.0)
        for model in (fixed, uncertainty, gradnorm):
            model.last_replay_class_count = 1
        self.assertAlmostEqual(fixed._weight_distillation(raw).item(), 6.0)
        self.assertAlmostEqual(
            uncertainty._weight_distillation(raw).item(),
            6.0 - math.log(2.0),
            places=6,
        )
        self.assertAlmostEqual(gradnorm._weight_distillation(raw).item(), 3.0)


class ConfigurationInterfaceTest(unittest.TestCase):
    def test_new_distillation_arguments_and_defaults_parse(self):
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.DATASETS = {"cifar100": object()}
        fake_methods = types.ModuleType("methods")
        fake_methods.METHODS = {"mlp_generator": object()}
        module_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "configuration"
            / "config.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_isolated_configuration", module_path
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
                sys.modules,
                {"datasets": fake_datasets, "methods": fake_methods}):
            spec.loader.exec_module(module)

        with mock.patch.object(sys, "argv", ["test-config"]):
            defaults = module.base_parser()
        self.assertEqual(defaults.distill_weight_mode, "fixed")
        self.assertEqual(defaults.distill_delay_samples, 100)
        self.assertEqual(defaults.distill_grad_ratio, 0.25)
        self.assertEqual(defaults.distill_grad_ema, 0.9)
        self.assertEqual(defaults.distill_weight_min, 1e-4)
        self.assertEqual(defaults.distill_weight_max, 1e4)

        with mock.patch.object(sys, "argv", [
                "test-config",
                "--distill_weight_mode", "gradnorm",
                "--distill_delay_samples", "64",
                "--distill_grad_ratio", "0.5",
        ]):
            configured = module.base_parser()
        self.assertEqual(configured.distill_weight_mode, "gradnorm")
        self.assertEqual(configured.distill_delay_samples, 64)
        self.assertEqual(configured.distill_grad_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
