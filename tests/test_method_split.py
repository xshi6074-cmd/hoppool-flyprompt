"""Contracts for the frozen-ViT mechanism methods split out of FlyPrompt.

Models use the repository's ``vit_tiny_patch16_224`` with random weights so the
suite runs on CPU. ``StreamSimulationTest`` drives the real trainer hooks the
way ``_Trainer.main_worker`` does: stream batches -> ``online_step`` (including
the internal step boundary) -> ``online_evaluate``.
"""
import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from methods.flyprompt import FlyPrompt as FlyPromptTrainer
from methods.frozen_vit import FrozenViTTrainer
from methods.hfpool import HFPool as HFPoolTrainer
from models.flyprompt import FlyPrompt
from models.frozen_vit import Baseline
from models.gate import Gate
from models.hfpool import HFPool
from models.shared_prompt import SharedPrompt

BACKBONE = "vit_tiny_patch16_224"
NUM_CLASSES = 6


def make_model(model_cls, **kwargs):
    torch.manual_seed(0)
    arguments = dict(
        task_num=4,
        num_classes=NUM_CLASSES,
        backbone_name=BACKBONE,
        pretrained=False,
    )
    arguments.update(kwargs)
    return model_cls(**arguments)


def make_images(batch_size=2, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, 3, 224, 224, generator=generator)


def trainable_names(model):
    return sorted(name for name, param in model.named_parameters() if param.requires_grad)


FC = ["backbone.fc.bias", "backbone.fc.weight"]


class ConfigurationTest(unittest.TestCase):
    """The README old -> new command table must parse, old flags must not."""

    def parse(self, *argv):
        from configuration import config
        with mock.patch.object(sys, "argv", ["main.py", *argv]):
            return config.base_parser()

    def test_new_commands_parse(self):
        args = self.parse("--method", "baseline")
        self.assertEqual(args.method, "baseline")

        args = self.parse("--method", "shared_prompt", "--use_ema")
        self.assertTrue(args.use_ema)

        args = self.parse("--method", "gate", "--gate_blocks", "0", "1", "2", "3", "4")
        self.assertEqual(args.gate_blocks, [0, 1, 2, 3, 4])

        args = self.parse(
            "--method", "hfpool", "--hopfield_trainable", "o", "k", "query",
            "--hopfield_grad_clip", "--hopfield_grad_clip_norm", "0.5",
            "--hopfield_input_mode", "full_vit", "--hopfield_start_layer", "3",
        )
        self.assertEqual(args.hopfield_trainable, ["o", "k", "query"])
        self.assertTrue(args.hopfield_grad_clip)
        self.assertEqual(args.hopfield_grad_clip_norm, 0.5)
        self.assertEqual(args.hopfield_input_mode, "full_vit")
        self.assertEqual(args.hopfield_start_layer, 3)

        self.assertEqual(self.parse("--method", "hfpool").hopfield_trainable, ["q"])

    def test_parsed_commands_build_registered_trainer_and_model(self):
        """main.py -> _Trainer(**vars(args)) -> select_model -> MODELS[method](**kwargs)."""
        from methods import METHODS
        from models import MODELS
        from utils.train_utils import STEP_AWARE_METHODS

        commands = {
            "baseline": ([], FrozenViTTrainer, Baseline),
            "shared_prompt": (["--len_prompt", "3", "--use_ema"], FrozenViTTrainer, SharedPrompt),
            "gate": (["--gate_blocks", "0", "1"], FrozenViTTrainer, Gate),
            "hfpool": (
                ["--hopfield_trainable", "o", "k", "--num_pooling_blocks", "2",
                 "--hopfield_start_layer", "1", "--hopfield_grad_clip"],
                HFPoolTrainer, HFPool,
            ),
            "flyprompt": (["--len_prompt", "3", "--rp_dim", "16"], FlyPromptTrainer, FlyPrompt),
        }
        for method, (extra, trainer_cls, model_cls) in commands.items():
            with self.subTest(method=method):
                args = self.parse("--method", method, "--backbone", BACKBONE, *extra)
                self.assertIs(METHODS[method], trainer_cls)
                self.assertIs(MODELS[method], model_cls)
                self.assertIn(method, STEP_AWARE_METHODS)

                kwargs = vars(args)
                torch.manual_seed(0)
                model = MODELS[method](
                    backbone_name=args.backbone,
                    pretrained=False,
                    num_classes=NUM_CLASSES,
                    task_num=4,
                    **{key: value for key, value in kwargs.items() if key != "backbone"},
                )
                self.assertEqual(model(make_images()).shape, (2, NUM_CLASSES))

        hfpool = MODELS["hfpool"](
            backbone_name=BACKBONE, pretrained=False, num_classes=NUM_CLASSES, task_num=4,
            **{key: value for key, value in vars(self.parse(
                "--method", "hfpool", "--hopfield_trainable", "o", "k",
                "--num_pooling_blocks", "2", "--hopfield_start_layer", "1",
            )).items() if key != "backbone"},
        )
        self.assertEqual(hfpool.adapter.trainable, frozenset({"o", "k"}))
        self.assertEqual(hfpool.adapter._resolve_prompt_range(12), (1, 3))

    def test_old_flags_are_rejected(self):
        for old_flag in (
                "--use_hopfield", "--shared_prompt", "--disable_prompt",
                "--use_attention_gate", "--use_attention_pool"):
            with self.subTest(flag=old_flag), mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    self.parse("--method", "flyprompt", old_flag)


class ModelContractTest(unittest.TestCase):
    def test_baseline_trains_only_fc_on_clean_cls(self):
        model = make_model(Baseline).eval()
        self.assertEqual(trainable_names(model), FC)
        x = make_images()
        with torch.no_grad():
            expected = model.backbone.fc(model.backbone.forward_features(x)[:, 0])
            torch.testing.assert_close(model(x), expected)

    def test_gate_starts_as_identity_and_trains_only_gains(self):
        gate = make_model(Gate, gate_blocks=[0, 2]).eval()
        baseline = make_model(Baseline).eval()
        baseline.load_state_dict(gate.state_dict(), strict=False)
        self.assertEqual(trainable_names(gate), sorted(FC + ["adapter.gate_logits"]))
        self.assertEqual(tuple(gate.adapter.gate_logits.shape), (2, gate.backbone.blocks[0].attn.num_heads))
        x = make_images()
        with torch.no_grad():
            torch.testing.assert_close(gate(x), baseline(x), rtol=1e-4, atol=1e-5)

    def test_shared_prompt_has_one_expert_and_a_global_ema_bank(self):
        model = make_model(SharedPrompt, len_prompt=3, pos_prompt=[0, 1], use_ema=True)
        self.assertEqual(model.adapter.prompts.shape[1], 1)
        self.assertFalse(hasattr(model, "rp_head"))
        self.assertEqual(trainable_names(model), sorted(FC + ["adapter.prompts"]))

        x = make_images()
        self.assertEqual(len(model.forward_with_ema(x)), 3)

        before = model.ema_fc[0].weight.detach().clone()
        with torch.no_grad():
            model.backbone.fc.weight.add_(1.0)
        model.process_task_count()
        model.update_ema_fc()
        self.assertEqual(model.adapter.prompts.shape[1], 1)
        self.assertFalse(torch.equal(before, model.ema_fc[0].weight))

    def test_flyprompt_keeps_task_experts_and_rear(self):
        model = make_model(FlyPrompt, len_prompt=3, pos_prompt=[0], rp_dim=16, use_ema=True)
        self.assertEqual(model.experts.prompts.shape[1], 4)
        x = make_images()
        self.assertEqual(model.forward_with_rp(x).shape, (2, 4))
        self.assertEqual(len(model.forward_with_ema(x, expert_ids=torch.tensor([0, 1]))), 3)

        model.process_task_count()
        self.assertEqual(model.task_count, 1)
        torch.testing.assert_close(model.experts.prompts[:, 1], model.experts.prompts[:, 0])


class HFPoolModelTest(unittest.TestCase):
    def make(self, **kwargs):
        arguments = dict(num_pooling_blocks=2, num_pooling_heads=1, prompt_length=2)
        arguments.update(kwargs)
        return make_model(HFPool, **arguments)

    def test_each_trainable_part_receives_only_its_gradient(self):
        cases = (["query"], ["q"], ["k"], ["v"], ["o"], ["q", "k", "o"], ["o", "k", "query"])
        for parts in cases:
            with self.subTest(parts=parts):
                model = self.make(hopfield_trainable=parts)
                model(make_images()).sum().backward()

                for layer in model.adapter.hopfield_pooling_layers:
                    core = layer.hopfield.association_core
                    self.assertEqual(layer.pooling_weights.grad is not None, "query" in parts)
                    self.assertEqual(core.out_proj.weight.grad is not None, "o" in parts)

                    projection_slices = model.adapter._projection_slices(core)
                    active = [projection_slices[name] for name in ("q", "k", "v") if name in parts]
                    self.assertEqual(sorted(projection_slices), ["k", "q", "v"] if "v" in parts else ["k", "q"])
                    if not active:
                        self.assertIsNone(core.in_proj_weight.grad)
                        continue
                    row_is_active = torch.zeros(core.in_proj_weight.shape[0], dtype=torch.bool)
                    for active_slice in active:
                        row_is_active[active_slice] = True
                        self.assertGreater(core.in_proj_weight.grad[active_slice].abs().sum().item(), 0.0)
                    self.assertEqual(core.in_proj_weight.grad[~row_is_active].abs().sum().item(), 0.0)

                    frozen = [
                        name for name, param in layer.named_parameters()
                        if not param.requires_grad and param.grad is not None
                    ]
                    self.assertEqual(frozen, [])

    def test_trainable_v_starts_equal_to_static_value(self):
        static = self.make(hopfield_trainable=["q"]).eval()
        learned_v = self.make(hopfield_trainable=["q", "v"]).eval()
        learned_v.backbone.load_state_dict(static.backbone.state_dict())
        for static_layer, v_layer in zip(
                static.adapter.hopfield_pooling_layers,
                learned_v.adapter.hopfield_pooling_layers):
            static_state = static_layer.state_dict()
            v_state = v_layer.state_dict()
            for name, tensor in static_state.items():
                if name.endswith("in_proj_weight"):
                    v_state[name][:tensor.shape[0]] = tensor
                else:
                    v_state[name] = tensor
            v_layer.load_state_dict(v_state)

        x = make_images()
        with torch.no_grad():
            torch.testing.assert_close(learned_v(x), static(x), rtol=1e-4, atol=1e-5)

    def test_full_vit_cached_context_matches_uncached_forward(self):
        model = self.make(hopfield_input_mode="full_vit", hopfield_start_layer=2).eval()
        x = make_images()
        with torch.no_grad():
            context = model.prepare_hopfield_context(x)
            torch.testing.assert_close(model(x, hopfield_context=context), model(x))

        local = self.make().eval()
        with self.assertRaisesRegex(ValueError, "full_vit"):
            local(x, hopfield_context=context)

    def test_prompt_block_placement(self):
        depth = 12
        self.assertEqual(self.make().adapter._resolve_prompt_range(depth), (0, 2))
        self.assertEqual(self.make(deep_prompts=1).adapter._resolve_prompt_range(depth), (10, 12))
        self.assertEqual(
            self.make(deep_prompts=1, hopfield_start_layer=3).adapter._resolve_prompt_range(depth), (3, 5)
        )
        with self.assertRaisesRegex(ValueError, "inside the backbone"):
            self.make(hopfield_start_layer=11).adapter._resolve_prompt_range(depth)

    def test_rejects_unknown_trainable_part(self):
        with self.assertRaisesRegex(ValueError, "trainable"):
            self.make(hopfield_trainable=["norm"])


def make_trainer(trainer_cls, model, **overrides):
    """Build a trainer around ``model`` without _Trainer's dataset/DDP setup."""
    trainer = trainer_cls.__new__(trainer_cls)
    attributes = dict(
        device=torch.device("cpu"),
        model=model,
        model_without_ddp=model,
        n_classes=NUM_CLASSES,
        exposed_classes=[],
        mask=torch.full((NUM_CLASSES,), -torch.inf),
        online_iter=3,
        no_batchmask=False,
        topk=1,
        use_amp=False,
        world_size=1,
        distributed=False,
        sched_name="default",
        step_num=2,
        samples_per_step=4,
        current_step=0,
        current_step_seen_samples=0,
        ensemble_method="softmax_max_prob",
        hopfield_grad_clip=False,
        hopfield_grad_clip_norm=1.0,
        task_id=0,
        label_to_task={},
        train_transform=lambda images: images,
        test_transform_tensor=lambda images: images,
        criterion=model.loss_fn,
        scaler=torch.cuda.amp.GradScaler(enabled=False),
    )
    attributes.update(overrides)
    trainer.__dict__.update(attributes)
    trainer.optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=0.01
    )
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda step: 1)
    return trainer


class StreamSimulationTest(unittest.TestCase):
    """Three stream batches cross one internal step boundary, then evaluate."""

    STREAM = ([0, 1], [2, 3], [0, 4])

    def run_stream(self, trainer):
        model = trainer.model_without_ddp
        backbone_before = {
            name: tensor.clone()
            for name, tensor in model.backbone.state_dict().items()
            if not name.startswith("fc.")
        }
        trainable_before = {
            name: param.detach().clone()
            for name, param in model.named_parameters() if param.requires_grad
        }

        for batch_index, labels in enumerate(self.STREAM):
            loss, accuracy = trainer.online_step(
                make_images(seed=10 + batch_index), torch.tensor(labels), None
            )
            self.assertTrue(torch.isfinite(torch.tensor(loss)))
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)

        self.assertEqual(trainer.current_step, 1)
        self.assertEqual(model.task_count, 1)
        self.assertEqual(sorted(trainer.exposed_classes), [0, 1, 2, 3, 4])

        for name, tensor in model.backbone.state_dict().items():
            if name in backbone_before:
                self.assertTrue(torch.equal(tensor, backbone_before[name]), name)
        changed = [
            name for name, param in model.named_parameters()
            if name in trainable_before and not torch.equal(param, trainable_before[name])
        ]
        self.assertIn("backbone.fc.weight", changed)

        test_loader = [(make_images(seed=99), torch.tensor([1, 4]))]
        result = trainer.online_evaluate(test_loader)
        self.assertTrue(0.0 <= result["avg_acc"] <= 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(result["avg_loss"])))
        return changed

    def test_baseline(self):
        self.run_stream(make_trainer(FrozenViTTrainer, make_model(Baseline)))

    def test_shared_prompt_with_ema(self):
        model = make_model(SharedPrompt, len_prompt=3, pos_prompt=[0], use_ema=True)
        trainer = make_trainer(FrozenViTTrainer, model)
        with mock.patch.object(trainer, "_ensemble_logits", wraps=trainer._ensemble_logits) as ensemble:
            changed = self.run_stream(trainer)
        self.assertIn("adapter.prompts", changed)
        ensemble.assert_called()

    def test_gate(self):
        changed = self.run_stream(make_trainer(FrozenViTTrainer, make_model(Gate, gate_blocks=[0, 1])))
        self.assertIn("adapter.gate_logits", changed)

    def test_hfpool_joint_training_with_grad_clip(self):
        model = make_model(
            HFPool, num_pooling_blocks=2, prompt_length=2, hopfield_trainable=["q", "k", "o"]
        )
        trainer = make_trainer(HFPoolTrainer, model, hopfield_grad_clip=True, hopfield_grad_clip_norm=1e-3)
        with mock.patch(
                "torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
            changed = self.run_stream(trainer)
        self.assertEqual(clip.call_count, len(self.STREAM) * 3)
        clipped = clip.call_args.args[0]
        self.assertEqual(
            {id(param) for param in clipped},
            {id(param) for param in model.adapter.parameters() if param.requires_grad},
        )
        self.assertLessEqual(
            torch.norm(torch.stack([param.grad.norm() for param in clipped])).item(), 1e-3 + 1e-6
        )
        self.assertTrue(any(name.startswith("adapter.") for name in changed))

    def test_hfpool_full_vit_reuses_one_clean_pass_per_batch(self):
        model = make_model(
            HFPool, num_pooling_blocks=2, prompt_length=2,
            hopfield_trainable=["o"], hopfield_input_mode="full_vit",
        )
        trainer = make_trainer(HFPoolTrainer, model)
        transforms = []
        trainer.train_transform = lambda images: transforms.append(1) or images
        with mock.patch.object(
                model, "prepare_hopfield_context", wraps=model.prepare_hopfield_context) as prepare:
            self.run_stream(trainer)
        self.assertEqual(prepare.call_count, len(self.STREAM))
        self.assertEqual(len(transforms), len(self.STREAM))

        trainer.online_iter = 1
        with self.assertRaisesRegex(ValueError, "online_iter 3"):
            trainer.online_step(make_images(), torch.tensor([0, 1]), None)

    def test_flyprompt_routes_with_rear_at_evaluation(self):
        model = make_model(FlyPrompt, len_prompt=3, pos_prompt=[0], rp_dim=16, use_ema=True)
        trainer = make_trainer(FlyPromptTrainer, model)
        with mock.patch.object(model, "forward_with_rp", wraps=model.forward_with_rp) as route:
            changed = self.run_stream(trainer)
        route.assert_called()
        self.assertIn("experts.prompts", changed)
        self.assertEqual(sorted(trainer.label_to_task), [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
