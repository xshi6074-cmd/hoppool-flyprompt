"""Contracts for the frozen-ViT mechanism methods split out of FlyPrompt.

Models use the repository's ``vit_tiny_patch16_224`` with random weights so the
suite runs on CPU. ``StreamSimulationTest`` drives the real trainer hooks the
way ``_Trainer.main_worker`` does: stream batches -> ``online_step`` (including
the internal step boundary) -> ``online_evaluate``.
"""
import os
import math
from pathlib import Path
import shlex
import subprocess
import sys
import unittest
from unittest import mock


import torch

from methods.flyprompt import FlyPrompt as FlyPromptTrainer
from methods.frozen_vit import FrozenViTTrainer
from methods.hfpool import HFPool as HFPoolTrainer
from methods.hfpool_lora import HFPoolLoRA as HFPoolLoRATrainer
from methods.spatial_gelu import SpatialGELU as SpatialGELUTrainer
from models.flyprompt import FlyPrompt
from models.frozen_vit import Baseline
from models.gate import Gate
from models.hfpool import HFPool
from models.hfpool_lora import HFPoolLoRA
from models.spatial_gelu import SpatialGELU
from models.shared_prompt import SharedPrompt
from utils.distill_gradnorm import DistillationDelayGate, GradNormLiteController

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
        args = self.parse(
            "--method", "hfpool", "--hopfield_trainable", "o",
            "--hopfield_o_mode", "lora_merge_1024", "--hopfield_lora_rank", "8",
            "--hopfield_lora_alpha", "8", "--hopfield_lora_lr", "0.005",
        )
        self.assertEqual(args.hopfield_o_mode, "lora_merge_1024")
        self.assertEqual(args.hopfield_lora_lr, 0.005)

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
            "hfpool_lora": (["--num_pooling_blocks", "1"], HFPoolLoRATrainer, HFPoolLoRA),
            "spatial_gelu": (["--num_pooling_blocks", "1"], SpatialGELUTrainer, SpatialGELU),
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

    def test_single_pool_generates_once_and_reuses_identical_prompts(self):
        for blocks in ([0], [1], [2], [0, 1, 2]):
            with self.subTest(blocks=blocks):
                model = self.make(num_pooling_blocks=1, hopfield_prompt_blocks=blocks,
                                  hopfield_trainable=["o"])
                layer = model.adapter.hopfield_pooling_layers[0]
                token_counts = []
                handles = [block.register_forward_pre_hook(
                    lambda module, inputs, index=index: token_counts.append((index, inputs[0].size(1)))
                ) for index, block in enumerate(model.backbone.blocks)]
                with mock.patch.object(layer, "forward", wraps=layer.forward) as pool:
                    features, prompts = model.adapter(model.backbone, make_images(), return_prompts=True)
                for handle in handles:
                    handle.remove()
                self.assertEqual(pool.call_count, 1)
                self.assertEqual(len(model.adapter.hopfield_pooling_layers), 1)
                self.assertEqual([i for i, count in token_counts if count == 199], blocks)
                self.assertTrue(all(count in (197, 199) for _, count in token_counts))
                for i in range(1, len(blocks)):
                    torch.testing.assert_close(prompts[:, 0], prompts[:, i], rtol=0, atol=0)
                features[:, 0].sum().backward()
                self.assertGreater(layer.hopfield.association_core.out_proj.weight.grad.abs().sum(), 0)

    def test_single_pool_replaces_cls_for_every_remaining_block(self):
        model = self.make(num_pooling_blocks=1, prompt_length=1,
                          hopfield_replace_cls=True, hopfield_trainable=["o"])
        layer = model.adapter.hopfield_pooling_layers[0]
        token_counts = []
        handles = [block.register_forward_pre_hook(
            lambda module, inputs: token_counts.append(inputs[0].size(1))
        ) for block in model.backbone.blocks]
        with mock.patch.object(layer, "forward", wraps=layer.forward) as pool:
            features, prompts = model.adapter(model.backbone, make_images(), return_prompts=True)
        for handle in handles:
            handle.remove()
        self.assertEqual(pool.call_count, 1)
        self.assertEqual(prompts.shape, (2, 1, 1, model.embed_dim))
        self.assertEqual(token_counts, [197] * len(model.backbone.blocks))
        model.backbone.fc(features).square().mean().backward()
        self.assertGreater(layer.hopfield.association_core.out_proj.weight.grad.abs().sum(), 0)

        for invalid in ({"num_pooling_blocks": 2}, {"prompt_length": 2},
                        {"hopfield_prompt_blocks": [0]}):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "CLS replacement"):
                options = dict(num_pooling_blocks=1, prompt_length=1,
                               hopfield_replace_cls=True)
                options.update(invalid)
                self.make(**options)

    def test_single_pool_source_is_identical_across_insertion_locations(self):
        sources = []
        for blocks in ([0], [1], [2], [0, 1, 2]):
            model = self.make(num_pooling_blocks=1, hopfield_prompt_blocks=blocks).eval()
            handle = model.adapter.hopfield_pooling_layers[0].register_forward_pre_hook(
                lambda module, inputs: sources.append(inputs[0].detach().clone()))
            with torch.no_grad():
                model(make_images())
            handle.remove()
        for source in sources[1:]:
            torch.testing.assert_close(source, sources[0], rtol=0, atol=0)

    def test_explicit_single_pool_preserves_legacy_block_zero_forward(self):
        legacy = self.make(num_pooling_blocks=1).eval()
        explicit = self.make(num_pooling_blocks=1, hopfield_prompt_blocks=[0]).eval()
        explicit.load_state_dict(legacy.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(legacy(make_images()), explicit(make_images()), rtol=0, atol=0)

    def test_output_lora_starts_at_full_weight_and_only_trains_ab(self):
        full = self.make(hopfield_trainable=["o"])
        lora = self.make(hopfield_trainable=["o"], hopfield_o_mode="lora")
        for full_layer, lora_layer in zip(
                full.adapter.hopfield_pooling_layers,
                lora.adapter.hopfield_pooling_layers):
            full_proj = full_layer.hopfield.association_core.out_proj
            lora_proj = lora_layer.hopfield.association_core.out_proj
            torch.testing.assert_close(lora_proj.weight, full_proj.weight)
            self.assertFalse(lora_proj.parametrizations.weight.original.requires_grad)
            self.assertFalse(lora_proj.bias.requires_grad)
            self.assertTrue(lora_proj.parametrizations.weight[0].lora_A.requires_grad)
            self.assertTrue(lora_proj.parametrizations.weight[0].lora_B.requires_grad)
        with self.assertRaisesRegex(ValueError, "hopfield_trainable o only"):
            self.make(hopfield_o_mode="lora")

    def test_output_lora_merge_preserves_forward(self):
        model = self.make(hopfield_trainable=["o"], hopfield_o_mode="lora_merge_batch").eval()
        for lora in model.adapter.lora_output_weights():
            with torch.no_grad():
                lora.lora_B.fill_(0.01)
        x = make_images()
        with torch.no_grad():
            before = model(x)
            stats = model.adapter.merge_lora_output_weights()
            after = model(x)
        torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)
        self.assertTrue(all(delta > 0 for delta, _ in stats))
        for lora in model.adapter.lora_output_weights():
            self.assertEqual(torch.count_nonzero(lora.lora_B).item(), 0)


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
        hopfield_lora_lr=0.005,
        lr=0.005,
        lr_gamma=0.9999,
        opt_name="adam",
        n_tasks=4,
        lora_seen_samples=0,
        lora_samples_since_merge=0,
        full_o_seen_samples=0,
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

    def test_hfpool_lora_batch_merges_after_three_updates(self):
        model = make_model(
            HFPool, num_pooling_blocks=1, prompt_length=2,
            hopfield_trainable=["o"], hopfield_o_mode="lora_merge_batch",
        )
        trainer = make_trainer(HFPoolTrainer, model, hopfield_lora_lr=0.002)
        trainer._configure_lora_optimizer()
        self.assertEqual([group["lr"] for group in trainer.optimizer.param_groups],
                         [0.005, 0.002])
        trainer.update_schedule(reset=True)
        self.assertEqual([group["lr"] for group in trainer.optimizer.param_groups],
                         [0.005, 0.002])
        out_proj = model.adapter.hopfield_pooling_layers[0].hopfield.association_core.out_proj
        base_before = out_proj.parametrizations.weight.original.detach().clone()
        with self.assertLogs(level="INFO") as logs:
            trainer.online_step(make_images(), torch.tensor([0, 1]), None)
        self.assertFalse(torch.equal(base_before, out_proj.parametrizations.weight.original))
        self.assertEqual(trainer.lora_seen_samples, 2)
        self.assertEqual(trainer.lora_samples_since_merge, 0)
        self.assertTrue(any("relative_delta_w=" in entry for entry in logs.output))
        self.assertTrue(any("relative_merge=" in entry for entry in logs.output))
        for lora in model.adapter.lora_output_weights():
            self.assertEqual(torch.count_nonzero(lora.lora_B).item(), 0)
            self.assertNotIn(lora.lora_A, trainer.optimizer.state)
            self.assertNotIn(lora.lora_B, trainer.optimizer.state)

    def test_hfpool_lora_without_merge_keeps_base_frozen(self):
        model = make_model(
            HFPool, num_pooling_blocks=1, prompt_length=2,
            hopfield_trainable=["o"], hopfield_o_mode="lora",
        )
        trainer = make_trainer(HFPoolTrainer, model)
        trainer._configure_lora_optimizer()
        out_proj = model.adapter.hopfield_pooling_layers[0].hopfield.association_core.out_proj
        base_before = out_proj.parametrizations.weight.original.detach().clone()
        trainer.online_step(make_images(), torch.tensor([0, 1]), None)
        torch.testing.assert_close(out_proj.parametrizations.weight.original, base_before)
        self.assertGreater(torch.count_nonzero(
            model.adapter.lora_output_weights()[0].lora_B).item(), 0)
        self.assertEqual(trainer.lora_samples_since_merge, 2)

    def test_hfpool_lora_1024_counts_complete_batches(self):
        model = make_model(
            HFPool, num_pooling_blocks=1, prompt_length=2,
            hopfield_trainable=["o"], hopfield_o_mode="lora_merge_1024",
        )
        trainer = make_trainer(HFPoolTrainer, model)
        trainer._configure_lora_optimizer()
        out_proj = model.adapter.hopfield_pooling_layers[0].hopfield.association_core.out_proj
        base_before = out_proj.parametrizations.weight.original.detach().clone()
        with torch.no_grad():
            model.adapter.lora_output_weights()[0].lora_B.fill_(0.01)
        for _ in range(15):
            trainer.after_online_updates(torch.empty(64, 1), None)
        self.assertEqual(trainer.lora_samples_since_merge, 960)
        torch.testing.assert_close(out_proj.parametrizations.weight.original, base_before)
        trainer.after_online_updates(torch.empty(64, 1), None)
        self.assertEqual(trainer.lora_seen_samples, 1024)
        self.assertEqual(trainer.lora_samples_since_merge, 0)
        self.assertFalse(torch.equal(out_proj.parametrizations.weight.original, base_before))

    def test_flyprompt_routes_with_rear_at_evaluation(self):
        model = make_model(FlyPrompt, len_prompt=3, pos_prompt=[0], rp_dim=16, use_ema=True)
        trainer = make_trainer(FlyPromptTrainer, model)
        with mock.patch.object(model, "forward_with_rp", wraps=model.forward_with_rp) as route:
            changed = self.run_stream(trainer)
        route.assert_called()
        self.assertIn("experts.prompts", changed)
        self.assertEqual(sorted(trainer.label_to_task), [0, 1, 2, 3, 4])



class ClassifierReplayTest(unittest.TestCase):
    def make(self, model_cls=SharedPrompt, mode="fixed", statistics="diagonal", **kwargs):
        arguments = dict(classifier_replay=True, classifier_replay_weight_mode=mode,
                         statistic_type=statistics, anchors_per_class=8,
                         distill_weight=1.0, use_ema=True)
        if model_cls is HFPool:
            arguments.update(num_pooling_blocks=1, prompt_length=2,
                             hopfield_trainable=["o"], hopfield_prompt_blocks=[0, 1, 2])
        else:
            arguments.update(len_prompt=2, pos_prompt=[0, 1, 2])
        arguments.update(kwargs)
        return make_model(model_cls, **arguments)

    def seed_history(self, model):
        replay = model.classifier_replay
        features = torch.randn(32, model.embed_dim)
        replay.update_statistics(features, torch.arange(32) % 2)
        replay.snapshot(model.backbone.fc)
        replay.snapshot(model.backbone.fc)
        with torch.no_grad():
            model.backbone.fc.weight[:2].add_(0.02)
        return replay

    def test_replay_has_only_head_gradients_and_excludes_future_logits(self):
        for statistics in ("diagonal", "low_rank"):
            model = self.make(model_cls=HFPool, statistics=statistics)
            replay = self.seed_history(model)
            torch.manual_seed(12)
            draw_before = replay.draw_count.clone()
            rng_before = torch.get_rng_state().clone()
            before = replay.loss(model.backbone.fc)
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
            with torch.no_grad():
                model.backbone.fc.weight[2:].add_(100)
            torch.manual_seed(12)
            replay.draw_count.copy_(draw_before)
            after = replay.loss(model.backbone.fc)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            after.backward()
            self.assertGreater(model.backbone.fc.weight.grad[:2].abs().sum(), 0)
            self.assertEqual(model.backbone.fc.weight.grad[2:].abs().sum(), 0)
            self.assertTrue(all(p.grad is None for p in model.adapter.parameters()))
            self.assertTrue(all(p.grad is None for p in replay.teachers.parameters()))

    def test_sample_and_closed_form_agree_for_both_statistics(self):
        for statistics in ("diagonal", "low_rank"):
            model = self.make(statistics=statistics, anchors_per_class=12000)
            replay = self.seed_history(model)
            sampled = replay.loss(model.backbone.fc).detach()
            replay.distill_mode = "closed_form"
            analytic = replay.loss(model.backbone.fc).detach()
            torch.testing.assert_close(sampled, analytic, rtol=0.04, atol=1e-4)

    def test_training_counts_each_stream_sample_once_and_updates_ema(self):
        model = self.make()
        trainer = make_trainer(FrozenViTTrainer, model, online_iter=3, samples_per_step=1000)
        self.configure_trainer(trainer)
        ema_before = model.ema_fc[0].weight.clone()
        trainer.online_step(make_images(), torch.tensor([0, 1]), None)
        self.assertEqual(model.classifier_replay.statistics.live_count.sum(), 2)
        self.assertFalse(torch.equal(ema_before, model.ema_fc[0].weight))

    def configure_trainer(self, trainer):
        trainer._classifier_replay_gate = DistillationDelayGate(0)
        trainer._classifier_replay_controller = GradNormLiteController(.25, .9, 1e-4, 1e4)
        trainer._replay_acc_steps = trainer._replay_acc_active = trainer._replay_acc_invalid = 0
        trainer._replay_acc_ce = trainer._replay_acc_raw = trainer._replay_acc_weighted = trainer._replay_acc_balance = 0.0

    def test_target_ratio_is_learned_and_prior_rejects_collapse(self):
        model = self.make(mode="learned_gradnorm")
        replay = self.seed_history(model)
        trainer = make_trainer(FrozenViTTrainer, model)
        self.configure_trainer(trainer)
        trainer._classifier_replay_gate.finish_batch(2, 2)
        features = torch.randn(8, model.embed_dim)
        ce = model.loss_fn(model.backbone.fc(features)[:, :2], torch.arange(8) % 2)
        raw = replay.loss(model.backbone.fc)
        weighted, balance, weight, ratio, reason = trainer._classifier_replay_objective(ce, raw)
        self.assertGreater(weight, 0)
        self.assertTrue(math.isfinite(ratio))
        self.assertAlmostEqual(float(replay.ratio_logit.detach().sigmoid()), .25, places=6)
        (weighted + balance).backward()
        self.assertGreater(replay.ratio_logit.grad, 0)
        self.assertTrue(torch.isfinite(replay.ratio_logit.grad))
        replay.ratio_logit.grad = None
        with torch.no_grad():
            replay.ratio_logit.fill_(-10)
        ce = model.loss_fn(model.backbone.fc(features)[:, :2], torch.arange(8) % 2)
        raw = replay.loss(model.backbone.fc)
        weighted, balance, *_ = trainer._classifier_replay_objective(ce, raw)
        balance.backward()
        self.assertLess(replay.ratio_logit.grad, 0)

    def test_active_replay_runs_three_real_updates_in_both_methods(self):
        for model_cls in (SharedPrompt, HFPool):
            for mode in ("gradnorm", "learned_gradnorm"):
                for statistics in ("diagonal", "low_rank"):
                    with self.subTest(model=model_cls, mode=mode, statistics=statistics):
                        model = self.make(model_cls=model_cls, mode=mode, statistics=statistics)
                        replay = self.seed_history(model)
                        trainer_cls = HFPoolTrainer if model_cls is HFPool else FrozenViTTrainer
                        trainer = make_trainer(trainer_cls, model, online_iter=3, samples_per_step=1000)
                        self.configure_trainer(trainer)
                        trainer._classifier_replay_gate.finish_batch(2, 32)
                        count_before = int(replay.statistics.live_count.sum())
                        ratio_before = (replay.ratio_logit.detach().clone()
                                        if replay.ratio_logit is not None else None)
                        with mock.patch.object(model, "update_ema_fc", wraps=model.update_ema_fc) as ema:
                            loss, _ = trainer.online_step(make_images(), torch.tensor([0, 1]), None)
                        self.assertTrue(math.isfinite(loss))
                        self.assertEqual(ema.call_count, 3)
                        self.assertEqual(replay.statistics.live_count.sum(), count_before + 2)
                        self.assertEqual(trainer._replay_acc_active, 3)
                        self.assertGreater(trainer.last_classifier_replay["weight"], 0)
                        if ratio_before is not None:
                            self.assertFalse(torch.equal(ratio_before, replay.ratio_logit))
                        state = model.state_dict()
                        restored = self.make(model_cls=model_cls, mode=mode, statistics=statistics)
                        restored.load_state_dict(state)
                        self.assertEqual(restored.classifier_replay.snapshot_count, 2)


if __name__ == "__main__":
    unittest.main()
