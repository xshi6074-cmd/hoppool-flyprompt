"""Small synthetic check of saved diagnostic data and both plotters."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.plot_forgetting import class_rows, plot_group
from scripts.plot_prompt_similarity import discover_probes, plot_pair, plot_single
from scripts.plot_run_f_last import JOB
from utils.prompt_probe import PromptProbeRecorder


class ProbeDataset:
    targets = [0, 0, 1, 1, 2]

    def __getitem__(self, index):
        return torch.full((1, 2, 2), float(index)), self.targets[index]


class ProbeModel:
    def __init__(self, shared):
        self.shared = shared
        self.offset = 0.0

    def capture_prompts(self, images):
        values = torch.ones(len(images), 2, 2, 3)
        if not self.shared:
            values[:, :, :, 0] = images[:, 0, 0, 0, None, None]
        values[:, :, :, 1] = self.offset
        return (0, 1), values


class DiagnosticPlotsTest(unittest.TestCase):
    def test_prompt_discovery_accepts_public_method_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for method in ("hfpool_lora", "spatial_gelu", "shared_prompt"):
                note = root / "logs" / "toy" / f"custom_{method}"
                note.mkdir(parents=True)
                np.savez(note / "seed_1_prompt_probe.npz", method=np.asarray(method))
            self.assertEqual(len(discover_probes(root)), 3)
            self.assertEqual(len(discover_probes(root / "logs", "custom_spatial")), 1)

    def test_f_last_plot_recognizes_lora_jobs(self):
        for method in ("hfpool_wo", "hfpool_lora", "hfpool_lora_merge_1024",
                       "hfpool_lora_merge_batch", "shared_prompt", "single_b0-1-2_hfpool_wo",
                       "replay_low_rank_learned_gradnorm_hfpool_wo",
                       "replay_diagonal_gradnorm_shared_prompt"):
            self.assertIsNotNone(JOB.fullmatch(
                f"job_001_cifar_ibot21k_{method}_seed1.log"
            ))

    def test_saved_probes_and_class_drops_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = ProbeDataset()
            for method in ("hfpool", "shared_prompt"):
                recorder = PromptProbeRecorder(dataset, 3, method)
                model = ProbeModel(shared=method == "shared_prompt")
                recorder.record(model, "cpu", 10)
                model.offset = 1.0
                recorder.record(model, "cpu", 20)
                recorder.save(root / f"{method}.npz")
            plot_pair(root / "hfpool.npz", root / "shared_prompt.npz", root / "prompt")
            plot_single(root / "hfpool.npz", root / "single")
            self.assertTrue((root / "single" / "prompt_similarity.png").exists())
            self.assertTrue((root / "prompt" / "prompt_similarity.png").exists())
            with np.load(root / "hfpool.npz") as probe:
                self.assertEqual(probe["same_image_first_cosine"].shape, (2, 2))
                self.assertTrue(np.all(probe["same_image_first_cosine"][1] < 1))

            history = root / "seed_1_class_history.npz"
            accuracy = np.array([[.8, 0], [.7, .6], [.5, .4]])
            np.savez_compressed(history, cls_accuracy=accuracy, class_ids=np.array([7, 9]),
                                seen_mask=np.array([[1, 0], [1, 1], [1, 1]], dtype=bool),
                                f_last=np.array(.25))
            rows = class_rows(history)
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(sum(row["accuracy_drop"] for row in rows) / 2, .25)
            plot_group([history], root / "class", "sample", "method")
            self.assertTrue((root / "class" / "sample" / "method" / "per_class_forgetting.png").exists())


if __name__ == "__main__":
    unittest.main()
