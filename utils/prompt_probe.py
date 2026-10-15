"""Small, deterministic prompt probe for paired HFPool/shared-prompt runs."""

import hashlib

import numpy as np
import torch
import torch.nn.functional as F


def fixed_test_probe(dataset, count):
    if count < 2:
        raise ValueError("prompt_probe_examples must be at least 2")
    targets = dataset.targets
    indices = []
    labels = []
    seen = set()
    for index, label in enumerate(targets):
        label = int(label)
        if label not in seen:
            indices.append(index)
            labels.append(label)
            seen.add(label)
            if len(indices) == count:
                break
    if len(indices) != count:
        raise ValueError(f"Only {len(indices)} distinct test classes for {count} probe images")
    images = torch.stack([dataset[index][0] for index in indices]).contiguous()
    image_hash = hashlib.sha256(images.numpy().tobytes()).hexdigest()
    return images, np.asarray(indices), np.asarray(labels), image_hash


class PromptProbeRecorder:
    def __init__(self, dataset, count, method):
        self.images, self.indices, self.labels, self.image_hash = fixed_test_probe(dataset, count)
        self.method = method
        self.layers = None
        self.samples = []
        self.mean_vectors = []
        self.between_image_cosine = []
        self.same_image_first_cosine = []
        self.same_image_previous_cosine = []
        self.relative_change_norm = []
        self._first_flat = None
        self._previous_flat = None
        self._first_prompt = None

    @torch.no_grad()
    def record(self, model, device, seen_samples):
        layers, prompts = model.capture_prompts(self.images.to(device))
        layers = np.asarray(layers, dtype=np.int64)
        if prompts.ndim != 4 or prompts.shape[0] != len(self.indices) or prompts.shape[1] != len(layers):
            raise ValueError(f"Unexpected prompt probe shape: {tuple(prompts.shape)}")
        if self.layers is None:
            self.layers = layers
        elif not np.array_equal(self.layers, layers):
            raise ValueError("Prompt layers changed during the run")

        prompts = prompts.detach().float()
        raw_flat = prompts.flatten(2).cpu()
        if self._first_prompt is None:
            self._first_prompt = raw_flat
        relative_change = (
            (raw_flat - self._first_prompt).norm(dim=-1)
            / self._first_prompt.norm(dim=-1).clamp_min(1e-12)
        ).mean(dim=0)
        # The full-token cosine measures input specificity within each method.
        flat = F.normalize(prompts.flatten(2), dim=-1)
        gram = torch.einsum("bld,cld->lbc", flat, flat)
        n = len(self.indices)
        off_diagonal = (gram.sum(dim=(1, 2)) - gram.diagonal(dim1=1, dim2=2).sum(dim=1)) / (n * (n - 1))
        if self._first_flat is None:
            self._first_flat = flat.cpu()
        if self._previous_flat is None:
            self._previous_flat = flat.cpu()
        first_cosine = (flat.cpu() * self._first_flat).sum(dim=-1).mean(dim=0)
        previous_cosine = (flat.cpu() * self._previous_flat).sum(dim=-1).mean(dim=0)
        self._previous_flat = flat.cpu()
        self.samples.append(int(seen_samples))
        self.mean_vectors.append(prompts.mean(dim=2).cpu().numpy())
        self.between_image_cosine.append(off_diagonal.cpu().numpy())
        self.same_image_first_cosine.append(first_cosine.numpy())
        self.same_image_previous_cosine.append(previous_cosine.numpy())
        self.relative_change_norm.append(relative_change.numpy())

    def save(self, path):
        if not self.samples:
            return
        np.savez_compressed(
            path,
            method=np.asarray(self.method),
            seen_samples=np.asarray(self.samples, dtype=np.int64),
            layer_indices=self.layers,
            probe_indices=self.indices,
            probe_labels=self.labels,
            probe_sha256=np.asarray(self.image_hash),
            mean_vectors=np.stack(self.mean_vectors).astype(np.float32),
            between_image_cosine=np.stack(self.between_image_cosine).astype(np.float32),
            same_image_first_cosine=np.stack(self.same_image_first_cosine).astype(np.float32),
            same_image_previous_cosine=np.stack(self.same_image_previous_cosine).astype(np.float32),
            relative_change_norm=np.stack(self.relative_change_norm).astype(np.float32),
        )
