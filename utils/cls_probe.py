"""Checkpoint-wise CLS geometry on a fixed cohort of actually observed images.

Variance is across images or checkpoints, never across one CLS's coordinates.
The first task-end state defines the cohort; all later states reuse it.
"""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.offline_probe import dataset_signature, mutable_state, restore_mutable_state


def choose_cohort(snapshot, dataset, seed, per_class):
    if dataset_signature(dataset) != snapshot["dataset_signature"]:
        raise ValueError("CLS cohort training split differs from saved stream")
    if per_class < 2:
        raise ValueError("Need at least two images per class")
    groups = {}
    for index in snapshot["indices"].tolist():
        groups.setdefault(int(dataset.targets[index]), []).append(index)
    groups = {cls: items for cls, items in groups.items() if len(items) >= 2}
    if len(groups) < 2:
        raise ValueError("CLS cohort needs two observed classes with two images each")
    count = min(per_class, min(map(len, groups.values())))
    generator = np.random.default_rng(seed)
    indices = np.concatenate([generator.choice(groups[cls], count, replace=False)
                              for cls in sorted(groups)]).astype(np.int64)
    labels = np.asarray([dataset.targets[index] for index in indices], dtype=np.int64)
    return indices, labels


def normalized(features):
    features = np.asarray(features, dtype=np.float64)
    return features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-12)


def geometry(features, labels):
    features = np.asarray(features, dtype=np.float64)
    classes = np.unique(labels)
    unit = normalized(features)
    similarities = unit @ unit.T
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    different = labels[:, None] != labels[None, :]
    centers = np.stack([features[labels == cls].mean(axis=0) for cls in classes])
    within = np.mean([features[labels == cls].var(axis=0, ddof=0).mean() for cls in classes])
    return dict(same_class_cos=float(similarities[same].mean()),
                different_class_cos=float(similarities[different].mean()),
                sample_variance=float(features.var(axis=0, ddof=0).mean()),
                within_class_variance=float(within),
                between_class_variance=float(centers.var(axis=0, ddof=0).mean()),
                norm_mean=float(np.linalg.norm(features, axis=-1).mean()))


def summarize_history(features, clean, labels, steps):
    features = np.asarray(features)
    if (features.ndim != 3 or features.shape[1:] != clean.shape or
            len(labels) != clean.shape[0] or len(steps) != len(features) or
            not np.isfinite(features).all() or not np.isfinite(clean).all()):
        raise ValueError("Invalid CLS history")
    unit = normalized(features)
    clean_unit = normalized(clean)
    rows = []
    for index, (values, step) in enumerate(zip(features, steps)):
        row = dict(checkpoint=index, samples=int(step), **geometry(values, labels))
        row.update({f"unit_{key}": value for key, value in geometry(unit[index], labels).items()
                    if "variance" in key})
        row["same_image_cos_first"] = float((unit[index] * unit[0]).sum(axis=-1).mean())
        row["same_image_cos_previous"] = float((unit[index] * unit[max(0, index-1)]).sum(axis=-1).mean())
        row["cos_clean"] = float((unit[index] * clean_unit).sum(axis=-1).mean())
        row["temporal_variance"] = float(features[:index+1].astype(np.float64).var(axis=0, ddof=0).mean())
        row["unit_temporal_variance"] = float(unit[:index+1].var(axis=0, ddof=0).mean())
        rows.append(row)
    return rows


def run_cls_probe(model, dataset, transform, directory, device, seed, per_class=16, batchsize=64):
    directory = Path(directory)
    paths = sorted(directory.glob("t[0-9][0-9].pt"))
    if not paths:
        raise ValueError("No CLS snapshots")
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    indices, labels = choose_cohort(first, dataset, seed, per_class)
    state = mutable_state(model)
    modes = {module: module.training for module in model.modules()}
    random_state, numpy_state = random.getstate(), np.random.get_state()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.device(device).type == "cuda" else []
    history, steps, tasks, clean = [], [], [], None
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batchsize,
                        shuffle=False, num_workers=0)
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad(), torch.autocast(
                device_type=torch.device(device).type, enabled=False):
            model.eval()
            for path in paths:
                snapshot = torch.load(path, map_location="cpu", weights_only=False)
                if (snapshot["dataset_signature"] != first["dataset_signature"] or
                        not set(indices.tolist()).issubset(snapshot["indices"].tolist())):
                    raise ValueError("CLS cohort was not observed at this checkpoint")
                restore_mutable_state(model, snapshot["state"])
                values, clean_values, actual_labels = [], [], []
                for images, target in loader:
                    images = images.to(device)
                    if transform is not None:
                        images = transform(images)
                    values.append(model.extract_features(images).float().cpu())
                    if clean is None:
                        clean_values.append(model.backbone.forward_features(images)[:, 0].float().cpu())
                    actual_labels.extend(target.tolist())
                if actual_labels != labels.tolist():
                    raise ValueError("CLS cohort label/order mismatch")
                history.append(torch.cat(values).numpy())
                if clean is None:
                    clean = torch.cat(clean_values).numpy()
                steps.append(snapshot["samples_cnt"])
                tasks.append(snapshot["task_id"])
    finally:
        restore_mutable_state(model, state)
        for module, training in modes.items():
            module.training = training
        random.setstate(random_state)
        np.random.set_state(numpy_state)
    metadata = dict(format_version=1, seed=int(seed), arguments=first["arguments"],
                    dataset_signature=first["dataset_signature"],
                    cohort_rule="balanced first-checkpoint actually observed training indices",
                    per_class=int(np.bincount(np.unique(labels, return_inverse=True)[1]).min()),
                    task_ids=tasks, variance_ddof=0)
    history = np.stack(history)
    np.savez_compressed(directory / "history.npz", features=history, clean=clean,
                        indices=indices, labels=labels, samples=np.asarray(steps),
                        metadata=np.asarray(json.dumps(metadata)))
    rows = summarize_history(history, clean, labels, steps)
    with (directory / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "clean_metrics.json").write_text(json.dumps(geometry(clean, labels), indent=2), encoding="utf-8")
    return rows
