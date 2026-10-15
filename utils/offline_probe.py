"""Post-stream linear readouts of immutable frozen-ViT method snapshots.

Only actual observed training indices enter the probe. Test labels select the
seen-class evaluation subset; they never enter fitting or epoch selection.
"""
import csv
import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeSettings:
    epochs: int = 100
    patience: int = 10
    lr: float = 0.001
    weight_decay: float = 0.0005
    batchsize: int = 256

    def __post_init__(self):
        if min(self.epochs, self.patience, self.batchsize) <= 0:
            raise ValueError("Probe epochs, patience and batchsize must be positive")
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Probe learning rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Probe weight decay must be finite and nonnegative")

    @classmethod
    def from_arguments(cls, arguments):
        return cls(**{name: arguments.get(f"offline_probe_{name}", default)
                      for name, default in asdict(cls()).items()})


class ObservedSamples:
    """Deduplicate stream visits; retaining a class does not reveal its future images."""
    def __init__(self):
        self.labels = {}
        self.visits = 0

    def add(self, indices, labels):
        indices = torch.as_tensor(indices).reshape(-1).tolist()
        labels = torch.as_tensor(labels).reshape(-1).tolist()
        if len(indices) != len(labels):
            raise ValueError("Stream indices and labels have different lengths")
        for index, label in zip(indices, labels):
            index, label = int(index), int(label)
            if index < 0 or (index in self.labels and self.labels[index] != label):
                raise ValueError("Invalid index or inconsistent observed label")
            self.labels[index] = label
        self.visits += len(indices)


def dataset_signature(dataset):
    digest = hashlib.sha256(np.asarray(dataset.targets, dtype=np.int64).tobytes())
    # CUB uses a fixed random Subset: also identify the source-image indices.
    subset = getattr(dataset, "dataset", None)
    if hasattr(subset, "indices"):
        digest.update(np.asarray(subset.indices, dtype=np.int64).tobytes())
    return digest.hexdigest()


def mutable_state(model):
    # Frozen ViT weights are reconstructed from the pinned pretrained file;
    # keep adapter buffers, head, EMA and replay state without duplicating ViT.
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
            if not key.startswith("backbone.") or key.startswith("backbone.fc.")}


def restore_mutable_state(model, state):
    expected = set(mutable_state(model))
    if set(state) != expected:
        raise ValueError("Snapshot adapter/head state does not match this model")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not key.startswith("backbone.")
                                     for key in result.missing_keys):
        raise ValueError("Unexpected incomplete snapshot state")


def save_snapshot(directory, model, dataset, observed, class_ids, task_id,
                  samples_cnt, online_eval, arguments, ensemble_method):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    indices = sorted(observed.labels)
    if not indices or indices[-1] >= len(dataset):
        raise ValueError("Observed training indices are outside the dataset")
    if any(int(dataset.targets[i]) != observed.labels[i] for i in indices):
        raise ValueError("Observed labels disagree with the training dataset")
    if observed.visits != samples_cnt:
        raise ValueError("Snapshot sample count does not match actual stream visits")
    payload = dict(format_version=1, state=mutable_state(model),
                   arguments=dict(arguments), dataset_signature=dataset_signature(dataset),
                   indices=torch.tensor(indices, dtype=torch.long), class_ids=list(class_ids),
                   task_id=int(task_id), samples_cnt=int(samples_cnt),
                   online_eval=online_eval, ensemble_method=ensemble_method,
                   evaluation_batchsize=int(arguments.get("batchsize", 128)) * 2)
    path = directory / f"t{task_id:02d}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    logger.info("Offline probe snapshot | task=%d | visits=%d | unique=%d | classes=%d",
                task_id, samples_cnt, len(indices), len(class_ids))


def split_validation(labels, seed):
    """Fixed per-class 10% holdout; retain at least one fitting example per class."""
    generator = torch.Generator().manual_seed(seed)
    training, validation = [], []
    for cls in labels.unique(sorted=True):
        indices = torch.where(labels == cls)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        count = min(max(1, int(len(indices) * .1)), len(indices) - 1)
        validation.extend(indices[:count].tolist())
        training.extend(indices[count:].tolist())
    return torch.tensor(training, dtype=torch.long), torch.tensor(validation, dtype=torch.long)


def linear_scores(head, features, device, batchsize):
    with torch.no_grad():
        return torch.cat([head(batch.to(device)).cpu()
                          for batch in features.split(batchsize)])


def fit_linear_probe(features, labels, num_classes, settings, seed, device):
    """Select epoch budget on training-only holdout, then refit ALL observed data."""
    features, labels = features.float().cpu(), labels.long().cpu()
    if (features.ndim != 2 or len(features) != len(labels) or
            not torch.isfinite(features).all() or
            labels.unique(sorted=True).tolist() != list(range(num_classes))):
        raise ValueError("Invalid features or incomplete seen-class labels")
    device = torch.device(device)
    training, validation = split_validation(labels, seed)
    cuda_devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        head = nn.Linear(features.shape[1], num_classes).to(device)
        initial = {key: value.detach().clone() for key, value in head.state_dict().items()}

        def train_epochs(indices, limit, select):
            optimizer = torch.optim.AdamW(head.parameters(), lr=settings.lr,
                                          weight_decay=settings.weight_decay)
            generator = torch.Generator().manual_seed(seed + 1)
            best_loss, best_epoch, stale = float("inf"), limit, 0
            curve = []
            for epoch in range(1, limit + 1):
                order = indices[torch.randperm(len(indices), generator=generator)]
                total = 0.
                for batch in order.split(settings.batchsize):
                    x, y = features[batch].to(device), labels[batch].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(head(x), y)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite offline probe objective")
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(batch)
                row = dict(epoch=epoch, train_loss=total / len(indices))
                if select and len(validation):
                    scores = linear_scores(head, features[validation], device, settings.batchsize)
                    row["validation_loss"] = F.cross_entropy(scores, labels[validation]).item()
                    if epoch >= min(20, settings.epochs):
                        if row["validation_loss"] < best_loss - 1e-6:
                            best_loss, best_epoch, stale = row["validation_loss"], epoch, 0
                        else:
                            stale += 1
                curve.append(row)
                if select and stale >= settings.patience:
                    break
            return best_epoch, curve

        budget, selection_curve = train_epochs(training, settings.epochs, True)
        head.load_state_dict(initial)
        _, refit_curve = train_epochs(torch.arange(len(labels)), budget, False)
    return head, dict(selected_epochs=budget, selection_curve=selection_curve,
                      refit_curve=refit_curve, fitting_count=len(labels),
                      selection_train_count=len(training), validation_count=len(validation))


def classification_metrics(scores, labels, num_classes):
    labels = labels.long().cpu()
    prediction = scores.argmax(dim=1).cpu()
    counts = torch.bincount(labels, minlength=num_classes).float()
    correct = torch.bincount(labels[prediction == labels], minlength=num_classes).float()
    return dict(accuracy=float((prediction == labels).float().mean()),
                cls_accuracy=(correct / (counts + 1e-5)).numpy(),
                class_counts=counts.numpy())


def online_scores(model, features, classes, device, batchsize, ensemble):
    scores = []
    with torch.no_grad():
        for batch in features.split(batchsize):
            x = batch.to(device)
            values = [model.backbone.fc(x)]
            if model.use_ema:
                values.extend(head(x) for head in model.ema_fc)
                mask = torch.full_like(values[0], -torch.inf)
                mask[:, :classes] = 0
                values = [value + mask for value in values]
                if "softmax" in ensemble:
                    values = [value.softmax(dim=1) for value in values]
                stack = torch.stack(values, dim=-1)
                if "mean" in ensemble:
                    value = stack.mean(dim=-1)
                elif "max_prob" in ensemble:
                    value = stack.max(dim=-1).values
                elif "min_entropy" in ensemble:
                    entropy = -(stack * (stack + 1e-8).log()).sum(dim=1)
                    choice = entropy.argmin(dim=-1)
                    value = stack[torch.arange(len(x), device=device), :, choice]
                else:
                    raise ValueError(f"Unknown EMA ensemble: {ensemble}")
            else:
                value = values[0]
            scores.append(value[:, :classes].cpu())
    return torch.cat(scores)


def extract_features(model, dataset, indices, device, batchsize, transform=None):
    features, labels = [], []
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batchsize,
                        shuffle=False, num_workers=0)
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for images, target in loader:
            images = images.to(device)
            if transform is not None:
                images = transform(images)
            features.append(model.extract_features(images).float().cpu())
            labels.append(target.long().cpu())
    if not features:
        raise ValueError("Empty probe dataset")
    return torch.cat(features), torch.cat(labels)


def make_cache(model, train_dataset, test_dataset, transform, snapshot, device, batchsize):
    if dataset_signature(train_dataset) != snapshot["dataset_signature"]:
        raise ValueError("Training split/order differs from the saved stream")
    classes = snapshot["class_ids"]
    # Match original evaluation batch shapes to minimize numerical argmax
    # differences in near-ties. Head-fit batching remains independently tunable.
    batchsize = snapshot.get("evaluation_batchsize", batchsize)
    mapping = {int(label): index for index, label in enumerate(classes)}
    train_x, train_y = extract_features(model, train_dataset, snapshot["indices"].tolist(),
                                        device, batchsize, transform)
    test_indices = [index for index, label in enumerate(test_dataset.targets) if int(label) in mapping]
    test_x, test_y = extract_features(model, test_dataset, test_indices, device, batchsize)
    train_y = torch.tensor([mapping[int(label)] for label in train_y])
    test_y = torch.tensor([mapping[int(label)] for label in test_y])
    scores = online_scores(model, test_x, len(classes), device, batchsize, snapshot["ensemble_method"])
    online = classification_metrics(scores, test_y, len(classes))
    expected = snapshot["online_eval"]
    if (not np.isclose(online["accuracy"], expected["avg_acc"], atol=1e-6, rtol=0) or
            not np.allclose(online["cls_accuracy"], expected["cls_acc"][:len(classes)],
                            atol=1e-6, rtol=0)):
        raise ValueError("Recomputed original-head evaluation differs from checkpoint evaluation")
    return dict(train_features=train_x, train_labels=train_y, test_features=test_x,
                test_labels=test_y, online_scores=scores,
                metadata={key: value for key, value in snapshot.items() if key != "state"})


def paired_forgetting(online, probe):
    online, probe = np.asarray(online), np.asarray(probe)
    if online.shape != probe.shape or len(online) < 2:
        raise ValueError("Need matched histories with at least two checkpoints")
    # Exact original F_last eligibility; reuse it for the probe, even if its
    # earlier accuracy is zero. Otherwise the two averages use different classes.
    eligible = online[:-1].max(axis=0) > 0
    online_drop = online[:-1].max(axis=0) - online[-1]
    probe_drop = probe[:-1].max(axis=0) - probe[-1]
    return (eligible, online_drop, probe_drop,
            float(online_drop[eligible].mean()) if eligible.any() else float("nan"),
            float(probe_drop[eligible].mean()) if eligible.any() else float("nan"))


def write_history(directory, rows, num_classes):
    directory = Path(directory)
    online = np.zeros((len(rows), num_classes))
    probe = np.zeros_like(online)
    seen = np.zeros_like(online, dtype=bool)
    for index, row in enumerate(rows):
        count = len(row["class_ids"])
        online[index, :count] = row["online"]["cls_accuracy"]
        probe[index, :count] = row["probe"]["cls_accuracy"]
        seen[index, :count] = True
    if len(rows) > 1:
        eligible, online_drop, probe_drop, original_f, probe_f = paired_forgetting(online, probe)
    else:
        eligible = np.zeros(num_classes, dtype=bool)
        online_drop = probe_drop = np.full(num_classes, np.nan)
        original_f = probe_f = float("nan")
    class_ids = np.full(num_classes, -1, dtype=np.int64)
    class_ids[:len(rows[-1]["class_ids"])] = rows[-1]["class_ids"]
    np.savez_compressed(directory / "history.npz", online_cls_accuracy=online,
                        probe_cls_accuracy=probe, seen_mask=seen, class_ids=class_ids,
                        task_end_samples=[row["samples_cnt"] for row in rows],
                        online_accuracy=[row["online"]["accuracy"] for row in rows],
                        probe_accuracy=[row["probe"]["accuracy"] for row in rows],
                        eligible=eligible, online_drop=online_drop, probe_drop=probe_drop,
                        online_f_last=original_f, probe_f_last=probe_f)
    summaries = [dict(task=row["task_id"], samples=row["samples_cnt"],
                      unique_samples=row["unique_samples"], classes=len(row["class_ids"]),
                      online_accuracy=row["online"]["accuracy"], probe_accuracy=row["probe"]["accuracy"],
                      recoverable_gap=row["probe"]["accuracy"] - row["online"]["accuracy"],
                      selected_epochs=row["fit"]["selected_epochs"]) for row in rows]
    with (directory / "accuracy.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    logger.info("Offline probe summary | online_F_last=%g | probe_F_last=%g | final_gap=%g",
                original_f, probe_f, summaries[-1]["recoverable_gap"])


def fit_caches(caches, directory, num_classes, settings, seed, device):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for cache in caches:
        metadata = cache["metadata"]
        if rows and (metadata["task_id"] <= rows[-1]["task_id"] or
                     metadata["class_ids"][:len(rows[-1]["class_ids"])] != rows[-1]["class_ids"]):
            raise ValueError("Probe checkpoints must preserve task and class-ID order")
        head, fit = fit_linear_probe(cache["train_features"], cache["train_labels"],
                                     len(metadata["class_ids"]), settings, seed, device)
        scores = linear_scores(head, cache["test_features"], device, settings.batchsize)
        probe = classification_metrics(scores, cache["test_labels"], len(metadata["class_ids"]))
        online = classification_metrics(cache["online_scores"], cache["test_labels"], len(metadata["class_ids"]))
        row = dict(task_id=metadata["task_id"], samples_cnt=metadata["samples_cnt"],
                   class_ids=metadata["class_ids"], unique_samples=len(cache["train_labels"]),
                   online=online, probe=probe, fit=fit)
        rows.append(row)
        tag = f"t{metadata['task_id']:02d}"
        torch.save({key: value.detach().cpu() for key, value in head.state_dict().items()}, directory / f"{tag}_head.pt")
        with (directory / f"{tag}_fit.json").open("w", encoding="utf-8") as file:
            json.dump(dict(settings=asdict(settings), seed=seed, **fit), file, indent=2)
        logger.info("Offline probe | task=%d | unique=%d | classes=%d | online=%g | probe=%g | epochs=%d",
                    row["task_id"], row["unique_samples"], len(row["class_ids"]),
                    online["accuracy"], probe["accuracy"], fit["selected_epochs"])
    write_history(directory, rows, num_classes)
    return rows


def run_probes(model, train_dataset, test_dataset, transform, directory, device,
               settings, seed=1, cache_features=False):
    directory = Path(directory)
    snapshots = sorted(directory.glob("t[0-9][0-9].pt"))
    if not snapshots:
        raise ValueError(f"No probe snapshots in {directory}")
    original = mutable_state(model)
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()

    def caches():
        for path in snapshots:
            snapshot = torch.load(path, map_location="cpu", weights_only=False)
            restore_mutable_state(model, snapshot["state"])
            model.eval()
            cache = make_cache(model, train_dataset, test_dataset, transform, snapshot,
                               device, settings.batchsize)
            if cache_features:
                torch.save(cache, directory / f"{path.stem}_features.pt")
            yield cache

    # Feature-loader iteration consumes torch RNG as well as head fitting.
    # Keep post-hoc analysis independent from other seeded diagnostics.
    cuda_devices = list(range(torch.cuda.device_count())) if torch.device(device).type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            return fit_caches(caches(), directory, model.num_classes, settings, seed, device)
    finally:
        restore_mutable_state(model, original)
        for module, was_training in modes:
            module.training = was_training
        random.setstate(python_state)
        np.random.set_state(numpy_state)
