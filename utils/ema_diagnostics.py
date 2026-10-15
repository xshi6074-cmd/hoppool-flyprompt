"""Head comparisons on exactly the same current prompted CLS and seen mask."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def probability_loss(scores, labels):
    probabilities = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return F.nll_loss(probabilities.clamp_min(1e-30).log(), labels)


def diagnostic_scores(logits, ratios):
    scores = {"online": (logits[0], False)}
    scores.update({f"ema_{ratio:g}": (value, False) for ratio, value in zip(ratios, logits[1:])})
    probabilities = torch.stack([value.float().softmax(dim=-1) for value in logits], dim=-1)
    scores["softmax_max_prob"] = (probabilities.max(dim=-1).values, True)
    scores["softmax_mean"] = (probabilities.mean(dim=-1), True)
    return scores


def save_head_metrics(path, task_id, class_ids, accumulators):
    record = {"task_id": int(task_id), "class_ids": list(class_ids), "heads": {}}
    for name, item in accumulators.items():
        record["heads"][name] = {"accuracy": item["correct"] / item["count"],
            "loss": item["loss"] / item["count"],
            "cls_accuracy": (item["cls_correct"] / (item["cls_count"] + 1e-5)).tolist()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_heads(path):
    path = Path(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    classes = records[-1]["class_ids"]
    histories = {}
    for name in records[0]["heads"]:
        history = np.zeros((len(records), len(classes)))
        for index, record in enumerate(records):
            for slot, cls in enumerate(record["class_ids"]):
                history[index, classes.index(cls)] = record["heads"][name]["cls_accuracy"][slot]
        histories[name] = history
    online_eligible = histories["online"][:-1].max(axis=0) > 0 if len(records) > 1 else np.zeros(len(classes), bool)
    rows = []
    for name, history in histories.items():
        eligible = history[:-1].max(axis=0) > 0 if len(records) > 1 else online_eligible
        drop = history[:-1].max(axis=0) - history[-1] if len(records) > 1 else np.zeros(len(classes))
        rows.append(dict(head=name, A_last=records[-1]["heads"][name]["accuracy"],
            A_avg=np.mean([record["heads"][name]["accuracy"] for record in records]),
            F_last=float(drop[eligible].mean()) if eligible.any() else float("nan"),
            F_online_eligible=float(drop[online_eligible].mean()) if online_eligible.any() else float("nan")))
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
