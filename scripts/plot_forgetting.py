"""Plot the exact per-class drops contributing to task-end F_last.

Usage: python scripts/plot_forgetting.py RESULTS_ROOT --output-dir OUTPUT
RESULTS_ROOT contains logs/<dataset>/<note>/seed_*_class_history.npz.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def class_rows(path):
    with np.load(path) as data:
        history = data["cls_accuracy"]
        class_ids = data["class_ids"]
        seen = data["seen_mask"]
        reported = float(data["f_last"])
    if history.ndim != 2 or history.shape != seen.shape or history.shape[1] != len(class_ids):
        raise ValueError(f"Invalid class history shape: {path}")
    if history.shape[0] < 2:
        return []
    prior_best = history[:-1].max(axis=0)
    eligible = prior_best > 0  # Identical to the trainer's F_last definition.
    drop = prior_best - history[-1]
    measured = float(drop[eligible].mean()) if eligible.any() else float("nan")
    if not np.isclose(measured, reported, atol=1e-6, equal_nan=True):
        raise ValueError(f"F_last mismatch in {path}: {measured} versus {reported}")
    seed = int(path.name.split("_")[1])
    rows = []
    for col in np.flatnonzero(eligible):
        rows.append({
            "seed": seed,
            "class_id": int(class_ids[col]),
            "first_seen_task": int(np.flatnonzero(seen[:, col])[0]) + 1,
            "best_prior_accuracy": float(prior_best[col]),
            "final_accuracy": float(history[-1, col]),
            "accuracy_drop": float(drop[col]),
        })
    return rows


def plot_group(paths, output_dir, dataset, note):
    rows = [row for path in paths for row in class_rows(path)]
    if not rows:
        return
    dest = output_dir / dataset / note
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "per_class_forgetting.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    seeds = sorted({row["seed"] for row in rows})
    fig, (ax, histogram) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1], layout="constrained")
    for index, seed in enumerate(seeds):
        points = [row for row in rows if row["seed"] == seed]
        ax.scatter([row["class_id"] + (index - (len(seeds)-1)/2) * .18 for row in points],
                   [row["accuracy_drop"] for row in points], s=14, alpha=.6, label=f"seed {seed}")
        histogram.hist([row["accuracy_drop"] for row in points], bins=np.linspace(-1, 1, 41),
                       alpha=.45, label=f"seed {seed}")
    ax.axhline(0, color="black", lw=.8)
    histogram.axvline(0, color="black", lw=.8)
    ax.set(title=f"{dataset} / {note}: class forgetting", xlabel="Original dataset class ID",
           ylabel="Best earlier task-end accuracy - final accuracy")
    histogram.set(xlabel="Per-class accuracy drop", ylabel="Count")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(dest / "per_class_forgetting.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--note-prefix", default="", help="Only include experiment notes beginning with this run name")
    parser.add_argument("--expected-files", type=int, default=None)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in args.results_root.rglob("seed_*_class_history.npz"):
        if not path.parent.name.startswith(args.note_prefix):
            continue
        groups[(path.parent.parent.name, path.parent.name)].append(path)
    if not groups:
        parser.error("No class-history files found; older runs saved only F_last and cannot reconstruct class drops")
    for (dataset, note), paths in sorted(groups.items()):
        plot_group(paths, args.output_dir, dataset, note)
    file_count = sum(len(paths) for paths in groups.values())
    print(f"Wrote {len(groups)} class-forgetting figures from {file_count} histories to {args.output_dir}")
    if args.expected_files is not None and file_count != args.expected_files:
        parser.error(f"Expected {args.expected_files} class histories, found {file_count}; figures are partial")


if __name__ == "__main__":
    main()
