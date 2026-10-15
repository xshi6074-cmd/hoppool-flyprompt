"""Plot paired original-head / refitted-head histories, and summarize seeds."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_history(path, output):
    output.mkdir(parents=True, exist_ok=True)
    with np.load(path) as data:
        steps = data["task_end_samples"]
        original, probe = data["online_accuracy"], data["probe_accuracy"]
        eligible = data["eligible"].astype(bool)
        ids = data["class_ids"][eligible]
        original_drop, probe_drop = data["online_drop"][eligible], data["probe_drop"][eligible]
        original_f, probe_f = float(data["online_f_last"]), float(data["probe_f_last"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    axes[0].plot(steps, original, "o-", label="Original head")
    axes[0].plot(steps, probe, "o-", label="Refitted linear head")
    axes[0].set(xlabel="Stream sample visits", ylabel="Seen-class accuracy", ylim=(0, 1))
    axes[0].legend()
    axes[1].scatter(ids, original_drop, s=14, alpha=.7, label="Original head")
    axes[1].scatter(ids, probe_drop, s=14, alpha=.7, label="Refitted linear head")
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set(xlabel="Dataset class ID", ylabel="Best earlier accuracy - final accuracy",
                title=f"F_last: original {original_f:.4f}; probe {probe_f:.4f}")
    axes[1].legend()
    axes[2].scatter(original_drop, probe_drop, s=18, alpha=.7)
    axes[2].plot([-1, 1], [-1, 1], "--", color="gray")
    axes[2].set(xlabel="Original-head class forgetting", ylabel="Refitted-head class forgetting",
                xlim=(-1, 1), ylim=(-1, 1))
    fig.savefig(output / "paired.png", dpi=180)
    plt.close(fig)
    with (output / "classes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class_id", "original_drop", "probe_drop"])
        writer.writerows(zip(ids, original_drop, probe_drop))
    return dict(online_f_last=original_f, probe_f_last=probe_f,
                online_last_accuracy=float(original[-1]), probe_last_accuracy=float(probe[-1]),
                recoverable_gap=float(probe[-1] - original[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-files", type=int)
    args = parser.parse_args()
    paths = sorted(args.root.rglob("seed_*_lp/history.npz"))
    if not paths and args.root.name.startswith("seed_") and (args.root / "history.npz").exists():
        paths = [args.root / "history.npz"]
    if args.expected_files is not None and len(paths) != args.expected_files:
        parser.error(f"Expected {args.expected_files} probe histories, found {len(paths)}")
    if not paths:
        parser.error("No probe histories found")
    rows = []
    for path in paths:
        relative = path.parent.relative_to(args.root) if path.parent != args.root else Path(path.parent.name)
        summary = plot_history(path, args.output_dir / relative)
        rows.append(dict(dataset=path.parents[2].name, method=path.parents[1].name,
                         seed=int(path.parent.name.split("_")[1]), **summary))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "by_seed.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = sorted({(row["dataset"], row["method"]) for row in rows})
    fig, ax = plt.subplots(figsize=(max(9, len(groups) * 2), 5), layout="constrained")
    for index, group in enumerate(groups):
        group_rows = [row for row in rows if (row["dataset"], row["method"]) == group]
        for offset, key, label, color in ((-.12, "online_f_last", "Original head", "C0"),
                                          (.12, "probe_f_last", "Refitted linear head", "C1")):
            values = [row[key] for row in group_rows]
            ax.scatter(np.full(len(values), index + offset), values, color=color,
                       label=label if index == 0 else None)
            ax.hlines(np.mean(values), index + offset - .08, index + offset + .08, color=color)
    ax.set(xticks=range(len(groups)), xticklabels=[f"{dataset}\n{method}" for dataset, method in groups],
           ylabel="F_last (same eligible classes)")
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend()
    fig.savefig(args.output_dir / "f_last.png", dpi=180)
    plt.close(fig)
    print(f"Rendered {len(rows)} paired probe histories")


if __name__ == "__main__":
    main()
