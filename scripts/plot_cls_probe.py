"""Render CLS diagnostics, retaining real sample times when seed schedules differ."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_records(root):
    records = []
    for path in sorted(Path(root).rglob("*_cls/history.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            record = {key: data[key].copy() for key in ("indices", "labels", "samples", "clean")}
        args = metadata["arguments"]
        if args["method"] == "shared_prompt":
            method = "shared_prompt"
        elif args["method"] == "spatial_gelu" or args.get("hopfield_pooling_mode") == "spatial_gelu":
            method = "SpatialGELU"
        elif args["method"] == "hfpool_lora":
            method = "HFPoolLoRA"
        elif args["method"] in {"baseline", "gate"}:
            method = args["method"]
        elif args.get("hopfield_prompt_blocks"):
            method = "HFPool_insert_" + "_".join(str(x + 1) for x in args["hopfield_prompt_blocks"])
        else:
            method = "HFPool"
        if args.get("hopfield_prompt_blocks") and not method.startswith("HFPool_insert_"):
            method += "_insert_" + "_".join(str(x + 1) for x in args["hopfield_prompt_blocks"])
        output_mode = args.get("hopfield_o_mode", "full")
        if output_mode != "full":
            method += "_" + output_mode
        with (path.parent / "metrics.csv").open(encoding="utf-8", newline="") as handle:
            metrics = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
        record.update(metadata=metadata, metrics=metrics, method=method,
                      dataset=args["dataset"], backbone=args["backbone"], path=str(path))
        records.append(record)
    return records


def check_pairs(records, require_pairs=False):
    groups = defaultdict(dict)
    for record in records:
        key = (record["dataset"], record["backbone"], record["metadata"]["seed"])
        if record["method"] in groups[key]:
            raise ValueError(f"Duplicate CLS comparison {key}/{record['method']}")
        groups[key][record["method"]] = record
    checked = 0
    for key, pair in groups.items():
        if "shared_prompt" not in pair or len(pair) == 1:
            if require_pairs:
                raise ValueError(f"Missing shared_prompt/reference partner: {key}")
            continue
        second = pair["shared_prompt"]
        for name, first in pair.items():
            if name == "shared_prompt":
                continue
            if first["metadata"]["dataset_signature"] != second["metadata"]["dataset_signature"]:
                raise ValueError(f"Different dataset split: {key}")
            for field in ("indices", "labels", "samples"):
                if not np.array_equal(first[field], second[field]):
                    raise ValueError(f"Mismatched cohort/checkpoint {field}: {key}")
            if not np.allclose(first["clean"], second["clean"], atol=1e-5, rtol=1e-5):
                raise ValueError(f"Different frozen backbone features: {key}")
            checked += 1
    return checked


def metric_series(items, metric, x_axis):
    """Aggregate only explicitly shared axes; otherwise retain each seed's visits."""
    if x_axis not in {"samples", "checkpoint"}:
        raise ValueError("x_axis must be samples or checkpoint")
    values = [np.asarray([row[metric] for row in item["metrics"]]) for item in items]
    if any(len(value) != len(item["samples"]) for value, item in zip(values, items)):
        raise ValueError("CLS metric count differs from its sample timeline")
    same_samples = all(np.array_equal(item["samples"], items[0]["samples"]) for item in items)
    if x_axis == "checkpoint" and len({len(value) for value in values}) != 1:
        raise ValueError("Checkpoint aggregation requires the same checkpoint count")
    if same_samples or x_axis == "checkpoint":
        array = np.asarray(values)
        mean = array.mean(axis=0)
        std = array.std(axis=0, ddof=1) if len(array) > 1 else np.zeros_like(mean)
        steps = np.arange(len(mean)) if x_axis == "checkpoint" else items[0]["samples"]
        return [(steps, mean, std, f"n={len(items)}")]
    return [(item["samples"], value, None, f"seed={item['metadata']['seed']}")
            for item, value in zip(items, values)]


def render(records, output, x_axis="samples"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        for metrics in record["metrics"]:
            rows.append(dict(dataset=record["dataset"], backbone=record["backbone"],
                             method=record["method"], seed=record["metadata"]["seed"], **metrics))
    with (output / "cls_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for record in records:
        groups[(record["dataset"], record["backbone"])].append(record)
    panels = [("same_image_cos_first", "Same image cosine to first checkpoint"),
              ("same_class_cos", "Within-class pair cosine (no diagonal)"),
              ("different_class_cos", "Between-class pair cosine"),
              ("cos_clean", "Cosine to frozen clean CLS"),
              ("temporal_variance", "Raw variance over checkpoints"),
              ("sample_variance", "Raw variance over images"),
              ("within_class_variance", "Raw within-class image variance"),
              ("between_class_variance", "Raw class-center variance")]
    for (dataset, backbone), group in groups.items():
        methods = defaultdict(list)
        for record in group:
            methods[record["method"]].append(record)
        for unit in (False, True):
            fig, axes = plt.subplots(2, 4, figsize=(19, 8), constrained_layout=True)
            for ax, (metric, title) in zip(axes.flat, panels):
                if unit and "variance" in metric:
                    metric = "unit_" + metric
                    title = title.replace("Raw", "L2-normalized")
                for method, items in sorted(methods.items()):
                    color = None
                    for steps, mean, std, label in metric_series(items, metric, x_axis):
                        line, = ax.plot(steps, mean, marker="o", color=color,
                                        label=f"{method} ({label})")
                        color = line.get_color()
                        if std is not None:
                            ax.fill_between(steps, mean-std, mean+std, color=color, alpha=.13)
                ax.set(title=title, xlabel=("Checkpoint index (seed times may differ)"
                                           if x_axis == "checkpoint" else "Stream sample visits"))
                if "variance" in metric:
                    ax.set_ylim(bottom=0)
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
                ax.grid(alpha=.2)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f"{dataset} / {backbone} | fixed first-checkpoint observed cohort | "
                         "matched axes: mean +/- seed SD; differing sample axes: individual seeds")
            label = backbone.replace("vit_base_patch16_224_mepo_", "")
            fig.savefig(output / f"{dataset}_{label}_{'unit' if unit else 'raw'}.png", dpi=170)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-files", type=int, default=0)
    parser.add_argument("--x-axis", choices=["samples", "checkpoint"], default="samples")
    parser.add_argument("--require-pairs", action="store_true",
                        help="Require and verify a shared_prompt reference for every dataset/backbone/seed.")
    args = parser.parse_args()
    records = load_records(args.root)
    if not records:
        raise ValueError("No completed CLS histories")
    if args.expected_files and len(records) != args.expected_files:
        raise ValueError(f"Expected {args.expected_files} CLS histories, found {len(records)}")
    pairs = check_pairs(records, require_pairs=args.require_pairs)
    render(records, args.output_dir or args.root / "plots" / "cls", args.x_axis)
    print(f"CLS histories={len(records)}, verified reference/shared_prompt pairs={pairs}")


if __name__ == "__main__":
    main()
