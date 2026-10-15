"""Plot logged F_last by seed for a run whose class histories are unavailable."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PATTERN = re.compile(r"A_auc ([0-9.eE+-]+) \| A_avg [0-9.eE+-]+ \| A_last [0-9.eE+-]+ \| F_last ([0-9.eE+-]+)")
JOB = re.compile(r"job_\d+_(cifar|cub)_(ibot21k|sup21k1k)_([A-Za-z0-9_-]+)_seed(\d+)\.log")


def expand_method(name):
    """Decode short filesystem labels; tables keep the complete recipe name."""
    match = re.match(r"^replay_(dcos|lrmse|dmse|lrcos8)_(gn|lgn)_(.+)$", name)
    if not match:
        return name
    recipe, weight, method = match.groups()
    recipes = {"dcos": "sample_diag_raw_cosine", "lrmse": "closed_form_low_rank_r16_mse",
               "dmse": "sample_diag_raw_mse", "lrcos8": "sample_low_rank_r16_k8_cosine"}
    weights = {"gn": "gradnorm", "lgn": "learned_gradnorm"}
    return f"replay_{recipes[recipe]}_{weights[weight]}_{method}"


def method_label(name):
    name = expand_method(name)
    if name.startswith("single_"):
        return name.removesuffix("_hfpool_wo").replace("single_", "single pool\n")
    return (name.replace("replay_", "").replace("_learned_gradnorm_", "\nlearned rho\n")
            .replace("_gradnorm_", "\nfixed rho\n").replace("hfpool_wo", "HFPool")
            .replace("shared_prompt", "shared").replace("hfpool_", ""))


def plot_methods(rows, methods, output):
    datasets = sorted({row["dataset"] for row in rows})
    backbones = sorted({row["backbone"] for row in rows})
    width = max(14, len(methods) * 1.15 * len(backbones))
    fig, axes = plt.subplots(len(datasets), len(backbones), figsize=(width, 9),
                             sharey="row", layout="constrained", squeeze=False)
    for i, dataset in enumerate(datasets):
        for j, backbone in enumerate(backbones):
            ax = axes[i, j]
            for x, method in enumerate(methods):
                vals = [row["f_last"] for row in rows if row["dataset"] == dataset and row["backbone"] == backbone and row["method"] == method]
                ax.scatter(np.full(len(vals), x) + np.linspace(-.08, .08, len(vals)), vals, s=28)
                if vals:
                    ax.hlines(np.mean(vals), x-.18, x+.18, color="black", lw=2)
            ax.set(xlim=(-.5, len(methods)-.5), xticks=range(len(methods)),
                   xticklabels=[method_label(name) for name in methods], title=f"{dataset} / {backbone}")
            ax.tick_params(axis="x", labelrotation=25)
            if j == 0:
                ax.set_ylabel("F_last (class accuracy drop)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-jobs", type=int, default=None)
    args = parser.parse_args()
    rows = []
    for path in sorted((args.run_dir / "logs").glob("job_*.log")):
        job = JOB.fullmatch(path.name)
        if not job:
            continue
        match = PATTERN.findall(path.read_text(encoding="utf-8", errors="replace"))
        if len(match) != 1:
            raise ValueError(f"Expected one summary in {path}, found {len(match)}")
        dataset, backbone, method, seed = job.groups()
        rows.append(dict(dataset=dataset, backbone=backbone, method=expand_method(method),
                         seed=int(seed), a_auc=float(match[0][0]), f_last=float(match[0][1])))
    if not rows:
        parser.error("No matching job logs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "f_last_by_seed.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    base = [method for method in (
        "hfpool_wo", "hfpool_lora", "hfpool_lora_merge_1024",
        "hfpool_lora_merge_batch", "shared_prompt"
    ) if any(row["method"] == method for row in rows)]
    known = set(base)
    extra = sorted({row["method"] for row in rows} - known)
    if base:
        plot_methods(rows, base, args.output_dir / "f_last_by_seed.png")
    for prefix, filename in (("single_", "f_last_single_pool.png"),
                             ("replay_", "f_last_classifier_replay.png"),
                             ("cls_", "f_last_cls_replacement.png"),
                             ("probe_", "f_last_probe_online_head.png")):
        methods = [method for method in extra if method.startswith(prefix)]
        if methods:
            plot_methods(rows, base + methods, args.output_dir / filename)
            known.update(methods)
    remaining = sorted({row["method"] for row in rows} - known)
    if remaining:
        plot_methods(rows, base + remaining, args.output_dir / "f_last_other.png")
    print(f"Wrote {len(rows)} job summaries to {args.output_dir}")
    if args.expected_jobs is not None and len(rows) != args.expected_jobs:
        parser.error(f"Expected {args.expected_jobs} job summaries, found {len(rows)}; figure is partial")


if __name__ == "__main__":
    main()
