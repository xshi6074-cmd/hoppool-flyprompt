"""Plot fixed-input prompt probes; optionally pair an explicitly named reference."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def normalize(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


def plot_single(path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = ("same_image_first_cosine", "same_image_previous_cosine",
            "relative_change_norm", "between_image_cosine")
    labels = ("Cosine to first", "Cosine to previous", "Relative L2 change", "Between-image cosine")
    rows = []
    with np.load(path) as probe:
        fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True, layout="constrained")
        for slot, layer in enumerate(probe["layer_indices"]):
            for axis, key, label in zip(axes, keys, labels):
                axis.plot(probe["seen_samples"], probe[key][:, slot], label=f"block {layer}")
                axis.set_ylabel(label)
            for index, seen in enumerate(probe["seen_samples"]):
                rows.append(dict(seen_samples=int(seen), layer=int(layer),
                                 **{key: float(probe[key][index, slot]) for key in keys}))
        axes[0].set_title(f"{str(probe['method'])}; fixed probe images")
        axes[0].legend()
        axes[-1].set_xlabel("Seen training samples")
        fig.savefig(output_dir / "prompt_similarity.png", dpi=180)
        plt.close(fig)
    with (output_dir / "prompt_similarity.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pair(hf_path, shared_path, output_dir):
    with np.load(hf_path) as hf, np.load(shared_path) as shared:
        if str(hf["probe_sha256"]) != str(shared["probe_sha256"]):
            raise ValueError(f"Fixed probe images differ: {hf_path} and {shared_path}")
        if not np.array_equal(hf["probe_indices"], shared["probe_indices"]):
            raise ValueError("Fixed probe indices differ")
        hf_steps = hf["seen_samples"]
        shared_steps = shared["seen_samples"]
        common_steps = np.intersect1d(hf_steps, shared_steps)
        common_layers = np.intersect1d(hf["layer_indices"], shared["layer_indices"])
        if not len(common_steps) or not len(common_layers):
            raise ValueError(f"No aligned samples/layers: {hf_path} and {shared_path}")
        hi = [int(np.flatnonzero(hf_steps == step)[0]) for step in common_steps]
        si = [int(np.flatnonzero(shared_steps == step)[0]) for step in common_steps]
        rows = []
        for layer in common_layers:
            hl = int(np.flatnonzero(hf["layer_indices"] == layer)[0])
            sl = int(np.flatnonzero(shared["layer_indices"] == layer)[0])
            h_vectors = normalize(hf["mean_vectors"][hi, :, hl, :])
            s_vectors = normalize(shared["mean_vectors"][si, :, sl, :])
            cross = (h_vectors * s_vectors).sum(axis=-1).mean(axis=-1)
            for index, step in enumerate(common_steps):
                rows.append({
                    "seen_samples": int(step), "layer": int(layer),
                    "hfpool_between_images_cosine": float(hf["between_image_cosine"][hi[index], hl]),
                    "shared_between_images_cosine": float(shared["between_image_cosine"][si[index], sl]),
                    "same_image_cross_method_cosine": float(cross[index]),
                    "hfpool_vs_first_cosine": float(hf["same_image_first_cosine"][hi[index], hl]),
                    "shared_vs_first_cosine": float(shared["same_image_first_cosine"][si[index], sl]),
                    "hfpool_vs_previous_cosine": float(hf["same_image_previous_cosine"][hi[index], hl]),
                    "shared_vs_previous_cosine": float(shared["same_image_previous_cosine"][si[index], sl]),
                    "hfpool_relative_change": float(hf["relative_change_norm"][hi[index], hl]),
                    "shared_relative_change": float(shared["relative_change_norm"][si[index], sl]),
                })
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "prompt_similarity.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, (temporal, change, within, cross) = plt.subplots(4, 1, figsize=(10, 12), sharex=True, layout="constrained")
    for layer in common_layers:
        selected = [row for row in rows if row["layer"] == layer]
        x = [row["seen_samples"] for row in selected]
        temporal.plot(x, [row["hfpool_vs_first_cosine"] for row in selected],
                      label=f"HFPool layer {layer}")
        temporal.plot(x, [row["shared_vs_first_cosine"] for row in selected],
                      linestyle="--", label=f"shared layer {layer}")
        change.plot(x, [row["hfpool_relative_change"] for row in selected],
                    label=f"HFPool layer {layer}")
        change.plot(x, [row["shared_relative_change"] for row in selected],
                    linestyle="--", label=f"shared layer {layer}")
        within.plot(x, [row["hfpool_between_images_cosine"] for row in selected],
                    label=f"HFPool layer {layer}")
        within.plot(x, [row["shared_between_images_cosine"] for row in selected],
                    linestyle="--", label=f"shared layer {layer}")
        cross.plot(x, [row["same_image_cross_method_cosine"] for row in selected],
                   label=f"layer {layer}")
    temporal.set(ylabel="Cosine to first snapshot", title="Same fixed images; full prompt tokens")
    change.set(ylabel="Relative L2 change", title="Norm of change / norm at first snapshot")
    within.set(ylabel="Cosine across probe images", title="Input specificity at each snapshot")
    cross.set(xlabel="Seen training samples", ylabel="HFPool vs shared cosine",
              title="Same image, mean-token direction (insertion positions differ)")
    temporal.legend(fontsize=8, ncol=2)
    within.legend(fontsize=8, ncol=2)
    cross.legend(fontsize=8)
    fig.savefig(output_dir / "prompt_similarity.png", dpi=180)
    plt.close(fig)


def discover_probes(root, note_prefix=""):
    """Discover saved probes independently of historical experiment note names."""
    records = []
    for path in sorted(Path(root).rglob("seed_*_prompt_probe.npz")):
        if not path.parent.name.startswith(note_prefix):
            continue
        with np.load(path, allow_pickle=False) as data:
            method = str(data["method"])
        records.append((path, method))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--note-prefix", default="", help="Only include experiment notes beginning with this run name")
    parser.add_argument("--reference-note", help="Exact shared_prompt note to pair in each dataset directory")
    parser.add_argument("--expected-pairs", type=int, default=None)
    parser.add_argument("--expected-files", type=int, default=None,
                        help="Expected total HFPool probes, paired or standalone")
    args = parser.parse_args()
    count = 0
    singles = 0
    records = discover_probes(args.results_root, args.note_prefix)
    candidates = [(path, method) for path, method in records if method != "shared_prompt"]
    if not candidates:
        candidates = records
    for hf_path, method in candidates:
        dataset = hf_path.parent.parent.name
        prefix = hf_path.parent.name.removesuffix("_hfpool_wo")
        reference = args.reference_note
        if reference is None and hf_path.parent.name.endswith("_hfpool_wo"):
            reference = f"{prefix}_shared_prompt"
        shared_path = hf_path.parent.parent / reference / hf_path.name if reference else None
        if shared_path is None or not shared_path.exists() or method == "shared_prompt":
            if args.reference_note and (shared_path is None or not shared_path.exists()):
                parser.error(f"Missing explicit reference for {hf_path}")
            plot_single(hf_path, args.output_dir / dataset / prefix / hf_path.stem)
            singles += 1
            continue
        plot_pair(hf_path, shared_path, args.output_dir / dataset / prefix / hf_path.stem)
        count += 1
    if not count and not singles:
        parser.error("No prompt probes found")
    print(f"Wrote {count} paired prompt figures to {args.output_dir}")
    print(f"Wrote {singles} standalone prompt figures")
    if args.expected_pairs is not None and count != args.expected_pairs:
        parser.error(f"Expected {args.expected_pairs} paired probes, found {count}; figures are partial")
    if args.expected_files is not None and count + singles != args.expected_files:
        parser.error(f"Expected {args.expected_files} HFPool probes, found {count + singles}; figures are partial")


if __name__ == "__main__":
    main()
