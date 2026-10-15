"""Rerun frozen-ViT method snapshots or fit persisted CLS caches without a GPU.

python scripts/offline_linear_probe.py LOG_DIR/seed_1_lp --device cuda:0
python scripts/offline_linear_probe.py LOG_DIR/seed_1_lp --device cpu --from-cache
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from utils.offline_probe import ProbeSettings, fit_caches, run_probes
from scripts.plot_offline_probe import plot_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="Default: DIRECTORY/refit")
    parser.add_argument("--data-dir", help="Override dataset root when moving hosts")
    parser.add_argument("--backbone-path", help="Override the pinned frozen-backbone file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.0005)
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1, help="CPU thread budget for cached-head fitting")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = ProbeSettings(args.epochs, args.patience, args.lr, args.weight_decay, args.batchsize)
    device = torch.device(args.device)
    paths = sorted(args.directory.glob("t[0-9][0-9].pt"))
    if not paths:
        parser.error("No evaluation snapshots found")
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    config = first["arguments"]
    seed = int(config["rnd_seed"])
    output = args.output_dir or args.directory / "refit"
    output.mkdir(parents=True, exist_ok=True)
    if args.from_cache:
        cache_paths = [path.with_name(f"{path.stem}_features.pt") for path in paths]
        if any(not path.is_file() for path in cache_paths):
            parser.error("Incomplete feature caches; enable --offline_probe_cache_features during training")
        caches = (torch.load(path, map_location="cpu", weights_only=False) for path in cache_paths)
        num_classes = len(torch.load(paths[-1], map_location="cpu", weights_only=False)["online_eval"]["cls_acc"])
        fit_caches(caches, output, num_classes, settings, seed, device)
    else:
        # Lazy imports keep CPU cached-head fitting independent of timm/hflayers.
        from datasets import DATASETS
        from utils.train_utils import select_model
        from torchvision import transforms
        from utils.data_loader import get_statistics
        if args.data_dir:
            config["data_dir"] = args.data_dir
        if args.backbone_path:
            config["backbone_path"] = args.backbone_path
        mean, std, num_classes, _, _ = get_statistics(config["dataset"])
        resize = transforms.Resize((224, 224))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean, std)
        load = ([resize, to_tensor] if any(name in config["dataset"] for name in ("imagenet", "cub", "car"))
                else [to_tensor])
        dataset_cls = DATASETS[config["dataset"]]
        train = dataset_cls(config["data_dir"], train=True, download=False,
                            transform=transforms.Compose(load))
        test = dataset_cls(config["data_dir"], train=False, download=False,
                           transform=transforms.Compose([resize, to_tensor, normalize]))
        model = select_model(config["method"], config["backbone"], num_classes,
                             config["n_tasks"], config).to(device)
        # A custom output directory contains copies of just the lightweight
        # snapshots; they retain source indices and pinned backbone metadata.
        for path in paths:
            destination = output / path.name
            if destination.resolve() != path.resolve():
                import shutil
                shutil.copy2(path, destination)
        run_probes(model, train, test, transforms.Compose([resize, normalize]), output,
                   device, settings, seed=seed)
    plot_history(output / "history.npz", output / "plots")


if __name__ == "__main__":
    main()
