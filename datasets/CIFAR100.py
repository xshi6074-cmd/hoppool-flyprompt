from pathlib import Path

from torchvision.datasets import CIFAR100 as TorchvisionCIFAR100


class CIFAR100(TorchvisionCIFAR100):
    """CIFAR-100 loader that accepts the extracted directory as ``root``.

    Torchvision normally expects ``root/cifar-100-python``. This adapter also
    accepts ``root`` itself being the extracted ``cifar-100-python`` directory.
    When that local directory exists, downloading is disabled so an incomplete
    or corrupted dataset produces an error instead of an unexpected network
    request.
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        requested_root = Path(root).expanduser()

        if requested_root.name == self.base_folder:
            torchvision_root = requested_root.parent
            extracted_dir = requested_root
        else:
            torchvision_root = requested_root
            extracted_dir = requested_root / self.base_folder

        if extracted_dir.is_dir():
            required_files = [
                filename for filename, _ in self.train_list + self.test_list
            ]
            required_files.append(self.meta["filename"])
            missing_files = [
                filename
                for filename in required_files
                if not (extracted_dir / filename).is_file()
            ]
            if missing_files:
                missing = ", ".join(missing_files)
                raise RuntimeError(
                    f"Found local CIFAR-100 directory at {extracted_dir}, "
                    f"but required files are missing: {missing}. "
                    "The loader will not download over an existing directory."
                )
            download = False

        super().__init__(
            root=str(torchvision_root),
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
