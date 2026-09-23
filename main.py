import argparse
import os
import sys


def _configure_visible_gpu(argv=None):
    """Restrict CUDA visibility before importing any torch-dependent modules."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    if args.gpu is not None:
        if args.gpu < 0:
            parser.error("--gpu must be a non-negative integer")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    return args.gpu


_REQUESTED_GPU = _configure_visible_gpu()

# CUDA_VISIBLE_DEVICES must be configured before these imports: importing the
# method registry also imports torch-dependent training modules.
import logging.config

import torch

from configuration import config
from methods import METHODS

logging.config.fileConfig('configuration/logging.conf')
logger = logging.getLogger()


def main():
    # Get Configurations
    args = config.base_parser()
    if args.gpu is not None:
        visible_gpu_count = torch.cuda.device_count()
        if visible_gpu_count != 1:
            raise RuntimeError(
                f"--gpu {args.gpu} was requested, but the process sees "
                f"{visible_gpu_count} CUDA devices. Check the GPU ID and CUDA driver."
            )
        logger.info(
            "Using requested physical GPU %s as logical cuda:0 (%s)",
            args.gpu,
            torch.cuda.get_device_name(0),
        )

    logger.info('Running for seeds: %s', args.seeds)
    for seed in args.seeds:
        setattr(args, 'rnd_seed', seed)
        logger.info('Configuration: %s', args)

        trainer = METHODS[args.method](**vars(args))
        trainer.run()

if __name__ == "__main__":
    main()
