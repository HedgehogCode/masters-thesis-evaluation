#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse

import numpy as np
import scipy.io
import dppp

sys.path.append(os.path.normpath(os.path.join(__file__, "..", "..")))
from eval_utils import sisr, metrics, utils

BICUBIC_KERNELS = np.stack(
    scipy.io.loadmat(
        os.path.normpath(
            os.path.join(utils.__file__, "..", "kernels", "kernels_bicubicx234.mat")
        )
    )["kernels"][0],
    axis=0,
)

DATASET_DEF = {"name": "Set5", "key": "set5", "split": "test"}
SCALE_FACTOR = 4
SEED = 100


def _hqs_superresolver(test_run, model, approximate_kernel):
    from methods import hqs

    sr, _ = hqs.sisr(test_run, model.name)

    def superresolver(lr, scale_factor, kernel):
        assert kernel is None, "Bicubic interpolation expected"
        if approximate_kernel:
            kernel = BICUBIC_KERNELS[scale_factor - 2]
        return sr(lr, scale_factor, kernel)

    return superresolver


def main(args):
    # Load the dataset
    dataset_name, _, dataset = sisr._load_sisr_dataset(DATASET_DEF, SCALE_FACTOR)
    if args.test_run:
        dataset = dataset.take(2)

    # Write the csv header
    args.result.write(
        f"dataset,image_index,approximate_kernel,{','.join(metrics.METRICS.keys())}\n"
    )

    # Evaluate once with gt disparity and once without
    for use_approximate_kernel in [True, False]:
        superresolver = _hqs_superresolver(
            args.test_run, args.model, use_approximate_kernel
        )
        model = sisr._define_sisr_model(superresolver, SCALE_FACTOR)
        utils.evaluate(
            model=model,
            dataset=dataset,
            dataset_name=dataset_name,
            csv_out=args.result,
            parameters=f"{use_approximate_kernel}",
            seed=SEED,
            test_run=args.test_run,
        )


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "model",
        help="Denoising model.",
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "result",
        help="Path to the file where results will be written to.",
        type=argparse.FileType("w"),
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Test run: Run each task only for a few steps.",
    )
    return parser.parse_args(arguments)


if __name__ == "__main__":
    sys.exit(main(parse_args(sys.argv[1:])))
