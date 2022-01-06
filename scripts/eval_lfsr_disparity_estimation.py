#!/usr/bin/env python

"""Evaluate the DMSP algorithm with different prior noise settings.
"""

from __future__ import print_function
import os
import sys
import argparse

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_datasets_bw as datasets
import dppp

sys.path.append(os.path.normpath(os.path.join(__file__, "..", "..")))
from eval_utils import utils, metrics


SCALE_FACTOR = 4
SEED = 100
BICUBIC_RESIZE_FN = dppp.create_resize_fn("bicubic", True)


def _define_lfsr_model(denoiser, denoiser_max, num_steps, gt_disparity):
    def model(example, example_seed):
        # Set the tf seed for models that use tf.random
        tf.random.set_seed(example_seed)

        # Get the high resolution and low resolution image
        hr = example["hr"]
        lr = example["lr"]
        disparity = example["disparity"] if gt_disparity else None

        # Super resolve
        sr = dppp.hqs_lf_super_resolve(
            degraded=lr,
            sr_factor=SCALE_FACTOR,
            denoiser=denoiser,
            max_denoiser_stddev=denoiser_max,
            resize_fn=BICUBIC_RESIZE_FN,
            kernel=None,
            disparity_hr=disparity,
            num_steps=num_steps,
        )[0]

        lf_grid = tf.shape(hr)[:2]
        lf_center = tf.math.floordiv(lf_grid, 2)

        # Compare with the center of the input light field
        return hr[lf_center[0], lf_center[1]], sr

    return model


def _load_dataset():
    # Load the dataset
    images = tfds.load("hci_lf", split="validation")

    def prepare_lf(example):
        hr = datasets.from_255_to_1_range(datasets.to_float32(example["lf"]))
        lr = dppp.resize_lf(hr, SCALE_FACTOR, False, BICUBIC_RESIZE_FN)
        disparity = example["disparity"]
        return {"hr": hr, "lr": lr, "disparity": disparity}

    return images.map(prepare_lf), "HCI"


def main(args):
    # Load the model
    denoiser, (denoiser_min, denoiser_max) = dppp.load_denoiser(args.model.name)

    # Check that the denoiser works on a range of noise levels
    if denoiser_min != 0:
        raise argparse.ArgumentError(
            "Model must be able to handle a range of noise levels starting with 0."
        )

    num_steps = 4 if args.test_run else 30

    # Load the dataset
    dataset, dataset_name = _load_dataset()
    if args.test_run:
        dataset = dataset.take(2)

    # Write the csv header
    args.result.write(
        f"dataset,image_index,gt_disparity,{','.join(metrics.METRICS.keys())}\n"
    )

    # Evaluate once with gt disparity and once without
    for gt_disparity in [True, False]:
        model = _define_lfsr_model(denoiser, denoiser_max, num_steps, gt_disparity)
        utils.evaluate(
            model=model,
            dataset=dataset,
            dataset_name=dataset_name,
            csv_out=args.result,
            parameters=f"{gt_disparity}",
            seed=SEED,
            test_run=args.test_run,
        )


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "model", help="Path to the model file.", type=argparse.FileType("r")
    )
    parser.add_argument(
        "result",
        help="Path to the folder where the results will be written to.",
        type=argparse.FileType("w"),
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Test run: Run each task only for 5 step.",
    )
    return parser.parse_args(arguments)


if __name__ == "__main__":
    sys.exit(main(parse_args(sys.argv[1:])))
