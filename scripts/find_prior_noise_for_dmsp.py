#!/usr/bin/env python

"""Evaluate the DMSP algorithm with different prior noise settings.
"""

from __future__ import print_function
import os
import sys
import argparse

import dppp

sys.path.append(os.path.normpath(os.path.join(__file__, "..", "..")))
import eval_utils as utils

PRIOR_NOISE_STDDEVS = [0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200]


def main(args):
    def open_file(denoiser_stddev):
        file_name = os.path.join(args.result, f"{denoiser_stddev:.3f}.csv")
        return open(file_name, "w")

    # Load the model
    denoiser, (_, _) = dppp.load_denoiser(args.model.name)

    num_steps = 8 if args.test_run else 300
    conv_mode = "wrap"

    for denoiser_stddev in PRIOR_NOISE_STDDEVS:

        # Define the deblurrer using dppp.dmsp_deblur
        def deblurrer(blurry, kernel, noise_stddev):
            return dppp.dmsp_deblur(
                degraded=blurry[None, ...],
                denoiser=denoiser,
                denoiser_stddev=denoiser_stddev,
                kernel=dppp.conv2D_filter_rgb(kernel),
                noise_stddev=noise_stddev,
                num_steps=num_steps,
                conv_mode=conv_mode,
            )[0]

        # Run evaluation of the model
        with open_file(denoiser_stddev) as f:
            utils.eval_nb_deblurring(
                deblurrer=deblurrer,
                csv_out=f,
                test_run=args.test_run,
                conv_mode=conv_mode,
                crop_for_eval=False,
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
        type=str,
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Test run: Run each task only for 5 step.",
    )
    return parser.parse_args(arguments)


if __name__ == "__main__":
    sys.exit(main(parse_args(sys.argv[1:])))
