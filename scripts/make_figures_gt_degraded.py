#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse

import tensorflow as tf
from tensorflow.python.data import util

sys.path.append(os.path.normpath(os.path.join(__file__, "..", "..")))
import eval_utils as utils


def main(args):

    # Denoising

    def denoiser(noisy, _):
        return noisy

    if args.denoising is not None:
        for i in range(utils.NUM_DENOISING_EXAMPLES):
            # Ground truth
            with open(os.path.join(args.denoising, f"{i:02d}_gt.png"), "w") as f:
                utils.example_denoising(denoiser, f, example_idx=i, save_gt=True)
            # Noisy
            with open(os.path.join(args.denoising, f"{i:02d}_noisy.png"), "w") as f:
                utils.example_denoising(denoiser, f, example_idx=i)

    # NB Deblurring

    def deblurrer(blurry, _, __):
        return blurry

    if args.nb_deblurring is not None:
        for i in range(utils.NUM_NB_DEBLURRING_EXAMPLES):
            # Ground truth
            with open(os.path.join(args.nb_deblurring, f"{i:02d}_gt.png"), "w") as f:
                utils.example_nb_deblurring(deblurrer, f, example_idx=i, save_gt=True)
            # Blurry
            with open(
                os.path.join(args.nb_deblurring, f"{i:02d}_blurry.png"), "w"
            ) as f:
                utils.example_nb_deblurring(deblurrer, f, example_idx=i)

    # SISR

    def superresolver(lr, _, __):
        return None

    if args.sisr is not None:
        for i in range(utils.NUM_SISR_EXAMPLES):
            # Ground truth
            with open(os.path.join(args.sisr, f"{i:02d}_gt.png"), "w") as f:
                utils.example_sisr(superresolver, f, example_idx=i, save_gt=True)

    # VSR

    def video_superresolver(lr, _, __, ___):
        return None

    if args.vsr is not None:
        for i in range(utils.NUM_VSR_EXAMPLES):
            # Ground truth
            with open(os.path.join(args.vsr, f"{i:02d}_gt.png"), "w") as f:
                utils.example_vsr(video_superresolver, f, example_idx=i, save_gt=True)

    # LFSR

    def lf_superresolver(lr, _, __):
        return None

    if args.lfsr is not None:
        for i in range(utils.NUM_LFSR_EXAMPLES):
            # Ground truth
            with open(os.path.join(args.lfsr, f"{i:02d}_gt.png"), "w") as f:
                utils.example_lfsr(lf_superresolver, f, example_idx=i, save_gt=True)

    # Inpainting

    def inpainter(image, mask):
        return image

    def inpainter_biharmonic(image, mask):
        import skimage.restoration

        return skimage.restoration.inpaint_biharmonic(
            image.numpy(),
            tf.cast(mask[..., 0] == False, tf.uint8).numpy(),
            multichannel=True,
        )

    def inpainter_border(image, mask):
        import dppp

        return dppp.inpaint_border(image[None, ...], mask[None, ...])[0]

    if args.inpainting is not None:
        for i in range(utils.NUM_INPAINTING_EXAMPLES):
            # Masked
            with open(os.path.join(args.inpainting, f"{i:02d}_masked.png"), "w") as f:
                utils.example_inpaint(inpainter, f, example_idx=i)

            # Skimage biharmonic
            with open(
                os.path.join(args.inpainting, f"{i:02d}_biharmonic.png"), "w"
            ) as f:
                utils.example_inpaint(inpainter_biharmonic, f, example_idx=i)

            # Border
            with open(os.path.join(args.inpainting, f"{i:02d}_border.png"), "w") as f:
                utils.example_inpaint(inpainter_border, f, example_idx=i)


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--denoising",
        help="Path to the denoising figures.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--nb-deblurring",
        help="Path to the nb-deblurring figures.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--sisr",
        help="Path to the sisr figures.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--vsr",
        help="Path to the vsr figures.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--lfsr",
        help="Path to the lfsr figures.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--inpainting",
        help="Path to the inpainting figures.",
        default=None,
        type=str,
    )
    return parser.parse_args(arguments)


if __name__ == "__main__":
    sys.exit(main(parse_args(sys.argv[1:])))
