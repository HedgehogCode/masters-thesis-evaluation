#!/usr/bin/env python

"""Evaluate VSR with different ways to compute the flow.
"""

from __future__ import print_function
import os
import sys
import argparse

import tensorflow as tf
import tensorflow_addons as tfa
import dppp

sys.path.append(os.path.normpath(os.path.join(__file__, "..", "..")))
from eval_utils import utils, metrics, vsr


DATASET_DEF = {"name": "Vid4", "key": "vid4", "split": "test"}
SCALE_FACTOR = 4
SEED = 100
BICUBIC_RESIZE_FN = dppp.create_resize_fn("bicubic", True)
NUM_FRAMES = 9


def ref_flows_sum(images, ref_idx, backwards=False, flow_fn=None):
    """Compute flows between the images and the reference image.

    Only flows between consecutive frames are computed and summed up to retrieve the final flows.
    """
    if flow_fn is None:
        flow_fn = dppp.flow_pyflow

    b, h, w, c = tf.shape(images)

    ref_flows = []

    # Before ref_index
    if backwards:
        # From images to ref frame -> forwards in time
        before_flows = flow_fn(images[:ref_idx], images[1 : ref_idx + 1])
    else:
        # From ref frame to images -> backwards in time
        before_flows = flow_fn(images[1 : ref_idx + 1], images[:ref_idx])

    for i in range(ref_idx):
        ref_flows.append(tf.reduce_sum(before_flows[i:], axis=0))

    # At ref_index
    ref_flows.append(tf.zeros((h, w, 2), dtype=tf.float64))

    # After ref_index
    if backwards:
        # From images to ref frame -> backwards in time
        after_flows = flow_fn(images[ref_idx + 1 :], images[ref_idx:-1])
    else:
        # From ref frame to images -> forwards in time
        after_flows = flow_fn(images[ref_idx:-1], images[ref_idx + 1 :])

    for i in range(ref_idx + 1, b):
        ref_flows.append(tf.reduce_sum(after_flows[: (i - ref_idx)], axis=0))

    return tf.stack(ref_flows, axis=0)


def ref_flows_warp(images, ref_idx, backwards=False, flow_fn=None):
    if flow_fn is None:
        flow_fn = dppp.flow_pyflow

    b, h, w, c = tf.shape(images)

    ref_flows = []

    def combine_flow(flow_0, flow_1):
        # Warp flow_1 to the grid of flow_0
        flow_1_warped_to_0 = tfa.image.dense_image_warp(
            flow_1[None, ...], -flow_0[None, ...]
        )[0]
        return flow_0 + flow_1_warped_to_0

    # Before ref_index
    if backwards:
        # From images to ref frame -> forwards in time
        before_flows = flow_fn(images[:ref_idx], images[1 : ref_idx + 1])

        for i in range(ref_idx):
            flow = before_flows[ref_idx - 1]
            for j in range(ref_idx - 2, i - 1, -1):
                flow = combine_flow(before_flows[j], flow)
            ref_flows.append(flow)

    else:
        # From ref frame to images -> backwards in time
        before_flows = flow_fn(images[1 : ref_idx + 1], images[:ref_idx])

        for i in range(ref_idx):
            flow = before_flows[i]
            for j in range(i + 1, ref_idx):
                flow = combine_flow(before_flows[j], flow)
            ref_flows.append(flow)

    # At ref_index
    ref_flows.append(tf.zeros((h, w, 2), dtype=tf.float64))

    # After ref_index
    if backwards:
        # From images to ref frame -> backwards in time
        after_flows = flow_fn(images[ref_idx + 1 :], images[ref_idx:-1])

        for i in range(b - ref_idx - 1):
            flow = after_flows[0]
            for j in range(1, i + 1):
                flow = combine_flow(after_flows[j], flow)
            ref_flows.append(flow)

    else:
        # From ref frame to images -> forwards in time
        after_flows = flow_fn(images[ref_idx:-1], images[ref_idx + 1 :])

        for i in range(b - ref_idx - 1):
            flow = after_flows[i]
            for j in range(i - 1, -1, -1):
                flow = combine_flow(after_flows[j], flow)
            ref_flows.append(flow)

    return tf.stack(ref_flows, axis=0)


def _define_vsr_model(denoiser, denoiser_max, num_steps, ref_flow_fn):
    def model(example, example_seed):
        # Set the tf seed for models that use tf.random
        tf.random.set_seed(example_seed)

        # Get the high resolution and low resolution image
        hr = example["hr"]
        lr = example["lr"]

        # Crop to NUM_FRAMES
        ref_idx = tf.shape(hr)[0].numpy() // 2
        video_frames = lr.shape[0]
        if video_frames > NUM_FRAMES:
            start_index = max(0, ref_idx - (NUM_FRAMES // 2))
            end_index = start_index + NUM_FRAMES
            lr_cropped = lr[start_index:end_index]
            ref_idx_cropped = ref_idx - start_index
        else:
            lr_cropped = lr
            ref_idx_cropped = ref_idx

        # Compute the flows
        forward_flows = ref_flow_fn(lr_cropped, ref_idx_cropped, backwards=False)
        forward_flows = dppp.resize_flow(
            forward_flows, SCALE_FACTOR, True, resize_fn=BICUBIC_RESIZE_FN
        )
        backward_flows = ref_flow_fn(lr_cropped, ref_idx_cropped, backwards=True)
        backward_flows = dppp.resize_flow(
            backward_flows, SCALE_FACTOR, True, resize_fn=BICUBIC_RESIZE_FN
        )

        sr = dppp.hqs_video_super_resolve(
            degraded=lr_cropped,
            ref_index=ref_idx_cropped,
            sr_factor=SCALE_FACTOR,
            denoiser=denoiser,
            max_denoiser_stddev=denoiser_max,
            resize_fn=BICUBIC_RESIZE_FN,
            num_steps=num_steps,
            forward_flows=forward_flows,
            backward_flows=backward_flows,
        )[0]

        # Compare with the center of the input light field
        return hr[ref_idx], sr

    return model


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
    dataset_name, _, dataset = vsr._load_vsr_dataset(DATASET_DEF, SCALE_FACTOR)
    if args.test_run:
        dataset = dataset.take(2)

    # Write the csv header
    args.result.write(
        f"dataset,image_index,flow_fn,{','.join(metrics.METRICS.keys())}\n"
    )

    flow_fns = {
        "default": dppp.reference_flows,
        "sum": ref_flows_sum,
        "warp": ref_flows_warp,
    }

    # Evaluate for all options of flow functions
    for flow_fn_name, ref_flow_fn in flow_fns.items():
        model = _define_vsr_model(denoiser, denoiser_max, num_steps, ref_flow_fn)
        utils.evaluate(
            model=model,
            dataset=dataset,
            dataset_name=dataset_name,
            csv_out=args.result,
            parameters=f"{flow_fn_name}",
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
