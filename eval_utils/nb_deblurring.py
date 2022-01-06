import os
import numpy as np
import h5py

import tensorflow as tf
from tensorflow.python import util
import tensorflow_datasets as tfds
import tensorflow_datasets_bw as datasets

from . import metrics
from . import utils


with h5py.File(os.path.normpath(os.path.join(__file__, "..", "kernels", "Levin09.mat")), "r") as f:
    NB_DEBLURRING_LEVIN_KERNELS = [
        f[k_ref[0]][()].astype("float32") for k_ref in f["kernels"][()]
    ]

NB_DEBLURRING_DATASETS = [
    {
        "name": "BSDS500",
        "key": "bsds500/dmsp",
        "split": "validation",
        "image_key": "image",
        "kernel_name": "schelten",
        "kernels_key": "schelten_kernels/dmsp",
        "kernels_split": "test",
    },
    *[
        {
            "name": "Set5",
            "key": "set5",
            "split": "test",
            "image_key": "hr",
            "kernel_name": f"levin_{i}",
            "kernel": k,
        }
        for i, k in enumerate(NB_DEBLURRING_LEVIN_KERNELS)
    ],
]
NB_DEBLURRING_NOISE_LEVELS = [0.01, 0.02, 0.03, 0.04]

EXAMPLES = [
    {
        "name": "Set14",
        "key": "set14",
        "split": "test",
        "image_key": "hr",
        "kernel_name": "levin_0",
        "kernel": NB_DEBLURRING_LEVIN_KERNELS[0],
        "image_idx": 0,
        "noise_stddev": 0.04,
        "figure_args": {
            "region_center": (410, 150),
            "region_extent": 30,
            "rel_width": 0.5,
            "place_border": 0.01,
            "place_top": True,
            "place_right": True,
        },
    },
    {
        "name": "Set5",
        "key": "set5",
        "split": "test",
        "image_key": "hr",
        "kernel_name": "levin_1",
        "kernel": NB_DEBLURRING_LEVIN_KERNELS[1],
        "image_idx": 2,
        "noise_stddev": 0.04,
        "figure_args": {
            "region_center": (200, 165),
            "region_extent": 40,
            "rel_width": 0.5,
            "place_border": 0.01,
            "place_top": False,
            "place_right": True,
        },
    },
    {
        "name": "Set5",
        "key": "set5",
        "split": "test",
        "image_key": "hr",
        "kernel_name": "levin_2",
        "kernel": NB_DEBLURRING_LEVIN_KERNELS[2],
        "image_idx": 3,
        "noise_stddev": 0.04,
        "figure_args": {
            "region_center": (190, 210),
            "region_extent": 40,
            "rel_width": 0.5,
            "place_border": 0.01,
            "place_top": False,
            "place_right": False,
        },
    },
    {
        "name": "Set14",
        "key": "set14",
        "split": "test",
        "image_key": "hr",
        "kernel_name": "levin_3",
        "kernel": NB_DEBLURRING_LEVIN_KERNELS[3],
        "image_idx": 10,
        "noise_stddev": 0.04,
        "figure_args": {
            "region_center": (80, 100),
            "region_extent": 50,
            "rel_width": 0.5,
            "place_border": 0.01,
            "place_top": False,
            "place_right": True,
        },
    },
    {
        "name": "BSDS500",
        "key": "bsds500/dmsp",
        "split": "validation",
        "image_key": "image",
        "kernel_name": "levin_4",
        "kernel": NB_DEBLURRING_LEVIN_KERNELS[4],
        "image_idx": 0,
        "noise_stddev": 0.04,
        "figure_args": {
            "region_center": (220, 180),
            "region_extent": 40,
            "rel_width": 0.6,
            "place_border": 0.01,
            "place_top": False,
            "place_right": True,
        },
    },
]
NUM_NB_DEBLURRING_EXAMPLES = len(EXAMPLES)


def _load_deblurring_dataset(dataset_def):
    dataset_name = dataset_def["name"]
    dataset_key = dataset_def["key"]
    dataset_split = dataset_def["split"]
    image_key = dataset_def["image_key"]
    kernel_name = dataset_def["kernel_name"]

    # Load the dataset
    images = (
        tfds.load(dataset_key, split=dataset_split)
        .map(datasets.get_value(image_key))
        .map(datasets.to_float32)
        .map(datasets.from_255_to_1_range)
    )

    # Get the kernels for a tfds
    if "kernels_key" in dataset_def:
        kernels_key = dataset_def["kernels_key"]
        kernels_split = dataset_def["kernels_split"]
        kernels = (
            tfds.load(kernels_key, split=kernels_split)
            .map(datasets.crop_kernel_to_size)
            .map(datasets.get_value("kernel"))
            .map(datasets.to_float32)
        )

        def interleave_with_kernels(img):
            def combine_with_image(ker):
                return (img, ker)

            return kernels.map(combine_with_image)

        return dataset_name, kernel_name, images.interleave(interleave_with_kernels)

    # One fixed kernel is given
    else:
        kernel = dataset_def["kernel"]

        def add_kernel(img):
            return (img, kernel)

        return dataset_name, kernel_name, images.map(add_kernel)


def example_nb_deblurring(deblurrer, result, example_idx=0, save_gt=False, seed=99):
    import dppp

    example = EXAMPLES[example_idx]
    _, __, dataset = _load_deblurring_dataset(example)
    image, kernel = datasets.get_one_example(dataset, index=example["image_idx"])

    # Blur
    tf.random.set_seed(seed + example_idx)
    noise_stddev = example["noise_stddev"]
    kernel_rgb = dppp.conv2D_filter_rgb(kernel)
    blurry = dppp.blur(image[None, ...], kernel_rgb, noise_stddev, "wrap")[0]

    # Run the deblurrer
    deblurred = deblurrer(blurry, kernel, noise_stddev)

    utils.save_example(
        gt=image,
        output=deblurred,
        result=result,
        save_gt=save_gt,
        figure_width=1.1,
        **example["figure_args"],
    )


def eval_nb_deblurring(
    deblurrer,
    csv_out,
    seed=99,
    test_run=False,
    conv_mode="wrap",
    crop_for_eval=False,
):
    """Evaluate the non-blind deblurring performace of a deblurrer method
    and write the results to a CSV file"""
    import dppp

    # Function to crop an image to the center part if crop_for_eval is true
    def do_crop_for_eval(x, kernel):
        if crop_for_eval:
            pad_y = np.floor(kernel.shape[0] / 2.0).astype(np.int64)
            pad_x = np.floor(kernel.shape[1] / 2.0).astype(np.int64)
            return x[pad_y:-pad_y, pad_x:-pad_x, :]
        return x

    # Create random seeds for reproducability:
    # Each combination of dataset and noise has one seed
    rng = np.random.default_rng(seed)
    noise_seeds = utils.seeds_for_values(rng, NB_DEBLURRING_NOISE_LEVELS)
    dataset_seeds = utils.seeds_for_values(
        rng, [d["name"] for d in NB_DEBLURRING_DATASETS]
    )

    # Write CSV file header
    csv_out.write(
        f"dataset,image_index,noise_stddev,kernel,{','.join(metrics.METRICS.keys())}\n"
    )

    # Loop over datasets
    for dataset_def in NB_DEBLURRING_DATASETS:
        dataset_name, kernel_name, dataset = _load_deblurring_dataset(dataset_def)

        # Only take 2 examples if this is a test run
        if test_run:
            dataset = dataset.take(2)

        # Loop over noise levels
        for noise_stddev in NB_DEBLURRING_NOISE_LEVELS:

            def model(data, example_seed):
                img, kernel = data

                # Create blurry example
                tf.random.set_seed(example_seed)
                kernel_rgb = dppp.conv2D_filter_rgb(kernel)
                blurry = dppp.blur(img[None, ...], kernel_rgb, noise_stddev, conv_mode)[
                    0
                ]

                # Run the deblurrer
                deblurred = deblurrer(blurry, kernel, noise_stddev)

                # For the evaluation we might crop the center of the image
                return (
                    do_crop_for_eval(img, kernel),
                    do_crop_for_eval(deblurred, kernel),
                )

            utils.evaluate(
                model=model,
                dataset=dataset,
                dataset_name=dataset_name,
                csv_out=csv_out,
                parameters=f"{noise_stddev},{kernel_name}",
                seed=noise_seeds[noise_stddev] * dataset_seeds[dataset_name],
                test_run=test_run,
            )
