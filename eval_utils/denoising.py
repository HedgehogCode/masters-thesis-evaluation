import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_datasets_bw as datasets

from . import utils
from . import metrics

DENOISING_DATASETS = [
    ("BSDS500", "bsds500/dmsp", "validation"),
    ("CBSD68", "cbsd68", "test"),
    ("Kodak24", "kodak24", "test"),
    ("McMaster", "mc_master", "test"),
]
DENOISING_NOISE_LEVELS = [0.01, 0.05, 0.1, 0.2]


EXAMPLES = [
    {
        "dataset_key": "cbsd68",
        "dataset_split": "test",
        "image_key": "image",
        "image_idx": 8,
        "noise_stddev": 0.35,
        "figure_args": {
            "region_center": (150, 110),
            "region_extent": 50,
            "rel_width": 0.5,
            "place_border": 0.01,
            "place_top": False,
            "place_right": True,
        },
    },
    {
        "dataset_key": "cbsd68",
        "dataset_split": "test",
        "image_key": "image",
        "image_idx": 9,
        "noise_stddev": 0.35,
        "figure_args": {
            "region_center": (70, 90),
            "region_extent": 40,
            "rel_width": 0.5,
            "place_border": 0.01,
            "place_top": False,
            "place_right": True,
        },
    },
    {
        "dataset_key": "cbsd68",
        "dataset_split": "test",
        "image_key": "image",
        "image_idx": 11,
        "noise_stddev": 0.35,
        "figure_args": {
            "region_center": (150, 130),
            "region_extent": 50,
            "rel_width": 0.5,
            "place_border": 0.01,
            "place_top": False,
            "place_right": True,
        },
    },
    {
        "dataset_key": "kodak24",
        "dataset_split": "test",
        "image_key": "image",
        "image_idx": 0,
        "noise_stddev": 0.35,
        "figure_args": {
            "region_center": (360, 520),
            "region_extent": 60,
            "rel_width": 0.5,
            "place_border": 0.01,
            "place_top": False,
            "place_right": False,
        },
    },
    {
        "dataset_key": "kodak24",
        "dataset_split": "test",
        "image_key": "image",
        "image_idx": 1,
        "noise_stddev": 0.35,
        "figure_args": {
            "region_center": (270, 550),
            "region_extent": 70,
            "rel_width": 0.5,
            "place_border": 0.01,
            "place_top": False,
            "place_right": False,
        },
    },
    {
        "dataset_key": "mc_master",
        "dataset_split": "test",
        "image_key": "image",
        "image_idx": 4,
        "noise_stddev": 0.35,
        "figure_args": {
            "region_center": (200, 300),
            "region_extent": 50,
            "rel_width": 0.5,
            "place_border": 0.01,
            "place_top": False,
            "place_right": False,
        },
    },
    {
        "dataset_key": "mc_master",
        "dataset_split": "test",
        "image_key": "image",
        "image_idx": 5,
        "noise_stddev": 0.35,
        "figure_args": {
            "region_center": (70, 400),
            "region_extent": 50,
            "rel_width": 0.5,
            "place_border": 0.01,
            "place_top": False,
            "place_right": False,
        },
    },
]
NUM_DENOISING_EXAMPLES = len(EXAMPLES)


def example_denoising(denoiser, result, example_idx=0, save_gt=False, seed=99):
    example = EXAMPLES[example_idx]

    # Get the example image
    dataset = (
        tfds.load(example["dataset_key"], split=example["dataset_split"])
        .map(datasets.get_value(example["image_key"]))
        .map(datasets.to_float32)
        .map(datasets.from_255_to_1_range)
    )
    image = datasets.get_one_example(dataset, index=example["image_idx"])

    # Add noise
    tf.random.set_seed(seed + example_idx)
    noise = tf.random.normal(tf.shape(image), stddev=example["noise_stddev"])
    noisy = image + noise

    # Run the denoiser
    denoised = denoiser(noisy, example["noise_stddev"])

    # Save the example
    utils.save_example(
        gt=image,
        output=denoised,
        result=result,
        save_gt=save_gt,
        figure_width=1.4,
        **example["figure_args"],
    )


def eval_denoising(denoiser, csv_out, seed=99, test_run=False):
    """Evaluate the denosing performace of a denoiser and write the results to a CSV file"""

    # Create random seeds for reproducability:
    # Each combination of dataset and noise has one seed
    rng = np.random.default_rng(seed)
    noise_seeds = utils.seeds_for_values(rng, DENOISING_NOISE_LEVELS)
    dataset_seeds = utils.seeds_for_values(rng, list(zip(*DENOISING_DATASETS))[0])

    # Write CSV file header
    csv_out.write(
        f"dataset,image_index,noise_stddev,{','.join(metrics.METRICS.keys())}\n"
    )

    # Loop over datasets
    for dataset_name, dataset_key, dataset_split in DENOISING_DATASETS:

        # Load the dataset
        dataset = (
            tfds.load(dataset_key, split=dataset_split)
            .map(datasets.get_value("image"))
            .map(datasets.to_float32)
            .map(datasets.from_255_to_1_range)
        )

        # Only take 2 examples if this is a test run
        if test_run:
            dataset = dataset.take(2)

        # Loop over noise levels
        for noise_stddev in DENOISING_NOISE_LEVELS:

            def model(img, example_seed):
                # Create noisy example
                tf.random.set_seed(example_seed)
                noise = tf.random.normal(tf.shape(img), stddev=noise_stddev)
                noisy = img + noise

                # Run the denoiser
                denoised = denoiser(noisy, noise_stddev)

                return img, denoised

            utils.evaluate(
                model=model,
                dataset=dataset,
                dataset_name=dataset_name,
                csv_out=csv_out,
                parameters=str(noise_stddev),
                seed=noise_seeds[noise_stddev] * dataset_seeds[dataset_name],
                test_run=test_run,
            )
