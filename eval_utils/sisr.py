import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_datasets_bw as datasets

from . import metrics
from . import utils

SISR_DATASETS = [
    {
        "name": "Set5",
        "key": "set5",
        "split": "test",
    },
    {
        "name": "Set14",
        "key": "set14",
        "split": "test",
    },
    {
        "name": "CBSD68",
        "key": "cbsd68",
        "split": "test",
        "hr_image_key": "image",
        "downscale": "bicubic",
    },
    *[
        {
            "name": "Set5",
            "key": "set5",
            "split": "test",
            "hr_image_key": "hr",
            "downscale": f"kernel_{i}",
            "kernel": utils.GAUSSIAN_KERNELS[i],
        }
        for i in range(len(utils.GAUSSIAN_KERNELS))
    ],
]
SISR_SCALE_FACTORS = [2, 3, 4, 5]

EXAMPLES = [
    {  # Set14: comic
        "dataset_def": SISR_DATASETS[1],
        "image_idx": 2,
        "scale_factor": 4,
        "figure_args": {
            "region_center": (120, 120),
            "region_extent": 30,
            "rel_width": 0.7,
            "place_border": 0.01,
            "place_top": False,
            "place_right": False,
        },
    },
    {  # CBSD68: Man with stick
        "dataset_def": {
            "name": "CBSD68",
            "key": "cbsd68",
            "split": "test",
            "hr_image_key": "image",
            "downscale": "kernel_6",
            "kernel": utils.GAUSSIAN_KERNELS[6],
        },
        "image_idx": 31,
        "scale_factor": 2,
        "figure_args": {
            "region_center": (60, 280),
            "region_extent": 20,
            "rel_width": 0.6,
            "place_border": 0.01,
            "place_top": False,
            "place_right": True,
        },
    },
    {  # Set 5 - bicubic
        "dataset_def": SISR_DATASETS[0],
        "image_idx": 0,
        "scale_factor": 4,
        "figure_args": {
            "region_center": (90, 140),
            "region_extent": 30,
            "rel_width": 0.6,
            "place_border": 0.01,
            "place_top": False,
            "place_right": False,
        },
    },
    {
        "dataset_def": SISR_DATASETS[0],
        "image_idx": 1,
        "scale_factor": 4,
        "figure_args": {
            "region_center": (50, 50),
            "region_extent": 40,
            "rel_width": 0.6,
            "place_border": 0.01,
            "place_top": False,
            "place_right": True,
        },
    },
    {
        "dataset_def": SISR_DATASETS[0],
        "image_idx": 2,
        "scale_factor": 4,
        "figure_args": {
            "region_center": (50, 256),
            "region_extent": 50,
            "rel_width": 0.6,
            "place_border": 0.01,
            "place_top": False,
            "place_right": True,
        },
    },
    {
        "dataset_def": SISR_DATASETS[0],
        "image_idx": 4,
        "scale_factor": 4,
        "figure_args": {
            "region_center": (40, 160),
            "region_extent": 30,
            "rel_width": 0.6,
            "place_border": 0.01,
            "place_top": False,
            "place_right": True,
        },
    },
    {  # Set 5 - blur
        "dataset_def": SISR_DATASETS[6],
        "image_idx": 0,
        "scale_factor": 2,
        "figure_args": {
            "region_center": (90, 140),
            "region_extent": 30,
            "rel_width": 0.6,
            "place_border": 0.01,
            "place_top": False,
            "place_right": False,
        },
    },
    {
        "dataset_def": SISR_DATASETS[7],
        "image_idx": 1,
        "scale_factor": 2,
        "figure_args": {
            "region_center": (50, 50),
            "region_extent": 40,
            "rel_width": 0.6,
            "place_border": 0.01,
            "place_top": False,
            "place_right": True,
        },
    },
    {
        "dataset_def": SISR_DATASETS[8],
        "image_idx": 2,
        "scale_factor": 3,
        "figure_args": {
            "region_center": (50, 256),
            "region_extent": 50,
            "rel_width": 0.6,
            "place_border": 0.01,
            "place_top": False,
            "place_right": True,
        },
    },
    {
        "dataset_def": SISR_DATASETS[9],
        "image_idx": 4,
        "scale_factor": 3,
        "figure_args": {
            "region_center": (40, 160),
            "region_extent": 30,
            "rel_width": 0.6,
            "place_border": 0.01,
            "place_top": False,
            "place_right": True,
        },
    },
]
NUM_SISR_EXAMPLES = len(EXAMPLES)


def _define_sisr_model(superresolver, scale_factor):
    def model(example, example_seed):
        # Set the tf seed for models that use tf.random
        tf.random.set_seed(example_seed)

        # Get the high resolution and low resolution image
        hr = example["hr"]
        lr = example["lr"]

        # Super resolve
        if "kernel" in example:
            sr = superresolver(lr, scale_factor, example["kernel"])
        else:
            sr = superresolver(lr, scale_factor, None)

        # Compare hr and super resolved
        return hr, sr

    return model


def _load_sisr_dataset(dataset_def, scale_factor):
    import dppp

    resize_fn_bicubic = dppp.create_resize_fn("bicubic", antialias=True)

    def bicubic_downscale(hr):
        return resize_fn_bicubic(hr[None, ...], scale_factor, False)[0]

    def conv_downscale(hr, kernel):
        resize_fn = dppp.create_convolve_resize_fn(kernel, mode="wrap")
        return resize_fn(hr[None, ...], scale_factor, False)[0]

    return utils.load_sr_dataset(
        dataset_def,
        scale_factor,
        bicubic_downscale=bicubic_downscale,
        conv_downscale=conv_downscale,
    )


def example_sisr(superresolver, result, example_idx=0, save_gt=False, seed=99):
    example = EXAMPLES[example_idx]
    scale_factor = example["scale_factor"]

    _, __, dataset = _load_sisr_dataset(example["dataset_def"], scale_factor)
    data = datasets.get_one_example(dataset, index=example["image_idx"])

    model = _define_sisr_model(superresolver, scale_factor)
    try:
        hr, sr = model(data, seed)
    except utils.MethodNotApplicableError:
        return

    # Save the example
    utils.save_example(
        gt=hr,
        output=sr,
        result=result,
        save_gt=save_gt,
        figure_width=1.1,
        **example["figure_args"],
    )


def eval_sisr(
    superresolver,
    csv_out,
    seed=99,
    test_run=False,
):
    """Evaluate the single image super-resolution performace of a superresolver method
    and write the results to a CSV file"""

    # Create random seeds for reproducability
    rng = np.random.default_rng(seed)
    dataset_seeds = utils.seeds_for_values(rng, [d["name"] for d in SISR_DATASETS])
    scales_seeds = utils.seeds_for_values(rng, SISR_SCALE_FACTORS)

    # Write CSV file header
    csv_out.write(
        f"dataset,image_index,scale_factor,downscale_method,{','.join(metrics.METRICS.keys())}\n"
    )

    # Loop over scales
    for scale_factor in SISR_SCALE_FACTORS:

        # Loop over datasets
        for dataset_def in SISR_DATASETS:
            # Load the dataset
            dataset_name, downscale_method, dataset = _load_sisr_dataset(
                dataset_def, scale_factor
            )

            # Only take 2 examples if this is a test run
            if test_run:
                dataset = dataset.take(2)

            model = _define_sisr_model(superresolver, scale_factor)
            utils.evaluate(
                model=model,
                dataset=dataset,
                dataset_name=dataset_name,
                csv_out=csv_out,
                parameters=f"{scale_factor},{downscale_method}",
                seed=scales_seeds[scale_factor] * dataset_seeds[dataset_name],
                test_run=test_run,
            )
