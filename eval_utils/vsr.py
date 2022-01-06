import functools
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_datasets_bw as datasets

from . import metrics
from . import utils

VSR_DATASETS = [
    {
        "name": "Vid4",
        "key": "vid4",
        "split": "test",
    },
    *[
        {
            "name": "Vid4",
            "key": "vid4",
            "split": "test",
            "hr_image_key": "hr",
            "downscale": f"kernel_{i}",
            "kernel": k,
        }
        for i, k in enumerate(utils.GAUSSIAN_KERNELS)
    ],
]
VSR_SCALE_FACTORS = [2, 3, 4, 5]

VSR_MDSP_DATASET = {
    "name": "MDSP Color",
    "key": "mdsp_color_sr",
    "split": "test",
    "lr_image_key": "video",
}

EXAMPLES = [
    {  # Image 0, bicubic
        "dataset_def": VSR_DATASETS[0],
        "image_idx": 0,
        "scale_factor": 4,
        "figure_args": {
            "region_center": (120, 360),
            "region_extent": 40,
            "rel_width": 0.4,
            "place_border": 0.01,
            "place_top": False,
            "place_right": False,
        },
    },
    {  # Image 2, isotropic blur
        "dataset_def": VSR_DATASETS[2],
        "image_idx": 2,
        "scale_factor": 4,
        "figure_args": {
            "region_center": (200, 490),
            "region_extent": 40,
            "rel_width": 0.4,
            "place_border": 0.01,
            "place_top": True,
            "place_right": False,
        },
    },
    {  # Image 3, anisotropic blur
        "dataset_def": VSR_DATASETS[7],
        "image_idx": 3,
        "scale_factor": 4,
        "figure_args": {
            "region_center": (200, 640),
            "region_extent": 50,
            "rel_width": 0.4,
            "place_border": 0.01,
            "place_top": False,
            "place_right": False,
        },
    },
    {  # Image 1, anisotropic blur
        "dataset_def": VSR_DATASETS[8],
        "image_idx": 1,
        "scale_factor": 4,
        "figure_args": {
            "region_center": (480, 450),
            "region_extent": 30,
            "rel_width": 0.4,
            "place_border": 0.01,
            "place_top": True,
            "place_right": False,
        },
    },
    {
        "dataset_def": VSR_MDSP_DATASET,
        "image_idx": 0,
        "scale_factor": 4,
        "figure_args": {
            "compute_metrics": False,
            "region_center": (90, 225),
            "region_extent": 50,
            "rel_width": 0.4,
            "place_border": 0.01,
            "place_top": False,
            "place_right": False,
        },
    },
    {
        "dataset_def": VSR_MDSP_DATASET,
        "image_idx": 2,
        "scale_factor": 4,
        "figure_args": {
            "compute_metrics": False,
            "region_center": (50, 50),
            "region_extent": 50,
            "rel_width": 0,
            "place_border": 0.01,
            "place_top": False,
            "place_right": False,
        },
    },
    {
        "dataset_def": VSR_MDSP_DATASET,
        "image_idx": 4,
        "scale_factor": 4,
        "figure_args": {
            "compute_metrics": False,
            "region_center": (50, 50),
            "region_extent": 50,
            "rel_width": 0,
            "place_border": 0.01,
            "place_top": False,
            "place_right": False,
        },
    },
]
NUM_VSR_EXAMPLES = len(EXAMPLES)


def _define_vsr_model(superresolver, scale_factor):
    def model(example, example_seed):
        # Set the tf seed for models that use tf.random
        tf.random.set_seed(example_seed)

        # Get the high resolution and low resolution video
        hr = example["hr"]
        lr = example["lr"]

        # Super resolve
        ref_idx = tf.shape(hr)[0].numpy() // 2
        if "kernel" in example:
            sr = superresolver(lr, ref_idx, scale_factor, example["kernel"])
        else:
            sr = superresolver(lr, ref_idx, scale_factor, None)

        # Compare with the high resolution at the ref index
        return hr[ref_idx], sr

    return model


def _load_vsr_dataset(dataset_def, scale_factor):
    import dppp

    bicubic_downscale = functools.partial(
        dppp.create_resize_fn("bicubic", antialias=True),
        factor=scale_factor,
        upscale=False,
    )

    def conv_downscale(hr, kernel):
        resize_fn = dppp.create_convolve_resize_fn(kernel, mode="wrap")
        return resize_fn(hr, scale_factor, False)

    return utils.load_sr_dataset(
        dataset_def,
        scale_factor,
        bicubic_downscale=bicubic_downscale,
        conv_downscale=conv_downscale,
    )


def example_vsr(superresolver, result, example_idx=0, save_gt=False, seed=99):
    example = EXAMPLES[example_idx]
    scale_factor = example["scale_factor"]
    image_idx = example["image_idx"]

    _, __, dataset = _load_vsr_dataset(example["dataset_def"], scale_factor)
    data = datasets.get_one_example(dataset, index=image_idx)

    model = _define_vsr_model(superresolver, scale_factor)
    hr, sr = model(data, seed + example_idx)

    # Save the example
    utils.save_example(
        gt=hr,
        output=sr,
        result=result,
        save_gt=save_gt,
        figure_width=1.1,  # TODO set according to size in pdf
        **example["figure_args"],
    )


def eval_vsr(
    superresolver,
    csv_out,
    seed=99,
    test_run=False,
):
    """Evaluate the video super-resolution performace of a superresolver method
    and write the results to a CSV file"""

    # Create random seeds for reproducability
    rng = np.random.default_rng(seed)
    dataset_seeds = utils.seeds_for_values(rng, [d["name"] for d in VSR_DATASETS])
    scales_seeds = utils.seeds_for_values(rng, VSR_SCALE_FACTORS)

    # Write CSV file header
    csv_out.write(
        f"dataset,image_index,scale_factor,downscale_method,{','.join(metrics.METRICS.keys())}\n"
    )

    # Loop over scales
    for scale_factor in VSR_SCALE_FACTORS:

        # Loop over datasets
        for dataset_def in VSR_DATASETS:
            # Load the dataset
            dataset_name, downscale_method, dataset = _load_vsr_dataset(
                dataset_def, scale_factor
            )

            # Only take 2 examples if this is a test run
            if test_run:
                dataset = dataset.take(2)

            model = _define_vsr_model(superresolver, scale_factor)
            utils.evaluate(
                model=model,
                dataset=dataset,
                dataset_name=dataset_name,
                csv_out=csv_out,
                parameters=f"{scale_factor},{downscale_method}",
                seed=scales_seeds[scale_factor] * dataset_seeds[dataset_name],
                test_run=test_run,
            )
