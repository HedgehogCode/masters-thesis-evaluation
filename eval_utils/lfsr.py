import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_datasets_bw as datasets

from . import metrics
from . import utils

LFSR_DATASETS = [
    {
        "name": "HCI",
        "key": "hci_lf",
        "split": "test",
        "hr_image_key": "lf",
        "downscale": "bicubic",
    },
    *[
        {
            "name": "HCI",
            "key": "hci_lf",
            "split": "test",
            "hr_image_key": "lf",
            "downscale": f"kernel_{i}",
            "kernel": k,
        }
        for i, k in enumerate(utils.GAUSSIAN_KERNELS)
    ],
]
LFSR_SCALE_FACTORS = [2, 3, 4, 5]

EXAMPLES = [
    {
        "dataset_def": LFSR_DATASETS[0],
        "image_idx": 0,
        "scale_factor": 4,
        "figure_args": {
            "region_center": (230, 300),
            "region_extent": 50,
            "rel_width": 0.5,
            "place_border": 0.01,
            "place_top": True,
            "place_right": False,
        },
    },
    {
        "dataset_def": LFSR_DATASETS[1],
        "image_idx": 1,
        "scale_factor": 4,
        "figure_args": {
            "region_center": (180, 120),
            "region_extent": 50,
            "rel_width": 0.5,
            "place_border": 0.01,
            "place_top": False,
            "place_right": True,
        },
    },
    {
        "dataset_def": LFSR_DATASETS[2],
        "image_idx": 2,
        "scale_factor": 4,
        "figure_args": {
            "region_center": (220, 70),
            "region_extent": 50,
            "rel_width": 0.5,
            "place_border": 0.01,
            "place_top": True,
            "place_right": True,
        },
    },
    {
        "dataset_def": LFSR_DATASETS[3],
        "image_idx": 3,
        "scale_factor": 4,
        "figure_args": {
            "region_center": (240, 160),
            "region_extent": 50,
            "rel_width": 0.5,
            "place_border": 0.01,
            "place_top": True,
            "place_right": True,
        },
    },
]
NUM_LFSR_EXAMPLES = len(EXAMPLES)


def _define_lfsr_model(superresolver, scale_factor):
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

        lf_grid = tf.shape(hr)[:2]
        lf_center = tf.math.floordiv(lf_grid, 2)

        # Compare with the center of the input light field
        return hr[lf_center[0], lf_center[1]], sr

    return model


def _load_lfsr_dataset(dataset_def, scale_factor):
    import dppp

    resize_fn_bicubic = dppp.create_resize_fn("bicubic", antialias=True)

    def bicubic_downscale(hr_lf):
        return dppp.resize_lf(hr_lf, scale_factor, False, resize_fn_bicubic)

    def conv_downscale(hr_lf, kernel):
        resize_fn = dppp.create_convolve_resize_fn(kernel, mode="wrap")
        return dppp.resize_lf(hr_lf, scale_factor, False, resize_fn)

    return utils.load_sr_dataset(
        dataset_def,
        scale_factor,
        bicubic_downscale=bicubic_downscale,
        conv_downscale=conv_downscale,
    )


def example_lfsr(superresolver, result, example_idx=0, save_gt=False, seed=99):
    example = EXAMPLES[example_idx]
    scale_factor = example["scale_factor"]
    image_idx = example["image_idx"]

    _, __, dataset = _load_lfsr_dataset(example["dataset_def"], scale_factor)
    data = datasets.get_one_example(dataset, index=image_idx)

    model = _define_lfsr_model(superresolver, scale_factor)
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


def eval_lfsr(
    superresolver,
    csv_out,
    seed=99,
    test_run=False,
):
    """Evaluate the light field super-resolution performace of a superresolver method
    and write the results to a CSV file"""

    # Create random seeds for reproducability
    rng = np.random.default_rng(seed)
    dataset_seeds = utils.seeds_for_values(rng, [d["name"] for d in LFSR_DATASETS])
    scales_seeds = utils.seeds_for_values(rng, LFSR_SCALE_FACTORS)

    # Write CSV file header
    csv_out.write(
        f"dataset,image_index,scale_factor,downscale_method,{','.join(metrics.METRICS.keys())}\n"
    )

    for scale_factor in LFSR_SCALE_FACTORS:

        # Loop over datasets
        for dataset_def in LFSR_DATASETS:
            # Load the dataset
            dataset_name, downscale_method, dataset = _load_lfsr_dataset(
                dataset_def, scale_factor
            )

            # Only take 2 examples if this is a test run
            if test_run:
                dataset = dataset.take(2)

            model = _define_lfsr_model(superresolver, scale_factor)
            utils.evaluate(
                model=model,
                dataset=dataset,
                dataset_name=dataset_name,
                csv_out=csv_out,
                parameters=f"{scale_factor},{downscale_method}",
                seed=scales_seeds[scale_factor] * dataset_seeds[dataset_name],
                test_run=test_run,
            )
