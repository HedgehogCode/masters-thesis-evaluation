import os
import numpy as np
import scipy.io

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_datasets_bw as datasets

from . import metrics

_MIN_SEED = 0
_MAX_SEED = np.iinfo(np.int32).max

# Gaussian kernels from Zhang 2021
GAUSSIAN_KERNELS = np.stack(
    scipy.io.loadmat(os.path.normpath(os.path.join(__file__, "..", "kernels", "kernels_12.mat")))[
        "kernels"
    ][0],
    axis=0,
)[:8]


class MethodNotApplicableError(Exception):
    """Models should raise this error if they are not applicable."""

    pass


def seeds_for_values(rng, values):
    """Generate random seeds for a list of values using the random number generator"""
    return dict(
        zip(
            values,
            rng.integers(_MIN_SEED, _MAX_SEED, [len(values)]),
        )
    )


def evaluate(model, dataset, dataset_name, csv_out, parameters, seed, test_run):
    """Evaluate the model on the dataset and write to a CSV file

    The CSV file has the columns
    [dataset_name, image_index, parameters..., metrics...]

    Args:
        model: A function taking data from the dataset, and a seed and
            returning a ground truth image and model output
        dataset
        dataset_name
        csv_out: A writer to a CSV file (header must be written already)
        parameters: A string with comma separated parameters that will be
            written to the CSV file
        example_seed
        test_run: Only evaluate the PSNR metric
    """
    print(
        f"\n{dataset_name} - {parameters} - seed: {seed} -- Results ("
        + ", ".join(metrics.METRICS.keys())
        + "):"
    )
    rng = np.random.default_rng(seed)

    metric_sums = {n: 0 for n in metrics.METRICS.keys()}
    num = 0

    for index, data in enumerate(dataset):
        example_seed = rng.integers(_MIN_SEED, _MAX_SEED, [1])[0]
        # Get the output of the model
        try:
            image, output = model(data, example_seed)
        except MethodNotApplicableError:
            # Skip the example
            print(f"{index:4d} -- not applicable")
            continue

        # Compute metrics
        values = []
        for n, fn in metrics.METRICS.items():
            v = fn(image, output)
            metric_sums[n] += v
            values.append(v)

            # Only compute PSNR for test runs
            if test_run:
                break

        # Print for this image
        print(f"{index:4d} -- ({', '.join([f'{v:.4f}' for v in values])})")

        # Write to csv
        csv_out.write(
            f"{dataset_name},{index},{parameters},"
            + ",".join([str(v) for v in values])
            + "\n"
        )
        csv_out.flush()

        num += 1

    if num > 0:
        print(f"MEAN -- ({', '.join([f'{v/num:.4f}' for v in metric_sums.values()])})")


def save_example(gt, output, result, save_gt, compute_metrics=True, **highlight_kwargs):
    if save_gt:
        output = gt
        label = "(PSNR, LPIPS)"
    elif compute_metrics:
        # Evaluate metrics
        metr_val = {n: fn(gt, output) for n, fn in metrics.METRICS.items()}
        label = f"({metr_val['PSNR']:.2f}dB, {metr_val['LPIPS_ALEX']:.4f})"
    else:
        label = None

    # Zoom in and add a label
    figure = _highlight(output, label=label, **highlight_kwargs)

    # Save the result
    figure.savefig(result.name, dpi=600)


def _highlight(
    image,
    region_center,
    region_extent,
    rel_width,
    place_top=False,
    place_right=True,
    place_border=0.01,
    figure_width=3,
    label="(PSNR, FSIM)",
):
    import matplotlib.pyplot as plt

    # Size of the figure
    # figsize = (image.shape[1] / 100, image.shape[0] / 100 * (1 + label_border))
    label_border = 0.15 if label is not None else 0.0
    fontsize = 6
    figsize = (
        figure_width,
        (image.shape[0] * figure_width / image.shape[1]) + label_border,
    )
    rel_label_border = label_border / figsize[1]

    fig = plt.figure(figsize=figsize)

    # Axis1 is the whole image
    ax1 = plt.axes([0, rel_label_border, 1, 1 - rel_label_border], frameon=False)
    ax1.imshow(image)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel(label, fontsize=fontsize)

    # Axis2 is the region
    if rel_width > 0:
        rel_height = figsize[0] * rel_width / figsize[1]
        pos_x = 1 - rel_width - place_border if place_right else place_border
        pos_y = (
            1 - rel_height - place_border
            if place_top
            else (place_border + rel_label_border)
        )
        ax2 = plt.axes([pos_x, pos_y, rel_width, rel_height])
        region = image[
            (region_center[0] - region_extent) : (region_center[0] + region_extent),
            (region_center[1] - region_extent) : (region_center[1] + region_extent),
            :,
        ]
        ax2.imshow(region)
        ax2.set_xticks([])
        ax2.set_yticks([])

    return fig


def load_sr_dataset(dataset_def, scale_factor, bicubic_downscale, conv_downscale):

    # Crop a high resolution image to the multiple of the scale factor
    def crop_hr(hr):
        hr_shape = tf.shape(hr)
        lr_size = (hr_shape[-3] // scale_factor, hr_shape[-2] // scale_factor)
        hr_size = (lr_size[0] * scale_factor, lr_size[1] * scale_factor)
        return hr[..., : hr_size[0], : hr_size[1], :]

    dataset_name = dataset_def["name"]
    dataset_key = dataset_def["key"]
    dataset_split = dataset_def["split"]

    # We only use the lr image
    if "lr_image_key" in dataset_def:

        images = (
            tfds.load(dataset_key, split=dataset_def["split"])
            .map(datasets.get_value(dataset_def["lr_image_key"]))
            .map(datasets.to_float32)
            .map(datasets.from_255_to_1_range)
        )

        def to_hr_lr_dict(lr):
            # Create a fake hr image with just zeros
            hr_shape = tf.concat(
                [
                    tf.shape(lr)[:-3],
                    [
                        tf.shape(lr)[-3] * scale_factor,
                        tf.shape(lr)[-2] * scale_factor,
                        tf.shape(lr)[-1],
                    ],
                ],
                axis=0,
            )
            hr = tf.zeros(hr_shape, dtype=tf.float32)
            return {"hr": hr, "lr": lr}

        return dataset_name, "Unknown", images.map(to_hr_lr_dict)

    # We have to downscale the images here (not done by tensorflow_datasets_bw)
    elif "downscale" in dataset_def:

        downscale_name = dataset_def["downscale"]
        images = (
            tfds.load(dataset_key, split=dataset_def["split"])
            .map(datasets.get_value(dataset_def["hr_image_key"]))
            .map(datasets.to_float32)
            .map(datasets.from_255_to_1_range)
        )

        # Use bicubic downsampling
        if downscale_name == "bicubic":

            def downscale(hr):
                hr = crop_hr(hr)
                lr = bicubic_downscale(hr)
                return {"hr": hr, "lr": lr}

        # Use a kernel for downsampling
        else:
            import dppp

            kernel = dataset_def["kernel"]

            def downscale(hr):
                hr = crop_hr(hr)
                kernel_rgb = dppp.conv2D_filter_rgb(kernel)
                lr = conv_downscale(hr, kernel_rgb)
                return {"hr": hr, "lr": lr, "kernel": kernel}

        return dataset_name, downscale_name, images.map(downscale)

    # Downscaling is done by the dataset (always bicubic)
    else:
        dataset_args = {
            "scale": scale_factor,
            "resize_method": "bicubic",
            "antialias": True,
        }

        dataset = (
            tfds.load(dataset_key, split=dataset_split, builder_kwargs=dataset_args)
            .map(datasets.map_on_dict(datasets.to_float32))
            .map(datasets.map_on_dict(datasets.from_255_to_1_range))
        )
        return dataset_name, "bicubic", dataset
