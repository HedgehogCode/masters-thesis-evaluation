import os
import numpy as np
import imageio

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_datasets_bw as datasets

from . import metrics
from . import utils

EXAMPLES = [
    "kate",
    "library",
    "vase",
    "vase2",
]
NUM_INPAINTING_EXAMPLES = len(EXAMPLES)


def _load_example(name):
    data_path = os.path.normpath(os.path.join(__file__, "..", "data_inpainting"))
    image_path = os.path.join(data_path, f"{name}.png")
    mask_path = os.path.join(data_path, f"{name}_mask.png")

    image = imageio.imread(image_path)
    mask = imageio.imread(mask_path)

    image_tf = datasets.from_255_to_1_range(datasets.to_float32(tf.constant(image)))
    mask_tf = tf.broadcast_to(
        datasets.from_255_to_1_range(datasets.to_float32(tf.constant(mask)))[..., None],
        shape=tf.shape(image_tf),
    )
    return image_tf, mask_tf


def example_inpaint(inpainter, result, example_idx=0):
    # Load the example
    image, mask = _load_example(EXAMPLES[example_idx])

    # Set the seed for reproducable results if random numbers are used
    tf.random.set_seed(example_idx)

    # Inpaint
    degraded = image * mask
    inpainted = inpainter(degraded, mask)

    # Save the result
    imageio.imsave(result.name, inpainted)
