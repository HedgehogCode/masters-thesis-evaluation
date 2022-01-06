import os
import sys
import argparse

import numpy as np

import tensorflow as tf

root = os.path.normpath(os.path.join(__file__, ".."))
sys.path.append(os.path.join(root, "fourier-deconvolution-network"))
from model import model_stacked
from fdn_predict import to_tensor, from_tensor

AVAILABLE_MODELS = [
    ([1.0, 3.0], "sigma_1.0-3.0"),  # First choise
    ([0.1, 12.75], "sigma_0.1-12.75"),  # Second choise
]

MODEL = None
MODEL_RANGE = [0, -1]


def find_model_dir(sigma):
    for range, model_dir in AVAILABLE_MODELS:
        if range[0] <= sigma <= range[1]:
            return model_dir, range
    raise ValueError(f"No model for sigma {sigma} available.")


def get_model(sigma):
    # Return the cached model if it fits
    global MODEL, MODEL_RANGE
    model_dir, range = find_model_dir(sigma)
    if MODEL_RANGE == range:
        return MODEL

    # Load the model that fits
    print(f"Loading model for sigma={sigma}...")

    tf.keras.backend.clear_session()
    n_stages = 10
    weights = os.path.join(
        root,
        "fourier-deconvolution-network",
        "models",
        model_dir,
        "stages_01-10_finetuned.hdf5",
    )
    m = model_stacked(n_stages)
    m.load_weights(weights)

    MODEL = m
    MODEL_RANGE = range

    return m


def nb_deblurring(test_run, args):
    def deblurrer(blurry, kernel, noise_stddev):
        kernel = kernel[::-1, ::-1]
        sigma = noise_stddev * 255.0
        model = get_model(sigma)

        # Channels to batch dimension
        # y = to_tensor(edgetaper(pad_for_kernel(blurry,kernel,'edge'),kernel))
        y = to_tensor(blurry)
        k = np.tile(kernel[np.newaxis], (y.shape[0], 1, 1))
        s = np.tile(sigma, (y.shape[0], 1)).astype(np.float32)
        x0 = y

        pred = model.predict_on_batch([x0, y, k, s])

        # Channels from batch to channels dimension
        pred = from_tensor(pred[-1])
        # pred = crop_for_kernel(from_tensor(pred[-1]), kernel)

        return pred

    return deblurrer, "fdn", False
