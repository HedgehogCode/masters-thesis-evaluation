import os
import sys

import numpy as np

import tensorflow as tf

from eval_utils.utils import MethodNotApplicableError

sys.path.append(os.path.normpath(os.path.join(__file__, "..")))
from DMSP import DMSPRestore


def _crop_to_valid(x, kernel):
    pad_y = np.floor(kernel.shape[0] / 2.0).astype(np.int64)
    pad_x = np.floor(kernel.shape[1] / 2.0).astype(np.int64)
    return x[pad_y:-pad_y, pad_x:-pad_x, :]

def _load_model():
    return tf.saved_model.load(os.path.join(__file__, "..", "DMSP", "DAE"))

def nb_deblurring(test_run, args):
    model = _load_model()

    def deblurrer(blurry, kernel, noise_stddev):
        kernel = kernel.numpy()
        params = {}
        params["denoiser"] = model
        params["sigma_dae"] = 11.0
        params["num_iter"] = 10 if test_run else 300
        params["mu"] = 0.9
        params["alpha"] = 0.1
        # params['gt'] = image * 255

        # Prepare degraded input
        degraded = _crop_to_valid(blurry, kernel) * 255.0

        # No downscaling -> No subsampling mask
        subsampling_mask = np.ones_like(degraded)

        # running DMSP
        restored = DMSPRestore.DMSP_restore(
            degraded=degraded,
            kernel=kernel,
            subsampling_mask=subsampling_mask,
            sigma_d=noise_stddev * 255,
            params=params,
        )

        return restored / 255.0

    return deblurrer, "dmsp_paper", True


def sisr(test_run, args):
    import dppp
    model = _load_model()

    def superresolver(lr, scale_factor, kernel=None):
        if kernel is None:
            raise MethodNotApplicableError()

        kernel = kernel.numpy()
        params = {}
        params["denoiser"] = model
        params["sigma_dae"] = 11.0
        params["num_iter"] = 10 if test_run else 300
        params["mu"] = 0.9
        params["alpha"] = 0.1

        # Prepare degraded input TODO

        degraded = dppp.upscale_no_iterpolation(lr[None, ...], scale_factor)[0]
        degraded = (_crop_to_valid(degraded, kernel) * 255.0).numpy()
        subsampling_mask = np.zeros_like(degraded)
        subsampling_mask[::scale_factor, ::scale_factor] = 1

        # running DMSP
        restored = DMSPRestore.DMSP_restore(
            degraded=degraded,
            kernel=kernel,
            subsampling_mask=subsampling_mask,
            sigma_d=0,
            params=params,
        )

        return restored / 255.0

    return superresolver, "dmsp_paper"