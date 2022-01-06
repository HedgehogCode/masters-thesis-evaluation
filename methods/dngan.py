import os
import re
import dppp

from eval_utils.utils import MethodNotApplicableError
from eval_utils.denoising import DENOISING_NOISE_LEVELS


def noise_level_for_model(name):
    groups = re.match(r".*?(\d\.\d+)?-?(\d\.\d+)", name)
    if not groups[1]:
        model_noise = float(groups[2])
        differences = [abs(n - model_noise) for n in DENOISING_NOISE_LEVELS]
        noise_level = DENOISING_NOISE_LEVELS[
            min(range(len(differences)), key=differences.__getitem__)
        ]
        return noise_level
    return None


def _model_path(args):
    if args is None:
        raise ValueError("A denoiser must be given.")
    return args


def _model_name(model_path):
    return os.path.basename(model_path)[:-3]


def denoising(test_run, args):
    # Load the model
    model_path = _model_path(args)
    model_name = _model_name(model_path)
    model_noise_stddev = noise_level_for_model(model_name)
    denoiser, (denoiser_min, denoiser_max) = dppp.load_denoiser(model_path)

    def denoise(x, noise_stddev):
        if model_noise_stddev is not None and noise_stddev != model_noise_stddev:
            raise MethodNotApplicableError()
        return denoiser(x[None, ...], noise_stddev)[0]

    return denoise, model_name
