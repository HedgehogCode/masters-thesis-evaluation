import os
import sys
import cv2

import numpy as np

import torch

from eval_utils.utils import MethodNotApplicableError

sys.path.append(os.path.normpath(os.path.join(__file__, "..", "DPIR")))
from models.network_unet import UNetRes as net
from utils import utils_model
from utils import utils_image as util
from utils import utils_pnp as pnp
from utils import utils_sisr as sr


# See https://github.com/cszn/DPIR/blob/master/main_dpir_deblur.py
def load_model(device):
    weights_path = os.path.normpath(
        os.path.join(__file__, "..", "drunet_color.pth")
    )
    model = net(
        in_nc=4,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    )
    model.load_state_dict(torch.load(weights_path), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    return model.to(device)


# See https://github.com/cszn/DPIR/blob/master/main_dpir_deblur.py
def run_dpir(degraded, kernel, noise_level, model, device, sf=1):
    k = kernel[::-1, ::-1].numpy()
    noise_level_img = noise_level
    noise_level_model = noise_level_img
    modelSigma1 = 49
    modelSigma2 = (
        max(sf, noise_level_model * 255.0) if sf > 1 else noise_level_model * 255.0
    )
    iter_num = 8
    rhos, sigmas = pnp.get_rho_sigma(
        sigma=max(0.255 / 255.0, noise_level_model),
        iter_num=iter_num,
        modelSigma1=modelSigma1,
        modelSigma2=modelSigma2,
        w=1.0,
    )
    rhos = torch.tensor(rhos).to(device)
    sigmas = torch.tensor(sigmas).to(device)

    if sf > 1:
        # Upscale degraded
        x = cv2.resize(
            degraded.numpy(),
            (degraded.shape[1] * sf, degraded.shape[0] * sf),
            interpolation=cv2.INTER_CUBIC,
        )
        x = util.single2tensor4(x).to(device)
    else:
        x = util.single2tensor4(degraded).to(device)

    img_L_tensor, k_tensor = util.single2tensor4(degraded), util.single2tensor4(
        np.expand_dims(k, 2)
    )
    [k_tensor, img_L_tensor] = util.todevice([k_tensor, img_L_tensor], device)
    FB, FBC, F2B, FBFy = sr.pre_calculate(img_L_tensor, k_tensor, sf)

    for i in range(iter_num):
        # Step 1
        tau = rhos[i].float().repeat(1, 1, 1, 1)
        x = sr.data_solution(x, FB, FBC, F2B, FBFy, tau, sf)

        # Step 2
        x = util.augment_img_tensor4(x, i % 8)
        x = torch.cat(
            (x, sigmas[i].float().repeat(1, 1, x.shape[2], x.shape[3])), dim=1
        )
        x = utils_model.test_mode(model, x, mode=2, refield=32, min_size=256, modulo=16)

        if i % 8 == 3 or i % 8 == 5:
            x = util.augment_img_tensor4(x, 8 - i % 8)
        else:
            x = util.augment_img_tensor4(x, i % 8)

    return util.tensor2single(x)


def nb_deblurring(test_run, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    def deblurrer(blurry, kernel, noise_stddev):
        return run_dpir(
            degraded=blurry,
            kernel=kernel,
            noise_level=noise_stddev,
            model=model,
            device=device,
        )

    return deblurrer, "dpir", False


def sisr(test_run, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    def superresolver(lr, scale_factor, kernel=None):
        if kernel is None:
            raise MethodNotApplicableError()

        return run_dpir(
            degraded=lr,
            kernel=kernel,
            noise_level=0,
            model=model,
            device=device,
            sf=scale_factor,
        )

    return superresolver, "dpir"
