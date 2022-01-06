from collections import namedtuple
import os
import sys
import argparse

import numpy as np
import tensorflow as tf
import skimage.color as color

import torch

sys.path.append(os.path.normpath(os.path.join(__file__, "..", "LFSSR-ATO")))
from model.model_LFSSR import LFSSRNet
from utils import util


# TODO add run at one exampe mode for visual result


def _load_model(scale, angular_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = {
        "feature_num": 64,
        "angular_num": angular_num,
        "scale": scale,
        "layer_num": [5, 2, 2, 3],
        "layer_num_refine": 3,
    }
    Opt = namedtuple("Opt", opt)
    opt = Opt(**opt)

    model = LFSSRNet(opt).to(device)

    checkpoint_path = os.path.normpath(
        os.path.join(
            __file__, "..", "LFSSR-ATO", "pretrained_models", f"LFSSRNet_{scale}x.pth"
        )
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    return model


def lfsr(test_run, args):
    scales = [2, 4]
    angular_num = 7
    models = {s: _load_model(s, angular_num) for s in scales}

    # Code adapted from https://github.com/jingjin25/LFSSR-ATO/blob/master/demo_LFSSR.py
    def superresolver(lr, scale_factor, kernel):
        model = models[scale_factor]

        lr = lr[1:-1, 1:-1]  # 9x9 views to 7x7
        lr_h, lr_w = tf.shape(lr)[2:4].numpy()
        sr_h, sr_w = lr_h * scale_factor, lr_w * scale_factor

        lr_ycbcr = color.rgb2ycbcr(lr.numpy())
        lr_ycbcr = lr_ycbcr.transpose([0, 1, 4, 2, 3])
        lr_ycbcr = torch.from_numpy(lr_ycbcr.astype(np.float32))

        # Interpolate color values
        lr_ycbcr_up = lr_ycbcr.reshape(1, -1, lr_h, lr_w)
        lr_ycbcr_up = torch.nn.functional.interpolate(
            lr_ycbcr_up, scale_factor=scale_factor, mode="bicubic", align_corners=False
        )
        lr_ycbcr_up = lr_ycbcr_up.view(-1, 3, sr_h, sr_w).numpy()

        lr_y = lr_ycbcr[:, :, 0, :, :].clone().view(1, -1, lr_h, lr_w) / 255.0

        with torch.no_grad():
            crop = 8
            length = 120
            lr_l, lr_m, lr_r = util.CropPatches(
                lr_y, length // scale_factor, crop // scale_factor
            )
            sr_l = model(lr_l).numpy()
            sr_m = np.zeros(
                (
                    lr_m.shape[0],
                    angular_num * angular_num,
                    lr_m.shape[2] * scale_factor,
                    lr_m.shape[3] * scale_factor,
                ),
                dtype=np.float32,
            )
            for i in range(lr_m.shape[0]):
                sr_m[i : i + 1] = model(lr_m[i : i + 1]).numpy()
            sr_r = model(lr_r).numpy()
            sr_y = util.MergePatches(
                sr_l,
                sr_m,
                sr_r,
                lr_y.shape[2] * scale_factor,
                lr_y.shape[3] * scale_factor,
                length,
                crop,
            )[0]

        lr_ycbcr_up[:, 0] = sr_y * 255.0

        sr = lr_ycbcr_up
        sr = sr[sr.shape[0] // 2, ...]
        sr = sr.transpose([1, 2, 0])
        sr = color.ycbcr2rgb(sr)

        return tf.constant(sr)

    return superresolver, "lfssr"
