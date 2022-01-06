import os
import sys
import argparse

import numpy as np
import tensorflow as tf
import torch

sys.path.append(os.path.normpath(os.path.join(__file__, "..", "KAIR")))
from models.network_dncnn import DnCNN as net


def load_model(device):
    weights_path = os.path.normpath(
        os.path.join(__file__, "..", "dncnn_color_blind.pth")
    )
    model = net(in_nc=3, out_nc=3, nc=64, nb=20, act_mode="R")
    model.load_state_dict(torch.load(weights_path), strict=True)
    for _, v in model.named_parameters():
        v.requires_grad = False
    return model.to(device)


def denoising(test_run, args):
    device = torch.device("cpu")
    model = load_model(device)

    def denoiser(img, _):
        in_tensor = torch.from_numpy(np.ascontiguousarray(img.numpy()))
        in_tensor = in_tensor.permute(2, 0, 1).float().unsqueeze(0)
        out_tensor = model(in_tensor.to(device))
        denoised = out_tensor.data.squeeze().permute(1, 2, 0).float().cpu().numpy()
        return tf.constant(denoised, dtype=tf.float32)

    return denoiser, "dncnn"
