import os

import numpy as np
import tensorflow as tf
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet

from eval_utils.utils import MethodNotApplicableError


PRETRAINED_ESRGAN = os.path.normpath(
    os.path.join(
        __file__,
        "..",
        "pretrained_models",
        "ESRGAN",
        "ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
    )
)


def sisr(test_run, args):
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
    )
    model.load_state_dict(torch.load(PRETRAINED_ESRGAN)["params"], strict=True)
    model.eval()

    def superresolver(lr, scale_factor, kernel=None):
        if scale_factor != 4:
            raise MethodNotApplicableError()

        # in_tensor = torch.from_numpy(np.ascontiguousarray(lr.numpy() * 255.0))
        in_tensor = torch.from_numpy(np.ascontiguousarray(lr.numpy()))
        in_tensor = in_tensor.permute(2, 0, 1).float().unsqueeze(0)
        out_tensor = model(in_tensor)
        out_numpy = out_tensor.data.squeeze().permute(1, 2, 0).float().cpu().numpy()
        # out_numpy = np.clip(out_numpy / 255.0, 0.0, 1.0)
        out_numpy = np.clip(out_numpy, 0.0, 1.0)
        return tf.constant(out_numpy, dtype=tf.float32)

    return superresolver, "esrgan"


def vsr(test_run, args):
    si_superresolver, _ = sisr(test_run, None)

    def video_superresolver(lr_video, ref_index, scale_factor, kernel):
        return si_superresolver(lr_video[ref_index], scale_factor, kernel)

    return video_superresolver, "esrgan"


def lfsr(test_run, args):
    si_superresolver, _ = sisr(test_run, None)

    def lf_superresolver(lr_lf, scale_factor, kernel):
        lf_grid = tf.shape(lr_lf)[:2]
        lf_center = tf.math.floordiv(lf_grid, 2)
        return si_superresolver(lr_lf[lf_center[0], lf_center[1]], scale_factor, kernel)

    return lf_superresolver, "esrgan"
