import os

import numpy as np
import tensorflow as tf
import torch

from basicsr.archs.edsr_arch import EDSR

from eval_utils.utils import MethodNotApplicableError


PRETRAINED_EDSR = {
    s: os.path.normpath(os.path.join(__file__, "..", "pretrained_models", "EDSR", f))
    for s, f in [
        (2, "EDSR_Lx2_f256b32_DIV2K_official-be38e77d.pth"),
        (3, "EDSR_Lx3_f256b32_DIV2K_official-3660f70d.pth"),
        (4, "EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f.pth"),
    ]
}


def sisr(test_run, args):
    scales = [2, 3, 4]
    models = {s: _load_model(s) for s in scales}

    def superresolver(lr, scale_factor, kernel=None):
        if not scale_factor in scales:
            raise MethodNotApplicableError()
        # in_tensor = torch.from_numpy(np.ascontiguousarray(lr.numpy() * 255.0))
        in_tensor = torch.from_numpy(np.ascontiguousarray(lr.numpy()))
        in_tensor = in_tensor.permute(2, 0, 1).float().unsqueeze(0)
        out_tensor = models[scale_factor](in_tensor)
        out_numpy = out_tensor.data.squeeze().permute(1, 2, 0).float().cpu().numpy()
        # out_numpy = np.clip(out_numpy / 255.0, 0.0, 1.0)
        out_numpy = np.clip(out_numpy, 0.0, 1.0)
        return tf.constant(out_numpy, dtype=tf.float32)

    return superresolver, "edsr"


def vsr(test_run, args):
    si_superresolver, _ = sisr(test_run, None)

    def video_superresolver(lr_video, ref_index, scale_factor, kernel):
        return si_superresolver(lr_video[ref_index], scale_factor, kernel)

    return video_superresolver, "edsr"


def lfsr(test_run, args):
    si_superresolver, _ = sisr(test_run, None)

    def lf_superresolver(lr_lf, scale_factor, kernel):
        lf_grid = tf.shape(lr_lf)[:2]
        lf_center = tf.math.floordiv(lf_grid, 2)
        return si_superresolver(lr_lf[lf_center[0], lf_center[1]], scale_factor, kernel)

    return lf_superresolver, "edsr"


def _load_model(scale):
    model = EDSR(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=256,
        num_block=32,
        upscale=scale,
        res_scale=0.1,
        img_range=255.0,
        rgb_mean=[0.4488, 0.4371, 0.4040],
    )
    model.load_state_dict(
        torch.load(PRETRAINED_EDSR[scale], map_location=torch.device("cpu"))["params"],
        strict=True,
    )
    model.eval()
    return model
