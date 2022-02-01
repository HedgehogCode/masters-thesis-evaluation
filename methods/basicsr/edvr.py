import os

import numpy as np
import tensorflow as tf
import torch

import tensorflow_datasets_bw as datasets

from basicsr.archs.edvr_arch import EDVR

from eval_utils.utils import MethodNotApplicableError


PRETRAINED_EDVR = os.path.normpath(
    os.path.join(
        __file__,
        "..",
        "pretrained_models",
        "EDVR",
        "EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth",
    )
)
NUM_FRAME = 7


def vsr(test_run, args):
    model = EDVR(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=128,
        num_frame=NUM_FRAME,
        deformable_groups=8,
        num_extract_block=5,
        num_reconstruct_block=40,
    )
    model.load_state_dict(torch.load(PRETRAINED_EDVR)["params"], strict=True)
    model.eval()

    def video_superresolver(lr_video, ref_index, scale_factor, kernel=None):
        if scale_factor != 4:
            raise MethodNotApplicableError()

        # Pad the input to be a multiple of 4
        t, h, w, c = tf.shape(lr_video)
        pad_h = -h % 4
        pad_w = -w % 4
        paddings = [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
        lr_video = tf.pad(lr_video, paddings)

        # Select frames from the video
        start_idx = ref_index - NUM_FRAME // 2
        end_idx = ref_index + NUM_FRAME // 2 + 1
        lr_video = lr_video[start_idx:end_idx, ...]

        # To torch + predict
        in_tensor = torch.from_numpy(np.ascontiguousarray(lr_video.numpy()))
        in_tensor = in_tensor.permute(0, 3, 1, 2).float().unsqueeze(0)
        out_tensor = model(in_tensor)
        out_numpy = out_tensor.data.squeeze().permute(1, 2, 0).float().cpu().numpy()
        out_numpy = np.clip(out_numpy, 0.0, 1.0)

        # Back to TF + crop back padding
        res = tf.constant(out_numpy, dtype=tf.float32)
        res = res[: (h * scale_factor), : (w * scale_factor), :]
        return res

    return video_superresolver, "edvr"


def lfsr(test_run, args):
    video_superresolver, _ = vsr(test_run, None)

    def lf_superresolver(lr_lf, scale_factor, kernel=None):
        lr_video = datasets.lf_to_batch(lr_lf)
        ref_index = tf.shape(lr_video)[0] // 2

        # Only take every 10th image (or less if we don't have enough frames)
        step = min(10, tf.shape(lr_video)[0] // NUM_FRAME)
        lr_video = lr_video[(ref_index % step) :: step]

        return video_superresolver(lr_video, ref_index // step, scale_factor, kernel)

    return lf_superresolver, "edsr"
