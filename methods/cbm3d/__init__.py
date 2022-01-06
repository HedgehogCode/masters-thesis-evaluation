import tensorflow as tf
import bm3d


def denoising(test_run, args):
    def denoiser(img, noise_stddev):
        denoised = bm3d.bm3d_rgb(img.numpy(), sigma_psd=noise_stddev)
        return tf.constant(denoised, dtype=tf.float32)

    return denoiser, "cbm3d"
