"""Implmenentation of bicubic upsampling for SISR, VSR, and LFSR using TensorFlow 2.
"""
import tensorflow as tf
import dppp


def sisr(test_run, args):
    resize_method = "bicubic"
    resize_antialias = True
    resize_fn = dppp.create_resize_fn(resize_method, resize_antialias)

    def superresolver(lr, scale_factor, kernel=None):
        return resize_fn(lr[None, ...], scale_factor, True)[0]

    return superresolver, "bicubic"


def vsr(test_run, args):
    si_superresolver, _ = sisr(test_run, None)

    def video_superresolver(lr_video, ref_index, scale_factor, kernel=None):
        return si_superresolver(lr_video[ref_index], scale_factor, kernel)

    return video_superresolver, "bicubic"


def lfsr(test_run, args):
    si_superresolver, _ = sisr(test_run, None)

    def lf_superresolver(lr_lf, scale_factor, kernel=None):
        lf_grid = tf.shape(lr_lf)[:2]
        lf_center = tf.math.floordiv(lf_grid, 2)
        return si_superresolver(lr_lf[lf_center[0], lf_center[1]], scale_factor, kernel)

    return lf_superresolver, "bicubic"
