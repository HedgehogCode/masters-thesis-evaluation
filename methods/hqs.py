import os
import dppp


def _model_path(args):
    if args is None:
        raise ValueError("A denoiser must be given.")
    return args


def _model_name(model_path):
    return os.path.basename(model_path)[:-3]


def _load_denoiser(model_path):
    # Load the model
    denoiser, (denoiser_min, denoiser_max) = dppp.load_denoiser(model_path)

    # Check that the denoiser works on a range of noise levels
    if denoiser_min != 0:
        raise ValueError(
            "Model must be able to handle a range of noise levels starting with 0."
        )
    return denoiser, denoiser_max


def nb_deblurring(test_run, args):
    model_path = _model_path(args)
    model_name = _model_name(model_path)
    denoiser, denoiser_max = _load_denoiser(model_path)
    num_steps = 3 if test_run else 8

    def deblurrer(blurry, kernel, noise_stddev):
        return dppp.hqs_deblur_nb(
            degraded=blurry[None, ...],
            kernel=dppp.conv2D_filter_rgb(kernel),
            noise_stddev=noise_stddev,
            denoiser=denoiser,
            max_denoiser_stddev=denoiser_max,
            num_steps=num_steps,
        )[0]

    return deblurrer, f"hqs_{model_name}", False


def sisr(test_run, args):
    model_path = _model_path(args)
    model_name = _model_name(model_path)
    denoiser, denoiser_max = _load_denoiser(model_path)
    bicubic_resize_fn = dppp.create_resize_fn("bicubic", True)
    num_steps = 5 if test_run else 30

    def superresolver(lr, scale_factor, kernel=None):
        if kernel is None:
            # Bicubic
            resize_fn = bicubic_resize_fn
            kernel_rgb = None
        else:
            # Using a downsampling kernel
            resize_fn = None
            kernel_rgb = dppp.conv2D_filter_rgb(kernel)

        return dppp.hqs_super_resolve(
            degraded=lr[None, ...],
            sr_factor=scale_factor,
            denoiser=denoiser,
            max_denoiser_stddev=denoiser_max,
            resize_fn=resize_fn,
            kernel=kernel_rgb,
            num_steps=num_steps,
        )[0]

    return superresolver, f"hqs_{model_name}"


def vsr(test_run, args):
    num_frames = 9  # Do not use all frames
    model_path = _model_path(args)
    model_name = _model_name(model_path)
    denoiser, denoiser_max = _load_denoiser(model_path)
    bicubic_resize_fn = dppp.create_resize_fn("bicubic", True)
    num_steps = 5 if test_run else 30

    # Define the deblurrer using dppp.
    def superresolver(lr_video, ref_index, scale_factor, kernel=None):
        if kernel is None:
            # Bicubic
            resize_fn = bicubic_resize_fn
            kernel_rgb = None
        else:
            # Using a downsampling kernel
            resize_fn = None
            kernel_rgb = dppp.conv2D_filter_rgb(kernel)

        # Crop the video to the desired number of frames
        video_frames = lr_video.shape[0]
        if video_frames > num_frames:
            start_index = max(0, ref_index - (num_frames // 2))
            end_index = start_index + num_frames
            lr_video_cropped = lr_video[start_index:end_index]
            ref_index_cropped = ref_index - start_index
        else:
            lr_video_cropped = lr_video
            ref_index_cropped = ref_index

        return dppp.hqs_video_super_resolve(
            degraded=lr_video_cropped,
            ref_index=ref_index_cropped,
            sr_factor=scale_factor,
            denoiser=denoiser,
            max_denoiser_stddev=denoiser_max,
            resize_fn=resize_fn,
            kernel=kernel_rgb,
            num_steps=num_steps,
        )[0]

    return superresolver, f"hqs_{model_name}"


def lfsr(test_run, args):
    model_path = _model_path(args)
    model_name = _model_name(model_path)
    denoiser, denoiser_max = _load_denoiser(model_path)
    bicubic_resize_fn = dppp.create_resize_fn("bicubic", True)
    num_steps = 5 if test_run else 30

    def superresolver(lr, scale_factor, kernel=None):
        if kernel is None:
            # Bicubic
            resize_fn = bicubic_resize_fn
            kernel_rgb = None
        else:
            # Using a downsampling kernel
            resize_fn = None
            kernel_rgb = dppp.conv2D_filter_rgb(kernel)

        return dppp.hqs_lf_super_resolve(
            degraded=lr,
            sr_factor=scale_factor,
            denoiser=denoiser,
            max_denoiser_stddev=denoiser_max,
            resize_fn=resize_fn,
            kernel=kernel_rgb,
            num_steps=num_steps,
        )[0]

    return superresolver, f"hqs_{model_name}"


def inpainting(test_run, args):
    model_path = _model_path(args)
    model_name = _model_name(model_path)
    denoiser, denoiser_max = _load_denoiser(model_path)
    num_steps = 5 if test_run else 30

    def inpainter(image, mask):
        return dppp.hqs_inpaint(
            image[None, ...],
            mask[None, ...],
            denoiser,
            denoiser_max,
            num_steps=num_steps,
        )[0]

    return inpainter, f"hqs_{model_name}"
