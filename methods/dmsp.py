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

    # Denoiser is only applicable to one noise stddev
    if denoiser_min == denoiser_max:
        denoiser_stddev = denoiser_max

    # Denoiser is applicable a range of noise stddev values
    else:
        denoiser_stddev = denoiser_max / 2.0

    return denoiser, denoiser_stddev


def nb_deblurring(test_run, args):
    model_path, noise_mode = args.split(",")
    model_path = _model_path(model_path)
    model_name = _model_name(model_path)
    denoiser, denoiser_stddev = _load_denoiser(model_path)
    num_steps = 10 if test_run else 300

    def deblurrer(blurry, kernel, noise_stddev):

        # Noise adaptive mode: We don't give the noise level to the function
        if noise_mode == "na":
            noise_stddev = None

        return dppp.dmsp_deblur(
            degraded=blurry[None, ...],
            denoiser=denoiser,
            denoiser_stddev=denoiser_stddev,
            kernel=dppp.conv2D_filter_rgb(kernel),
            noise_stddev=noise_stddev,
            num_steps=num_steps,
            conv_mode="wrap",
        )[0]

    return deblurrer, f"dmsp_{model_name}{'-na' if noise_mode == 'na' else ''}", False


def sisr(test_run, args):
    model_path = _model_path(args)
    model_name = _model_name(model_path)
    denoiser, denoiser_stddev = _load_denoiser(model_path)
    bicubic_resize_fn = dppp.create_resize_fn("bicubic", True)
    num_steps = 10 if test_run else 300

    def superresolver(lr, scale_factor, kernel=None):
        if kernel is None:
            # Bicubic
            resize_fn = bicubic_resize_fn
            kernel_rgb = None
        else:
            # Using a downsampling kernel
            resize_fn = None
            kernel_rgb = dppp.conv2D_filter_rgb(kernel)

        return dppp.dmsp_super_resolve(
            degraded=lr[None, ...],
            sr_factor=scale_factor,
            denoiser=denoiser,
            denoiser_stddev=denoiser_stddev,
            resize_fn=resize_fn,
            kernel=kernel_rgb,
            num_steps=num_steps,
            conv_mode="wrap",
        )[0]

    return superresolver, f"dmsp_{model_name}"


def inpainting(test_run, args):
    model_path = _model_path(args)
    model_name = _model_name(model_path)
    denoiser, denoiser_stddev = _load_denoiser(model_path)
    num_steps = 10 if test_run else 300

    def inpainter(image, mask):
        return dppp.dmsp_inpaint(
            image[None, ...],
            mask[None, ...],
            denoiser,
            denoiser_stddev,
            num_steps=num_steps,
        )[0]

    return inpainter, f"dmsp_{model_name}"
