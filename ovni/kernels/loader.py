import os

import cupy as cp



_cuda_comp_dir = os.path.join(os.path.dirname(__file__), "compiled")


# Loading all functions
mod = cp.RawModule(path=os.path.join(_cuda_comp_dir, "all_kernels.ptx"), backend='ptx')

class Kernels:
    # Pixmft
    nv12_to_rgb = mod.get_function("nv12_to_rgb")
    rgb_to_nv12 = mod.get_function("rgb_to_nv12")

    # Transform
    scale_translate = mod.get_function("scale_translate")
    resize = mod.get_function("resize")
    overlay_opacity = mod.get_function("overlay_opacity")
    blend = mod.get_function("blend")
    blend_4c = mod.get_function("blend_4c")
    rotate = mod.get_function("rotate")

    # Interpolation
    bilinear_blend = mod.get_function("bilinear_blend")
    linear_blend = mod.get_function("linear_blend")

    # Filter
    chroma_key = mod.get_function("chroma_key")
    round_mask = mod.get_function("round_mask")
    
    # Libass
    blend_ass_image = mod.get_function("blend_ass_image")

    # Gaussian blur
    gaussian_blur_horizontal = mod.get_function("gaussian_blur_horizontal")
    gaussian_blur_vertical = mod.get_function("gaussian_blur_vertical")
