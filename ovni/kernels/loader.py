import os

import cupy as cp



_cuda_comp_dir = os.path.join(os.path.dirname(__file__), "compiled")


# Loading all functions
mod = cp.RawModule(path=os.path.join(_cuda_comp_dir, "all_kernels.ptx"), backend='ptx')

class Kernels:
    nv12_to_rgb = mod.get_function("nv12_to_rgb")
    rgb_to_nv12 = mod.get_function("rgb_to_nv12")

    scale_translate = mod.get_function("scale_translate")
    resize = mod.get_function("resize")