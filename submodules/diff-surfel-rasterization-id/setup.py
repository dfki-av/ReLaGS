#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
# os.path.dirname(os.path.abspath(__file__))


glm_path = os.environ.get("GLM_PATH")
if glm_path is None or not os.path.exists(glm_path):
    raise RuntimeError("Set the GLM_PATH environment variable to the absolute path of third_party/glm")

setup(
    name="diff_surfel_rasterization_id",
    packages=['diff_surfel_rasterization_id'],
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            name="diff_surfel_rasterization_id._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + glm_path]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
# "-I" + os.path.abspath(os.path.join(os.path.dirname(__file__), "../third_party/glm"))