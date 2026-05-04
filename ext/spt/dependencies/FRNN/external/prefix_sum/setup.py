import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

class BuildExtension(torch.utils.cpp_extension.BuildExtension):
    def __init__(self, *args, **kwargs):
        super().__init__(use_ninja=False, *args, **kwargs)

setup(
  name="prefix_sum",
  author="Matt Dean, Lixin Xue",
  description="Parallel Prefix Sum on CUDA with Pytorch API",
  ext_modules=[
    CUDAExtension('prefix_sum', ['prefix_sum.cu'])
  ],
  cmdclass={"build_ext": BuildExtension},
)