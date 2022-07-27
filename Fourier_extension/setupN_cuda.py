from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fourier_N',
    ext_modules=[
        CUDAExtension(
            name = 'fourier_N_cuda', 
            sources = ['fourier_N_cuda.cu']),
    ],
    cmdclass={'build_ext': BuildExtension}
    )