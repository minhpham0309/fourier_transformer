from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fourier_layerR',
    ext_modules=[
        CUDAExtension(
            name = 'fourier_R_cuda', 
            sources = ['fourier_R_cuda.cu']),
    ],
    cmdclass={'build_ext': BuildExtension}
    )