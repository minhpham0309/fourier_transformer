from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fourier_layer',
    ext_modules=[
        CUDAExtension(
            name = 'fourier_cuda', 
            sources = ['fourier_cuda.cu']),
    ],
    cmdclass={'build_ext': BuildExtension}
)