from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, load

setup(
    name='convolution',
    ext_modules=[
        CUDAExtension('convolution_cuda', [
            'mesh_conv3d.cpp',
            'mesh_conv3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='decimation',
    ext_modules=[
        CUDAExtension('decimation_cuda', [
            'decimate.cpp',
            'decimate_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='pool',
    ext_modules=[
        CUDAExtension('pool_cuda', [
            'mesh_pool3d.cpp',
            'mesh_pool3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='unpool',
    ext_modules=[
        CUDAExtension('unpool_cuda', [
            'mesh_unpool3d.cpp',
            'mesh_unpool3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

