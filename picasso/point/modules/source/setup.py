from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='buildkernel',
    ext_modules=[
        CUDAExtension('buildkernel_cuda', [
            'buildkernel.cpp',
            'buildkernel_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='convolution',
    ext_modules=[
        CUDAExtension('convolution_cuda', [
            'pcloud_conv3d.cpp',
            'pcloud_conv3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='nnquery',
    ext_modules=[
        CUDAExtension('nnquery_cuda', [
            'nnquery.cpp',
            'nnquery_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='pool',
    ext_modules=[
        CUDAExtension('pool_cuda', [
            'pcloud_pool3d.cpp',
            'pcloud_pool3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='sample',
    ext_modules=[
        CUDAExtension('sample_cuda', [
            'sample.cpp',
            'sample_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='unpool',
    ext_modules=[
        CUDAExtension('unpool_cuda', [
            'pcloud_unpool3d.cpp',
            'pcloud_unpool3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })