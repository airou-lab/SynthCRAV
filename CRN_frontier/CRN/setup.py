import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)
def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


setup(
    name='CRN',
    version='0.0.1',
    author='KAIST',
    author_email='youngseok.kim@kaist.ac.kr',
    description='Code for CRN',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[],
    ext_modules=[
        make_cuda_ext(
            name='average_voxel_pooling_ext',
            module='ops.average_voxel_pooling_v2',
            sources=['src/average_voxel_pooling_forward.cpp'],
            sources_cuda=['src/average_voxel_pooling_forward_cuda.cu'],
        ),
        make_cuda_ext(
            name='voxel_pooling_ext',
            module='ops.voxel_pooling_v2',
            sources=['src/voxel_pooling_forward.cpp'],
            sources_cuda=['src/voxel_pooling_forward_cuda.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
