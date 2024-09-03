from setuptools import setup, find_packages, Extension
import subprocess
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
subprocess.run(["pip install pybind11"], shell=True)

proc = subprocess.Popen(["python3 -m pybind11 --includes"], stdout=subprocess.PIPE, shell=True)
(out, err) = proc.communicate()
out = out.decode('ascii').strip().split()

reqs = open('requirements.txt', 'r').read().splitlines()
readme = open('README.md', 'r').read()

setup(
    name='sili',
    version='0.0.3',
    description='SILi: Sparse Intelligence Library',
    long_description=readme,
    url='https://github.com/simleek/SILi',
    author='SimLeek',
    author_email='simulator.leek@gmail.com',
    license='MIT License',
    packages=find_packages(exclude=['test', 'test.*']),
    ext_modules=[
        CppExtension(
            'backend',
            [
                'sili/cpu_backend.cpp'
            ],
            # most optimization work will be done here and in the compiler.
            # Let the compiler use immintrin.h, unless you have extensive tests to prove you beat the compiler.
            extra_compile_args=['-O3', '-Wall', '-shared', '-std=c++20', '-fPIC', *out, '-march=native', '-fopenmp', '-ffast-math'],
            extra_link_args=['lgomp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=reqs,
    include_package_data=True,
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
    ],
)