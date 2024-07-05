from setuptools import find_packages, setup

required_packages=[
    'flax==0.8.4', 
    'optax>=0.2.2', 
    'jax==0.4.28',
]

setup(
    name='FlaxDiff',
    packages=find_packages(),
    version='0.1.0',
    description='A complete and easy to understand Diffusion library for Generating Images',
    author='Ashish Kumar Singh',
    author_email='ashishkmr472@gmail.com',
    install_requires=required_packages,
)