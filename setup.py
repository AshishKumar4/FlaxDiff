from setuptools import find_packages, setup

required_packages=[
    'flax>=0.8.4', 
    'optax>=0.2.2', 
    'jax>=0.4.28',
    'orbax', 
    'clu', 
]

setup(
    name='flaxdiff',
    packages=find_packages(),
    version='0.1.25',
    description='A versatile and easy to understand Diffusion library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Ashish Kumar Singh',
    author_email='ashishkmr472@gmail.com',
    install_requires=required_packages,
)