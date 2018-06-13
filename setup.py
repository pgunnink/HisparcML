from setuptools import setup, find_packages
setup(
    name="ProcessDataML",
    version="0.1",
    install_requires=['tables', 'h5py', 'sapphire', 'matplotlib', 'fast_histogram',
              'numpy', 'keras'],
    packages=find_packages()
)