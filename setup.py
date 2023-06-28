from setuptools import setup, find_packages

setup(
    name='sen2cr',
    version='0.0.1',
    url='https://github.com/aoyono/dsen2-cr',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'scipy',
        'rasterio',
        'pydot',
        'h5py',
        "matplotlib",
        "click"
    ],
)