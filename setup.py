from setuptools import setup, find_packages

setup(
    name='sen2cr',
    version='0.0.1',
    url='https://github.com/aoyono/dsen2-cr',
    packages=find_packages(),
    install_requires=[
        'tensorflow-gpu==1.15.0',
        'keras==2.2.4',
        'numpy==1.19.5',
        'scipy==1.7.3',
        'rasterio==1.2.10',
        'pydot==1.4.2',
        'h5py==3.7.0',
    ],
)