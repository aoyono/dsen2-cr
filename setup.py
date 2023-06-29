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
    entry_points={
        "console_scripts": [
            "sen2cr-rm-clouds = sen2cr.dsen2cr_main:remove_clouds",
        ],
    },
)