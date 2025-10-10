from setuptools import setup, find_packages

setup(
    name='texbank',
    version='0.5.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
)
