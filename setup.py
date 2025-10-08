from setuptools import setup, find_packages

setup(
    name='texbank',
    version='0.4.5',
    packages=find_packages('src'),
    package_dir={'': 'src'},
)
