from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='artssat',  # Required
    version='0.0.2',  # Required
    description='High-level simulation and retrieval framework for microwave and IR remote sensing.',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/simonpf/artssat',  # Optional
    author='Simon Pfreundschuh',  # Optional
    author_email='simon.pfreundschuh@chalmers.se',  # Optional
    install_requires=["pyarts", "typhon", "netCDF4"],
    packages=find_packages(exclude=['examples', 'doc', 'misc', 'tests']),
    python_requires='>=3.6',
    project_urls={  # Optional
        'Source': 'https://github.com/simonpf/artssat/',
    })
