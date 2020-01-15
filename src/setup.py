#! /usr/bin/env python
import os
import re
import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages

descr = """Convolutional dictionary learning for noisy signals"""

DISTNAME = 'alphacsc_trend'
DESCRIPTION = descr
MAINTAINER = 'Mainak Jas'
MAINTAINER_EMAIL = 'mainakjas@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/alphacsc/alphacsc.git'


# Function to parse __version__ in `alphacsc`
def find_version():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'alphacsc_trend', '__init__.py'), 'r') as fp:
        version_file = fp.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version()
modules = []
# Create the alphacsc.cython modules
other_modules = [
    "compute_ztX",
    "compute_ztz",
    "coordinate_descent",
    "sparse_conv",
]
for m in other_modules:
    modules.append(
        Extension("alphacsc_trend.cython_code.{}".format(m),
                  sources=["alphacsc_trend/cython_code/{}.pyx".format(m)]))
ext_modules = cythonize(modules)

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        classifiers=[
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],
        platforms='any',
        ext_modules=ext_modules,
        packages=find_packages(exclude=["tests"]),
        setup_requires=['Cython', 'numpy'],
        install_requires=[
            'mne',
            'numba',
            'numpy',
            'scipy',
            'joblib',
            'prox_tv',
            'matplotlib',
            'scikit-learn',
        ],
        include_dirs=[np.get_include()], )
