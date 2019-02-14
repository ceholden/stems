#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from setuptools import find_packages, setup

import versioneer


DISTNAME = 'stems'
VERSION = versioneer.get_version()
CMDCLASS = versioneer.get_cmdclass()
LICENSE="BSD license",

AUTHOR = "Chris Holden"
AUTHOR_EMAIL = 'ceholden@gmail.com'

URL = 'https://github.com/ceholden/stems'
DESCRIPTION = "Spatio-temporal Tools for Earth Monitoring Science"
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: BSD License',
    'Natural Language :: English',
    "Programming Language :: Python :: 2",
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
]

with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
with open('CHANGELOG.rst') as f:
    CHANGELOG = f.read()
ENTRY_POINTS = {
    'console_scripts': [
        'stems=stems.cli:main'
    ]
}



PYTHON_REQUIRES = '>=3.5'
INSTALL_REQUIRES = [
    'dask', 'distributed', 'numpy', 'pandas', 'toolz', 'xarray',
    'affine', 'cartopy', 'fiona', 'gdal', 'rasterio>=1.0.14', 'shapely',
    'pyyaml',
]
INSTALL_REQUIRES.extend(['click>=6.0', 'click-plugins', 'cligj>=0.5'])
SETUP_REQUIRES = ['pytest-runner']
TESTS_REQUIRE = [
    'pytest', 'pytest-cov', 'pytest-lazy-fixture', 'coverage',
    'sphinx', 'sphinx_rtd_theme', 'sphinxcontrib-bibtex'
]
EXTRAS_REQUIRE = {
    'core': INSTALL_REQUIRES,
    'tests': TESTS_REQUIRE
}
EXTRAS_REQUIRE['all'] = sorted(set(sum(EXTRAS_REQUIRE.values(), [])))


setup(
    name=DISTNAME,
    version=VERSION,
    license=LICENSE,
    description=DESCRIPTION,
    long_description='\n'.join([LONG_DESCRIPTION, CHANGELOG]),
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    url=URL,
    packages=find_packages(include=['stems']),
    entry_points=ENTRY_POINTS,
    zip_safe=False,
    include_package_data=True,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    tests_require=TESTS_REQUIRE,
    extras_require=EXTRAS_REQUIRE
)
