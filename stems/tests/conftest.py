""" Test fixtures for :py:mod:`stems`
"""
from pathlib import Path

import pytest


DIR_TESTS = Path(__file__).parent.absolute()
LANDSAT_ARD_SUBSET_FNAME = 'LANDSAT_ARD_US_003009_2010-2018_C01_V01.subset.nc'


@pytest.fixture(scope='session')
def landsat_ard_subset_file(request):
    f = DIR_TESTS.joinpath('data', LANDSAT_ARD_SUBSET_FNAME)
    return f


@pytest.fixture(scope='session')
def landsat_ard_subset_dataset(request, landsat_ard_subset_file):
    ds = xr.open_dataset(
        str(landsat_ard_subset_file),
        chunks={
            'y': 2,
            'x': 2,
            'time': 25
        }
    )
    return ds
