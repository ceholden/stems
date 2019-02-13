""" Example data for GIS testing
"""
import pytest

from . import data


# ============================================================================
# Projections
EXAMPLE_CRS = [
    data.EXAMPLE_WGS84,
    data.EXAMPLE_AEA_NA,
    data.EXAMPLE_LAEA_NA
]


@pytest.fixture(params=EXAMPLE_CRS)
def example_crs(request):
    # has keys "name", "longname", "wkt"
    return request.param


@pytest.fixture(params=EXAMPLE_CRS)
def example_crs_wkt(request):
    return request.param['wkt']


@pytest.fixture(params=EXAMPLE_CRS)
def example_crs_OSR(request):
    from osgeo import osr
    sr = osr.SpatialReference()
    sr.ImportFromWkt(request.param['wkt'])
    return sr


@pytest.fixture(params=EXAMPLE_CRS)
def example_crs_CRS(request):
    from rasterio.crs import CRS
    sr = CRS.from_wkt(request.param['wkt'])
    return sr

@pytest.fixture(params=EXAMPLE_CRS)
def example_crs_longname(request):
    return request.param['longname']


@pytest.fixture(params=EXAMPLE_CRS)
def example_crs_cf_name(request):
    return request.param['name']
