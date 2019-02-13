""" Tests for :py:mod:`stems.gis.projections`
"""
from osgeo import osr
from rasterio.crs import CRS
import pytest

from stems.gis import projections
from stems.gis.tests import data

PROJ_EPSG = [4326, 3857, 32619]


# ============================================================================
# EPSG code
@pytest.mark.parametrize('code', [5070, 4326, 32619])
def test_epsg_code(code):
    # test rasterio
    crs_ = CRS.from_epsg(code)
    test = projections.epsg_code(crs_)
    assert test == f'epsg:{code}'
    # test osr
    crs_ = osr.SpatialReference()
    crs_.ImportFromEPSG(code)
    test = projections.epsg_code(crs_)
    assert test == f'epsg:{code}'


@pytest.mark.parametrize('wkt', [
    data.EXAMPLE_LAEA_NA['wkt'],
    data.EXAMPLE_AEA_NA['wkt'],
])
def test_epsg_code_fail(wkt):
    crs_ = CRS.from_wkt(wkt)
    with pytest.raises(ValueError, match=r'Cannot decode.*'):
        projections.epsg_code(crs_)


def test_crs_longname(example_crs):
    crs_ = CRS.from_wkt(example_crs['wkt'])
    test = projections.crs_longname(crs_)
    assert test == example_crs['longname']


# =============================================================================
# CF info / parameters
def test_cf_crs_name(example_crs):
    crs_ = CRS.from_string(example_crs['wkt'])
    test = projections.cf_crs_name(crs_)
    assert test == example_crs['cf_name']
