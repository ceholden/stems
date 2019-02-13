""" Tests for :py:mod:`stems.gis.utils`
"""
from rasterio.crs import CRS
from osgeo import osr
import pytest

from stems.gis import utils
from stems.gis.tests import data

osr.UseExceptions()


# crs2osr
@pytest.mark.parametrize('code', [32619, 4326, 3857])
def test_crs2osr(code):
    # Post WKT change in rasterio=1.0.14 we can test roundtrip
    crs_ = CRS.from_epsg(code)
    sr_ = osr.SpatialReference()
    sr_.ImportFromEPSG(code)

    crs_sr = utils.crs2osr(crs_)
    assert sr_.IsSame(crs_sr)


# same_crs
def test_same_crs_1():
    # literally the same
    crs1 = CRS.from_wkt(data.EXAMPLE_LAEA_NA['wkt'])
    test = utils.same_crs(*(crs1, ) * 3)
    assert test is True

def test_same_crs_2():
    # from 3 different approaches
    crs1 = CRS.from_epsg(3857)
    crs2 = CRS.from_wkt(crs1.wkt)
    crs3 = CRS.from_string(crs1.to_string())

    test = utils.same_crs(crs1, crs2, crs3)
    assert test is True


# osr_same_crs
def test_osr_same_crs():
    crs1 = CRS.from_epsg(3857)
    crs2 = CRS.from_wkt(crs1.wkt)
    crs3 = CRS.from_string(crs1.to_string())

    test = utils.osr_same_crs(crs1, crs2, crs3)
    assert test is True
