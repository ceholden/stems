""" Tests for :py:mod:`stems.gis.convert`
"""
from affine import Affine
import numpy as np
import pytest
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
import shapely.geometry

from stems.gis import convert


# ============================================================================
# to_crs
EPSG_CODES = [4326, 5070, 32619]
@pytest.fixture(params=EPSG_CODES)
def ex_crs(request):
    return CRS.from_epsg(request.param)


@pytest.mark.parametrize('code', EPSG_CODES)
def test_to_crs_int(code):
    assert convert.to_crs(code) == CRS.from_epsg(code)


def test_to_crs_str(ex_crs):
    # proj4
    assert convert.to_crs(ex_crs.to_string()) == ex_crs
    # wkt
    assert convert.to_crs(ex_crs.wkt) == ex_crs


def test_to_crs_str_error():
    bad = '+proj=wrong'
    msg = 'Could not interpret CRS.*WKT or Proj4'
    with pytest.raises(rasterio.errors.CRSError, match=msg):
        convert.to_crs(convert.to_crs(bad))


def test_to_crs_dict():
    crs = CRS.from_epsg(4326)
    assert convert.to_crs(dict(crs)) == crs


def test_to_crs_OSR(ex_crs):
    from stems.gis import utils
    osr_crs = utils.crs2osr(ex_crs)
    assert convert.to_crs(osr_crs) == ex_crs


def test_to_crs_CRS():
    # should just return same object
    crs = CRS.from_epsg(4326)
    assert convert.to_crs(crs) is crs


# ============================================================================
# to_transform
def test_to_transform_Affine():
    # should just return same object
    transform = Affine(30.0, 0.0, -2102235.0, 0.0, -30.0, 1939455.0)
    assert convert.to_transform(transform) is transform


def test_to_transform_iter():
    transform = Affine(30.0, 0.0, -2102235.0, 0.0, -30.0, 1939455.0)
    # test as a list / tuple / np.ndarray
    assert convert.to_transform(list(transform)) == transform
    assert convert.to_transform(tuple(transform)) == transform
    assert convert.to_transform(np.array(transform)) == transform
    # with "from_gdal"
    geotransform = transform.to_gdal()
    assert convert.to_transform(geotransform, from_gdal=True) == transform


# ============================================================================
# to_bounds
def test_to_bounds_bounds():
    b = BoundingBox(0, 0, 10, 10)
    b_ = convert.to_bounds(b)
    # equal and same object
    assert b == b_ and b is b_


def test_to_bounds_list():
    b = [0, 0, 10, 10]
    bounds = convert.to_bounds(b)
    assert isinstance(bounds, BoundingBox)
    assert list(bounds) == b


def test_to_bounds_geom():
    box = shapely.geometry.box(0, 10, 0, 10)
    bounds = convert.to_bounds(box)
    assert isinstance(bounds, BoundingBox)
    assert bounds == BoundingBox(0, 10, 0, 10)


# ============================================================================
# to_bbox
