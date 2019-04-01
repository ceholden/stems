""" Tests for :py:mod:`stems.xarray_accessor`
"""
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
import pytest
import xarray as xr

from stems import xarray_accessor  # registers with xarray here


# Shorter variable name
@pytest.fixture
def ds(landsat_ard_subset_dataset):
    return landsat_ard_subset_dataset


def test_xarray_accessor_1(ds):
    crs_ = CRS.from_wkt(ds.coords['crs'].attrs['spatial_ref'])
    xform_ = Affine(30.0, 0.0, -2106255.0, 0.0, -30.0, 1858905.0)
    bounds_ = BoundingBox(-2106255.0, 1858815.0, -2106105.0, 1858905.0)

    # Already georeferenced
    is_georef = ds.stems.is_georeferenced()
    assert is_georef is True

    # Test attributes
    xr.testing.assert_equal(ds.stems.grid_mapping, ds.coords['crs'])
    xr.testing.assert_equal(ds.stems.coord_x, ds.coords['x'])
    xr.testing.assert_equal(ds.stems.coord_y, ds.coords['y'])
    assert ds.stems.crs == crs_
    assert ds.stems.transform == xform_
    assert ds.stems.bounds == bounds_
    bbox_ = ds.stems.bbox
    assert bbox_.area == (bounds_[2] - bounds_[0]) * (bounds_[3] - bounds_[1])


def test_xarray_accessor_2(ds):
    crs_ = CRS.from_wkt(ds.coords['crs'].attrs['spatial_ref'])
    xform_ = Affine(30.0, 0.0, -2106255.0, 0.0, -30.0, 1858905.0)
    bounds_ = BoundingBox(-2106255.0, 1858815.0, -2106105.0, 1858905.0)

    del ds.coords['crs']

    # Not georeferenced anymore...
    is_georef = ds.stems.is_georeferenced()
    assert is_georef is False

    # Add georeferencing
    ds_ = ds.stems.georeference(crs_, xform_)

    # Test attributes
    xr.testing.assert_equal(ds_.stems.grid_mapping, ds_.coords['crs'])
    xr.testing.assert_equal(ds_.stems.coord_x, ds_.coords['x'])
    xr.testing.assert_equal(ds_.stems.coord_y, ds_.coords['y'])
    assert ds_.stems.crs == crs_
    assert ds_.stems.transform == xform_
    assert ds_.stems.bounds == bounds_
    bbox_ = ds_.stems.bbox
    assert bbox_.area == (bounds_[2] - bounds_[0]) * (bounds_[3] - bounds_[1])
