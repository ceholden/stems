""" Tests for :py:mod:`stems.gis.conventions`
"""
from affine import Affine
import numpy as np
from rasterio.crs import CRS
import pytest
import xarray as xr

from stems.gis import conventions


# ----------------------------------------------------------------------------
# georeference / is_georeferenced
def test_is_georeferenced():
    # Create ungeoreferenced data
    a = xr.DataArray(np.ones((5, 5)), dims=('y', 'x', ),
                     coords={'x': np.arange(5), 'y': np.arange(5)})
    ds = xr.Dataset({'a': a})

    ans = conventions.is_georeferenced(a)
    assert ans is False
    ans = conventions.is_georeferenced(ds)
    assert ans is False

    crs_ = CRS.from_epsg(32619)
    transform_ = Affine(1, 0, 0, 0, -1., 5)

    # Create georeferenced data
    a_ = conventions.georeference(a, crs_, transform_)
    ds_ = conventions.georeference(ds, crs_, transform_)

    ans = conventions.is_georeferenced(a_)
    assert ans is True
    ans = conventions.is_georeferenced(ds_)
    assert ans is True


# ----------------------------------------------------------------------------
# create_grid_mapping
utm19n = {
    'crs': CRS.from_epsg(32619),
    'attrs': {
        'grid_mapping_name': 'transverse_mercator',
        'projected_coordinate_system_name': 'WGS 84 / UTM zone 19N',
        'inverse_flattening': 298.257223563,
        'reference_ellipsoid_name': 'WGS 84',
        'false_easting': 500000,
    }
}
conus_usgs = {
    'crs': CRS.from_epsg(5070),
    'attrs': {
        'grid_mapping_name': 'albers_conic_equal_area',
        'projected_coordinate_system_name': 'NAD83 / Conus Albers',
        'inverse_flattening': 298.257222101,
        'reference_ellipsoid_name': 'GRS 1980',
        'false_easting': 0.0,
    }
}

@pytest.mark.parametrize(('crs', 'attrs', ), [
    (utm19n['crs'], utm19n['attrs'], ),
    (conus_usgs['crs'], conus_usgs['attrs'], )
])
@pytest.mark.parametrize('name', ['one', 'two'])
def test_create_grid_mapping_utm(crs, attrs, name):
    transform = Affine(30, 0, -130,
                       0, -30, 170)
    gridmap = conventions.create_grid_mapping(crs, transform, name)

    assert gridmap.name == name

    if crs.to_epsg():
        assert gridmap.item() == crs.to_epsg()

    for k, v in attrs.items():
        assert k in gridmap.attrs
        assert gridmap.attrs[k] == v

    # Check GDAL attributes
    assert gridmap.attrs['spatial_ref'] == crs.wkt
    assert np.array_equal(gridmap.attrs['GeoTransform'],
                          transform.to_gdal())


# ----------------------------------------------------------------------------
# xarray_coords
yx = pytest.mark.parametrize(('y', 'x', ), [
    (np.arange(0, 10), np.arange(-10, -20)),
    (np.arange(50, 65), np.arange(-10, -20))
])


@yx
def test_create_coordinates_utm19n(y, x):
    crs = CRS.from_epsg(32619)
    y_, x_ = conventions.create_coordinates(y, x, crs)

    assert np.array_equal(x_.data, x)
    assert x_.data is x
    assert np.array_equal(y_.data, y)
    assert y_.data is y

    for s, v in [('y', y_, ), ('x', x_, )]:
        assert v.attrs['standard_name'] == f'projection_{s}_coordinate'
        assert v.attrs['long_name'] == f'{s} coordinate of projection'
        assert v.attrs['units'].startswith('m')


# ============================================================================
# Test integration with GDAL
# TODO: write to NetCDF4 and try to retrieve info from GDAL
