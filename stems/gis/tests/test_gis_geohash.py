""" Tests for :py:mod:`stems.gis.geohash`
"""
import dask.array as da
import numpy as np
import pytest
from rasterio.crs import CRS
import xarray as xr

from stems.gis import geohash

needs_geohash = pytest.importorskip('geohash')


CRS_4326 = CRS.from_epsg(4326)
EXAMPLES = pytest.mark.parametrize(('yx', 'crs', 'hash_'), [
    ((42.37453, -71.03193), None, 'drt3pdh1s'),
    ((4693360.95, 332707.50), CRS.from_epsg(32619), 'drt3pdh1s'),
    ((-3.733313, -73.249991), CRS_4326, '6r7fv0kgz'),
])


@EXAMPLES
def test_geohash_encode(yx, crs, hash_):
    z_max = len(hash_)
    for z in range(z_max, 0, -1):
        ans = geohash.geohash_encode(yx[0], yx[1],
                                     crs=crs, precision=z)
        assert ans.item() == hash_[:z]


@EXAMPLES
def test_geohash_decode(yx, crs, hash_):
    ys, xs = geohash.geohash_decode(hash_, crs=crs)

    # error is ~6.8e-4 in lat/lon or ~5m for 9 length
    # https://www.movable-type.co.uk/scripts/geohash.html
    # https://en.wikipedia.org/wiki/Geohash#Algorithm_and_example
    if crs is None or crs == CRS_4326:
        err = 6.8e-4
    else:
        err = 5

    assert abs(ys - yx[0]) < err
    assert abs(xs - yx[1]) < err


param_precision = pytest.mark.parametrize('precision', [8, 10, 12])
param_lat = pytest.mark.parametrize('lat', [da.arange(40, 41, 0.05)])
param_lon = pytest.mark.parametrize('lon', [da.arange(-72, -71, 0.05)])


@param_precision
@param_lat
@param_lon
def test_geohash_dask(precision, lat, lon):
    # Test running as dask
    gh = geohash.geohash_encode(lat, lon, precision=precision)
    assert isinstance(gh, da.Array)
    gh_ = np.array(gh)

    # Should be same as run with ndarray
    compare = geohash.geohash_encode(np.array(lat), np.array(lon),
                                     precision=precision)
    np.testing.assert_equal(np.array(gh_), np.array(compare))

    # Should work both ways
    lat_, lon_ = geohash.geohash_decode(gh)
    np.testing.assert_almost_equal(np.array(lat), np.array(lat_),
                                   decimal=precision // 3)
    np.testing.assert_almost_equal(np.array(lon), np.array(lon_),
                                   decimal=precision // 3)


@param_precision
@param_lat
@param_lon
def test_geohash_xarray(precision, lat, lon):
    lat, lon = xr.DataArray(lat), xr.DataArray(lon)

    # Test running as dask
    gh = geohash.geohash_encode(lat, lon, precision=precision)
    assert isinstance(gh, xr.DataArray)
    gh_ = np.array(gh)

    # Should be same as run with ndarray
    compare = geohash.geohash_encode(np.array(lat), np.array(lon),
                                     precision=precision)
    np.testing.assert_equal(np.array(gh_), np.array(compare))

    # Should work both ways
    lat_, lon_ = geohash.geohash_decode(gh)
    np.testing.assert_almost_equal(np.array(lat), np.array(lat_),
                                   decimal=precision // 3)
    np.testing.assert_almost_equal(np.array(lon), np.array(lon_),
                                   decimal=precision // 3)

    # Should work as scalar
    gh = geohash.geohash_encode(lat[0], lon[0], precision=precision)
    assert isinstance(gh, xr.DataArray)
    np.testing.assert_equal(gh.values, gh_[0])
