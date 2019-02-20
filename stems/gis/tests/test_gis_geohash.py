""" Tests for :py:mod:`stems.gis.geohash`
"""
import pytest
from rasterio.crs import CRS

from stems.gis import geohash

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
        assert ans[0].item() == hash_[:z]


@EXAMPLES
def test_geohash_deencode(yx, crs, hash_):
    ys, xs = geohash.geohash_decode(hash_, crs=crs)

    # error is ~6.8e-4 in lat/lon or ~5m for 9 length
    # https://www.movable-type.co.uk/scripts/geohash.html
    # https://en.wikipedia.org/wiki/Geohash#Algorithm_and_example
    if crs is None or crs == CRS_4326:
        err = 6.8e-4
    else:
        err = 5

    assert abs(ys[0] - yx[0]) < err
    assert abs(xs[0] - yx[1]) < err
