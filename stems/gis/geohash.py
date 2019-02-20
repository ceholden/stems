""" Tools for encoding and decoding geohashes

Geohashes are a way of encoding latitude and longitude information into
a single value (hash as a str or uint64) by dividing up the world into
hierarchical quadrants for a given level of precision. The higher the
precision, the more tiers of quadrants are created. When using just one
level of precision, for example, divides the globe into 36 zones (a-z, 0-9).
By adding letters to the hash, the quandrants get increasingly small to the
point where they are useful as global, unique identifiers for pixel coordinates
(assuming you pick a precision that creates quadrant cells smaller than your
image pixel size).

References
----------
.. [1] https://en.wikipedia.org/wiki/Geohash
.. [2] https://www.movable-type.co.uk/scripts/geohash.html


TODO
----
* Implement geohash within this submodule to cut dependencies
* Make faster using Numba (look for 0.43 to deal with dict / str)

"""
_HAS_GEOHASH = True
try:
    import geohash
except ImportError:
    _HAS_GEOHASH = False

import numpy as np
from rasterio.crs import CRS
from rasterio.warp import transform

from ..compat import requires_module

_CRS_4326 = CRS.from_epsg(4326)


@requires_module('geohash')
def geohash_encode(y, x, crs=None, precision=12):
    """ Encode Y/X coordinates into a geohash, reprojecting as needed

    Parameters
    ----------
    y : np.ndarray
        Y coordinates
    x : np.ndarray
        X coordinates
    crs : rasterio.crs.CRS, optional
        If Y/X aren't in latitude/longitude (EPSG:4326), then provide their
        coordinate reference system
    precision : int, optional
        Characters of precision for the geohash

    Returns
    -------
    np.ndarry
        The geohashes for all Y/X
    """
    y = np.atleast_1d(y)
    assert y.ndim == 1
    x = np.atleast_1d(x)
    assert x.ndim == 1

    if crs is not None:
        assert isinstance(crs, CRS)
        x, y = transform(crs, _CRS_4326, x, y)

    geohashes = []
    for x_, y_ in zip(x, y):
        gh = geohash.encode(y_, x_, precision=precision)
        geohashes.append(gh)

    geohashes_ = np.asarray(geohashes)
    return geohashes_


@requires_module('geohash')
def geohash_decode(geohashes, crs=None):
    """ Encode Y/X coordinates into a geohash, reprojecting as needed

    Parameters
    ----------
    np.ndarry
        The geohashes for all Y/X
    crs : rasterio.crs.CRS, optional
        Reproject Y/X latitude/longitude values for each geohash into this CRS
        before returning (to help with end-to-end encode/decode)

    Returns
    -------
    y : np.ndarray
        Y coordinates
    x : np.ndarray
        X coordinates
    """
    if isinstance(geohashes, str):
        geohashes = [geohashes]

    lats, lons = [], []
    for gh in geohashes:
        lat, lon = geohash.decode(gh, delta=False)
        lats.append(lat)
        lons.append(lon)

    if crs is not None:
        assert isinstance(crs, CRS)
        x, y = transform(_CRS_4326, crs, lons, lats)
    else:
        x, y = lons, lats

    return y, x
