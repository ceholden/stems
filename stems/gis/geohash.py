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
from functools import singledispatch

_HAS_GEOHASH = True
try:
    import geohash
except ImportError:
    _HAS_GEOHASH = False

_HAS_DASK = True
try:
    import dask.array as da
except ImportError:
    _HAS_DASK = False

import numpy as np
import pandas as pd
from rasterio.crs import CRS
from rasterio.warp import transform

_HAS_XARRAY = True
try:
    import xarray as xr
except ImportError:
    _HAS_XARRAY = False

from ..compat import requires_module
from ..utils import register_multi_singledispatch

_CRS_4326 = CRS.from_epsg(4326)
_MEM_TYPES = (np.ndarray, pd.Series, )


# =============================================================================
# ENCODE
@requires_module('geohash')
@singledispatch
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
        The geohashes for all Y/X (dtype=``np.dtype(('U', precision))``)
    """
    raise TypeError('Only works on array types')


def _geohash_encode_kernel(y, x, crs=None, precision=12):
    y, x, is_scalar = _guard_scalar_yx(y, x)
    if crs is not None:
        assert isinstance(crs, CRS)
        x, y = transform(crs, _CRS_4326, x, y)

    geohashes = []
    for x_, y_ in zip(x, y):
        gh = geohash.encode(y_, x_, precision=precision)
        geohashes.append(gh)
    geohashes_ = np.asarray(geohashes, dtype=np.dtype(('U', precision)))

    if is_scalar:
        return geohashes_[0]
    else:
        return geohashes_


@register_multi_singledispatch(geohash_encode, _MEM_TYPES + (int, float, ))
def _geohash_encode_mem(y, x, crs=None, precision=12):
    return _geohash_encode_kernel(y, x, crs=crs, precision=precision)


if _HAS_DASK:
    @geohash_encode.register(da.Array)
    def _geohash_encode_dask(y, x, crs=None, precision=12):
        dtype_ = np.dtype(('U', precision))
        sig = '(i),(i)->(i)' if y.shape else '(),()->()'

        ans = da.map_blocks(_geohash_encode_kernel,
                            y, x,
                            dtype=dtype_,
                            crs=crs,
                            precision=precision)
        return ans


if _HAS_XARRAY:
    @geohash_encode.register(xr.DataArray)
    def _geohash_encode_xarray(y, x, crs=None, precision=12):
        dtype_ = np.dtype(('U', precision))
        ans = geohash_encode(y.data, x.data, crs=crs, precision=precision)
        if np.ndim(ans):
            return xr.DataArray(ans, dims=('geohash', ),
                                coords={'geohash': ans},
                                name='geohash')
        else:
            return xr.DataArray(ans, name='geohash')


@requires_module('geohash')
@singledispatch
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
    raise TypeError('Only works for array types')


def _geohash_decode_kernel(geohashes, crs=None):
    geohashes, is_scalar = _guard_scalar_gh(geohashes)
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

    y_ = np.asarray(y, dtype=np.float32)
    x_ = np.asarray(x, dtype=np.float32)

    if is_scalar:
        return y_[0], x_[0]
    else:
        return y_, x_


@register_multi_singledispatch(geohash_decode, _MEM_TYPES + (str, ))
def _geohash_decode_mem(geohashes, crs=None):
    return _geohash_decode_kernel(geohashes, crs=crs)


if _HAS_DASK:
    @geohash_decode.register(da.Array)
    def _geohash_decode_dask(geohashes, crs=None):
        sig = '(i)->(i),(i)' if geohashes.shape else '()->(),()'
        y, x = da.apply_gufunc(_geohash_decode_kernel,
                               sig,
                               geohashes,
                               output_dtypes=[np.float32, np.float32],
                               crs=crs)
        return y, x


if _HAS_XARRAY and _HAS_DASK:
    @geohash_decode.register(xr.DataArray)
    def _geohash_decode_xarray(geohashes, crs=None):
        if isinstance(geohashes.data, da.Array):
            y, x = _geohash_decode_dask(geohashes.data, crs=crs)
        else:
            y, x = _geohash_decode_mem(geohashes.data, crs=crs)

        y_ = xr.DataArray(y, name='y', coords=geohashes.coords)
        x_ = xr.DataArray(x, name='x', coords=geohashes.coords)

        return y_, x_


def _guard_scalar_yx(y, x):
    is_scalar = not getattr(y, 'shape', ())
    if is_scalar:
        y = np.atleast_1d(y)
        x = np.atleast_1d(x)
    assert y.shape == x.shape
    assert y.ndim == 1 and x.ndim == 1
    return y, x, is_scalar


def _guard_scalar_gh(geohashes):
    is_scalar = not getattr(geohashes, 'shape', ())
    if is_scalar:
        geohashes = [geohashes]
    return geohashes, is_scalar
