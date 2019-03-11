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

import dask.array as da
import numpy as np
import pandas as pd
from rasterio.crs import CRS
from rasterio.warp import transform
import xarray as xr

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

    geohashes_ = np.asarray(geohashes, dtype=np.dtype(('U', precision)))
    return geohashes_


@register_multi_singledispatch(geohash_encode, _MEM_TYPES + (int, float, ))
def _geohash_encode_mem(y, x, crs=None, precision=12):
    return _geohash_encode_kernel(y, x, crs=crs, precision=precision)


@geohash_encode.register(da.Array)
def _geohash_encode_dask(y, x, crs=None, precision=12):
    dtype_ = np.dtype(('U', precision))
    ans = da.apply_gufunc(_geohash_encode_kernel, '(i),(i)->(i)',
                          y, x,
                          output_dtypes=[dtype_],
                          crs=crs,
                          precision=precision)
    return ans


@geohash_encode.register(xr.DataArray)
def _geohash_encode_xarray(y, x, crs=None, precision=12):
    dtype_ = np.dtype(('U', precision))
    ans = xr.apply_ufunc(
        _geohash_encode_kernel, y, x,
        input_core_dims=[(), (),],
        output_core_dims=[()],
        output_dtypes=[dtype_],
        dask='parallelized',
        kwargs={
            'crs': crs,
            'precision': precision
        }
    )
    return ans


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

    y_ = np.asarray(y, dtype=np.float32)
    x_ = np.asarray(x, dtype=np.float32)

    return y_, x_


@register_multi_singledispatch(geohash_decode, _MEM_TYPES + (str, ))
def _geohash_decode_mem(geohashes, crs=None):
    return _geohash_decode_kernel(geohashes, crs=crs)


@geohash_decode.register(da.Array)
def _geohash_decode_dask(geohashes, crs=None):
    y, x = da.apply_gufunc(_geohash_decode_kernel,
                          '(i)->(i),(i)',
                          geohashes,
                          output_dtypes=[np.float32, np.float32],
                          crs=crs)
    return y, x


@geohash_decode.register(xr.DataArray)
def _geohash_decode_xarray(geohashes, crs=None):
    # Can't use xr.apply_ufunc with multiple outputs yet
    # https://github.com/pydata/xarray/issues/1815
#    y, x = xr.apply_ufunc(
#        _geohash_decode_kernel,
#        geohashes,
#        input_core_dims=[()],
#        output_core_dims=[(), ()],
#        output_dtypes=[np.float32, np.float32],
#        dask='parallelized',
#        kwargs={
#            'crs': crs,
#        }
#    )

    if isinstance(geohashes.data, da.Array):
        y, x = _geohash_decode_dask(geohashes.data, crs=crs)
    else:
        y, x = _geohash_decode_mem(geohashes.data, crs=crs)

    y_ = xr.DataArray(y, name='y', coords=geohashes.coords)
    x_ = xr.DataArray(x, name='x', coords=geohashes.coords)

    return y_, x_
