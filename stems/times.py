""" Module for handling dates and times
"""
import datetime as dt
from functools import singledispatch
import logging

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from .utils import register_multi_singledispatch

logger = logging.getLogger(__name__)

_NUMBERS = (int, float, )
_LIST_LIKE = (list, tuple, )
_ARRAY_LIKE = (np.ndarray, da.Array, pd.Series, xr.DataArray, )
_PANDAS = (pd.Series, pd.Index, pd.DatetimeIndex, )
_ARRAYS = (np.ndarray, da.Array, )
_XARRAYS = (xr.DataArray, xr.Variable, )


# -----------------------------------------------------------------------------
@singledispatch
def ordinal_to_datetime64(x):
    """ Convert ordinal to datetime64, handling NaN/NaT
    """
    raise TypeError(f'Not supported for type "{type(x)}"')


@register_multi_singledispatch(ordinal_to_datetime64, _NUMBERS)
def _ordinal_to_datetime64_scalar(x):
    if x > 0:
        return np.datetime64(dt.datetime.fromordinal(x))
    else:
        return np.datetime64('NaT')


@register_multi_singledispatch(ordinal_to_datetime64, _LIST_LIKE)
def _ordinal_to_datetime64_sequence(x):
    x_ = [_ordinal_to_datetime64_scalar(x_) for x_ in x]
    return type(x)(x_)


@ordinal_to_datetime64.register(np.ndarray)
def _ordinal_to_datetime64_np(x):
    if x.ndim == 0:
        return _ordinal_to_datetime64_scalar(x.item())
    else:
        return np.array([_ordinal_to_datetime64_scalar(x_) for x_ in x],
                        dtype=np.datetime64)


@register_multi_singledispatch(ordinal_to_datetime64, _PANDAS)
def _ordinal_to_datetime64_pd(x):
    x_ = _ordinal_to_datetime64_np(x.values)
    if isinstance(x, pd.Index):
        return pd.Index(x_, name=x.name)
    else:
        return pd.Series(x_, name=x.name)


@ordinal_to_datetime64.register(da.Array)
def _ordinal_to_datetime64_da(x):
    return da.map_blocks(_ordinal_to_datetime64_np, x, )


@register_multi_singledispatch(ordinal_to_datetime64, _XARRAYS)
def _ordinal_to_datetime64_xarr(x):
    x_ = _ordinal_to_datetime64_array(x.data)
    return xr.DataArray(x_, dims=x.dims, coords=x.coords,
                        attrs=x.attrs, name=x.name)


# -----------------------------------------------------------------------------
@singledispatch
def datetime64_to_pydatetime(x):
    """ Convert datetime64 to Python datetime.datetime
    """
    raise TypeError(f'Not supported for type "{type(x)}"')


@register_multi_singledispatch(datetime64_to_pydatetime, _ARRAYS)
def _datetime64_to_pydatetime_array(x):
    return x.astype('M8[ms]').astype('O')


@register_multi_singledispatch(datetime64_to_pydatetime, _XARRAYS)
def _datetime64_to_pydatetime_xarray(x):
    x_ = _datetime64_to_pydatetime_array(x.data)
    return xr.DataArray(x_, dims=x.dims, coords=x.coords,
                        attrs=x.attrs, name=x.name)


# -----------------------------------------------------------------------------
@singledispatch
def datetime64_to_ordinal(x, name=None):
    """ Convert datetime to ordinal
    """
    raise TypeError(f'Not supported for type "{type(x)}"')


@register_multi_singledispatch(datetime64_to_ordinal, _LIST_LIKE)
def _datetime64_to_ordinal_list(x):
    x_ = [_datetime64_to_ordinal_scalar(x_) for x_ in x]
    return type(x)(x_)


@datetime64_to_ordinal.register(np.ndarray)
def _datetime64_to_ordinal_np(x):
    # Convert to pydatetime first if datetime
    if np.issubdtype(x.dtype, np.datetime64):
        x = datetime64_to_pydatetime(x)
    if x.ndim == 0:
        return np.array(x.item().toordinal())
    else:
        return np.array([x_.toordinal() for x_ in x])


@register_multi_singledispatch(datetime64_to_ordinal, _PANDAS)
def datetime64_to_ordinal_pd(x, name=None):
    x_ = _datetime64_to_ordinal_np(x.values)
    if isinstance(x, pd.Index):
        return pd.Index(x_, name=name or x.name)
    else:
        return pd.Series(x_, name=name or x.name)


@datetime64_to_ordinal.register(da.Array)
def _datetime64_to_ordinal_da(x):
    return da.map_blocks(_datetime64_to_ordinal_np, x, dtype=np.int)


@register_multi_singledispatch(datetime64_to_ordinal, _XARRAYS)
def _datetime64_to_ordinal_xarr(x, name=None):
    x_ = datetime64_to_ordinal(x.data)
    return xr.DataArray(x_, dims=x.dims, coords=x.coords,
                        attrs=x.attrs, name=name or x.name)

# -----------------------------------------------------------------------------
@singledispatch
def datetime64_to_strftime(time, strf='%Y%m%d', cast=np.int32, fill=-9999):
    """ Convert time data to some string format (e.g., 20000101)
    Parameters
    ----------
    time : array-like
        Array of time data
    strf : str, optional
        String format to be used with ``strftime``
    cast : callable
        Function to cast the string format to a number
    fill : int or float
        Fill value to use when ``time`` is NaN or NaT
    Returns
    -------
    array-like
        Ordinal date formatted into a date string numeric representation
    """
    raise TypeError('`time` must be a NumPy, Dask, or Xarray array')


@datetime64_to_strftime.register(np.ndarray)
def _datetime64_to_strftime_np(time, strf='%Y%m%d', cast=np.int32,
                               fill=-9999):
    # to_datetime needs 1D, so reshape after func
    time_ = pd.to_datetime(time.ravel())
    strf_time_ = np.asarray(time_.strftime(strf)).reshape(time.shape)

    # NaN or NaT in `time_` is now 'NaT' in strf_time_, so we need to fill it
    na = pd.isna(time_).reshape(time.shape)
    strf_time_[na] = fill

    # Now it's safe to cast to numeric
    strf_time_ = cast(strf_time_)

    return strf_time_


@datetime64_to_strftime.register(da.Array)
def _datetime64_to_strftime_da(time, strf='%Y%m%d', cast=np.int32,
                               fill=-9999):
    return da.map_blocks(_datetime64_to_strftime_np, time,
                         dtype=cast,
                         strf=strf, cast=cast, fill=fill)


@datetime64_to_strftime.register(xr.DataArray)
def _datetime64_to_strftime_xarray(time, strf='%Y%m%d', cast=np.int32,
                                   fill=-9999):
    time_ = datetime64_to_strftime(time.data, strf=strf, cast=cast, fill=fill)
    return xr.DataArray(time_, dims=time.dims, coords=time.coords,
                        name=time.name, attrs=time.attrs)
