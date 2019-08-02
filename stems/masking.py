""" Helpers with data masks (clouds, qa/qc, etc)


References
----------

* https://github.com/USGS-EROS/landsat-ldope-tools/blob/master/src/unpack_collection_bits.c

"""
from functools import singledispatch
import logging

import dask.array as da
import numpy as np
import xarray as xr
from xarray.core.computation import apply_ufunc

from .utils import register_multi_singledispatch


@singledispatch
def checkbit(data, offset, width=1, value=3):
    """ Unpack a bit into True/False

    Parameters
    ----------
    data : np.ndarray, da.Array, xr.DataArray
        Bitpacked array data
    offset : int
        Bit offset
    width : int, optional
        Width of bitpacking (e.g., 1=>[0, 1], 2=>[0, 1, 2, 3])
    value : int, optional
        If ``width > 1``, specify the value required to be True (e.g., 3 for
        'high', 2 for 'medium', etc. in some codings)

    Returns
    -------
    array-like, dtype=bool
        True or False value of unpacked bit(s)
    """
    raise TypeError('Only supported for NumPy/Dask arrays or xarray.DataArray')


@checkbit.register(np.ndarray)
def _checkbit_nparray(data, offset, width=1, value=3):
    if data.dtype.kind != 'i':
        data = data.astype(int)
    if width == 1:
        return (np.right_shift(data, offset) & 0x01) == 1
    elif width == 2:
        return (np.right_shift(data,offset) & 0x03) >= value
    else:
        raise ValueError("Only works on 1-2 bit sizes")


@checkbit.register(da.Array)
def _checkbit_darray(data, offset, width=1, value=3):
    return da.map_blocks(_checkbit_nparray,
                         data, offset,
                         width=width, value=value,  # kwargs to function
                         dtype=bool)


@checkbit.register(xr.DataArray)
def _checkbit_xarray(data, offset, width=1, value=3):
    return xr.apply_ufunc(
        _checkbit_nparray,
        data, offset,
        dask='allowed',
        kwargs={'width': width, 'value': value}
    )


def bitpack_to_coding(bitpack, offsets, coding, dtype=None):
    """ Unpack a bipacked QA/QC band to some coding (e.g. CFMask)

    Parameters
    ----------
    bitpack : np.ndarray, dask.array.Array, or xr.DataArray
        Bitpacked data
    offsets : dict
        Dict mapping output mask coding to bitpack (offset, width)
    coding : dict
        Dict mapping offsets to coding values (e.g., ``{4: 4}``)
    dtype : np.dtype, optional
        Output NumPy datatype. If ``None``, output will be same
        datatype as input ``bitpack``

    Returns
    -------
    array
        CFmask look-alike array

    """
    func_where = da.where if isinstance(bitpack, da.Array) else np.where

    offsets_ = list(offsets.values())
    unpack = unpack_bitpack(bitpack, offsets_, dtype=dtype)

    coding_ = empty_like(unpack[..., 0])
    for offset, code in coding.items():
        try:
            idx_ = offsets_.index(offset)
        except ValueError:
            logger.debug(f'Offset label "{label}" not found in coding')
        else:
            coding_ = func_where(unpack[..., idx_], code, coding_)

    return coding_
