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

from .compat import toolz as tz
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
        checkbit,
        data,
        dask='allowed',
        kwargs={'offset': offset, 'width': width, 'value': value}
    )


@singledispatch
def bitpack_to_coding(bitpack, bitinfo, fill=0, dtype=None):
    """ Unpack a bipacked QA/QC band to some coding (e.g. CFMask)

    Parameters
    ----------
    bitpack : np.ndarray, dask.array.Array, or xr.DataArray
        Bitpacked data
    bitinfo : Dict[int, Sequence[Tuple[Int, Int, Int]]
        A dict mapping output codes to bit unpacking info(s)
        (offsets, widths, and values) for use in unpacking. Should be ordered
        in order of preference with values listed first possibly overwriting
        later ones
    fill : int, float, etc, optional
        Fill value to initialize array with (e.g., your "clear" value, since
        qa/qc bands usually indicate issues with data)
    dtype : np.dtype, optional
        Output NumPy datatype. If ``None``, output will be same
        datatype as input ``bitpack``

    Returns
    -------
    array
        CFmask look-alike array
    """
    raise TypeError('Only supported for NumPy/Dask arrays or xarray.DataArray')


@bitpack_to_coding.register(np.ndarray)
def _bitpack_to_coding_nparray(bitpack, bitinfo, fill=0, dtype=None):
    coding_ = np.full_like(bitpack, fill, dtype=dtype)

    for code, offsets in reversed(list(bitinfo.items())):
        for offset in offsets:
            unpack = checkbit(bitpack, *offset)
            coding_[unpack] = code
    return coding_


@bitpack_to_coding.register(da.Array)
def _bitpack_to_coding_darray(bitpack, bitinfo, fill=0, dtype=None):
    func = tz.curry(_bitpack_to_coding_nparray)(dtype=dtype, fill=fill)
    coding_ = da.map_blocks(
        func, bitpack, bitinfo,
        dtype=dtype)  # this dtype goes to da.map_blocks
    return coding_


@bitpack_to_coding.register(xr.DataArray)
def _bitpack_to_coding_xrarray(bitpack, bitinfo, fill=0, dtype=None):
    out = xr.core.computation.apply_ufunc(
        bitpack_to_coding,
        bitpack,
        dask='allowed',
        kwargs={'bitinfo': bitinfo, 'fill': fill, 'dtype': dtype}
    )
    out.attrs['bitinfo'] = bitinfo
    return out
