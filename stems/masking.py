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
    if width == 1:
        return ((data >> offset) & 0x01) == 1
    elif width == 2:
        return ((data >> offset) & 0x03) >= value
    else:
        raise ValueError("Only works on 1-2 bit sizes")


@singledispatch
def unpack_bitpack(data, offsets, dtype=None):
    """ Unpack bitpacked data (e.g., a QA/QC band)

    Parameters
    ----------
    data : np.ndarray, dask.array.Array, or xr.DataArray
        Bitpacked data
    offsets : Sequence[(Int, Int)]
        Bit offsets and widths to unpack
    dtype : dtype, optional
        Return dtype. If not specified, defaults to input
        data datatype

    Returns
    -------
    unpacked_data : np.ndarray, dask.array.Array, or xr.DataArray
        Data unpacked from bits into multiple arrays (last dim). Return type
        dependent on input.
    """
    raise TypeError('Cannot unpack data of type "{0}"'.format(type(data)))


@register_multi_singledispatch(unpack_bitpack, (np.ndarray, da.Array))
def _unpack_bitpack_array(data, offsets, dtype=None):
    stack_func = da.stack if isinstance(data, da.Array) else np.stack
    dtype = dtype if dtype is not None else data.dtype

    out = []
    for offset, width in offsets:
        match = checkbit(data, offset, width=width)
        out.append(match)
    out = stack_func(out, axis=-1)
    return out


@unpack_bitpack.register(xr.DataArray)
def _unpack_bitpack_xarray(data, offsets, dtype=None):
    out = xr.core.computation.apply_ufunc(
        unpack_bitpack,
        data,
        offsets,
        dask='allowed',
        output_core_dims=[['offset']],
        kwargs={'dtype': dtype}
    )
    out.coords['offset'] = [i[0] for i in offsets]
    return out


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
