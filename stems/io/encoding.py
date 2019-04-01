from collections import OrderedDict
from functools import singledispatch
import logging

import numpy as np
import xarray as xr

from ..utils import list_like
from .chunk import chunks_to_chunksizes

logger = logging.getLogger(__name__)


#: NumPy string types (str, bytes, unicode)
_NP_STRING_TYPES = (np.str_, np.bytes_, np.unicode_, )


@singledispatch
def netcdf_encoding(data,
                    chunks=None,
                    zlib=True,
                    complevel=4,
                    nodata=None,
                    **encoding_kwds):
    """ Return "good" NetCDF encoding information for some data

    The returned encoding is the default or "good" known standard for data
    used in ``stems``. Each default determined in this function is given
    as a keyword argument to allow overriding, and you can also pass additional
    encoding items via ``**encoding``. You may pass one override for all
    data, or overrides for each data variable (as a dict).

    For more information, see the NetCDF4 documentation for the
    ``createVariable`` [1].

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Define encoding for this data. If ``xr.Dataset``, map function across
        all ``xr.DataArray`` in ``data.data_vars``
    dtype : np.dtype, optional
        The data type used for the encoded data. Defaults to the input data
        type(s), but can be set to facilitate discretization based compression
        (typically alongside scale_factor and _FillValue)
    chunks : None, tuple or dict, optional
        Chunksizes used to encode NetCDF. If given as a `tuple`, chunks should
        be given for each dimension. Chunks for dimensions not specified when
        given as a `dict` will default to 1.
    zlib : bool, optional
        Use compression
    complevel : int, optional
        Compression level
    nodata : int, float, or sequence, optional
        NoDataValue(s). Specify one for each ``DataArray`` in ``data``
        if a :py:class:`xarray.Dataset`. Used for ``_FillValue``
    encoding_kwds : dict
        Additional encoding data to pass

    Returns
    -------
    dict
        Dict mapping band name (e.g., variable name) to relevant encoding
        information

    See Also
    --------
    xarray.Dataset.to_netcdf
        Encoding information designed to be passed to
        :py:meth:`xarray.Dataset.to_netcdf`.

    References
    ----------
    .. [1] http://unidata.github.io/netcdf4-python/#netCDF4.Dataset.createVariable
    """
    raise TypeError(f'Unknown type for input ``data`` "{type(data)}"')


@netcdf_encoding.register(xr.DataArray)
def _netcdf_encoding_dataarray(data,
                               dtype=None,
                               chunks=None,
                               zlib=True,
                               complevel=4,
                               nodata=None,
                               **encoding_kwds):
    name = encoding_name(data)
    encoding = {name: {}}

    # dtype: Determine & guard/fixup
    dtype_ = guard_dtype(data, dtype or encoding_dtype(data))
    assert isinstance(dtype_, dict)
    encoding[name].update(dtype_)

    # chunksizes: Determine and guard/fixup
    chunks = encoding_chunksizes(data, chunks=chunks)
    chunks = guard_chunksizes_str(data, guard_chunksizes(data, chunks))
    assert isinstance(chunks, tuple)
    assert all(isinstance(i, int) for i in chunks)
    if chunks:
        encoding[name]['chunksizes'] = tuple(chunks)

    # _FillValue
    if nodata is not None:
        assert isinstance(nodata, (float, int, np.integer, np.floating))
        encoding[name]['_FillValue'] = nodata

    # complevel & zlib: compression
    encoding[name]['complevel'] = complevel
    encoding[name]['zlib'] = zlib

    # Fill in user input
    encoding[name].update(encoding_kwds)

    return encoding


@netcdf_encoding.register(xr.Dataset)
def _netcdf_encoding_dataset(data,
                             dtype=None,
                             chunks=None,
                             zlib=True,
                             complevel=4,
                             nodata=None,
                             **encoding_kwds):
    encoding = OrderedDict()
    for var in data.data_vars:
        # Allow user to specify local (key exists) or global (KeyError) kwds
        kwds = encoding_kwds.get(var, encoding_kwds)

        # Construct, unpacking if needed
        var_encoding = _netcdf_encoding_dataarray(
            data[var],
            dtype=dtype[var] if _is_dict(dtype) else dtype,
            chunks=chunks[var] if _has_key(chunks, var) else chunks,
            zlib=zlib[var] if _is_dict(zlib) else zlib,
            complevel=complevel[var] if _is_dict(complevel) else complevel,
            nodata=nodata[var] if _is_dict(nodata) else nodata,
            **kwds
        )
        encoding[var] = var_encoding[var]

    return encoding


# ----------------------------------------------------------------------------
# Encoding components for DataArray(s)
def encoding_name(xarr):
    """ Return the name of the variable to provide encoding for

    Either returns the name of the DataArray, or the name that XArray will
    assign it when writing to disk
    (:py:data:`xarray.backends.api.DATAARRAY_VARIABLE`).

    Parameters
    ----------
    xarr : xarray.DataArray
        Provide the name of this DataArray used for encoding

    Returns
    -------
    str
        Encoding variable name
    """
    return xarr.name or xr.backends.api.DATAARRAY_VARIABLE


def encoding_dtype(xarr):
    """Get dtype encoding info

    Parameters
    ----------
    xarr : xarray.DataArray or np.ndarray
        DataArray to consider

    Returns
    -------
    dict[str, np.dtype]
        Datatype information for encoding (e.g., ``{'dtype': np.float32}``)

    """
    return {'dtype': xarr.dtype}


def encoding_chunksizes(xarr, chunks=None):
    """ Find/resolve chunksize for a DataArray

    Parameters
    ----------
    xarr : xarray.DataArray
        DataArray to consider
    chunks : tuple[int] or Mapping[str, int]
        Chunks per dimension

    Returns
    -------
    tuple[int]
        Chunksizes per dimension
    """
    if chunks is None:
        # Grab chunks from DataArray
        chunksize = chunks_to_chunksizes(xarr)
    elif isinstance(chunks, dict):
        # Default to 1 chunk per dimension if none found
        chunksize = tuple(chunks.get(dim, len(xarr.coords[dim]))
                          for dim in xarr.dims)
    return chunksize


# ----------------------------------------------------------------------------
# Encoding checks, safeguards, and fixes
def guard_chunksizes(xarr, chunksizes):
    """ Guard chunksize to be <= dimension sizes

    Parameters
    ----------
    xarr : xarray.DataArray
        DataArray to consider
    chunksizes : tuple[int]
        Chunks per dimension

    Returns
    -------
    tuple[int]
        Guarded chunksizes
    """
    assert isinstance(xarr, xr.DataArray) and list_like(chunksizes)
    assert len(chunksizes) == xarr.ndim or not chunksizes

    chunksizes_ = []
    for csize, dim, size in zip(chunksizes, xarr.dims, xarr.shape):
        if csize > size:
            logger.warning(f'Chunk size for dim {dim} is larger than dim '
                           f'size ({size}). Resetting ({csize}->{size})')
            chunksizes_.append(size)
        else:
            chunksizes_.append(csize)

    return tuple(chunksizes_)


def guard_chunksizes_str(xarr, chunksizes):
    """Guard chunk sizes for str datatypes

    Chunks for ``str`` need to include string length dimension since
    python-netcdf represents, for example, 1d char array as 2d array

    Parameters
    ----------
    xarr : xarray.DataArray
        DataArray to consider
    chunksizes : tuple[int]
        Chunk sizes per dimension

    Returns
    -------
    tuple[int]
        Guarded chunk sizes
    """
    if xarr.dtype.type in _NP_STRING_TYPES and chunksizes is not None:
        if len(chunksizes) == xarr.ndim:
            logger.debug('Adding chunk size to `str` dtype variable '
                         f'"{xarr.name}" corresponding to the string length')
            chunksizes = chunksizes + (xarr.dtype.itemsize, )
    elif xarr.dtype.kind == 'O':
        logger.debug('Removing chunks for `dtype=object` variable '
                     f'"{xarr.name}"')
        chunksizes = ()

    return chunksizes


def guard_dtype(xarr, dtype_):
    """Guard dtype encoding for datetime datatypes

    Parameters
    ----------
    xarr : xarray.DataArray or np.ndarray
        DataArray to consider
    dtype_ : dict[str, np.dtype]
        Datatype information for encoding (e.g., ``{'dtype': np.float32}``)

    Returns
    -------
    dict[str, np.dtype]
        Datatype information for encoding (e.g., ``{'dtype': np.float32}``),
        if valid. Otherwise returns empty dict
    """
    # Don't encode datatype for datetime types, since xarray changes it
    if xarr.dtype.kind == 'M':
        dtype_ = {}
    return dtype_


def _is_dict(x):
    return isinstance(x, dict)


def _has_key(d, k):
    return _is_dict(d) and k in d
