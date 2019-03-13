""" Handle chunks/chunksize related logic

Chunks vs chunksizes::

"Chunks" refers to a collection of chunk sizes organized by dimension

    * e.g., ``{'time': (3, 3, 3, 1, )}``
    * For :py:class:`dask.array.Array` and :py:class:`xarray.DataArray`,
      ``.chunks`` is a tuple
    * :py:class:`xarray.Dataset` ``.chunks`` is a mapping

"Chunksizes" refers to a scalar size (an integer) organized by dimension

    * e.g., ``{'time': 3}``
    * ``chunksizes`` is used in encoding for NetCDF4 xarray backend

"""
from collections import Counter, OrderedDict, defaultdict
from functools import singledispatch
import logging
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Read chunks from files
def read_chunks(filename, variables=None):
    """ Return chunks associated with each variable if possible

    Parameters
    ----------
    filename : str
        Read chunks from this file
    variables : Sequence
        Subset of variables to retrieve ``chunking`` for

    Returns
    -------
    Mapping[str, Mapping[str, int]]
        Mapping of variable names to chunks. Chunks are stored
        mapping dimension name to chunksize (e.g., ``{'x': 250}``)

    Raises
    ------
    ValueError
        Raised if no chunks can be determined (unknown file format, etc.)
    """
    read_funcs = (read_chunks_netcdf4, )
    for func in read_funcs:
        try:
            var_chunks = func(filename, variables=variables)
        except Exception as e:
            logger.debug(f'Could not determine chunks for "{filename}" '
                         f'using "{func.__name__}"', exc_info=True)
        else:
            return var_chunks
    raise ValueError(f'Could not determine chunks for "{filename}"')


def read_chunks_netcdf4(filename, variables=None):
    """ Return chunks associated with each variable

    Parameters
    ----------
    filename : str
        Filename of NetCDF file
    variables : Sequence
        Subset of variables to retrieve `chunking` for

    Returns
    -------
    Mapping[str, Mapping[str, int]]
        Mapping of variable names to chunks. Chunks are stored
        mapping dimension name to chunksize (e.g., ``{'x': 250}``)
    """
    # Keep this import inside incase user doesn't have library
    # (e.g., with a minimal install of xarray)
    from netCDF4 import Dataset

    logger.debug(f'Opening "{filename}" as a `netCDF4.Dataset` to read '
                 'saved chunksizes')
    with Dataset(filename, mode='r') as nc:
        # Store info on each chunk: what vars use, and how many
        chunks = OrderedDict()
        variables = variables or nc.variables.keys()
        for name in variables:
            var = nc.variables[name]
            dims = var.dimensions
            chunking = var.chunking()
            if isinstance(chunking, list):
                chunking = OrderedDict((
                    (_dim, _chunk) for _dim, _chunk in
                    zip(dims, chunking)
                    if not _dim.startswith('string')
                ))
            else:
                chunking = None
            chunks[name] = chunking

    return chunks


def read_chunks_rasterio(riods):
    """ Returns chunks for rasterio dataset formatted for xarray

    Parameters
    ----------
    riods : str, pathlib.Path, or rasterio.DatasetReader
        Rasterio dataset or path to dataset

    Returns
    -------
    dict
        Chunks as expected by xarray (e.g., ``{'x': 50, 'y': 50}``)
    """
    if isinstance(riods, (str, Path, )):
        # Keep this import inside incase user doesn't have library
        # (e.g., with a minimal install of xarray)
        import rasterio
        # Open it for ourselves
        with rasterio.open(str(riods), 'r') as riods:
            return read_chunks_rasterio(riods)

    chunks = riods.block_shapes
    if len(set(chunks)) != 1:
        warnings.warn('Block shapes inconsistent across bands. '
                      'Using block shapes from first band')
    chunks = dict(y=chunks[0][0], x=chunks[0][1])

    return chunks


# ----------------------------------------------------------------------------
# Chunk heuristics
def best_chunksizes(chunks, tiebreaker=max):
    """Decide which chunksize to use for each dimension from variables

    Parameters
    ----------
    chunks : Mapping[str, Mapping[str, int]]
        Mapping of variable names to variable chunksizes
    tiebreaker : callable, optional
        Controls what chunksize should be used for a dimension in the event
        of a tie. For example, if 3 variables had a chunksize of 250 and
        another 3 had a chunksize of 500, the guess is determined by
        ``callable([250, 500])``. By default, prefer the larger chunksize
        (i.e., :py:func:`max`)

    Returns
    -------
    dict
        Chunksize per dimension

    Examples
    --------

    >>> chunks = {
    ... 'blu': {'x': 5, 'y': 5},
    ... 'grn': {'x': 5, 'y': 10},
    ... 'red': {'x': 5, 'y': 5},
    ... 'ordinal': None
    }
    >>> best_chunksizes(chunks)
    {'x': 5, 'y': 5}

    """
    # Collect all chunksizes as {dim: [chunksizes, ...]}
    dim_chunks = defaultdict(list)
    for var, var_chunks in chunks.items():
        if var_chunks:
            # Guard if chunks/chunksizes
            dims = list(var_chunks.keys())
            chunksizes = chunks_to_chunksizes(var_chunks)
            for dim, chunksize in zip(dims, chunksizes):
                dim_chunks[dim].append(chunksize)

    guess = {}
    for dim, chunksizes in dim_chunks.items():
        # Use most frequently used chunksize
        counter = Counter(chunksizes)
        max_n = max(counter.values())
        max_val = tuple(k for k, v in counter.items() if v == max_n)

        # If multiple, prefer biggest value (by default)
        if len(max_val) > 1:
            logger.debug('Multiple common chunksizes found. Breaking tie using'
                         f'`{tiebreaker}`')
            pick = tiebreaker(max_val)
        else:
            pick = max_val[0]

        logger.debug(f'Guessing value "{pick}" for dim "{dim}"')
        guess[dim] = pick

    return guess


def auto_determine_chunks(filename):
    """ Try to guess the best chunksizes for a filename

    Parameters
    ----------
    filename : str
        File to read

    Returns
    -------
    dict
        Best guess for chunksizes to use for each dimension
    """
    try:
        var_chunks = read_chunks(str(filename))
    except ValueError:
        logger.debug('"auto" chunk determination failed')
        chunks = None
    else:
        chunks = best_chunksizes(var_chunks)

    return chunks


# ----------------------------------------------------------------------------
# Chunk format handling
@singledispatch
def get_chunksizes(xarr):
    """ Return the chunk sizes used for each dimension in `xarr`

    Parameters
    ----------
    xarr : xr.DataArray or xr.Dataset
        Chunked data

    Returns
    -------
    dict
        Dimensions (keys) and chunk sizes (values)

    Raises
    ------
    TypeError
        Raised if input is not a Dataset or DataArray
    """
    raise TypeError('Input `xarr` must be an xarray Dataset or DataArray, '
                    f'not "{type(xarr)}"')


@get_chunksizes.register(xr.DataArray)
def _get_chunksizes_dataarray(xarr):
    if not xarr.chunks:
        return {}
    return OrderedDict((
        (dim, xarr.chunks[i][0]) for i, dim in enumerate(xarr.dims)
    ))


@get_chunksizes.register(xr.Dataset)
def _get_chunksizes_dataset(xarr):
    if not xarr.chunks:
        return {}
    return OrderedDict((
        (dim, chunks[0]) for dim, chunks in xarr.chunks.items()
    ))


@singledispatch
def chunks_to_chunksizes(data, dims=None):
    """ Convert an object to chunksizes (i.e., used in encoding)

    Parameters
    ----------
    data : xarray.DataArray, dict, or xarray.Dataset
        Input data containing chunk information
    dims : Sequence[str], optional
        Optionally, provide the order in which dimension chunksizes
        should be returned. Useful when asking for chunksizes from
        not-necessarily-ordered data (dicts and Datasets)

    Returns
    -------
    tuple
        Chunk sizes for each dimension. Returns an empty tuple if there
        are no chunks.
    """
    raise TypeError(f'Unknown type for input ``data`` "{type(data)}"')


@chunks_to_chunksizes.register(dict)
def _chunks_to_chunksizes_dict(data, dims=None):
    dims_ = dims or data.keys()
    return tuple(data[d] if isinstance(data[d], int) else data[d][0]
                 for d in dims_)


@chunks_to_chunksizes.register(xr.Dataset)
def _chunks_to_chunksizes_dataset(data, dims=None):
    if not data.chunks:
        return ()
    return _chunks_to_chunksizes_dict(data.chunks, dims=dims)


@chunks_to_chunksizes.register(xr.DataArray)
def _chunks_to_chunksizes_dataarray(data, dims=None):
    if not data.chunks:
        return ()
    if dims:
        dim_idx = [data.dims.index(d) for d in dims]
    else:
        dim_idx = dims or range(len(data.chunks))

    return tuple(data.chunks[i][0] for i in dim_idx)
