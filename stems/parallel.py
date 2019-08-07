""" Helper tools to facilitate turning computation into dask-able gufuncs

References
----------
[1] https://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
[2] http://numba.pydata.org/numba-doc/dev/user/vectorize.html
"""
from collections import defaultdict
import functools
from itertools import product
import logging

import numpy as np
import six
import xarray as xr

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Dimension helpers
def iter_chunks(dim_sizes, chunk_sizes=None):
    """ Return slices that help iterate over dimensions in chunks/blocks

    Parameters
    ----------
    dim_sizes : dict[str, int]
        Dimension sizes
    chunk_sizes : None, int, or dict[str, int]
        Sizes of chunks. If ``int``, will use the same size for all dimensions.
        If ``None``, will iter over each element in all dimensions (e.g., one
        pixel)

    Yields
    -------
    dict[str, slice or int]
        Mapping of dimension name to select statement (e.g.,
        ``{'x': slice(0, 50), 'y': slice(0, 50)}``)
    """
    # TODO: dispatch on dict/tuple so this works more easily with DataArray
    dims = tuple(dim_sizes.keys())
    if chunk_sizes is not None:
        # Parse chunks
        if isinstance(chunk_sizes, int):
            size = {d: chunk_sizes for d in dims}
        elif isinstance(chunk_sizes, dict):
            # Ensure all dims are represented, defaulting to 1
            size = {d: chunk_sizes.get(d, 1) for d in dims}
        else:
            raise TypeError('`chunk_sizes` must be None, int, or dict')
        # Create all possible slices
        coords = product(*(
            (slice(start, stop) for start, stop in
             _zip_iter_seq_window(np.arange(dim_sizes[dim]), size[dim]))
            for dim in dims
        ))
    else:
        # Just 1 item -- no slice needed
        coords = product(*(range(dim_sizes[dim]) for dim in dims))

    # Yield out coordinates
    for coords_ in coords:
        yield {dim: coord for dim, coord in zip(dims, coords_)}


def iter_noncore_chunks(data, core_dims, chunk_sizes=None):
    """ Yield data in chunks selected out from non-core dimensions

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Data
    core_dims : str or list[str]
        One or more dimensions to exclude from selection. These dimensions
        will be the only dimensions on the data returned
    chunk_size : int, dict[str, int], or None
        Chunk/block/window size when iterating over noncore dimensions. If
        ``None``, will return a slice for each element in each dimension.

    Yields
    ------
    dict[str, slice]
        ``isel`` select statements to retrieve data
    """
    if isinstance(core_dims, str):
        core_dims = (core_dims, )

    # "Non-core" are remaining dims on data
    noncore = set(data.dims) - set(core_dims)
    # Keep same order as data.dims so this is deterministic
    noncore_sizes = {dim: data[dim].size for dim in data.dims if dim in noncore}

    for i in iter_chunks(noncore_sizes, chunk_sizes):
        yield i


def map_collect_1d(core_dims, arg_idx=0, concat_axis=0,
                   concat_func=np.concatenate):
    """ Decorator that maps a function across pixels and concatenates the result

    Parameters
    ----------
    core_dims : Sequence[str]
        Core dimensions to exclude from broadcast/selection. For example,
        if applying a time series model over all x/y in a DataArray,
        the ``core_dim`` could be 'time'.
    arg_idx : int or tuple[int], optional
        Index(es) of ``*args`` to pick as the data to iterate over.
    concat_axis : int, optional
        Axis along which the arrays will be joined (passed to ``concat_func``)
    concat_func : callable, optional
        Function used to concatenate the data. Should take the ``axis`` keyword

    Returns
    -------
    callable
        Decorator for function (parametrized by ``core_dims`` and ``arg_idx``)
    """
    if isinstance(arg_idx, (int, )):
        arg_idx = (arg_idx, )

    def decorator(func):
        @functools.wraps(func)
        def inner(*args, **kwds):
            # Extract from args and check
            data = [args[i] for i in arg_idx]
            assert all(isinstance(d, (xr.Dataset, xr.DataArray, ))
                       for d in data)

            # Find noncore dim names and sizes to iterate over
            dim_sizes = _check_data_dims(data)
            noncore_dims = tuple(d for dat in data for d in dat.dims
                                 if d not in core_dims)
            noncore_dims_sizes = {k: dim_sizes[k] for k in noncore_dims}

            # `None` for chunk_sizes defaults to 1 and avoids slices
            iter_ = iter_chunks(noncore_dims_sizes, None)

            # Store arguments so we can replace some with data post-selection
            func_args = list(args)

            results = []
            for window in iter_:
                # Select data
                data_ = [_isel_n_squeeze(dat, noncore_dims, **window)
                         for dat in data]

                # Replace `*args` with subset version(s) of `data`
                for dat_i, arg_i in enumerate(arg_idx):
                    func_args[arg_i] = data_[dat_i]

                # TODO: delayed?
                # Run and collect
                result = func(*func_args, **kwds)
                results.append(result)

            # Concatenate and return
            results_ = concat_func(results, axis=concat_axis)
            return results_

        return inner

    return decorator


def _iter_seq(d, offset, spacing):
    i = range(offset * spacing, len(d) + offset * spacing, spacing)
    return (min(i_, len(d)) for i_ in i)


def _zip_iter_seq_window(d, spacing):
    return zip(_iter_seq(d, 0, spacing), _iter_seq(d, 1, spacing))


def _check_data_dims(data):
    # Populate all dimensions organized by name
    dims_ = defaultdict(list)
    for dat in data:
        for key in dat.dims:
            dims_[key].append(dat[key])

    # Ensure all dimensions match
    for key, dim in dims_.items():
        eq = [np.array_equal(a, b) for a, b in zip(dim, dim[1:])]
        if not all(eq):
            raise ValueError(f'Data dimensions for "{key}" do not all match')

    # Finally just return the sizes for each dimension
    dim_sizes = {k: vals[0].size for k, vals in dims_.items()}
    return dim_sizes


def _isel_n_squeeze(xarr, noncore_dims, **isel):
    isel_ = {k: i for k, i in isel.items() if k in xarr.dims}
    item = xarr.isel(**isel_)
    squeeze_dims = tuple(d for d in noncore_dims if
                         (d in item.dims and item[d].size == 1))
    if squeeze_dims:
        item = item.squeeze(dim=squeeze_dims)
    return item
