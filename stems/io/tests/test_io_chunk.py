""" Tests for :py:mod:`stems.io.chunk`
"""
from collections import OrderedDict
import string

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from stems.io import chunk
from stems.tests import build_data


TEST_PARAMS_CHUNKS = [
    {'y': 5, 'x': 3, 'time': 25},
]



# ----------------------------------------------------------------------------
# read_chunks 
@pytest.mark.parametrize('chunks', TEST_PARAMS_CHUNKS)
def test_read_chunks_netcdf4(tmpdir, chunks):
    data_vars = list('bgrn')
    dst = str(tmpdir.join('test.nc'))
    dst_ = build_data.create_test_netcdf4(dst=dst,
                                          data_vars=data_vars,
                                          chunk_x=chunks['x'],
                                          chunk_y=chunks['y'],
                                          chunk_time=chunks['time'])
    test = chunk.read_chunks(dst_)
    for dv in data_vars:
        assert test[dv] == chunks


# ----------------------------------------------------------------------------
# read_chunks_netcdf4
@pytest.mark.parametrize('chunks', TEST_PARAMS_CHUNKS)
def test_read_chunks_netcdf4(tmpdir, chunks):
    data_vars = list('bgrn')
    dst = str(tmpdir.join('test.nc'))
    dst_ = build_data.create_test_netcdf4(dst=dst,
                                          data_vars=data_vars,
                                          chunk_x=chunks['x'],
                                          chunk_y=chunks['y'],
                                          chunk_time=chunks['time'])
    test = chunk.read_chunks_netcdf4(dst_)
    for dv in data_vars:
        assert test[dv] == chunks


# ----------------------------------------------------------------------------
# read_chunks_rasterio
@pytest.mark.parametrize('chunks', TEST_PARAMS_CHUNKS)
def test_read_chunks_rasterio(tmpdir, chunks):
    dst = str(tmpdir.join('test.tif'))
    dst_, meta, info = build_data.create_test_raster(
        dst,
        height=99, width=49,
        blockysize=chunks['y'],
        blockxsize=49
    )

    # Try letting function open the file
    test = chunk.read_chunks_rasterio(dst_)
    assert test == {'y': chunks['y'], 'x': 49}

    # Try using file handle
    import rasterio
    with rasterio.open(dst_) as riods:
        test = chunk.read_chunks_rasterio(riods)
    assert test == {'y': chunks['y'], 'x': 49}


# ----------------------------------------------------------------------------
# best_chunksizes
def test_best_chunksizes_1():
    # All the same
    d = {
        'blu': {'x': 5, 'y': 5, 'time': 1},
        'grn': {'x': 5, 'y': 5, 'time': 1},
        'red': {'x': 5, 'y': 5, 'time': 1},
    }
    best = chunk.best_chunksizes(d)
    assert best == d['blu']


def test_best_chunksizes_2():
    # One obvious answer
    d = {
        'blu': {'x': 10, 'y': 5, 'time': 5},
        'grn': {'x': 10, 'y': 10, 'time': 1},
        'red': {'x': 5, 'y': 10, 'time': 5},
    }
    best = chunk.best_chunksizes(d)
    assert best == {'x': 10, 'y': 10, 'time': 5}


def test_best_chunksizes_3():
    # Break tie using max
    d = {
        'blu': {'x': 10, 'y': 5, 'time': 5},
        'grn': {'x': 10, 'y': 10, 'time': 10},
        'red': {'x': 5, 'y': 10, 'time': 5},
        'nir': {'x': 5, 'y': 5, 'time': 10}
    }
    best = chunk.best_chunksizes(d)
    assert best == {'x': 10, 'y': 10, 'time': 10}


# ----------------------------------------------------------------------------
# get_chunksizes
@pytest.mark.parametrize('chunks', (
    (1, 5, 10, ),
    (5, 5, ),
))
def test_get_chunksizes_dataarray_1(chunks):
    ndim = len(chunks)
    xarr = xr.DataArray(da.ones((15, ) * ndim, chunks=chunks))
    ans = {dim: n for dim, n in zip(xarr.dims, chunks)}

    test = chunk.get_chunksizes(xarr)
    assert test == ans


@pytest.mark.parametrize('chunks', TEST_PARAMS_CHUNKS)
def test_get_chunksizes_dataarray_2(chunks):
    data_vars = ['blue', 'green', 'red', 'nir']

    ds = build_data.create_test_dataset(data_vars=data_vars,
                                        chunk_x=chunks['x'],
                                        chunk_y=chunks['y'],
                                        chunk_time=chunks['time'])
    for dv in data_vars:
        test = chunk.get_chunksizes(ds[dv])
        assert test == chunks


def test_get_chunksizes_dataarray_3():
    xarr = xr.DataArray(np.ones((15, ) * 3))
    test = chunk.get_chunksizes(xarr)
    assert test == {}


def test_get_chunksizes_dataset_1():
    xarr = xr.DataArray(np.ones((15, ) * 3))
    ds = xr.Dataset({'a': xarr})

    test = chunk.get_chunksizes(ds)
    assert test == {}


@pytest.mark.parametrize('test', ({'ten': 10}, (10, ), 10, '10', ))
def test_get_chunksizes_TypeError(test):
    err = r'Input.*must be an xarray Dataset or DataArray.*'
    with pytest.raises(TypeError, match=err):
        chunk.get_chunksizes(test)


# ----------------------------------------------------------------------------
# chunks_to_chunksizes
@pytest.mark.parametrize('chunksizes', (
    (10, 5, 1, ),
    (5, 1, )
))
def test_chunks_to_chunksizes_dict(chunksizes):
    keys = string.ascii_letters[:len(chunksizes)]
    d = OrderedDict(((k, (size, ) * 3) for k, size in zip(keys, chunksizes)))
    test = chunk.chunks_to_chunksizes(d)
    assert test == chunksizes


def test_chunks_to_chunksizes_dataset():
    ds = build_data.create_test_dataset(chunk_x=2, chunk_y=4, chunk_time=30)
    test = chunk.chunks_to_chunksizes(ds, dims=('y', 'time', 'x', ))
    assert test == (4, 30, 2)


@pytest.mark.parametrize('chunksizes', (
    (10, 5, 1, ),
    (5, 1, )
))
def test_chunks_to_chunksizes_dataarray(chunksizes):
    ndim = len(chunksizes)
    xarr = xr.DataArray(da.ones((100, ) * ndim, chunks=chunksizes))
    test = chunk.chunks_to_chunksizes(xarr)
    assert test == chunksizes


def test_chunks_to_chunksizes_none():
    # DataArray
    xarr = xr.DataArray(np.ones(10))
    test = chunk.chunks_to_chunksizes(xarr)
    assert test == ()
    # Dataset
    ds = xr.Dataset({'x': xarr})
    test = chunk.chunks_to_chunksizes(ds)
    assert test == ()


@pytest.mark.parametrize('test', ((10, ), 10, '10', ))
def test_chunks_to_chunksizes_TypeError(test):
    with pytest.raises(TypeError, match=r'Unknown type.*'):
        chunk.chunks_to_chunksizes(test)
