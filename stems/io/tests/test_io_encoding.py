""" Tests for :py:mod:`stems.io.encoding`
"""
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from stems.io import encoding


# ----------------------------------------------------------------------------
# default_encoding
def test_default_encoding_dataarray(dataset_11w7h100t4v):
    data_vars = list(dataset_11w7h100t4v.data_vars.keys())
    for dv in data_vars:
        xarr = dataset_11w7h100t4v[dv]
        enc = encoding.netcdf_encoding(xarr)
        # ensure variable in encoding
        assert dv in enc
        # ensure dtype is correct
        assert enc[dv]['dtype'] == xarr.dtype
        # ensure chunking is correct
        assert enc[dv]['chunksizes'] == (3, 5, 25)
        # no nodata so no _FillValue
        assert '_FillValue' not in enc[dv]
        # compression by default
        assert 'complevel' in enc[dv]
        assert enc[dv]['zlib'] is True


def test_default_encoding_dataset(dataset_11w7h100t4v):
    data_vars = list(dataset_11w7h100t4v.data_vars.keys())
    enc = encoding.netcdf_encoding(dataset_11w7h100t4v)
    for dv in data_vars:
        enc_ = enc[dv]
        # ensure dtype is correct
        assert enc_['dtype'] == dataset_11w7h100t4v[dv].dtype
        # ensure chunking is correct
        assert enc[dv]['chunksizes'] == (3, 5, 25)
        # no nodata so no _FillValue
        assert '_FillValue' not in enc[dv]
        # compression by default
        assert 'complevel' in enc[dv]
        assert enc[dv]['zlib'] is True


def test_default_encoding_dataset(dataset_11w7h100t4v):
    data_vars = list(dataset_11w7h100t4v.data_vars.keys())

    chunks = {'y': 5, 'x': 2, 'time': 20}
    nodata = dict(zip(data_vars, range(len(data_vars))))

    enc = encoding.netcdf_encoding(
        dataset_11w7h100t4v,
        chunk=chunks,
        zlib=False,
        nodata=nodata
    )

    for dv in data_vars:
        enc_ = enc[dv]
        # ensure dtype is correct
        assert enc_['dtype'] == dataset_11w7h100t4v[dv].dtype
        # ensure chunking is correct
        assert enc[dv]['chunksizes'] == (3, 5, 25)
        # _FillValue different in each variable
        assert enc[dv]['_FillValue'] == nodata[dv]
        # compression by default
        assert enc[dv]['zlib'] is False


# -----------------------------------------------------------------------------
# encoding_name
def test_encoding_name():
    xarr = xr.DataArray(np.ones(5))
    assert encoding.encoding_name(xarr) == xr.backends.api.DATAARRAY_VARIABLE
    xarr.name = 'test'
    assert encoding.encoding_name(xarr) == 'test'


# -----------------------------------------------------------------------------
# encoding_dtype
@pytest.mark.parametrize('dtype', (np.int16, np.float32, np.byte,
                                   np.datetime64))
def test_encoding_dtype(dtype):
    xarr = xr.DataArray(np.ones(5, ).astype(dtype))
    ans = encoding.encoding_dtype(xarr)
    assert ans == {'dtype': xarr.dtype}


# -----------------------------------------------------------------------------
# encoding_chunksize
def test_encoding_chunksize_None():
    chunks = (10, 10, )
    xarr = xr.DataArray(da.ones((100, 100), chunks=chunks))
    ans = encoding.encoding_chunksize(xarr, chunks=None)
    assert ans == chunks


def test_encoding_chunksize_dict():
    chunks = (10, 10, )
    xarr = xr.DataArray(da.ones((100, 100), chunks=chunks))
    chunks_ = {dim: 50 for dim in xarr.dims}
    ans = encoding.encoding_chunksize(xarr, chunks=chunks_)
    assert ans == (50, ) * xarr.ndim


# -----------------------------------------------------------------------------
# guard_chunksize
def test_guard_chunksize():
    chunks = (10, 10, )
    xarr = xr.DataArray(da.ones((100, 100), chunks=chunks))
    # Test too big
    ans = encoding.guard_chunksize(xarr, (200, 200, ))
    assert ans == xarr.shape
    # Test passing sizes (no change)
    ans = encoding.guard_chunksize(xarr, chunks)
    assert ans == chunks


# -----------------------------------------------------------------------------
# guard_chunksize_str
def test_guard_chunksize_str():
    # 1D (but really 2D in NetCDF world) array of character
    xarr = xr.DataArray(np.repeat(['asdf'], 5))
    # Should get 2D of chunks out for 1D in
    # (since stored as multiple 1D arrays)
    chunks = (1, )
    test = encoding.guard_chunksize_str(xarr, chunks)
    assert len(test) == 2
    assert test[0] == 1 and test[1] == xarr.dtype.itemsize

    # If object type, just punt
    chunks = (1, )
    test = encoding.guard_chunksize_str(xarr.astype(object), chunks)
    assert len(test) == 0


# -----------------------------------------------------------------------------
# guard_dtype
def test_guard_dtype():
    # some float
    dtype = np.float32
    xarr = xr.DataArray(np.ones(5, ).astype(dtype))
    ans = encoding.guard_dtype(xarr, {'dtype': xarr.dtype})
    assert ans == {'dtype': dtype}

    # datetime should be ignored
    dtype = np.datetime64
    xarr = xr.DataArray(np.ones(5, ).astype(dtype))
    ans = encoding.guard_dtype(xarr, {'dtype': xarr.dtype})
    assert ans == {}
