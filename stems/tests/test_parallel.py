""" Tests for :py:mod:`stems.parallel`
"""
import numpy as np
import pytest
import xarray as xr

from stems import parallel


def test_iter_noncore_chunks_nosize_1(ex_da):
    ans = list(parallel.iter_noncore_chunks(ex_da, ('band', 'time', )))
    assert len(ans) == (ex_da['y'].size * ex_da['x'].size)
    shape = (ex_da['band'].size, ex_da['time'].size)
    assert all([ex_da.isel(**a).shape == shape for a in ans])


def test_iter_noncore_chunks_nosize_2(ex_da):
    ans = list(parallel.iter_noncore_chunks(ex_da, 'time'))
    assert len(ans) == (ex_da['band'].size * ex_da['y'].size * ex_da['x'].size)
    shape = (ex_da['time'].size, )
    assert all([ex_da.isel(**a).shape == shape for a in ans])


def test_iter_noncore_chunks_nosize_3(ex_da):
    # don't consider any dims 'core'
    ans = list(parallel.iter_noncore_chunks(ex_da, ()))
    assert len(ans) == (ex_da['band'].size * ex_da['time'].size *
                        ex_da['y'].size * ex_da['x'].size)
    assert all([ex_da.isel(**a).shape == () for a in ans])


def test_iter_noncore_chunks_size_1(ex_da):
    # time (size=10) split into chunks of 3
    ans = list(parallel.iter_noncore_chunks(ex_da, ('y', 'x', 'band', ), 3))
    assert len(ans) == 4
    ans_ = [ex_da.isel(**a) for a in ans]
    assert sum([_['time'].size for _ in ans_]) == ex_da['time'].size
    ans_concat = xr.concat(ans_, dim='time')
    xr.testing.assert_equal(ex_da, ans_concat)


def test_iter_noncore_chunks_size_2(ex_da):
    # time (size=10) split into chunks of 1
    ans = list(parallel.iter_noncore_chunks(ex_da, ('y', 'x', 'band', ), 1))
    ans_ = list([ex_da.isel(**a) for a in ans])
    assert len(ans) == 10
    assert sum([_['time'].size for _ in ans_]) == ex_da['time'].size
    ans_concat = xr.concat(ans_, dim='time')
    xr.testing.assert_equal(ex_da, ans_concat)


def test_iter_noncore_chunks_size_dict(ex_da):
    # time (size=10) split into chunks of 5 and 'band' in 1
    size = {'time': 5, 'band': 1}
    ans = list(parallel.iter_noncore_chunks(ex_da, ('y', 'x', ), size))
    assert len(ans) == 4
    ans_ = list([ex_da.isel(**a) for a in ans])

    # Same properties as in `test_iter_noncore_chunks_size_2`, but by band
    ans_all_concat = []
    for b in ex_da['band'].values:
        ans_b = [_ for _ in ans_ if _['band'] == b]
        assert len(ans_b) == 2
        assert sum([_['time'].size for _ in ans_b]) == ex_da['time'].size
        # Need `.squeeze` since we're not concat-ing 2 dims
        ans_concat = xr.concat(ans_b, dim='time').squeeze()
        xr.testing.assert_equal(ex_da.sel(drop=False, band=b), ans_concat)
        ans_all_concat.append(ans_concat)

    # Concat all the dims back together -- should match original
    ans_all_concat = xr.concat(ans_all_concat, dim='band')
    xr.testing.assert_equal(ans_all_concat, ex_da)


# =============================================================================
# FIXTURES
@pytest.fixture
def ex_da(request):
    nband = 2
    ntime = 10
    nx, ny = 5, 5
    return xr.DataArray(
        np.ones((nband, ntime, ny, nx, )),
        dims=('band', 'time', 'y', 'x'),
        coords={
            'band': ['B%i' % (i + 1) for i in range(nband)],
            'time': range(ntime),
            'x': range(10, 10 + nx),
            'y': range(-100, -100 + ny)
        }
    )
