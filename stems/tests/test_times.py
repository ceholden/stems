""" Tests for :py:mod:`stems.times`
"""
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from stems import times


# =============================================================================
# ordinal_to_datetime64

# =============================================================================
# datetime64_to_pydatetime

# =============================================================================
# datetime64_to_ordinal

# =============================================================================
# datetime64_to_strftime
def test_datetime64_to_strftime():
    times_ = pd.date_range('2000', '2009', freq='YS')
    times_np = times_.values.reshape(2, -1)
    times_np[0, 1] = np.datetime64('NaT')
    times_da = da.from_array(times_np, chunks=2)
    times_xr = xr.DataArray(times_np)
    times_xr_da = xr.DataArray(times_da)

    ans = np.array([
        [20000101, -9999, 20020101, 20030101, 20040101],
        [20050101, 20060101, 20070101, 20080101, 20090101]
    ], dtype=np.int32)

    test = times.datetime64_to_strftime(times_np, strf='%Y%m%d', cast=np.int32)
    np.testing.assert_equal(test, ans)

    test = times.datetime64_to_strftime(times_da, strf='%Y%m%d', cast=np.int32)
    np.testing.assert_equal(test.compute(), ans)

    test = times.datetime64_to_strftime(times_xr, strf='%Y%m%d', cast=np.int32)
    np.testing.assert_equal(test.values, ans)

    test = times.datetime64_to_strftime(times_xr_da, strf='%Y%m%d',
                                        cast=np.int32)
    np.testing.assert_equal(test.values, ans)
