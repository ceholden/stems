""" Tests for :py:mod:`stems.io.rasterio_`
"""
from affine import Affine
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS

from stems.gis.tests import data as gis_data
from stems.io import rasterio_
from stems.tests import build_data


EXAMPLE_CRS = [
    gis_data.EXAMPLE_WGS84,
    gis_data.EXAMPLE_AEA_NA,
    gis_data.EXAMPLE_LAEA_NA
]


@pytest.fixture(params=EXAMPLE_CRS)
def example_crs(request):
    return CRS.from_wkt(request.param['wkt'])


EXAMPLE_TRANSFORM = [
    Affine(30., 0., 100., 0., -30., 200),
    Affine(5., 0., 1000., 0., -5., 1500)
]
@pytest.fixture(params=EXAMPLE_TRANSFORM)
def example_transform(request):
    return request.param


# =============================================================================
# xarray_to_rasterio
def test_xarray_to_rasterio_1(tmpdir, example_crs, example_transform):
    kwds = {'ntime': 1}
    ds = build_data.create_test_dataset(crs=example_crs,
                                        transform=example_transform,
                                        **kwds)
    img = ds.squeeze().to_array(dim='band')

    dest = str(tmpdir.join('test.gtif'))
    dest_ = rasterio_.xarray_to_rasterio(img, dest)

    _test_rasterio_xarray(img, example_crs, example_transform, dest_)


def _test_rasterio_xarray(xarr, crs, transform, filename):
    # Determine bands and y/x shape
    bands = list(map(str, np.atleast_1d(xarr.coords['band'])))
    yx_shape = xarr.shape[-2:]

    with rasterio.open(str(filename)) as src:
        # Test shape and count
        assert src.shape == yx_shape
        n_band = len(bands)
        assert src.count == n_band
        # Test georeferencing
        assert src.crs == crs
        assert src.transform == transform
        # Test band descriptions
        descriptions = list(src.descriptions)
        assert descriptions == bands
        # Test data (prefer 2D if shape[0]=1
        data = src.read().squeeze()
        np.testing.assert_equal(data, xarr.values)
