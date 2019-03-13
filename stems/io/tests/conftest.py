""" Test fixtures for data IO
"""
import pytest
import rasterio
import rasterio.windows

from stems.tests import build_data


# ----------------------------------------------------------------------------
# Fixtures
@pytest.fixture()
def image_11w7h4b(tmpdir):
    dst = str(tmpdir.join('image_11w7h4b.tif'))
    dst_, rasterio_kwds, meta = build_data.create_test_raster(dst)
    return dst_


@pytest.fixture()
def image_11w7h4b_chopped(tmpdir, image_11w7h4b):
    return build_data.chop_test_image(tmpdir, image_11w7h4b)


@pytest.fixture
def dataset_11w7h100t4v(tmpdir):
    dst = str(tmpdir.join('dataset_11w7h100t4v.nc'))
    dst_ = build_data.create_test_netcdf4(dst)
    return dst_
