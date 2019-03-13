"""Tests for :py:mod:`stems.io.vrt`
"""
from collections import OrderedDict

from affine import Affine
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
import six

from stems.io import vrt
from stems.tests import build_data


# ----------------------------------------------------------------------------
# VRTDataset
def test_VRTDataset_1(image_11w7h4b):
    # Test dataset properties (crs/transform/bounds)
    ds = vrt.VRTDataset(separate=True)
    ds.add_band(image_11w7h4b, 1, vrt_bidx=None)

    with rasterio.open(image_11w7h4b, 'r') as src:
        assert ds.crs == src.crs
        assert ds.bounds == src.bounds
        assert ds.transform == src.transform
        assert ds.shape == src.shape
        assert ds.count == 1


def test_VRTDataset_2(image_11w7h4b):
    # Test vrt_bidx autoincrement
    ds = vrt.VRTDataset(separate=True)
    vrt_bidx = ds.add_band(image_11w7h4b, 1, vrt_bidx=None)
    assert vrt_bidx == 1
    vrt_bidx = ds.add_band(image_11w7h4b, 2, vrt_bidx=None)
    assert vrt_bidx == 2
    assert ds.count == 2


def test_VRTDataset_3(image_11w7h4b_chopped):
    # Test vrt_bidx autoincrement -- don't increment for mosaic
    ds = vrt.VRTDataset(separate=False)
    vrt_bidx = ds.add_band(image_11w7h4b_chopped['ul'], 1, vrt_bidx=None)
    assert vrt_bidx == 1
    vrt_bidx = ds.add_band(image_11w7h4b_chopped['ur'], 1, vrt_bidx=None)
    assert vrt_bidx == 1
    assert ds.count == 1
    assert len(ds._bands[1]) == 2


def test_VRTDataset_5(tmpdir, image_11w7h4b):
    # Test use of `relative`/`relative_to_vrt`
    vrtf = str(tmpdir.join('test.vrt'))

    ds = vrt.VRTDataset(separate=True)
    ds.add_band(image_11w7h4b)

    vrtf_out = ds.write(vrtf, relative=True)

    with rasterio.open(image_11w7h4b) as src:
        dat_ans = src.read(indexes=[1])
    with rasterio.open(vrtf_out) as src:
        dat_test = src.read()

    np.testing.assert_equal(dat_test, dat_ans)


def test_VRTDataset_4(image_11w7h4b):
    # Test ``ds.bands`` order
    ds = vrt.VRTDataset(separate=True)
    ds.add_band(image_11w7h4b, 1, vrt_bidx=2)
    assert ds.count == 1
    ds.add_band(image_11w7h4b, 2, vrt_bidx=1)
    assert ds.count == 2

    bands = ds.bands
    assert isinstance(bands, OrderedDict)
    assert list(bands.keys()) == [1, 2]
    assert all([len(b) == 1 for b in bands.values()])

    # Check insertion is in correct key
    ds.add_band(image_11w7h4b, 3, vrt_bidx=1)
    assert len(ds.bands[1]) == 2
    # Check count is still 2, despite there being 3 input bands
    assert ds.count == 2


def test_VRTDataset_error_nobands():
    ds = vrt.VRTDataset()
    err = r'Cannot determine dataset properties.*'
    for prop in ('crs', 'transform', 'bounds', 'shape', ):
        with pytest.raises(ValueError, match=err):
            getattr(ds, prop)


def test_VRTDataset_error_badbidx(image_11w7h4b):
    # Test vrt_bidx ValueError for bad vrt_bidx
    ds = vrt.VRTDataset(separate=True)
    with pytest.raises(ValueError, match=r'.*greater than 0'):
        ds.add_band(image_11w7h4b, 1, vrt_bidx=0)


def test_VRTDataset_error_notstack(image_11w7h4b):
    # Test vrt_bidx ValueError for mosaic
    ds = vrt.VRTDataset(separate=False)
    with pytest.raises(ValueError, match=r'.*if not stacking.*'):
        ds.add_band(image_11w7h4b, 1, vrt_bidx=2)


def test_VRTDataset_error_diffcrs(tmpdir, image_11w7h4b):
    # Fail on projection difference
    img1 = image_11w7h4b
    img2 = str(tmpdir.join('img.tif'))

    with rasterio.open(img1, 'r') as src:
        meta = src.meta
        meta['crs'] = CRS.from_epsg(4326)
        with rasterio.open(img2, 'w', **meta) as dst:
            dst.write(src.read())

    ds = vrt.VRTDataset(separate=True)
    ds.add_band(img1)
    with pytest.raises(ValueError, match=r'must have same.*crs.*'):
        ds.add_band(img2)


def test_VRTDataset_stack(tmpdir, image_11w7h4b):
    vrtf = str(tmpdir.join('test.vrt'))

    # Create and test count
    ds = vrt.VRTDataset(separate=True)
    for i in range(4):
        ds.add_band(image_11w7h4b, src_bidx=i + 1, vrt_bidx=i + 1)
    assert ds.count == 4

    # Test writing to str
    xml_str = ds.write()
    assert isinstance(xml_str, str)

    # Test writing to file
    vrtf_out = ds.write(vrtf)
    with rasterio.open(vrtf_out) as src:
        assert src.count == ds.count
        assert src.transform == ds.transform
        assert src.bounds == ds.bounds
        assert src.crs == ds.crs
        dat = src.read()

    # Ensure data comes out the same
    dat_test = np.empty_like(dat)
    with rasterio.open(image_11w7h4b) as src:
        dat_test = src.read()
    np.testing.assert_equal(dat, dat_test)


def test_VRTDataset_mosaic_1(image_11w7h4b_chopped):
    # Test bounds with upper half
    info = {}
    for q in ('ul', 'ur', ):
        with rasterio.open(image_11w7h4b_chopped[q]) as src:
            info[q] = {
                'bounds': src.bounds,
                'height': src.height,
                'width': src.width
            }

    ds = vrt.VRTDataset(separate=False)
    ds.add_band(image_11w7h4b_chopped['ul'])
    ds.add_band(image_11w7h4b_chopped['ur'])
    assert ds.bounds.bottom == info['ul']['bounds'].bottom
    assert ds.bounds.top == info['ur']['bounds'].top
    assert ds.bounds.left == info['ul']['bounds'].left
    assert ds.bounds.right == info['ur']['bounds'].right


def test_VRTDataset_mosaic_2(tmpdir, image_11w7h4b_chopped):
    vrtf = str(tmpdir.join('test.vrt'))
    quads = ('ul', 'ur', 'll', 'lr', )
    quad_imgs = list([v for k, v in image_11w7h4b_chopped.items()
                      if k in quads])

    # Should look like the 'all' image
    ds = vrt.VRTDataset.from_bands(quad_imgs, separate=False)
    with rasterio.open(image_11w7h4b_chopped['all']) as src:
        assert ds.bounds == src.bounds
        assert ds.transform == src.transform
        assert ds.shape == src.shape

        # Now write it and test again
        vrtf_out = ds.write(vrtf)
        with rasterio.open(vrtf_out) as vrt_src:
            assert vrt_src.bounds == src.bounds
            assert vrt_src.transform == src.transform
            assert vrt_src.shape == src.shape


# ----------------------------------------------------------------------------
# VRTSourceBand
@pytest.mark.parametrize('params', [
    {}  # default, but parametrized here to make room for later
])
def test_VRTSourceBand_1(tmpdir, params):
    dst = str(tmpdir.join('test.tif'))
    dst_, kwds, meta = build_data.create_test_raster(dst, **params)
    for b in range(meta['count']):
        vrtb = vrt.VRTSourceBand(dst, b + 1)
        # crs
        assert vrtb.crs == meta['crs']
        # transform
        assert vrtb.transform == meta['transform']
        # bounds
        assert vrtb.bounds == meta['bounds']
        # width / height
        assert vrtb.height == meta['height']
        assert vrtb.width == meta['width']
        # dtype
        assert vrtb.dtype == meta['dtype']
        # blockxsize / blockysize
        assert vrtb.blockxsize == meta['blockxsize']
        assert vrtb.blockysize == meta['blockysize']
        # nodata
        assert vrtb.nodata == meta['nodata']
        # description
        assert vrtb.description == meta['description'][b]
        # colorinterp
        assert vrtb.colorinterp == meta['colorinterp'][b]


def test_VRTSourceBand_2(image_11w7h4b):
    # Test overrides
    vrtb = vrt.VRTSourceBand(image_11w7h4b, 1,
                              description='rusrev', nodata=1918)
    assert vrtb.description == 'rusrev'
    assert vrtb.nodata == 1918


def test_VRTSourceBand_open_close(image_11w7h4b):
    vrtb = vrt.VRTSourceBand(image_11w7h4b, 1, keep_open=True)

    # Should have _ds after a `start`
    vrtb.start()
    assert isinstance(vrtb._ds, rasterio.DatasetReader)
    # Should be None after `close`
    vrtb.close()
    assert vrtb._ds is None
    # Context manager -- kept open
    with vrtb.open() as ds:
        assert not ds.closed
    assert not vrtb._ds.closed
    # Close
    vrtb.close()
