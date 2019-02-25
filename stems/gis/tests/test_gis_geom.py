""" Tests for :py:mod:`stems.gis.geom`
"""
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.windows import Window
import pytest

from stems.gis import geom

BOX_1 = [0, 0, 10, 10]
BOX_2 = [-5, -5, 5, 5]
BOX_3 = [5, 5, 15, 15]

AFFINE_1m = Affine(1., 0., 0., 0., -1., 10.)
AFFINE_5m = Affine(5., 0., 0., 0., -5., 10.)


# ----------------------------------------------------------------------------
# bounds_union
@pytest.mark.parametrize(('data', 'ans', ), [
    ((BOX_1, ), BoundingBox(0, 0, 10, 10)),
    ((BOX_1, BOX_2, ), BoundingBox(-5, -5, 10, 10)),
    ((BOX_1, BOX_2, BOX_3), BoundingBox(-5, -5, 15, 15)),
])
def test_bounds_union(data, ans):
    test = geom.bounds_union(*data)
    assert isinstance(test, BoundingBox)
    assert test == ans


# ----------------------------------------------------------------------------
# bounds_intersection
@pytest.mark.parametrize(('data', 'ans', ), (
    ((BOX_1, ), BoundingBox(0, 0, 10, 10)),
    ((BOX_1, BOX_2, ), BoundingBox(0, 0, 5, 5)),
    ((BOX_1, BOX_2, BOX_3), BoundingBox(5, 5, 5, 5)),
))
def test_bounds_intersection(data, ans):
    test = geom.bounds_intersection(*data)
    assert isinstance(test, BoundingBox)
    assert test == ans


# ----------------------------------------------------------------------------
# union_bounds_transforms
def test_bounds_tranform_union_1():
    # Should all be the same
    n = 3
    bounds = BoundingBox(365385.0, 2901915.0, 380385.0, 2916915.0)
    transform = Affine(30.0, 0.0, 365385.0, 0.0, -30.0, 2916915.0)

    test = geom.bounds_transform_union((bounds, ) * n, transform)
    assert test[0] == bounds
    assert test[1] == transform


@pytest.mark.parametrize(('bounds', 'transform', 'result'), [
    [(BOX_1, BOX_2,), AFFINE_1m,
     (BoundingBox(-5, -5, 10, 10), Affine(1., 0., -5, 0, -1., 10), (15, 15))],
    [(BOX_1, BOX_2,), AFFINE_5m,
     (BoundingBox(-5, -5, 10, 10), Affine(5., 0., -5, 0, -5., 10), (3, 3))],
    [(BOX_1, BOX_2, BOX_3), AFFINE_1m,
     (BoundingBox(-5, -5, 15, 15), Affine(1., 0., -5, 0, -1., 15), (20, 20))]
])
def test_bounds_tranform_union_2(bounds, transform, result):
    ans = geom.bounds_transform_union(bounds, transform)
    assert len(ans) == 2
    assert isinstance(ans[0], BoundingBox)
    assert isinstance(ans[1], Affine)
    assert ans[0] == result[0]
    assert ans[1] == result[1]


# ----------------------------------------------------------------------------
# calculate_src_window
@pytest.mark.parametrize(('bounds', 'transform', 'dst_bounds', 'result'), [
    (BOX_1, AFFINE_1m, BOX_2, (Window(0, 5, 5, 5), BoundingBox(0, 0, 5, 5))),
    (BOX_1, AFFINE_5m, BOX_2, (Window(0, 1, 1, 1), BoundingBox(0, 0, 5, 5))),
    (BOX_1, AFFINE_5m, BOX_3, (Window(1, 0, 1, 1), BoundingBox(5, 5, 10, 10)))
])
def test_calculate_src_window_exact(bounds, transform, dst_bounds, result):
    ans = geom.calculate_src_window(bounds, transform, dst_bounds)
    assert ans[0] == result[0]
    assert ans[1] == result[1]


def test_calculate_src_window_nonexact():
    bounds = BoundingBox(0, 0, 10, 10)
    transform = Affine(5., 0., 0., 0., -5., 10.)
    dst_bounds = BoundingBox(-1., 0., 12, 13)
    ans = geom.calculate_src_window(bounds, transform, dst_bounds)
    assert ans[0] == Window(0, 0, 2, 2)
    assert ans[1] == bounds


# ----------------------------------------------------------------------------
# calculate_dst_window
def test_calculate_dst_window():
    src_bounds = BoundingBox(0, 0, 120, 120)
    dst_transform = Affine(30., 0., 0,
                           0., -30., 120)
    ans = geom.calculate_dst_window(src_bounds, dst_transform)
    assert ans == Window(0, 0, 4, 4)

    # dst_transform further left & up
    src_bounds = BoundingBox(0, 0, 65, 95)
    dst_transform = Affine(30., 0., -35,
                           0., -30., 155)
    ans = geom.calculate_dst_window(src_bounds, dst_transform)
    assert ans == Window(1, 2, 2, 3)


# ----------------------------------------------------------------------------
# is_null
def test_is_null():
    pass


# ----------------------------------------------------------------------------
# fix_geojson_rfc7946
def test_fix_geojson_rfc7946():
    pass
