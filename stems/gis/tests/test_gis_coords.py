""" Tests for :py:mod:`stems.gis.coords`
"""
import math

import affine
from rasterio.coords import BoundingBox
import numpy as np
import pytest

from stems.gis import coords


TRANSFORM_1 = affine.Affine(30.0, 0.0, 365385.0,
                            0.0, -30.0, 2916915.0)
TRANSFORM_2 = affine.Affine(0.5, 0.0, -180.0,
                            0.0, -0.5, 90.0)
TRANSFORMS = [TRANSFORM_1, TRANSFORM_2]

SHAPE_1 = (500, 500)
SHAPE_2 = (360, 180)
SHAPES = [SHAPE_1, SHAPE_2]


param_centered = pytest.mark.parametrize('center', [True, False])


# ============================================================================
@pytest.mark.parametrize(('transform', 'shape', ),
                         zip(TRANSFORMS, SHAPES))
@param_centered
def test_transform_to_coords_shape(transform, shape, center):
    # Run
    y, x = coords.transform_to_coords(transform,
                                      height=shape[1],
                                      width=shape[0],
                                      center=center)
    # Check size
    assert len(y) == shape[1]
    assert len(x) == shape[0]
    # Check upper left
    if center:
        assert y.max() == transform.f + transform.e / 2
        assert x.min() == transform.c + transform.a / 2
    else:
        assert y.max() == transform.f
        assert x.min() == transform.c
    # Check spacing
    assert y[2] - y[1] == transform.e
    assert x[2] - x[1] == transform.a


@pytest.mark.parametrize('transform', TRANSFORMS)
@pytest.mark.parametrize(('off_y', 'off_x'), [
    (10, 5, ),
    (10.3, 5.7, )
])
def test_transform_to_coords_bbox(transform, off_y, off_x):
    left = transform.c + off_x * transform.a
    top = transform.f + off_y * transform.e
    right = left + off_x * transform.a
    bottom = top + off_y * transform.e

    bbox = BoundingBox(left, bottom, right, top)

    y, x = coords.transform_to_coords(transform, bbox=bbox,
                                      center=False)

    # Was the bbox evenly offset?
    if int(off_y) == off_y:
        assert y.max() == top
        assert y.min() + transform.e == bottom
        assert len(y) == off_y
    # Or was it not aligned to grid set by transform?
    else:
        # Should cover extent and have more area
        assert y.max() > top
        assert y.min() + transform.e < bottom
        assert y.max() - y.min() > top - (bottom - transform.e)

    if int(off_x) == off_x:
        assert x.max() + transform.a == right
        assert x.min() == left
        assert len(x) == off_x
    else:
        # Should cover extent and have more area
        assert x.max() + transform.a > right
        assert x.min() < left
        assert x.max() - x.min() > (right - transform.a) - left


def test_transform_to_coords_error():
    transform = TRANSFORMS[0]
    errmsg = r'Must provide either.*bbox.*height.*width.*'
    with pytest.raises(ValueError, match=errmsg):
        y, x = coords.transform_to_coords(transform)


# ============================================================================
@pytest.mark.parametrize('transform', TRANSFORMS)
def test_transform_to_res(transform):
    res = coords.transform_to_res(transform)
    assert res == (transform.a, -transform.e)


# ============================================================================
@pytest.mark.parametrize(('transform', 'shape', ),
                         zip(TRANSFORMS, SHAPES))
def test_transform_to_bounds(transform, shape):
    bounds = coords.transform_to_bounds(transform, shape[1], shape[0])
    assert (bounds.right - bounds.left) / transform.a == shape[0]
    assert (bounds.bottom - bounds.top) / transform.e == shape[1]
    assert bounds.left == transform.c
    assert bounds.top == transform.f


# ============================================================================
@pytest.mark.parametrize(('dy', 'dx', ), [
    [-30, 30],
    [-1, 1],
    [-5, 5]
])
@pytest.mark.parametrize(('ul_y', 'ul_x', ), [
    [0, 0],
    [90, -180],
    [500, 1500]
])
@param_centered
@pytest.mark.parametrize('assume_unique', [True, False])
def test_coords_to_transform(dy, dx, ul_y, ul_x,
                             center, assume_unique):
    y = np.arange(ul_y, ul_y + dy * 7, dy)
    x = np.arange(ul_x, ul_x + dx * 11, dx)
    xform = coords.coords_to_transform(y, x,
                                       center=center,
                                       assume_unique=assume_unique)

    offset = 0.5 if center else 0.0

    assert xform.a == dx
    assert xform.e == dy
    assert xform.c == ul_x - dx * offset
    assert xform.f == ul_y - dy * offset


def test_coords_to_transform_noneven():
    y = np.array([90, 85., 80., 77.5, 75., 72.5, 70.0])
    x = np.array([-180., -175., -170., -165.])
    with pytest.warns(UserWarning, match=r'"y" coord.*equal spacing'):
        xform = coords.coords_to_transform(y, x, center=True)
    with pytest.warns(UserWarning, match=r'"x" coord.*equal spacing'):
        xform = coords.coords_to_transform(x, y, center=True)


@pytest.mark.parametrize(('dy', 'dx', ), [
    (-10, 15, ),
    (-30, 30, )
])
@param_centered
def test_coords_to_bounds(dy, dx, center):
    y = np.arange(115, 10, dy)
    x = np.arange(-500, -300, dx)

    bounds = coords.coords_to_bounds(y, x, center=center)

    if center:
        assert bounds.left == x.min() - dx / 2
        assert bounds.bottom == y.min() + dy / 2
        assert bounds.right == x.max() + dx / 2
        assert bounds.top == y.max() - dy / 2
    else:
        assert bounds.left == x.min()
        assert bounds.bottom - dy == y.min()
        assert bounds.right - dx == x.max()
        assert bounds.top == y.max()


# ============================================================================
@pytest.mark.parametrize(('transform', 'shape', ),
                         zip(TRANSFORMS, SHAPES))
@param_centered
def test_coord_transform_integration(transform, shape, center):
    y, x = coords.transform_to_coords(transform,
                                      height=shape[1],
                                      width=shape[0],
                                      center=center)
    transform_ = coords.coords_to_transform(y, x, center=center)
    assert transform == transform_
