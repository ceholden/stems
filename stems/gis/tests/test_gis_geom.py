""" Tests for :py:mod:`stems.gis.geom`
"""
from rasterio.coords import BoundingBox
import pytest

from stems.gis import geom

BOX_1 = [0, 0, 10, 10]
BOX_2 = [-5, -5, 5, 5]
BOX_3 = [5, 5, 15, 15]


# ----------------------------------------------------------------------------
# bbox_union
@pytest.mark.parametrize(('data', 'ans', ), [
    ((BOX_1, ), BoundingBox(0, 0, 10, 10)),
    ((BOX_1, BOX_2, ), BoundingBox(-5, -5, 10, 10)),
    ((BOX_1, BOX_2, BOX_3), BoundingBox(-5, -5, 15, 15)),
])
def test_bbox_union(data, ans):
    test = geom.bbox_union(*data)
    assert isinstance(test, BoundingBox)
    assert test == ans


# ----------------------------------------------------------------------------
# bbox_intersection
@pytest.mark.parametrize(('data', 'ans', ), (
    ((BOX_1, ), BoundingBox(0, 0, 10, 10)),
    ((BOX_1, BOX_2, ), BoundingBox(0, 0, 5, 5)),
    ((BOX_1, BOX_2, BOX_3), BoundingBox(5, 5, 5, 5)),
))
def test_bbox_intersection(data, ans):
    test = geom.bbox_intersection(*data)
    assert isinstance(test, BoundingBox)
    assert test == ans
