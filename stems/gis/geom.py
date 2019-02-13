""" Geometry related functions
"""
from rasterio.coords import BoundingBox

from ..utils import list_like


def bbox_union(*bounds):
    """Return the union of some BoundingBox(s)

    Parameters
    ----------
    bounds : BoundingBox
        Bounding boxes

    Returns
    -------
    BoundingBox
        The union of all input BoundingBox
    """
    assert len(bounds) > 0
    assert all([isinstance(b, BoundingBox) or
                (list_like(b) and len(b) == 4)
                for b in bounds])

    xs, ys = [], []
    for b in bounds:
        left, bottom, right, top = b
        xs.extend([left, right])
        ys.extend([bottom, top])
    return BoundingBox(min(xs), min(ys), max(xs), max(ys))


def bbox_intersection(*bounds):
    """Return the intersection of some BoundingBox(s)

    Parameters
    ----------
    bounds : BoundingBox
        Bounding boxes

    Returns
    -------
    BoundingBox
        The intersection of all input BoundingBox
    """
    assert len(bounds) > 0
    assert all([isinstance(b, BoundingBox) or
                (list_like(b) and len(b) == 4)
                for b in bounds])

    left = max([b[0] for b in bounds])
    bottom = max([b[1] for b in bounds])
    right = min([b[2] for b in bounds])
    top = min([b[3] for b in bounds])

    return BoundingBox(left, bottom, right, top)
