""" Coordinate and geotransforms
"""
import logging
import math
import warnings

from affine import Affine
import numpy as np
from rasterio.coords import BoundingBox

logger = logging.getLogger(__name__)


# ============================================================================
# From transform
def transform_to_coords(transform,
                        bbox=None,
                        width=None, height=None,
                        center=True):
    """ Return the coordinates for a given transform

    This function needs to know how many pixels are in the raster,
    so either a :py:class:`rasterio.coords.BoundingBox` or
    height and width arguments must be supplied.

    Parameters
    ----------
    transform : affine.Affine
        Affine transform
    bbox : BoundingBox
        Return the coordinates for this transform that are contained inside
        the provided bounds (left, bottom, right, up). Coordinates will
        be aligned/spaced according to the transform and will not necessarily
        match the upper-left of this ``bbox``
    height : int
        Number of pixels tall
    width : int
        Number of pixels wide
    center : bool, optional
        Return coordinates for the center of each pixel

    Returns
    -------
    y, x : tuple[np.ndarray, np.ndarray]
        Y/X coordinate pairs (e.g., latitude/longitude)
    """
    # TODO Guard type of transform and bbox using convert func
    # transform = convert.to_transform(transform)

    if bbox is not None:
        # TODO Guard bbox
        # bbox = convert.to_bbox(bbox)

        # Find nearest y/x for bbox given posting from transform
        off_y = math.floor((bbox.top - transform.f) / transform.e)
        off_x = math.floor((bbox.left - transform.c) / transform.a)
        # Determine (potentially) new upper-left for bbox
        top = transform.f + off_y * transform.e
        left = transform.c + off_x * transform.a

        # Calculate size needed to cover bounds
        height = int(math.ceil(abs((top - bbox.bottom) / transform.e)))
        width = int(math.ceil(abs((bbox.right - left) / transform.a)))

        # Move the transform to this location
        transform = transform * Affine.translation(off_x, off_y)
    elif not (height and width):
        raise ValueError("Must provide either a BoundingBox (`bbox`) "
                         "or the dimensions (`height` and `width`) "
                         "of the raster grid.")

    offset = 0.5 if center else 0.0

    x, _ = (np.arange(width) + offset, np.zeros(width) + offset) * transform
    _, y = (np.zeros(height) + offset, np.arange(height) + offset) * transform

    return (y, x)


def transform_to_res(transform):
    """ Return the resolution from a transform

    Parameters
    ----------
    transform : affine.Affine
        Affine transform

    Returns
    -------
    tuple[float, float]
        Width and height
    """
    a, b, c, d, e, f = transform[:6]
    if b == d == 0:  # no rotation
        return a, -e
    else:
        return math.sqrt(a * a + d * d), math.sqrt(b * b + e * e)


def transform_to_bounds(transform, height, width):
    """ Return a BoundingBox for some transform and shape

    Parameters
    ----------
    transform : affine.Affine
        Affine transformation
    height : int
        Height in pixels
    width : int
        Width in pixels

    Returns
    -------
    BoundingBox
        left, bottom, right, & top bounding box coordinates
    """
    a, b, c, d, e, f = transform[:6]
    if b == d == 0:
        return BoundingBox(c, f + e * height, c + a * width, f)
    else:
        raise ValueError("Not implemented for rotated transformations (TODO)")


# ============================================================================
# From coordinates
def coords_to_transform(y, x, center=True, assume_unique=True):
    """ Calculate Affine transform from coordinates

    Note: y/x will be loaded into memory

    Parameters
    ----------
    y : array-like
        Y coordinates (latitude, y, etc)
    x : array-like
        X coordinates (longitude, x, etc)
    center : bool, optional
        Are coordinates provided pixel center (True) or
        upper-left (False)
    assume_unique : bool, optional
        Assume coordinates are sorted and unique. Otherwise, will run
        ``np.unique`` on each

    Returns
    -------
    affine.Affine
        Affine transformation
    """
    xform, bounds, shape = _inspect_coords(y, x, center=center,
                                           assume_unique=assume_unique)
    return xform


def coords_to_bounds(y, x, center=True, assume_unique=True):
    """ Calculate the BoundingBox of the coordinates

    If the Affine ``transform`` of the data is known, it can be provided.
    Otherwise this function runs :py:func:`coords_to_transform` to determine
    the pixel sizing.

    Parameters
    ----------
    y : array-like
        Y coordinates (latitude, y, etc)
    x : array-like
        X coordinates (longitude, x, etc)
    center : bool, optional
        Are coordinates provided pixel center (True) or
        upper-left (False)
    assume_unique : bool, optional
        Assume coordinates are sorted and unique. Otherwise, will run
        ``np.unique`` on each

    Returns
    -------
    BoundingBox
        Bounds expressed as left, bottom, right, top
    """
    xform, bounds, shape = _inspect_coords(y, x, center=center,
                                           assume_unique=assume_unique)
    return bounds


# TODO: @warn_dask?
# TODO: @lru_cache?
def _inspect_coords(y, x, center=True, assume_unique=True):
    # returns transform, bounds, shape
    y_ = np.atleast_1d(y)
    x_ = np.atleast_1d(x)

    if not assume_unique:
        logger.warning('Computing unique values of coordinates...')
        y_ = np.unique(y_)
        x_ = np.unique(x_)

    if not _check_spacing(y_):
        warnings.warn('"y" coordinate does not have equal spacing')
    if not _check_spacing(x_):
        warnings.warn('"x" coordinate does not have equal spacing')

    minmax_y = y_[0], y_[-1]
    minmax_x = x_[0], x_[-1]

    if minmax_y[0] > minmax_y[1]:
        logger.warning('Unreversing y coordinate min/max')
        minmax_y = (minmax_y[1], minmax_y[0])

    nx, ny = len(x_), len(y_)
    dy = (minmax_y[0] - minmax_y[1]) / (ny - 1)
    dx = (minmax_x[1] - minmax_x[0]) / (nx - 1)

    transform = Affine(
        dx, 0.0, minmax_x[0],
        0., dy, minmax_y[1]
    )

    if center:
        # affine transform is relative to upper-left, not pixel center
        transform = transform * Affine.translation(-0.5, -0.5)
        # create bounds that cover pixel area
        bounds = BoundingBox(minmax_x[0] - dx / 2,
                             minmax_y[0] + dy / 2,
                             minmax_x[1] + dx / 2,
                             minmax_y[1] - dy / 2)
    else:
        # create bounds that cover pixel area
        bounds = BoundingBox(minmax_x[0],
                             minmax_y[0] + dy,
                             minmax_x[1] + dx,
                             minmax_y[1])

    return transform, bounds, (nx, ny)


def _check_spacing(coord):
    """ Check for equal spacing (see GDAL NetCDF driver)
    """
    n = len(coord)
    beg = coord[1] - coord[0]
    mid = coord[n // 2 + 1] - coord[n // 2]
    end = coord[n - 1] - coord[n - 2]

    def cmp(a, b, tol=2e-3):
        return abs(a - b) < tol

    if cmp(beg, mid) and cmp(mid, end) and cmp(beg, end):
        return True
    else:
        return False
