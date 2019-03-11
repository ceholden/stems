""" Geometry related functions
"""
import json
import logging
import math

import affine
from rasterio.coords import BoundingBox
from rasterio.windows import Window, from_bounds
import shapely.geometry

from . import coords
from ..utils import list_like

logger = logging.getLogger(__name__)


# =============================================================================
# Bounds / BoundingBox
def bounds_union(*bounds):
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


def bounds_intersection(*bounds):
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


def bounds_transform_union(bounds, transform):
    """ Calculate union of bounds for some transform

    Parameters
    ----------
    bounds : Sequence[BoundingBox]
        Bounding boxes
    transform : Affine
        Affine transform to use

    Returns
    -------
    rasterio.coords.BoundingBox
        Union of BoundingBox
    affine.Affine
        Unified affine transform
    tuple[int, int]
        Shape of union (nrow, ncol)
    """
    res = coords.transform_to_res(transform)
    # Union of bounds
    out_bounds = bounds_union(*bounds)
    logger.debug('Initial output bounds: %r', out_bounds)

    # Create transform
    out_transform = affine.Affine.translation(out_bounds.left, out_bounds.top)
    out_transform *= affine.Affine.scale(res[0], -res[1])
    logger.debug('Output transform: %r', out_transform)

    # Calculate width / height
    out_w = int(math.ceil((out_bounds.right - out_bounds.left) / res[0]))
    out_h = int(math.ceil((out_bounds.top - out_bounds.bottom) / res[1]))
    logger.debug('Output width / height: %d / %d', out_w, out_h)

    # Adjust bounds to fit (given integer width / height)
    right, bottom = out_transform * (out_w, out_h)
    out_bounds = BoundingBox(out_bounds.left, bottom, right, out_bounds.top)
    logger.debug('Adjusted bounds: %r', out_bounds)

    return out_bounds, out_transform, (out_h, out_w)


def calculate_src_window(src_bounds, src_transform, dst_bounds):
    """ Calculate the source window for some output bounds

    Parameters
    ----------
    src_bounds : rasterio.coords.BoundingBox
        Input bounds for the source raster image
    src_transform : affine.Affine
        Affine transform for the source raster image
    dst_bounds : rasterio.coords.BoundingBox
        Destination bounds

    Returns
    -------
    rasterio.windows.Window
        Window of source that contains ``dst_bounds``
    rasterio.coords.BoundingBox
        The BoundingBox representing the destination window for the source
    """
    # TODO: refactor to use bbox_union/intersect?
    # Calculate source window and bounds (x/y offsets and sizes)
    src_w, src_s, src_e, src_n = src_bounds
    dst_w, dst_s, dst_e, dst_n = dst_bounds

    left = src_w if src_w > dst_w else dst_w
    bottom = src_s if src_s > dst_s else dst_s
    right = src_e if src_e < dst_e else dst_e
    top = src_n if src_n < dst_n else dst_n
    adj_bounds = BoundingBox(left, bottom, right, top)

    src_window = from_bounds(left, bottom, right, top,
                             src_transform)
    logger.debug('Computed source window: %r', src_window)

    # Round the sizes
    src_window = _round_window(src_window)
    logger.debug('Rounded to source window: %r', src_window)

    return src_window, adj_bounds


def calculate_dst_window(src_bounds, dst_transform):
    """ Calculate the rounded source window for some other transform

    Parameters
    ----------
    src_bounds : rasterio.coords.BoundingBox
        Input bounds for the source raster image
    dst_transform : affine.Affine
        Destination Affine transform

    Returns
    -------
    rasterio.windows.Window
        Window of source that conforms (pixel size, upper left) to
        ``dst_tranform``
    """
    dst_window = from_bounds(*src_bounds, dst_transform)
    logger.debug('Computed destination window: %r', dst_window)
    dst_window = _round_window(dst_window)
    logger.debug('Rounded to destination window: %r', dst_window)

    return dst_window


def _round_window(window):
    return window.round_shape().round_offsets()


# =============================================================================
# Geometry
def is_null(geom):
    """ Return True if a geometry is not valid / null

    Parameters
    ----------
    shapely.geometry.BaseGeometry
        Some shapely geometry (or something that supports
        __geo_interface__)

    Returns
    -------
    bool
        Null or not
    """
    # asShape doesn't copy vs shape
    geom_ = shapely.geometry.asShape(geom)
    return getattr(geom_, '_geom', None) is None


# =============================================================================
# GEOJSON
_CO_RFC7946 = ['RFC7946=YES', 'WRITE_BBOX=YES']
def fix_geojson_rfc7946(geojson):
    """ Fixes issues with dateline and meridian cutting

    Parameters
    ----------
    geojson : dict
        GeoJSON as a dict

    Returns
    -------
    dict
        Fixed up GeoJSON

    TODO
    ----
    * Q1: Can we do this ourself using shapely/maths?
    * Q2: Is solution to Q1 faster? Avoids dumps/loads
    """
    from osgeo import gdal, ogr
    gdal.UseExceptions()
    ogr.UseExceptions()

    def read_vsi(filename):
        vsifile = gdal.VSIFOpenL(filename,'r')
        gdal.VSIFSeekL(vsifile, 0, 2)
        vsileng = gdal.VSIFTellL(vsifile)
        gdal.VSIFSeekL(vsifile, 0, 0)
        return gdal.VSIFReadL(1, vsileng, vsifile)

    # Fix up GeoJSON while writing to memory
    _ = gdal.VectorTranslate('/vsimem/out.json',
                             json.dumps(geojson),
                             format='GeoJSON',
                             layerCreationOptions=_CO_RFC7946)
    _ = None  # make sure to delete so it flushes

    text = (read_vsi('/vsimem/out.json')
            .decode('utf-8')
            .replace("'", '"'))

    return json.loads(text)
