""" GIS variable conversion library

Functions here are convenient ways of going from various representations
of GIS information used in this stack (e.g., WKT) to the following
representations:


* Coordinate Reference System
    * :py:class:`rasterio.crs.CRS`
* Geotransform
    * :py:class:`affine.Affine`
* Bounding Box
    * :py:class:`rasterio.coords.BoundingBox`
* Bounds
    * :py:class:`shapely.geom.Polygon`

"""
from functools import singledispatch
import logging

from affine import Affine
import numpy as np
from osgeo import osr
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.errors import CRSError
import shapely.geometry

from ..utils import (find_subclasses,
                     register_multi_singledispatch)

logger = logging.getLogger()

LIST_TYPE = (tuple, list, np.ndarray, )
# XARRAY_TYPE = (xr.Dataset, xr.DataArray)
GEOM_TYPE = find_subclasses(shapely.geometry.base.BaseGeometry)


# ============================================================================
# Affine geotransform
@singledispatch
def to_transform(value, from_gdal=False):
    """ Convert input into an :py:class:`affine.Affine` transform

    Parameters
    ----------
    value : Affine or iterable
        6 numbers representing affine transform
    from_gdal : bool, optional
        If `value` is a tuple or list, specifies if transform
        is GDAL variety (True) or rasterio/affine (False)

    Returns
    -------
    affine.Affine
        Affine transform
    """
    raise _CANT_CONVERT(value)


@to_transform.register(Affine)
def _to_transform_affine(value, from_gdal=False):
    return value


@register_multi_singledispatch(to_transform, LIST_TYPE)
def _to_transform_iter(value, from_gdal=False):
    if from_gdal:
        return Affine.from_gdal(*value[:6])
    else:
        return Affine(*value[:6])


# ============================================================================
# CRS
# TODO: Dispatch function for Cartopy
@singledispatch
def to_crs(value):
    """ Convert a CRS representation to a :py:class:`rasterio.crs.CRS`

    Parameters
    ----------
    value : str, int, dict, or osr.SpatialReference
        Coordinate reference system as WKT, Proj.4 string, EPSG code,
        rasterio-compatible proj4 attributes in a dict, or OSR definition

    Returns
    -------
    rasterio.crs.CRS
        CRS
    """
    raise _CANT_CONVERT(value)


@to_crs.register(CRS)
def _to_crs_crs(value):
    return value


@to_crs.register(str)
def _to_crs_str(value):
    # After rasterio=1.0.14 WKT is backbone so try it first
    try:
        crs_ = CRS.from_wkt(value)
        crs_.is_valid
    except CRSError as err:
        logger.debug('Could not parse CRS as WKT', err)
        try:
            crs_ = CRS.from_string(value)
            crs_.is_valid
        except CRSError as err:
            logger.debug('Could not parse CRS as Proj4', err)
            raise CRSError('Could not interpret CRS input as '
                           'either WKT or Proj4')
    return crs_


@to_crs.register(int)
def _to_crs_epsg(value):
    return CRS.from_epsg(value)


@to_crs.register(dict)
def _to_crs_dict(value):
    return CRS(value)


@to_crs.register(osr.SpatialReference)
def _to_crs_osr(value):
    return CRS.from_wkt(value.ExportToWkt())


# ============================================================================
# BoundingBox
@singledispatch
def to_bounds(value):
    """ Convert input to a :py:class:`rasterio.coords.BoundingBox`

    Parameters
    ----------
    value : iterable, or Polygon
        Input containing some geographic information

    Returns
    -------
    BoundingBox
        Bounding box (left, bottom, right, top). Also described as
        (minx, miny, maxx, maxy)
    """
    raise _CANT_CONVERT(value)


@to_bounds.register(BoundingBox)
def _to_bounds_bounds(value):
    return value


@register_multi_singledispatch(to_bounds, LIST_TYPE)
def _to_bounds_iter(value):
    return BoundingBox(*value)


@register_multi_singledispatch(to_bounds, GEOM_TYPE)
def _to_bounds_geom(value):
    return BoundingBox(*value.bounds)


# ============================================================================
# Polygon
@singledispatch
def to_bbox(value):
    """ Convert input a bounding box :py:class:`shapely.geometry.Polygon`

    Parameters
    ----------
    value : BoundingBox
        Object representing a bounding box, or an xarray object with coords
        we can use to calculate one from

    Returns
    -------
    shapely.geometry.Polygon
        BoundingBox as a polygon
    """
    raise _CANT_CONVERT(value)


@register_multi_singledispatch(to_bbox, GEOM_TYPE)
def _to_bbox_geom(value):
    return _to_bbox_bounds(BoundingBox(*value.bounds))


@to_bbox.register(BoundingBox)
def _to_bbox_bounds(value):
    return shapely.geometry.box(*value)


# ============================================================================
# UTILITIES
def _CANT_CONVERT(obj):
    return TypeError("Don't know how to convert this type: {type(obj)}")
