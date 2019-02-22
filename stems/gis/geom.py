""" Geometry related functions
"""
import json

from rasterio.coords import BoundingBox
import shapely.geometry

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
