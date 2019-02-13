""" Assorted GIS utilities
"""
import logging

from osgeo import osr
from rasterio.crs import CRS

osr.UseExceptions()


def crs2osr(crs):
    """ Return `osgeo.osr.SpatialReference` of a `rasterio.crs.CRS`

    Parameters
    ----------
    crs : rasterio.crs.CRS
        Rasterio coordinate reference system

    Returns
    -------
    osr.SpatialReference
        CRS as OSR object
    """
    crs_osr = osr.SpatialReference()
    crs_osr.ImportFromWkt(crs.wkt)
    crs_osr.Fixup()
    return crs_osr


def same_crs(*crs):
    """ Check if multiple CRS are the same

    Parameters
    ----------
    crs : rasterio.crs.CRS
        Multiple CRS to compare

    Returns
    -------
    bool
        True if all of the CRS inputs are the same (according to rasterio)

    See Also
    --------
    osr_same_crs
    """
    assert len(crs) >= 1
    base = crs[0]
    for _crs in crs[1:]:
        if base != _crs:
            return False
    return True


def osr_same_crs(*crs):
    """ Use OSR to compare for CRS equality

    Converts Rasterio ``CRS`` to ``OSRSpatialReference``
    and compares using OSR because comparing CRS from
    WKT with CRS from EPSG/Proj4 can be buggy.

    Parameters
    ----------
    *crs : rasterio.crs.CRS
        Two or more CRS

    Returns
    -------
    bool
        True if all CRS are equivalent
    """
    assert len(crs) >= 1
    sr_crs = [crs2osr(crs_) for crs_ in crs]
    base = sr_crs[0]
    for other in sr_crs[1:]:
        if not bool(base.IsSame(other)):
            return False
    return True
