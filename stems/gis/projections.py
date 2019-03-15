""" Projection handling - OSGEO/Rasterio, CF/NetCDF, and Cartopy

See Also
--------

* https://trac.osgeo.org/gdal/wiki/NetCDF_ProjectionTestingStatus
* http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/cf-conventions.html#appendix-grid-mappings

"""
from collections import OrderedDict
from functools import singledispatch
import logging

from rasterio.crs import CRS
from osgeo import osr

from .utils import crs2osr
from .. import errors

logger = logging.getLogger(__name__)


#: dict: Mapping between OGC WKT <-> CF projection names
CF_PROJECTION_NAMES = {
    'Albers_Conic_Equal_Area': 'albers_conic_equal_area',
    'Azimuthal_Equidistant': 'azimuthal_equidistant',
    'Lambert_Azimuthal_Equal_Area': 'lambert_azimuthal_equal_area',
    # ...
    'Sinusoidal': 'sinusoidal',
    'Transverse_Mercator': 'transverse_mercator',
}

#: dict: Mapping between CF <-> OGC WKT projection parameters for projections
CF_PROJECTION_DEFS = {
    # http://cfconventions.org/cf-conventions/cf-conventions.html#lambert-azimuthal-equal-area
    'albers_conic_equal_area': (
        ('latitude_of_projection_origin', 'latitude_of_center'),
        ('longitude_of_central_meridian', 'longitude_of_center'),
        ('standard_parallel', ('standard_parallel_1', 'standard_parallel_2')),
        ('false_easting', 'false_easting'),
        ('false_northing', 'false_northing'),
    ),
    'lambert_azimuthal_equal_area': (
        ('longitude_of_projection_origin', 'longitude_of_center'),
        ('latitude_of_projection_origin', 'latitude_of_center'),
        ('false_easting', 'false_easting'),
        ('false_northing', 'false_northing'),
    ),
    # http://cfconventions.org/cf-conventions/cf-conventions.html#_transverse_mercator
    'transverse_mercator': (
        ('latitude_of_projection_origin', 'latitude_of_origin'),
        ('longitude_of_central_meridian', 'central_meridian'),
        ('scale_factor_at_central_meridian', 'scale_factor'),
        ('false_easting', 'false_easting'),
        ('false_northing', 'false_northing'),
    ),
    'universal_transverse_mercator': (
        ('utm_zone_number', 'utm_zone_number')
    ),
    # http://cfconventions.org/cf-conventions/cf-conventions.html#_sinusoidal
    'sinusoidal': (
        ('longitude_of_central_meridian', 'central_meridian'),
        ('false_northing', 'false_northing'),
        ('false_easting', 'false_easting'),
    ),
    'latitude_longitude': (),  # None
}

#: tuple: Mapping between CF <-> OGC WKT projection attribute name definitions
# https://trac.osgeo.org/gdal/wiki/NetCDF_ProjectionTestingStatus#CoordinatesystemkeywordsstartingwithCF-1.7
CF_CRS_NAMES = (
    ('horizontal_datum_name', 'GEOGCS|DATUM'),
    ('reference_ellipsoid_name', 'GEOGCS|DATUM|SPHEROID'),
    ('towgs84', 'GEOGCS|DATUM|TOWGS84'),
    ('prime_meridian_name', 'GEOGCS|PRIMEM')
)

#: tuple: mapping from CF ellipsoid parameter to SpatialReference method calls
CF_ELLIPSOID_DEFS = (
    ('semi_major_axis', 'GetSemiMajor'),
    ('semi_minor_axis', 'GetSemiMinor'),
    ('inverse_flattening', 'GetInvFlattening')
)
# TODO: support more...
# TODO: Cartopy mappings


# ============================================================================
# EPSG code
@singledispatch
def epsg_code(crs):
    """Return EPSG code for a CRS

    Parameters
    ----------
    crs : CRS or osr.SpatialReference
        Coordinate reference system defined by an EPSG code

    Returns
    -------
    str
        EPSG code string

    Raises
    ------
    ValueError
        Raised if projection does not correspond to an EPSG code
    """
    raise TypeError('Input `crs` needs to be rasterio `CRS` or '
                    'osr `SpatialReference`')


@epsg_code.register(osr.SpatialReference)
def _epsg_code_osr(crs):
    try:
        status = crs.AutoIdentifyEPSG()
    except RuntimeError as e:
        raise ValueError('Cannot decode EPSG code from CRS')
    else:
        if status == 0:
            auth = crs.GetAuthorityName(None)
            code = crs.GetAuthorityCode(None)
            return '%s:%s' % (auth.lower(), code)


@epsg_code.register(CRS)
def _epsg_code_rasterio(crs):
    osr_crs = crs2osr(crs)
    return _epsg_code_osr(osr_crs)


def crs_longname(crs):
    """ Return name of a CRS / ellipsoid pair

    Parameters
    ----------
    crs : rasterio.crs.CRS
        CRS

    Returns
    -------
    str
        Lowercase projection name (see keys of :py:data:`CF_PROJECTION_DEFS`)

    Examples
    --------
    >>> crs_longname(CRS.from_epsg(3857))  # Web Mercator
    'WGS 84 / Pseudo-Mercator'
    >>> crs_longname(CRS.from_epsg(32619))  # UTM19N
    'WGS 84 / UTM zone 19N'
    """
    # This doesn't necessarily relate to CF but it's nice to have
    crs_osr = crs2osr(crs)
    if crs.is_projected:
        return crs_osr.GetAttrValue('PROJCS')
    elif crs.is_geographic:
        return crs_osr.GetAttrValue('GEOGCS')


# ============================================================================
# CF info / parameters
def cf_crs_name(crs):
    """ Return CF name of a CRS projection

    Parameters
    ----------
    crs : rasterio.crs.CRS
        CRS

    Returns
    -------
    str
        Lowercase projection name (see keys of :py:mod:`CF_PROJECTION_DEFS`)

    Examples
    --------
    >>> cf_crs_attrs(CRS.from_epsg(4326))  # WGS84
    'latitude_longitude'
    >>> cf_crs_attrs(CRS.from_epsg(32619))  # UTM19N
    'transverse_mercator'
    """
    crs_osr = crs2osr(crs)
    if crs.is_projected:
        name = crs_osr.GetAttrValue('PROJECTION')
        if name not in CF_PROJECTION_NAMES:
            if crs.is_valid:
                raise errors.TODO(f'Cannot handle "{name}" CRS types '
                                  f'yet: {crs.wkt}')
            else:
                raise KeyError('Unsupported CRS "{0!r}"'.format(crs))
        return CF_PROJECTION_NAMES[name]
    else:
        return 'latitude_longitude'


def cf_crs_attrs(crs):
    """ Return CF-compliant CRS info to prevent "unknown" CRS/Ellipse/Geoid

    Parameters
    ----------
    crs: rasterio.crs.CRS
        CRS

    Returns
    -------
    OrderedDict
        CF attributes

    References
    ----------
    .. [1] https://cf-pcmdi.llnl.gov/trac/wiki/Cf2CrsWkt#Table2-FutureCF-1.7CFGridMappingAttributes

    Examples
    --------
    >>> cf_crs_attrs(CRS.from_epsg(4326))  # WGS84
    OrderedDict([('geographic_coordinate_system_name', 'WGS 84'),
             ('horizontal_datum_name', 'WGS_1984'),
             ('reference_ellipsoid_name', 'WGS 84'),
             ('prime_meridian_name', 'Greenwich')])
    >>> cf_crs_attrs(CRS.from_epsg(32619))  # UTM19N
    OrderedDict([('projected_coordinate_system_name', 'WGS 84 / UTM zone 19N'),
                 ('horizontal_datum_name', 'WGS_1984'),
                 ('reference_ellipsoid_name', 'WGS 84'),
                 ('prime_meridian_name', 'Greenwich')])
    """
    osr_crs = crs2osr(crs)
    attrs = OrderedDict()

    long_name = crs_longname(crs)
    if crs.is_projected:
        attrs['projected_coordinate_system_name'] = long_name
    elif crs.is_geographic:
        attrs['geographic_coordinate_system_name'] = long_name

    for cf_parm, osgeo_parm in CF_CRS_NAMES:
        value = osr_crs.GetAttrValue(osgeo_parm)
        if value:  # only record if not None so we can serialize
            attrs[cf_parm] = value
    return attrs


def cf_proj_params(crs):
    """ Return projection parameters for a CRS

    Parameters
    ----------
    crs: rasterio.crs.CRS
        CRS

    Returns
    -------
    OrderedDict
        CRS parameters and values

    Raises
    ------
    stems.errors.TODO
        Raise if CRS isn't supported yet

    Examples
    --------
    >>> cf_crs_attrs(CRS.from_epsg(4326))  # WGS84
    OrderedDict()
    >>> cf_crs_attrs(CRS.from_epsg(32619))  # UTM19N
    OrderedDict([('latitude_of_projection_origin', 0.0),
                 ('longitude_of_central_meridian', -69.0),
                 ('scale_factor_at_central_meridian', 0.9996),
                 ('false_easting', 500000.0),
                 ('false_northing', 0.0)])
    """
    name = cf_crs_name(crs)
    osr_crs = crs2osr(crs)

    if name not in CF_PROJECTION_DEFS:
        raise errors.TODO(f'Cannot handle "{name}" CRS types yet')

    parms = OrderedDict()  # "parm" is eww, but ref to `GetProjParm`
    for cf_parm, osgeo_parm in CF_PROJECTION_DEFS[name]:
        if isinstance(osgeo_parm, (list, tuple)):
            parms[cf_parm] = tuple(osr_crs.GetProjParm(p) for p in osgeo_parm)
        else:
            parms[cf_parm] = osr_crs.GetProjParm(osgeo_parm)

    return parms


def cf_ellps_params(crs):
    """ Return ellipsoid parameters for a CRS

    Parameters
    ----------
    crs: rasterio.crs.CRS
        CRS

    Returns
    -------
    OrderedDict
        Ellipsoid parameters and values

    Examples
    --------
    >>> cf_crs_attrs(CRS.from_epsg(4326))  # WGS84
    OrderedDict([('semi_major_axis', 6378137.0),
                 ('semi_minor_axis', 6356752.314245179),
                 ('inverse_flattening', 298.257223563)])
    >>> cf_crs_attrs(CRS.from_epsg(32619))  # UTM19N
    OrderedDict([('semi_major_axis', 6378137.0),
                 ('semi_minor_axis', 6356752.314245179),
                 ('inverse_flattening', 298.257223563)])
    """
    osr_crs = crs2osr(crs)

    return OrderedDict(
        (key, getattr(osr_crs, func)())
        for (key, func) in CF_ELLIPSOID_DEFS
    )


def cf_xy_coord_names(crs):
    """ Returns appropriate names for coordinates given CRS

    Parameters
    ----------
    crs : rasterio.crs.CRS
        Coordinate reference system

    Returns
    -------
    str : x_coord_name
        X coordinate name
    str : y_coord_name
        Y coordinate name

    Examples
    --------
    >>> cf_crs_attrs(CRS.from_epsg(4326))  # WGS84
    ('longitude', 'latitude')
    >>> cf_crs_attrs(CRS.from_epsg(32619))  # UTM19N
    ('x', 'y')
    """
    # CRSError is raised if neither is true, so we don't handle
    if crs.is_geographic:
        return 'longitude', 'latitude'
    elif crs.is_projected:
        return 'x', 'y'
    else:
        raise ValueError('CRS must either be geographic or projected')
