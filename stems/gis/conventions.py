""" Conventions for referencing XArray/NetCDF data

Includes functions useful for managing CF conventions (variable and
coordinating naming, grid mapping variables, etc)
"""
from collections import OrderedDict
import logging

import numpy as np
from rasterio.crs import CRS
from rasterio.coords import BoundingBox
import xarray as xr

logger = logging.getLogger(__name__)

# Names of x/y dimensions, ordered with some preference
_X_DIMENSIONS = ['x', 'longitude', 'lon', 'long']
_Y_DIMENSIONS = ['y', 'latitude', 'lat']

#: dict: CF coordinate attribute metadata
# http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#coordinate-types
COORD_DEFS = {
    'longitude': {
        'standard_name': 'longitude',
        'long_name': 'longitude',
        'units': 'degrees_east',
    },
    'latitude': {
        'standard_name': 'latitude',
        'long_name': 'latitude',
        'units': 'degrees_north',
    },
    'x': {
        'standard_name': 'projection_x_coordinate',
        'long_name': 'x coordinate of projection',
    },
    'y': {
        'standard_name': 'projection_y_coordinate',
        'long_name': 'y coordinate of projection',
    },
    'time': {
        'standard_name': 'time',
        'long_name': 'Time, unix time-stamp',
        'axis': 'T',
        'calendar': 'standard'
    }
}

#: dict: CF NetCDF attributes
# http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#identification-of-conventions
CF_NC_ATTRS = OrderedDict((
    ('Conventions', 'CF-1.7'),
))


# =============================================================================
# Projection
def xarray_crs(crs, transform, grid_mapping='crs'):
    """ Return an :py:class:`xarray.DataArray` of CF-compliant CRS info

    Parameters
    ----------
    crs : rasterio.crs.CRS
        Coordinate reference system information
    transform : affine.Affine
        Affine transform
    grid_mapping : str, optional
        Name of grid mapping variable. Defaults to 'crs'

    Returns
    -------
    xarray.DataArray
        "crs" variable holding CRS information
    """
    name = projections.cf_crs_name(crs)

    # This part is entirely unnecessary!
    epsg_code = projections.epsg_code(crs) or 0
    if epsg_code:
        epsg_auth, epsg_code = epsg_code.split(':')
    epsg_code = np.array(int(code), dtype=np.int32)

    da = xr.DataArray(epsg_code, name=grid_mapping)
    da.attrs['grid_mapping_name'] = name

    da.attrs.update(projections.cf_crs_attrs(crs))
    da.attrs.update(projections.cf_proj_params(crs))
    da.attrs.update(projections.cf_ellps_params(crs))

    # For GDAL in case CF doesn't work
    # http://www.gdal.org/frmt_netcdf.html
    for attr, value in _georeference_attrs(crs, transform).items():
        da.attrs[attr] = value

    for attr, value in da.attrs.items():
        # every list/tuple should be np.ndarray to look like CRS variables
        # that have been written to disk (otherwise comparisons fail)
        if isinstance(value, (list, tuple)):
            da.attrs[attr] = np.asarray(value)

    return da


# =============================================================================
# Coordinates
def xarray_coords(y, x, crs):
    """ Return `y` and `x` as `xr.Variable` useful for coordinates

    Parameters
    ----------
    y : np.ndarray
        Y coordinate
    x : np.ndarray
        X coordinate
    crs : rasterio.crs.CRS
        Coordinate reference system

    Returns
    -------
    xr.Variable : y_coord
        X coordinate
    xr.Variable : x_coord
        Y coordinate

    References
    ----------

    .. [1] http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#coordinate-types
    """
    x_var, y_var = xy_coordinate_names(crs)
    y_attrs = COORD_DEFS[y_var].copy()
    x_attrs = COORD_DEFS[x_var].copy()

    if crs.is_projected:
        crs_osr = crs2osr(crs)
        units = crs_osr.GetLinearUnitsName().lower()
        y_attrs['units'], x_attrs['units'] = units, units

    y = xr.Variable((y_var, ), y, attrs=y_attrs)
    x = xr.Variable((x_var, ), x, attrs=x_attrs)

    return y, x


def xy_coordinate_names(crs):
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
    """
    # CRSError is raised if neither is true, so we don't handle
    if crs.is_geographic:
        return 'longitude', 'latitude'
    elif crs.is_projected:
        return 'x', 'y'
    else:
        raise ValueError('CRS must either be geographic or projected')
