""" Rasterio IO helpers
"""
import logging
from pathlib import Path

import rasterio
import xarray as xr

from .. import xarray_accessor
from ..gis import projections

logger = logging.getLogger(__name__)


#: Default Rasterio driver format
DEFAULT_RASTERIO_DRIVER = 'GTiff'
#: Attributes to keep from output of ``xarray.open_rasterio``
RASTERIO_ATTR_WHITELIST = ('nodatavals', )


def xarray_to_rasterio(xarr, path, driver=DEFAULT_RASTERIO_DRIVER,
                       crs=None, transform=None,
                       nodata=None,
                       **meta):
    """ Save a DataArray to a rasterio/GDAL dataset
    Parameters
    ----------
    xarr : xarray.DataArray
        2D or 3D DataArray to save. Shape is assumed to be
        ``(width, height, )`` for 2D or  ``(count, width, height, )`` for
        3D arrays
    path : str or Path
        Save DataArray to this file path
    driver : str, optional
        Rasterio dataset driver
    crs : str, dict, or rasterio.crs.CRS, optional
        Optionally, provide CRS information about ``xarr``. Will
        try to read from ``xarr`` if not provided
    transform : affine.Affine, optional
        Optionally, provide affine transform information about ``xarr``. Will
        try to read from ``xarr`` if not provided
    nodata : int or float, optional
        No data value to set
    **meta
        Additional keyword arguments to :py:func:`rasterio.open`. Useful for
        specifying block sizes, color interpretation, and other metadata.

    Returns
    -------
    path : Path
        Saved file path

    Raises
    ------
    ValueError
        Raised if ``xarr`` is not 2D or 3D
    """
    if not isinstance(xarr, xr.DataArray):
        raise TypeError('Can only save 2D or 3D ``xarray.DataArray``s to '
                        'rasterio/GDAL datasets')

    xarr, meta_ = _prepare_xarray_for_rasterio(xarr, crs, transform,
                                               driver=driver, **meta)
    dim_band = xarr.dims[0]

    # TODO: support some kind of block writing if chunked (dask array)
    with rasterio.open(str(path), 'w', **meta_) as dst:
        # Write data
        dst.write(xarr.values)

        # Write 1st dim ("band") coordinate names as band descriptions
        dst.descriptions = xarr.coords[dim_band].values

        # Write attrs as tags (except "grid_mapping")
        tags = {
            k: v for k, v in xarr.attrs.items()
            if k not in ('grid_mapping', )
        }
        dst.update_tags(**tags)

        if nodata is not None:
            dst.nodata = nodata

    return Path(path)


def _prepare_xarray_for_rasterio(xarr, crs=None, transform=None,
                                 dim_y=None, dim_x=None, dim_band=None,
                                 **meta):
    # Expand 2D to 3D before processing
    dim_band = dim_band or 'band'
    if xarr.ndim == 2:
        xarr = xarr.expand_dims(dim_band)
        xarr.coords['band'] = [xarr.name] if xarr.name else ['Band_1']
    elif xarr.ndim != 3:
        raise ValueError('Can only save 2D or 3D DataArrays')

    if dim_band not in xarr.dims:
        raise KeyError(f'Cannot find band dimension "{dim_band}" in dims')

    if crs is None:
        crs = xarr.stems.crs

    if transform is None:
        transform = xarr.stems.transform

    if dim_x is None and dim_y is None:
        dim_x, dim_y = projections.cf_xy_coord_names(crs)

    dims_ = dict(zip(xarr.dims, xarr.shape))

    meta_ = {
        'driver': DEFAULT_RASTERIO_DRIVER,
        'count': dims_[dim_band],
        'width': dims_[dim_x],
        'height': dims_[dim_y],
        'dtype': xarr.dtype,
        'crs': crs,
        'transform': transform
    }
    meta_.update(meta)

    return xarr, meta_
