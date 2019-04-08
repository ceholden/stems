""" xarray IO standards and helpers
"""
import logging

import numpy as np
import xarray as xr

from ..gis import convert
from ..gis.conventions import is_georeferenced, georeference
from ..gis.projections import cf_xy_coord_names
from .chunk import auto_determine_chunks
from .utils import parse_paths

logger = logging.getLogger(__name__)


def open_dataset(paths, chunks='auto', concat_dim=None, crs=None, **kwds):
    """ Open an xarray dataset (somewhat) intelligently

    Parameters
    ----------
    paths : str or List[str]
        Either a string glob in the form "path/to/my/files/*.nc" or an explicit
        list of files to open.
    chunks : 'auto', int or dict
        Number of chunks or the chunk size for each dimension. If 'auto',
        reads chunks from first file in ``paths`` and uses
        :py:func:`plants.io.chunking_.best_chunksizes` to guess an appropriate
        chunksize  based on the most frequently used chunksize per dimension.
        Otherwise, pass `chunks` onto :py:func:`xr.open_mfdataset`.
    concat_dim : None, str, DataArray or Index, optional
        Dimension to concatenate files along. This argument is passed on to
        :py:func:`xarray.auto_combine` along with the dataset objects. You only
        need to provide this argument if the dimension along which you want to
        concatenate is not a dimension in the original datasets, e.g., if you
        want to stack a collection of 2D arrays along a third dimension.
        By default, xarray attempts to infer this argument by examining
        component files. Set ``concat_dim=None`` explicitly to disable
        concatenation.
    crs : rasterio.crs.CRS, int, or str, optional
        Coordinate reference system to assign, if not already georeferenced.
        If type is not ``CRS``, will be converted using
        :py:func:`plants.gis.convert.to_crs`.
    kwds : optional
        Options passed to :py:func:`xarray.open_mfdataset`

    Returns
    -------
    xarray.Dataset
        Multi-file dataset

    Raises
    ------
    IOError
        Raise if ``paths`` does not parse into any filenames (e.g., a bad glob)
    """
    # Parse path from input
    paths_ = parse_paths(paths)
    if not paths:
        raise IOError('Could not find a file to read using input "{paths}"'
                      .format(paths=paths))

    # Guess at chunks
    if chunks == 'auto':
        chunks = auto_determine_chunks(paths_[0])

    # The xarray call we're wrapping
    ds = xr.open_mfdataset(paths_,
                           chunks=chunks,
                           concat_dim=concat_dim,
                           **kwds)

    # TODO: what to do if no geocoding (?)
    if not is_georeferenced(ds):
        if crs is not None:
            logger.debug('Adding CRS information provided')
            crs_ = convert.to_crs(crs)
            transform_ = convert.to_transform(ds)
            ds = georeference(ds, crs_, transform_, inplace=True)
        else:
            logger.debug('No georeference information found on file.')

    # TODO: in GIS, figure out if we can go CF -> proj, not just other way
    #       around
    return ds


def xarray_map(y, x, count, dtype, crs=None, nodata=-9999, dim_band='band'):
    """ A raster map with ``count`` bands
    Parameters
    ----------
    y : array-like
        Y coordinates
    x : array-like
        X coordinates
    count : int
        Number of bands
    dtype : np.dtype
        Data type
    crs : CRS, optional
        Optionally, give the raster a CRS
    nodata : float or int
        No Data Value to initialize data with

    Returns
    -------
    xr.DataArray
        3D (band, y, x,) "map" to write data into
    """
    if crs is not None:
        crs_ = convert.to_crs(crs)
        dim_yx = cf_xy_coord_names(crs_)[::-1]
    else:
        dim_yx = ('y', 'x', )

    shape = (count, len(y), len(x), )
    dims = (dim_band, ) + dim_yx

    coords = {
        dim_yx[0]: y,
        dim_yx[1]: x,
        dim_band: [f'{dim_band}_{i:0{len(str(count))}d}'
                   for i in range(count)]
    }

    arr = np.full(shape, nodata, dtype=dtype)
    xarr = xr.DataArray(arr, dims=dims, coords=coords)

    if crs is not None:
        xarr = georeference(xarr, crs, inplace=True)

    return xarr
