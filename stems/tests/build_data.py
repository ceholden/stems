""" Test data simulation helpers
"""
from affine import Affine
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
import xarray as xr


BANDS_BGRN = ['blu', 'grn', 'red', 'nir']


# =============================================================================
# GDAL / Rasterio
def create_test_raster(dst='test.tif',
                       driver='GTiff', count=4, dtype='int16',
                       height=7, width=11,
                       blockysize=5, blockxsize=11,
                       nodata=-1234,
                       crs=CRS.from_epsg(32619),
                       transform=Affine(30., 0., 100., 0., -30., 200)):
    """ Create a random raster file for testing

    Returns
    -------
    str
        Filename to raster
    dict
        Rasterio metadata used to write image
    dict
        All relevant metadata
    """
    meta = {
        'driver': driver,
        'count': count,
        'dtype': dtype,
        'height': height,
        'width': width,
        'blockxsize': blockxsize,  # will be width since <256
        'blockysize': blockysize,
        'nodata': nodata,
        'crs': crs,
        'transform': transform,
    }

    # Create data
    shape = (meta['count'], meta['height'], meta['width'], )
    dat = np.random.randint(0, 10, shape).astype(meta['dtype'])

    # Create descriptions
    desc = list(['b_%d' % b for b in range(count)])
    # Create color interpretation
    colorinterp = list(ColorInterp(i) for i in range(5, 5 + count))

    # Write to file
    with rasterio.open(dst, 'w', **meta) as dst_:
        dst_.write(dat)
        dst_.descriptions = desc
        dst_.colorinterp = colorinterp
        bounds = dst_.bounds

    # Add remaining metadata
    meta_ = meta.copy()
    meta_.update({
        'shape': shape,
        'dat': dat,
        'description': desc,
        'colorinterp': colorinterp,
        'bounds': bounds
    })

    return dst, meta, meta_


def chop_test_image(tmpdir, src_fname):
    files = {'all': src_fname}

    with rasterio.open(src_fname) as src:
        midx, midy = int(src.width // 2), int(src.height // 2)
        quarters = {
            'ul': ((0, midx), (0, midy)),
            'ur': ((midx, src.width), (0, midy)),
            'll': ((0, midx), (midy, src.height)),
            'lr': ((midx, src.width), (midy, src.height))
        }

        meta_ = src.meta.copy()

        for name, (xwin, ywin) in quarters.items():
            meta_['width'] = xwin[1] - xwin[0]
            meta_['height'] = ywin[1] - ywin[0]

            fname = str(tmpdir.join(name + '.tif'))
            _meta = meta_.copy()
            window = rasterio.windows.Window(xwin[0], ywin[0],
                                             xwin[1], ywin[1])
            _meta['transform'] = rasterio.windows.transform(
                window, meta_['transform']
            )
            with rasterio.open(fname, 'w', **_meta) as dst_qrt:
                dat = src.read(window=window)
                dst_qrt.write(dat)
            files[name] = fname

    return files


# =============================================================================
# NetCDF4
def create_test_dataset(compute=False, data_vars=BANDS_BGRN,
                        dtype='int16',
                        ny=7, nx=11, ntime=100,
                        chunk_y=3, chunk_x=5, chunk_time=25,
                        nodata=-1234,
                        crs=CRS.from_epsg(32619),
                        transform=Affine(30., 0., 100., 0., -30., 200)):
    """ Create a test xarray.Dataset
    """
    from stems.gis import conventions, coords, projections

    # y/x dim names
    dim_x, dim_y = projections.cf_xy_coord_names(crs)

    # Create coordinates
    y, x = coords.transform_to_coords(transform, width=nx, height=ny)
    time = pd.date_range('2000-01-01', '2010-01-01', periods=ntime).values

    # Create georeferencing
    y_, x_ = conventions.create_coordinates(y, x, crs)
    grid_mapping = conventions.create_grid_mapping(crs, transform, 'crs')

    # Create data
    data = {}
    dims=(dim_y, dim_x, 'time', )
    coords = {dim_y: y_, dim_x: x_, 'time': time, 'crs': grid_mapping}
    chunks = {dim_y: chunk_y, dim_x: chunk_x, 'time': chunk_time}
    for dv in data_vars:
        dat = np.random.randint(0, 10, (ny, nx, ntime)).astype(dtype)
        xarr = xr.DataArray(dat, dims=dims, coords=coords, name=dv)
        data[dv] = xarr.chunk(chunks)
    # Create XArray
    xarr = xr.Dataset(data)
    return xarr


def create_test_netcdf4(dst='test.nc', data_vars=BANDS_BGRN,
                        dtype='int16',
                        ny=7, nx=11, ntime=100,
                        chunk_y=3, chunk_x=5, chunk_time=25,
                        nodata=-1234,
                        crs=CRS.from_epsg(32619),
                        transform=Affine(30., 0., 100., 0., -30., 200)):
    """ Create a test xarray.Dataset and write it to a NetCDF4 file
    """
    xarr = create_test_dataset(
        data_vars=data_vars,
        dtype=dtype,
        ny=ny, nx=nx, ntime=ntime,
        chunk_y=chunk_y, chunk_x=chunk_x, chunk_time=chunk_time,
        nodata=nodata, crs=crs, transform=transform
    )
    chunks = {'y': chunk_y, 'x': chunk_x, 'time': chunk_time}
    encoding = {
        dv: {
            'chunksizes': tuple(chunks.get(d) for d in xarr[dv].dims)
        } for dv in data_vars
    }
    xarr.to_netcdf(dst, encoding=encoding)
    return dst
