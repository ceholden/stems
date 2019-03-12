""" Test data simulation helpers
"""
from affine import Affine
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp


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
