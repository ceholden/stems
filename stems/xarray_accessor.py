""" Extend XArray with the ``.stems`` accessor

See Also
--------
http://xarray.pydata.org/en/stable/internals.html#extending-xarray

"""
import xarray as xr

from .gis import convert, conventions, coords, projections


class _STEMSAccessor(object):
    """ Base class for xarray.DataArray and xarray.Dataset accessors
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._grid_mapping = 'crs'

    def georeference(self, crs, transform, grid_mapping='crs'):
        """ Apply georeferencing to XArray data

        Parameters
        ----------
        crs : rasterio.crs.CRS
            Rasterio CRS
        transform : affine.Affine
            Affine transform of the data
        grid_mapping : str, optional
            Name to use for grid mapping variable

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Georeferenced data
        """
        self._grid_mapping = grid_mapping
        obj = conventions.georeference(self._obj, crs, transform,
                                       grid_mapping=grid_mapping,
                                       inplace=False)
        return obj

    def is_georeferenced(self):
        """ Check if data is georeferenced

        Returns
        -------
        bool
            True if XArray data is georeferenced
        """
        return conventions.is_georeferenced(self._obj)

    @property
    def coord_x(self):
        """ np.ndarray: X coordinates
        """
        x, _ = projections.cf_xy_coord_names(self.crs)
        return self._obj.coords[x]

    @property
    def coord_y(self):
        """ np.ndarray: Y coordinates
        """
        _, y = projections.cf_xy_coord_names(self.crs)
        return self._obj.coords[y]

    @property
    def crs(self):
        """ rasterio.crs.CRS: Coordinate reference system
        """
        # TODO: parse based on CF information, not just GDAL
        var_gm = self.grid_mapping
        return convert.to_crs(var_gm.attrs['spatial_ref'])

    @property
    def transform(self):
        """ affine.Affine: Affine transform
        """
        # TODO: assume unique -> yes if 2D, otherwise no
        assume_unique = True
        xform = coords.coords_to_transform(self.coord_y, self.coord_x,
                                           assume_unique=assume_unique)
        return xform

    @property
    def bounds(self):
        """ BoundingBox: Bounding box of data
        """
        assume_unique = True
        bounds = coords.coords_to_bounds(self.coord_y, self.coord_x,
                                         assume_unique=assume_unique)
        return bounds

    @property
    def bbox(self):
        """ Polygon: Bounding box polygon
        """
        return convert.to_bbox(self.bounds)

    @property
    def grid_mapping(self):
        """ xarray.DataArray: Georeferencing variable
        """
        try:
            var_grid_mapping = conventions.get_grid_mapping(
                self._obj, grid_mapping=self._grid_mapping
            )
        except KeyError as ke:
            raise KeyError('XArray data object is not georeferenced')
        else:
            return var_grid_mapping


@xr.register_dataarray_accessor('stems')
class DataArrayAccessor(_STEMSAccessor):
    """ XArray.DataArray accessor for STEMS project
    """
    pass


@xr.register_dataset_accessor('stems')
class DatasetAccessor(_STEMSAccessor):
    """ XArray.Dataset accessor for STEMS project
    """
    pass
