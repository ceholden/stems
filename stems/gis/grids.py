""" Predefined tile specifications and utilities for working with tile systems
"""
import collections
import inspect
import itertools
import logging
from pathlib import Path
import warnings

from affine import Affine
from rasterio.coords import BoundingBox
import shapely.geometry

from .coords import transform_to_coords
from . import convert


logger = logging.getLogger(__name__)

_DEFAULT_TILEGRID_UNSIZED_LIMITS = 50
_GEOJSON_EPSG_4326_STRING = '+init=epsg:4326'


class TileGrid(collections.abc.Mapping):
    """ A tile grid specification for gridding data

    Attributes
    ----------
    ul : tuple
        Upper left X/Y coordinates
    crs : rasterio.crs.CRS
        Coordinate system information
    res : tuple
        Pixel X/Y resolution
    size : tuple
        Number of pixels in X/Y dimensions for each tile
    limits : tuple[tuple, tuple]
        Maximum and minimum rows (vertical) and columns (horizontal)
        given as ((row_start, row_stop), (col_start, col_stop)). Used
        to limit access to Tiles beyond domain.
    name : str, optional
        Name of this tiling scheme
    """

    def __init__(self, ul, crs, res, size, limits=None, name='Grid'):
        """ A tile specification or tile scheme

        Parameters
        ----------
        ul : tuple
            Upper left X/Y coordinates
        crs : str, int, dict, or rasterio.crs.CRS
            Coordinate system information, given as a proj4 string, EPSG code,
            rasterio-compatible `dict`, or a :py:class:`rasterio.crs.CRS`
        res : tuple
            Pixel X/Y resolution
        size : tuple
            Number of pixels in X/Y dimensions for each tile
        limits : Sequence, optional
            Optionally, provide the minimum/maximum row and column indices for
            this grid (e.g., ``limits=[(0, 10), (0, 25)]``)
        name : str, optional
            Name of this tiling scheme
        """
        crs_ = convert.to_crs(crs)
        assert crs_.is_valid

        self.ul = tuple(ul)
        self.crs = crs_
        self.crs_wkt = self.crs.wkt
        self.res = tuple(res)
        self.size = tuple(size)
        self.limits = limits
        self.name = name
        self._tiles = {}

    def __repr__(self):
        return '\n'.join([
            f'<{self.__class__.__name__} at {hex(id(self))}>',
            f'    * name: {self.name}',
            f'    * ul={self.ul}',
            f'    * crs={self.crs_wkt}',
            f'    * res={self.res}',
            f'    * size={self.size}',
            f'    * limits={self.limits}'
        ])

    def _guard_limits(self):
        if not self.limits:
            warnings.warn(
                f"{self.__class__.__name__} '{self.name}' does not specify a "
                f"``limits``, so has an unlimited number of rows/columns. "
                f"Defaulting to {_DEFAULT_TILEGRID_UNSIZED_LIMITS}"
            )
            return ((0, _DEFAULT_TILEGRID_UNSIZED_LIMITS, ), ) * 2
        else:
            return self.limits

    @property
    def rows(self):
        limits = self._guard_limits()
        return range(limits[0][0], limits[0][1] + 1)

    @property
    def cols(self):
        limits = self._guard_limits()
        return range(limits[1][0], limits[1][1] + 1)

    @property
    def nrow(self):
        return len(self.rows)

    @property
    def ncol(self):
        return len(self.cols)

    def geojson(self, rows=None, cols=None, crs=_GEOJSON_EPSG_4326_STRING):
        """ Returns this grid of tiles as GeoJSON

        Parameters
        ----------
        rows : Sequence[int], optional
            If this TileGrid was not given ``limits`` or if you want a subset
            of the tiles, specify the rows to map
        cols : Sequence[int], optional
            If this TileGrid was not given ``limits`` or if you want a subset
            of the tiles, specify the rows to map
        crs : rasterio.crs.CRS
            The coordinate reference system to use for the GeoJSON. Defaults to
            EPSG:4326

        Returns
        -------
        dict
            GeoJSON
        """
        return {
            "type": "FeatureCollection",
            "features": list([tile.geojson(crs=crs) for tile in self.values()])
        }

# `Mapping` ABC requirement
    def __getitem__(self, index):
        """ Return a Tile for the grid row/column specified by index
        """
        if isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError('TileSpec only has two dimensions (row/col)')
            if not isinstance(index[0], int) or not isinstance(index[1], int):
                raise TypeError('Only support indexing int/int for now')
            row, col = index
            return self._index_to_tile(index)
        else:
            raise IndexError('Unknown index type')

    def __len__(self):
        return self.nrow * self.ncol

    def __iter__(self):
        rows = range(self.nrow)
        cols = range(self.ncol)
        for idx in itertools.product(rows, cols):
            yield idx

# TILE ACCESS METHODS
    def point_to_tile(self, point):
        """ Return a Tile containing a given point (x, y)

        Parameters
        ----------
        point : tuple
            X/Y coordinates in tile specification's CRS

        Returns
        -------
        tile : stems.gis.grids.Tile
            The intersecting :class`Tile`
        """
        px, py = self.size[0] * self.res[0], self.size[1] * self.res[1]
        _x = int((point[0] - self.ul[0]) // px)
        _y = int((self.ul[1] - point[1]) // py)

        return self._index_to_tile((_y, _x))

    def bounds_to_tiles(self, bounds):
        """ Yield Tile objects for this grid within a given bounds

        .. note::

            It is required that the input ``bounds`` be in the same
            coordinate reference system as ``crs``.

        Parameters
        ----------
        bounds : BoundingBox
            Input bounds

        Yields
        ------
        tile : Tile
            Tiles that are within within provided bounds
        """
        grid_ys, grid_xs = self._frame_bounds(bounds)
        return self._yield_tiles(grid_ys, grid_xs, bounds)

    def roi_to_tiles(self, roi):
        """ Yield tiles within a Region of Interest (ROI) ``shapely`` geometry

        Parameters
        ----------
        roi : Polygon, MultiPolygon, etc.
            A shapely geometry in the tile specifications' crs

        Yields
        ------
        iterable[Tile]
            Yields ``Tile``s within provided Region of Interest
        """
        bounds = BoundingBox(*roi.bounds)
        grid_ys, grid_xs = self._frame_bounds(bounds)
        return self._yield_tiles(grid_ys, grid_xs, roi)

# HELPERS
    def _index_to_bounds(self, index):
        """ Return Tile footprint bounds for given index

        Parameters
        ----------
        index : tuple
            Tile row/column index

        Returns
        -------
        bbox : BoundingBox
            Bounding box of tile
        """
        return BoundingBox(
            left=self.ul[0] + index[1] * self.size[0] * self.res[0],
            right=self.ul[0] + (index[1] + 1) * self.size[0] * self.res[0],
            top=self.ul[1] - index[0] * self.size[1] * self.res[1],
            bottom=self.ul[1] - (index[0] + 1) * self.size[1] * self.res[1]
        )

    def _guard_index(self, index):
        row, col = index
        if self.limits:
            row_lim, col_lim = self.limits
            if (row < min(row_lim) or row > max(row_lim) or
                    col < min(col_lim) or col > max(col_lim)):
                raise IndexError(f'Tile at index "{index}" is outside of '
                                 f'"{self.name}" limits ({self.limits}).')

    def _index_to_tile(self, index):
        """ Return the Tile for given index

        Parameters
        ----------
        index : tuple
            Tile row/column index

        Returns
        -------
        tile : Tile
            Tile at index

        Raises
        ------
        IndexError
            Raise if requested Tile is out of domain (outside ``limits``)
        """
        self._guard_index(index)
        if index not in self._tiles:
            bounds = self._index_to_bounds(index)
            self._tiles[index] = Tile(index, self.crs, bounds,
                                      self.res, self.size)
        return self._tiles[index]

    def _yield_tiles(self, grid_ys, grid_xs, geom_or_bounds):
        # yield `Tile`s that intersect 
        bbox_poly = convert.to_bbox(geom_or_bounds)
        for index in itertools.product(grid_ys, grid_xs):
            tile = self._index_to_tile(index)
            # TODO - this should really a check that we can turn off
            if (tile.bbox.intersects(bbox_poly) and not
                    tile.bbox.touches(bbox_poly)):
                yield tile

    def _frame_bounds(self, bounds):
        # return y/x tile index that intersect `bounds` 
        px, py = self.size[0] * self.res[0], self.size[1] * self.res[1]
        min_grid_x = int((bounds.left - self.ul[0]) // px)
        max_grid_x = int((bounds.right - self.ul[0]) // px)
        min_grid_y = int((self.ul[1] - bounds.top) // py)
        max_grid_y = int((self.ul[1] - bounds.bottom) // py)
        return (range(min_grid_y, max_grid_y + 1),
                range(min_grid_x, max_grid_x + 1))


class Tile(object):
    """ A Tile
    """
    def __init__(self, index, crs, bounds, res, size):
        """ A Tile

        Parameters
        ----------
        index : tuple[int, int]
            The (row, column) index of this tile in the larger tile
            specification
        crs : rasterio.crs.CRS
            The coordinate reference system of the tile
        bounds : BoundingBox
            The bounding box of the tile
        res : tuple[float, float]
            Pixel X/Y resolution of tile
        size : tuple[int, int]
            Number of pixels in X/Y dimensions for each tile
        """

        self.index = index
        self.crs = crs
        self.bounds = bounds
        self.res = res
        self.size = size

    @property
    def vertical(self):
        """ int: The horizontal index of this tile in its tile specification
        """
        return self.index[0]

    @property
    def horizontal(self):
        """ int: The horizontal index of this tile in its tile specification
        """
        return self.index[1]

    @property
    def transform(self):
        """ Affine: The ``Affine`` transform for the tile
        """
        return Affine(self.res[0], 0, self.bounds.left,
                      0, -self.res[1], self.bounds.top)

    @property
    def bbox(self):
        """ shapely.geometry.Polygon: This tile's bounding box geometry
        """
        return convert.to_bbox(self.bounds)

    @property
    def width(self):
        """ int : The number of columns in this Tile
        """
        return self.size[0]

    @property
    def height(self):
        """ int : The number of columns in this Tile
        """
        return self.size[0]

    def coords(self, center=True):
        """ Return y/x pixel coordinates

        Parameters
        ----------
        center : bool, optional
            Return coordinates for pixel centers (default)

        Returns
        -------
        tuple[np.ndarray]
            Y/X coordinates
        """
        return transform_to_coords(self.transform,
                                   width=self.width,
                                   height=self.height,
                                   center=center)

    def geojson(self, crs=_GEOJSON_EPSG_4326_STRING):
        """ Return this Tile's geomtry as GeoJSON

        Parameters
        ----------
        crs : rasterio.crs.CRS
            Coordinate reference system of output. Defaults to EPSG:4326
            per GeoJSON standard

        Returns
        -------
        dict
            This tile's geometry and crs represented as GeoJSON

        References
        ----------
        .. [1] https://tools.ietf.org/html/rfc7946#page-12
        """
        from rasterio.warp import transform_geom
        crs_ = convert.to_crs(crs)
        geom = shapely.geometry.mapping(self.bbox)
        geom_epsg4326 = transform_geom(self.crs, crs_, geom)
        return {
            'type': 'Feature',
            'properties': {
                'horizontal': self.horizontal,
                'vertical': self.vertical
            },
            'geometry': geom_epsg4326
        }


def load_grids(filename=None):
    """ Retrieve tile grid specifications from a YAML file

    Parameters
    ----------
    filename : str
        Filename of YAML data containing specifications. If ``None``,
        will load grids packaged with module

    Returns
    -------
    dict
        Mapping of (name, TileGrid) pairs for tile grid specifications. By
        default, returns tile specifications included with this package
    """
    import yaml
    from .grids import TileGrid

    if filename is None:
        filename = Path(__file__).parent.joinpath('tilegrids.yml')

    with open(str(filename), 'r') as f:
        specs = yaml.load(f)

    tilegrids = {}
    for name in specs:
        kwds = specs[name]
        kwds['name'] = kwds.get('name', name)
        tilegrids[name] = TileGrid(**kwds)

    return tilegrids
