""" Create and work with geospatial data tiling schemes

A tile scheme defines the grid which many products are based on. Tiling
schemes define a projection, usually a projection applicable over a wide
area, the size (in pixels) of each tile, and georeferencing information
that defines the upper left coordinate and pixel sizes (thus defining the
posting of each pixel). Tiles are defined by the tile grid coordinate
(horizontal & vertical indices) that determines the number of tiles
offset from the upper left coordinate of the tiling scheme, such that
one can retrieve the real-world coordinate of a pixel if the tile
grid index and pixel row/column within the tile is known.
"""
import collections
import inspect
import itertools
import logging
from pathlib import Path
import warnings

from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
import shapely.geometry

from .coords import transform_to_coords
from . import convert, geom


logger = logging.getLogger(__name__)

_DEFAULT_TILEGRID_UNSIZED_LIMITS = 50
_GEOJSON_EPSG_4326_STRING = 'epsg:4326'


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

    def to_dict(self):
        """ Return this TileGrid as a dictionary (e.g., to serialize)

        Returns
        -------
        dict
            TileGrid attributes needed to re-initialize class
        """
        return {
            'ul': tuple(self.ul),
            'crs': self.crs_wkt,
            'res': tuple(self.res),
            'size': tuple(self.size),
            'limits': tuple(self.limits),
            'name': self.name
        }

    @classmethod
    def from_dict(cls, d):
        """ Return a ``TileGrid`` from a dictionary

        Parameters
        ----------
        d : dict
            Dictionary of class attributes (see __init__)

        Returns
        -------
        TileGrid
            A new TileGrid according to parameters in ``d``
        """
        return cls(**d)

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
        """ list[int]: the vertical/row numbers for all tiles
        """
        limits = self._guard_limits()
        return list(range(limits[0][0], limits[0][1] + 1))

    @property
    def cols(self):
        """ list[int]: the horizontal/column numbers for all tiles
        """
        limits = self._guard_limits()
        return list(range(limits[1][0], limits[1][1] + 1))

    @property
    def nrow(self):
        """ int: the number of rows in the TileGrid
        """
        return len(self.rows)

    @property
    def ncol(self):
        """ int: the number of columns in the TileGrid
        """
        return len(self.cols)

    @property
    def transform(self):
        """ affine.Affine : the affine transform for this TileGrid
        """
        return Affine(self.res[0], 0, self.ul[0],
                      0, -self.res[1], self.ul[1])

    def geojson(self, crs=_GEOJSON_EPSG_4326_STRING,
                rows=None, cols=None,
                rfc7946=False, skip_invalid=False):
        """ Returns this grid of tiles as GeoJSON

        Parameters
        ----------
        crs : rasterio.crs.CRS
            Coordinate reference system of output. Defaults to EPSG:4326 per
            GeoJSON standard (RFC7946). If ``None``, will return
            geometries in TileGrid's CRS
        rows : Sequence[int], optional
            If this TileGrid was not given ``limits`` or if you want a subset
            of the tiles, specify the rows to map
        cols : Sequence[int], optional
            If this TileGrid was not given ``limits`` or if you want a subset
            of the tiles, specify the rows to map
        rfc7946 : bool, optional
            Return GeoJSON compliant with RFC7946. Helps fix GeoJSON
            that crosses the anti-meridian/datelines by splitting
            polygons into multiple as needed
        skip_invalid : bool, optional
            If ``True``, checks for tile bounds for invalid geometries and will
            only include valid tile geometries.

        Returns
        -------
        dict
            GeoJSON
        """
        rows_ = rows or self.rows
        cols_ = cols or self.cols
        tile_rc = itertools.product(rows_, cols_)

        features = []
        for r, c in tile_rc:
            tile = self[r, c]
            feat = tile.geojson(crs=crs)
            if skip_invalid and geom.is_null(feat['geometry']):
                continue
            features.append(feat)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        if rfc7946:
            return geom.fix_geojson_rfc7946(geojson)
        else:
            return geojson

# `Mapping` ABC requirement
    def __getitem__(self, index):
        """ Return a Tile for the grid row/column specified by index
        """
        index = self._guard_index(index)
        return self._index_to_tile(index)

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
            X/Y coordinates. Coordinates must be in the same CRS as the
            TileGrid.

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


        Parameters
        ----------
        bounds : BoundingBox
            Input bounds. Bounds must be in the same CRS as the TileGrid.

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
            A shapely geometry. Must be in the same CRS as the TileGrid.

        Yields
        ------
        iterable[Tile]
            Yields ``Tile`` objects within provided Region of Interest
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
        """ Return a valid tile index, or raise an error
        """
        # TODO: this could be simpler if we store (h, v) in a np.array
        #       because we could borrow fancy indexing to support returning
        #       a scalar (row/col), or a Sequence (e.g., grid[:, 0],
        #       grid[[0, 1], [0]], etc)

        # Only support row/col for now
        if isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError('TileSpec only has two dimensions (row/col)')
            # Need to be ints, and we'll cast as built-in int
            row, col = index
            if not _isint(row) or not _isint(col):
                raise TypeError('Only support indexing int/int for now')
            index_ = (int(row), int(col), )
        else:
            raise IndexError('Unknown index type')

        # Check if index is outside of limits
        if self.limits:
            row, col = index_
            row_lim, col_lim = self.limits
            if (row < min(row_lim) or row > max(row_lim) or
                    col < min(col_lim) or col > max(col_lim)):
                raise IndexError(f'Tile at index "{index}" is outside of '
                                 f'"{self.name}" limits ({self.limits}).')

        return index_

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

    Attributes
    ----------
    index : tuple[int, int]
        The (row, column) index of this tile in the grid
    crs : rasterio.crs.CRS
        The tile coordinate reference system
    bounds : BoundingBox
        The bounding box of the tile
    res : tuple[float, float]
        Pixel X/Y resolution
    size : tuple[int, int]
        Number of columns and rows in tile
    """
    def __init__(self, index, crs, bounds, res, size):
        self.index = tuple(index)
        self.crs = crs
        self.bounds = BoundingBox(*bounds)
        self.res = tuple(res)
        self.size = tuple(size)

    def __eq__(self, other):
        for attr in ('index', 'crs', 'bounds', 'res', 'size', ):
            if getattr(self, attr) != getattr(other, attr, object()):
                return False
        return True

    def __hash__(self):
        return hash((self.index, self.crs.wkt, self.bounds, self.res,
                     self.size))

    def to_dict(self):
        """ Return a ``dict`` representing this Tile

        Returns
        -------
        dict
            This Tile as a dict (CRS will be a WKT for portability)
        """
        return {
            'index': self.index,
            'crs': self.crs.wkt,
            'bounds': self.bounds,
            'res': self.res,
            'size': self.size
        }

    @classmethod
    def from_dict(cls, d):
        """ Create a Tile from a dict

        Parameters
        ----------
        d : dict
            Tile info, including "index", "crs" (a WKT), "bounds", "res",
            and "size"
        """
        return cls(tuple(d['index']),
                   convert.to_crs(d['crs']),
                   d['bounds'],
                   tuple(d['res']),
                   tuple(d['size']))

    @property
    def vertical(self):
        """ int: The horizontal index of this tile in its tile specification
        """
        return int(self.index[0])

    @property
    def horizontal(self):
        """ int: The horizontal index of this tile in its tile specification
        """
        return int(self.index[1])

    @property
    def transform(self):
        """ affine.Affine: The ``Affine`` transform for the tile
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
        np.ndarray
            Y coordinates
        np.ndarray
            X coordinates
        """
        return transform_to_coords(self.transform,
                                   width=self.width,
                                   height=self.height,
                                   center=center)

    def geojson(self, crs=_GEOJSON_EPSG_4326_STRING):
        """ Return this Tile's geometry as GeoJSON

        Parameters
        ----------
        crs : rasterio.crs.CRS
            Coordinate reference system of output. Defaults to EPSG:4326 per
            GeoJSON standard (RFC7946). If ``None``, will return
            geometries in Tile's CRS

        Returns
        -------
        dict
            This tile's geometry and crs represented as GeoJSON

        References
        ----------
        .. [1] https://tools.ietf.org/html/rfc7946#page-12
        """
        gj = shapely.geometry.mapping(self.bbox)

        if crs is not None:
            from rasterio.warp import transform_geom
            crs_ = convert.to_crs(crs)
            if crs_ != self.crs:
                gj = transform_geom(self.crs, crs_, gj)
            else:
                logger.debug('Not reprojecting GeoJSON since output CRS '
                             'is the same as the Tile CRS')

        return {
            'type': 'Feature',
            'properties': {
                'horizontal': self.horizontal,
                'vertical': self.vertical
            },
            'geometry': gj
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
        specs = yaml.safe_load(f)

    tilegrids = {}
    for name in specs:
        kwds = specs[name]
        kwds['name'] = kwds.get('name', name)
        tilegrids[name] = TileGrid(**kwds)

    return tilegrids


def _isint(v):
    # ints of many types, if they work...
    try:
        v_ = int(v)
    except:
        return False
    else:
        return v == v_
