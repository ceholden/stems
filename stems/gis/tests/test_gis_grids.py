""" Tests for :py:mod:`stems.gis.grids`
"""
from pathlib import Path
import re

from rasterio.coords import BoundingBox
from shapely.geometry import Polygon
import pytest

from stems.gis import grids


# =============================================================================
# TileGrid
def test_tilegrid_examples(example_kwds):
    grid = grids.TileGrid(**example_kwds)

    assert grid[0, 0].horizontal == 0
    assert grid[0, 0].vertical == 0

    repr_ = repr(grid).split('\n')
    assert re.match(f'^<TileGrid at .*>', repr_[0])
    assert re.search(f'.*name: {grid.name}',
                     repr_[1])
    assert re.search(f'.*ul=.*{grid.ul[0]}, {grid.ul[1]}.*',
                     repr_[2])
    assert re.search(f'.*crs=.*', repr_[3])
    assert re.search(f'.*res=.*{grid.res[0]}, {grid.res[1]}.*',
                     repr_[4])
    assert re.search(f'.*size=.*{grid.size[0]}, {grid.size[1]}.*',
                     repr_[5])
    assert re.search(f'.*limits=.*{grid.limits[-1]}.*{grid.limits[1]}',
                     repr_[6])

    if grid.limits:
        limits = example_kwds['limits']
        assert grid.nrow == (max(limits[0]) - min(limits[0])) + 1
        assert grid.ncol == (max(limits[1]) - min(limits[1])) + 1
        assert len(grid) == (grid.nrow * grid.ncol)


def test_tilegrid_props():
    grid = grids.TileGrid(
        ul=[0, 100],
        crs='+init=epsg:5070',
        res=[5, 5],
        size=[10, 10],
        limits=[[0, 11], [0, 7]]
    )
    assert grid.ncol == 8
    assert grid.cols == range(0, 8)
    assert grid.nrow == 12
    assert grid.rows == range(0, 12)

    tile = grid[0, 0]
    assert isinstance(tile, grids.Tile)
    assert tile.bounds[0] == grid.ul[0]
    assert tile.bounds[-1] == grid.ul[1]
    assert tile.crs == grid.crs
    assert tile.size == grid.size

    last = grid[11, 7]
    with pytest.raises(IndexError, match=r'.*outside of.*limits'):
        grid[12, 8]

    all_tiles = list(grid)
    assert len(all_tiles) == len(grid)


def test_tilegrid_roi_multiple(example_kwds_GEOG):
    # example_grid_GEOG is 10x10 degree starting at -180x80, easy to test with
    grid = grids.TileGrid(**example_kwds_GEOG)
    roi = Polygon.from_bounds(0.5, 0.5, 10.5, 10.5)
    tiles = list(grid.roi_to_tiles(roi))
    assert len(tiles) == 4


def test_tilegrid_roi_exact(example_kwds_GEOG):
    grid = grids.TileGrid(**example_kwds_GEOG)
    roi = Polygon.from_bounds(0, 0, 10, 10)  # 1 tile exactly
    tiles = list(grid.roi_to_tiles(roi))
    assert len(tiles) == 1
    assert tiles[0].bbox.area == roi.area


def test_tilegrid_point(example_kwds_GEOG):
    grid = grids.TileGrid(**example_kwds_GEOG)
    point = (-165.0, 30.00001)  # 1 tile exactly
    tile = grid.point_to_tile(point)
    assert isinstance(tile, grids.Tile)
    assert tile.horizontal == 1
    assert tile.vertical == 4

    point = (-165.0, 30.0)  # just on start of other tile
    tile = grid.point_to_tile(point)
    assert tile.vertical == 5


def test_tilegrid_bounds(example_kwds_GEOG):
    grid = grids.TileGrid(**example_kwds_GEOG)
    bounds = BoundingBox(-74, 41, -69, 44)
    tiles = list(grid.bounds_to_tiles(bounds))
    assert len(tiles) == 2


# Failure: indexing problems
def test_tilegrid_fail_1(example_grid_GEOG):
    with pytest.raises(IndexError, match=r'.*Unknown index type.*'):
        example_grid_GEOG[-1]


def test_tilegrid_fail_2(example_grid_GEOG):
    match = r'Only support indexing int/int for now'
    with pytest.raises(TypeError, match=match):
        example_grid_GEOG[([0, 1], [0, 1])]


def test_tilegrid_fail_oob(example_grid_GEOG):
    # Force out of bounds by adding/knowing grid limits
    h = example_grid_GEOG.limits[1][1] + 1
    v = example_grid_GEOG.limits[0][1] + 1
    match = r'.*Tile at index.*is outside.*limits.*'
    with pytest.raises(IndexError, match=match):
        example_grid_GEOG[v, h]


# =============================================================================
# load_grids
def test_load_grids():
    ex_grids = grids.load_grids()
    assert isinstance(ex_grids, dict)
    assert len(ex_grids) == 6
    assert 'LANDSAT_ARD_CU' in ex_grids
    assert 'LANDSAT_ARD_AL' in ex_grids
    assert 'LANDSAT_ARD_HI' in ex_grids


# =============================================================================
# Fixtures
@pytest.fixture
def example_kwds_GEOG(request):
    return {
        'name': 'GEOG',
        'crs': 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]',
        'ul': [-180.0, 80.0],
        'res': [0.00025, 0.00025],
        'size': [40000, 40000],
        'limits': [[0, 13], [0, 36]]
    }


@pytest.fixture
def example_grid_GEOG(example_kwds_GEOG):
    return grids.TileGrid(**example_kwds_GEOG)


@pytest.fixture
def example_kwds_AEA(request):
    return {
        'name': 'Landsat ARD - CONUS',
        'crs': 'PROJCS["CONUS_WGS84_Albers_Equal_Area_Conic",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1]]',
        'ul': [-2565585.0, 3314805],
        'res': [30, 30],
        'size': [5000, 5000],
        'limits': [[0, 21], [0, 32]]
    }


@pytest.fixture
def example_grid_AEA(example_kwds_AEA):
    return grids.TileGrid(**example_kwds_AEA)


@pytest.fixture(params=[
    pytest.lazy_fixture('example_kwds_GEOG'),
    pytest.lazy_fixture('example_kwds_AEA')
])
def example_kwds(request):
    return request.param


@pytest.fixture(params=[
    pytest.lazy_fixture('example_kwds_GEOG'),
    pytest.lazy_fixture('example_kwds_AEA')
])
def example_grids(request):
    return grids.TileGrid(**request.param)



