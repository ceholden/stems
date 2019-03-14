.. _gis_grids:

=================
Tile Grid Systems
=================


.. currentmodule:: stems.gis.grids


TileGrid
========

.. autosummary::
   
   TileGrid
   load_grids

Attributes
----------

.. autosummary::

   TileGrid.ul
   TileGrid.crs
   TileGrid.res
   TileGrid.size
   TileGrid.limits
   TileGrid.name
   TileGrid.rows
   TileGrid.cols
   TileGrid.nrow
   TileGrid.ncol
   TileGrid.transform

Geometry Tools
--------------

Methods for finding :py:class:`Tile`\s that intersect with your geospatial data.

.. autosummary::

   TileGrid.point_to_tile
   TileGrid.bounds_to_tiles
   TileGrid.roi_to_tiles

Export
------

.. autosummary::

    TileGrid.to_dict
    TileGrid.from_dict
    TileGrid.geojson


Dictionary Interface
--------------------

You can grab :py:class:`Tile` from the :py:class:`TileGrid` by
indexing the row and column (or vertical, horizontal) as you would a ``dict``.

.. autosummary::
   
   TileGrid.__getitem__
   TileGrid.__len__
   TileGrid.__iter__

Tile
====

.. autosummary::

    Tile.index
    Tile.crs
    Tile.bounds
    Tile.res
    Tile.size
    Tile.vertical
    Tile.horizontal
    Tile.transform
    Tile.bbox
    Tile.width
    Tile.height
    Tile.coords
    Tile.geojson
