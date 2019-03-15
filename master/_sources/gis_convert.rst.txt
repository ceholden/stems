.. _gis_convert:

GIS Conversion Tools
====================

.. currentmodule:: stems.gis.convert

The :py:mod:`stems.gis.convert` module contains dispatch functions to
convert representations of fundamental GIS data (projections, transforms,
bounding boxes, and polygons) to the standard types used internally to this
package.

These "standard types" for GIS data variables are the following:

+-----------------------------+-----------------------------------------+
| **GIS**                     | **Object Type**                         |
+-----------------------------+-----------------------------------------+
| Affine transform            | :py:class:`affine.Affine`               |
+-----------------------------+-----------------------------------------+
| Coordinate Reference System | :py:class:`rasterio.crs.CRS`            |
+-----------------------------+-----------------------------------------+
| Bounding Box (extent)       | :py:class:`rasterio.coords.BoundingBox` |
+-----------------------------+-----------------------------------------+
| Bounding Box (polygon)      | :py:class:`shapely.geometry.Polygon`    |
+-----------------------------+-----------------------------------------+

The conversion functions:

.. autosummary::

   to_transform
   to_crs
   to_bounds
   to_bbox
