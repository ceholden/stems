.. _gis_projections:
   
   
Projection Format Conversion Tools
==================================

.. currentmodule:: stems.gis.projections

The :py:mod:`stems.gis.projections` module contains tools to help convert
between representations of projections, including between the Open Geospatial
Consortium (OGC) and the Climate Forecast (CF) convention systems.


There are also functions that help retrieve useful information from a Rasterio
:py:class:`rasterio.crs.CRS`, including the "long name" (e.g.,
``"WGS 84 / Pseudo-Mercator"``)

.. autosummary::

   crs_longname


Climate Forecast (CF) Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following functions are useful for retrieving CF
parameters from :py:class:`rasterio.crs.CRS` representations of coordinate
reference systems:

.. autosummary::

   cf_crs_name
   cf_crs_attrs
   cf_proj_params
   cf_ellps_params
   cf_xy_coord_names


These functions rely on a few data variables that contain the translations
between the OGC and CF systems:

.. autosummary::

   CF_CRS_NAMES
   CF_PROJECTION_NAMES
   CF_ELLIPSOID_DEFS
   CF_PROJECTION_DEFS


EPSG Codes
~~~~~~~~~~

You can also try to retrieve the European Petroleum Survey Group (EPSG) code
from a :py:class:`rasterio.crs.CRS`:

.. autosummary::

   epsg_code
