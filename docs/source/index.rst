.. _index:

Spatio-temporal Tools for Earth Monitoring Science (STEMS)
==========================================================

STEMS is a library to help with analysis of geospatial time series data. It
tries to integrate geospatial libraries (rasterio_, fiona_, and
shapely_) with the xarray library to make it easier to use netCDF4_
files for large data processing. This library also contains functions and tools
to facilitate processing these data with dask_ and distributed_. 


.. toctree::
   :maxdepth: 1
   :caption: Getting Started 

   install
   history
   faq

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   gis
   gis_grids
   parallel


.. toctree::
   :caption: Help and Reference

   dev
   api

.. _rasterio: https://rasterio.readthedocs.io
.. _fiona: https://fiona.readthedocs.io/
.. _shapely: https://shapely.readthedocs.io/
.. _xarray: http://xarray.pydata.org
.. _dask: http://docs.dask.org/en/latest/
.. _distributed: http://distributed.dask.org/en/latest/
.. _netCDF4: http://unidata.github.io/netcdf4-python/
