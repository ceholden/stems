.. _xarray:

==================
XArray Integration
==================

.. currentmodule:: stems.xarray_accessor


The stems project tries to help better integrate XArray and NetCDF4 data
with GDAL/rasterio, especially with respect to handling geospatial attributes
and data. While you may import various functions from this module to use
within your own code, there is also a convenient way of accessing many of these
integrations that uses `xarray`_ "accessors".

For more information on XArray accessors, please refer to their documentation on
`extending xarray`_.


Initialize The Accessor
=======================

In order to use the ``.stems`` xarray accessor, you need to import the module
from stems:


.. ipython:: python

   import stems.xarray_accessor


Examples
========

Attributes
----------

In the following example, we use some functions from our test suite that
generate example data:


.. ipython:: python

   from stems.tests.build_data import create_test_dataset
   test_ds = create_test_dataset()
   test_ds


We can inspect the "crs" coordinate, which are Climate Forecast (CF) formatted
`Grid Mapping`_ variables. These conventions govern how various projection and
ellipsoid parameters should be represented. The conventions are different than
what the Open Geospatial Consortium (OSGEO/GDAL/etc) use, but can be
converted.


.. ipython:: python

   test_ds['crs']


The ``stems`` XArray accessor makes accessing geospatial metadata much easier:


.. ipython:: python

   print('CRS:\n', test_ds.stems.crs)
   print('Affine transform:\n', test_ds.stems.transform)
   print('Bounds:\n', test_ds.stems.bounds)
   print('Bounding box:\n', test_ds.stems.bbox)


Georeferencing
--------------

We can also assign georeferencing information for new data that we might
create:


.. ipython:: python

   # Calculate (and drop the CRS to show example)
   ratio = (test_ds['nir'] / test_ds['red']).drop('crs')
   ratio
   
Now we can use the ``stems`` accessor to georeference the data:


.. ipython:: python

   ratio_ = ratio.stems.georeference(test_ds.stems.crs,
                                     test_ds.stems.transform)
   ratio_


For comparison, you could also directly use components from stems to do the
same thing:

.. ipython:: python

   from stems.gis import convert, conventions, coords
   # Get the CRS/transform from source data "test_ds"
   grid_mapping = conventions.get_grid_mapping(test_ds)
   crs_ = convert.to_crs(grid_mapping.attrs['spatial_ref'])
   transform_ = coords.coords_to_transform(ratio['y'], ratio['x'])
   # Apply
   ratio_ = conventions.georeference(ratio, crs_, transform_)
   ratio_

As you can see, it's a lot faster and more straightforward to use the accessors!


.. _extending xarray: http://xarray.pydata.org/en/stable/internals.html#extending-xarray
.. _Grid Mapping: http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/apf.html
