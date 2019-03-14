.. _parallel:


Parallel Computing Helpers
==========================

The stems project has tries to help facilitate running geospatial time series
analysis in parallel using dask_ and `dask.distributed`_.


Distributed Cluster Setup
-------------------------

This package contains a few functions that help create and shutdown
:py:class:`distributed.LocalCluster`

.. autofunction:: stems.executor.setup_executor
.. autofunction:: stems.executor.executor_info

Command Line Applications
-------------------------

When building command line applications, you can use some of the Click_-based
callbacks and program arguments or options.

.. autodata:: stems.cli.options.opt_nprocs
.. autodata:: stems.cli.options.opt_nprocs
.. autodata:: stems.cli.options.opt_nprocs
.. autofunction:: stems.cli.options.cb_executor
.. autofunction:: stems.cli.options.close_scheduler


Block Mapping of Functions
--------------------------

The :py:mod:`stems.parallel` module contains functions that help you run
functions on chunks of data.

The functions, :py:func:`stems.parallel.iter_chunks` and
:py:func:`stems.parallel.iter_noncore_chunks` assist you in iterating across
blocks of data by yielding array dimension slices.

.. autofunction:: stems.parallel.iter_chunks
.. autofunction:: stems.parallel.iter_noncore_chunks

You can nest these to run an inner function on each pixel within a block on a
worker (e.g., ``chunksize=1`` for each dimension), and process many blocks in
parallel using the slices (e.g., ``chunksize=100`` in x/y) you calculate using
the same :py:func:`stems.parallel.iter_chunks` function.

Sometimes you want to run a function across some arbitrary dimensions and
collect the results in a single list. One common example is running the CCDC
algorithm on pixels from an image (``dims=('band', 'time', 'y', 'x', )``) or 
from a list of samples (``dims=('band', 'time', 'sample_id', )``) and
collecting all the segments estimated into a single dimension array
(``dims=('segment_id', )``).

The function :py:func:`stems.parallel.map_collect_1d` is designed to help with
this task:

.. autofunction:: stems.parallel.map_collect_1d


.. _dask: http://docs.dask.org/en/latest/
.. _dask.distributed: http://distributed.dask.org/en/latest/
.. _Click: https://click.palletsprojects.com/en/7.x/
