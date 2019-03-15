""" IO utilities
"""
import datetime as dt
import glob
from pathlib import Path

import numpy as np


def parse_paths(paths):
    """ Return a list of path(s)

    Parameters
    ----------
    paths : str or sequence
        Either a string glob in the form "path/to/my/files/*.nc" or an explicit
        list of files to open.

    Returns
    -------
    list[Path]
        Paths determined from ``paths``
    """
    if isinstance(paths, str):
        # Filename or glob, either way turn into list
        paths = glob.glob(paths)
    elif isinstance(paths, Path):
        if '*' in str(paths):
            paths = glob.glob(str(paths))
        else:
            paths = [paths]

    return list([Path(p) for p in paths])


def parse_filename_attrs(paths, index=None, sep=None):
    """ Parse filenames into an array of dates

    Parameters
    ----------
    paths : str or Sequence
        Glob style search pattern, or a list of filenames
    index : slice or int
        Either a ``slice`` used to index on the stem of the filenames
        in `paths`, or together with `sep` an `int` used to index on
        the filename stem split by `sep` (e.g., ``filename.split(sep)[index]``)
    sep : str, optional
        String field seperator

    Returns
    -------
    list[str]
        Attribute extracted from each filename path
    """
    if index is None and sep is None:
        raise TypeError("Must provide either `index`, or `index` and `sep`")

    paths = parse_paths(paths)
    if not paths:
        raise IOError('No files to open')

    attrs = []
    for path in paths:
        # Extract filename attributes
        if sep:
            string = path.stem.split(sep)[index]
            if isinstance(index, slice):
                # Asked for multiple components - rejoin into single str
                string = ''.join(string)
        else:
            string = path.stem[index]
        assert isinstance(string, str)
        attrs.append(string)

    return attrs


def parse_filename_dates(paths, index=None, sep=None, date_format='%Y%m%d'):
    """ Parse filenames into an array of dates

    Parameters
    ----------
    paths : str or Sequence
        Glob style search pattern, or a list of filenames
    index : slice or int
        Either a ``slice`` used to index on the stem of the filenames
        in `paths`, or together with `sep` an `int` used to index on
        the filename stem split by `sep` (e.g., ``filename.split(sep)[index]``)
    sep : str, optional
        String field seperator
    date_format : str
        Date format used by :py:func:`datetime.datetime.strptime`

    Returns
    -------
    np.ndarray
        Array of datetime64

    See Also
    --------
    """
    attrs = parse_filename_attrs(paths, index=index, sep=sep)
    times = [np.datetime64(dt.datetime.strptime(attr, date_format))
             for attr in attrs]
    return np.array(times)
