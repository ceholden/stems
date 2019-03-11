"""
"""
import collections
import contextlib
import errno
import fnmatch
import functools
import importlib
import logging
import os
from pathlib import Path
import re
import shutil

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# STANDARD DATATYPE HELPERS
def to_number(string):
    """ Convert string to most appropriate number
    """
    try:
        n = int(string)
    except ValueError:
        n = float(string)
    return n


def list_like(obj):
    """ Return True if ``obj`` is list-like

    List-like includes lists, tuples, and numpy.ndarrays, but not
    other sequences like str or Mapping.

    Parameters
    ----------
    obj : object
        An object

    Returns
    -------
    bool
        True if ``obj`` is a list (sequence but not str)
    """
    return (hasattr(obj, '__iter__')
            and not isinstance(obj, collections.abc.Mapping)
            and not isinstance(obj, str))


def squeeze(l):
    """ Squeeze Sequences with 1 item into a scalar

    Parameters
    ----------
    l : Sequence
        List, tuple, etc

    Returns
    -------
    object
        Either original object if ``len(l) != 1`` else ``l[0]``
    """
    return l[0] if len(l) == 1 else l


def concat_lists(l):
    """ Concatenate all list-like items in ``l``

    Parameters
    ----------
    l : list or tuple
        Sequence (but not a str)

    Yields
    ------
    list
        Concatenated list items

    See Also
    --------
    toolz.concat
        Similar function, but concatenates all iterables
    flatten_lists
        Recursive version of this function
    """
    for ele in l:
        if list_like(ele):
            for i in ele:
                yield i
        else:
            yield ele


def flatten_lists(l):
    """ Flatten all list-like items in ``l``

    Parameters
    ----------
    l : list or tuple
        Sequence (but not a str)

    Yields
    ------
    list
        Flattened list items
    """
    for ele in l:
        if list_like(ele):
            for i in flatten_lists(ele):
                yield i
        else:
            yield ele


def update_nested(d, other):
    """ Update a potentially nested dict with another

    Parameters
    ----------
    d : dict
        Dictionary to update
    other : dict
        Other dict with replacement values

    Returns
    -------
    dict
        Updated dict
    """
    d_ = d.copy()
    for k, v in other.items():
        d_v = d.get(k, {})
        if (isinstance(v, collections.abc.Mapping) and
                isinstance(d_v, collections.abc.Mapping)):
            d_[k] = update_nested(d.get(k, {}) or {}, v)
        else:
            d_[k] = v
    return d_


class FrozenKeyDict(collections.abc.MutableMapping):
    """ A dict that doesn't allow new keys
    """

    def __init__(self, *args, **kwds):
        self._data = collections.OrderedDict(*args, **kwds)
        super(FrozenKeyDict, self).__init__()

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError('Cannot create new keys')
        self._data[key] = value

    def __delitem__(self, key):
        raise KeyError('Cannot delete keys')

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, repr(self._data))

    def copy(self):
        return FrozenKeyDict(**self)


# ============================================================================
# STANDARD LIBRARY HELPERS
# ============================================================================
def cached_property(prop):
    """ Cache a class property (e.g., that requires a lookup)
    """
    prop_name = f'_{prop.__name__}'

    @functools.wraps(prop)
    def wrapper(self):
        if not hasattr(self, prop_name):
            setattr(self, prop_name, prop(self))
        return getattr(self, prop_name)

    return property(wrapper)


def register_multi_singledispatch(func, types):
    """ Register multiple types for singledispatch

    Parameters
    ----------
    func : callable
        Function
    types : tuple
        Multiple types to register

    Returns
    -------
    func : callable
        Decorated function
    """
    if not hasattr(func, 'registry'):
        raise TypeError("Function must be dispatchable (missing "
                        "`func.registry` from wrapping with `singledispatch`)")

    def decorate(dispatch_func):
        for type_ in types:
            dispatch_func = func.register(type_, dispatch_func)
        return dispatch_func

    return decorate


# ============================================================================
# MODULE HELPERS
# ============================================================================
def find_subclasses(cls_):
    """ Find subclasses of an object

    Parameters
    ----------
    cls_ : class
        A Python class

    Returns
    -------
    set[class]
        Classes that inherit from ``cls_``
    """
    subcls = set()
    for sub in cls_.__subclasses__():
        subsub = find_subclasses(sub)
        subcls.add(sub)
        subcls.update(subsub)
    return subcls


def import_object(string):
    """ Import a Python module/class/function from its string

    Parameters
    ----------
    string : str
        A Python object path

    Returns
    -------
    object
        The imported object (class/module/function)
    """
    parts = string.split('.')
    module, item = '.'.join(parts[:-1]), parts[-1]
    try:
        mod = importlib.import_module(module)
        return getattr(mod, item)
    except (ImportError, AttributeError) as e:
        logger.exception(f'Could not import and find object "{string}"')
        raise


# ============================================================================
# NUMPY HELPERS
# ============================================================================
def np_promote_all_types(*dtypes):
    """ Return the largest NumPy datatype required to hold all types

    Parameters
    ----------
    dtypes : iterable
        NumPy datatypes to promote

    Returns
    -------
    np.dtype
        Smallest NumPy datatype required to store all input datatypes

    See Also
    --------
    np.promote_types
    """
    dtype = dtypes[0]
    if not all([dt == dtype for dt in dtypes[1:]]):
        logger.debug('Promoting memory allocation to largest datatype of '
                     'source bands')
        for _dtype in dtypes[1:]:
            dtype = np.promote_types(dtype, _dtype)
    return dtype


def dtype_info(arr):
    """Return integer or float dtype info

    Parameters
    ----------
    arr : np.ndarray
        Array

    Returns
    -------
    numpy.core.getlimits.finfo or numpy.core.getlimits.iinfo
        NumPy type information

    Raises
    ------
    TypeError
        Raised if ``arr`` is not a int or float type
    """
    if arr.dtype.kind == 'i':
        return np.iinfo(arr.dtype)
    elif arr.dtype.kind == 'f':
        return np.finfo(arr.dtype)
    else:
        raise TypeError('Only valid for NumPy int or float '
                        f'(got "{arr.dtype.kind}")')


# ============================================================================
# FILE HELPERS
# ============================================================================
def find(location, pattern, regex=False):
    """ Return a sorted list of files matching pattern

    Parameters
    ----------
    location : str or pathlib.Path
        Directory location to search
    pattern : str
        Search pattern for files
    regex : bool
        True if ``pattern`` is a regular expression

    Returns
    --------
    list
        List of file paths for files found

    """
    if not regex:
        pattern = fnmatch.translate(pattern)
    regex = re.compile(pattern)

    files = []
    for root, dirnames, filenames in os.walk(str(location)):
        for filename in filenames:
            if regex.search(filename):
                files.append(os.path.join(root, filename))

    return sorted(files)


def relative_to(one, two):
    """ Return the relative path of a file compared to another

    Parameters
    ----------
    one : str or Path
        File to return relative path for
    two : str or Path
        File ``one`` will be relative to

    Returns
    -------
    Path
        Relative path of ``one``
    """
    one, two = Path(one), Path(two).absolute()

    # Ensure could be a file (is_file or doesn't exist yet)
    assert one.is_file() or not one.exists()
    assert two.is_file() or not two.exists()

    root = os.path.abspath(os.sep)
    for parent in one.absolute().parents:
        if parent in two.parents:
            root = parent
            break

    fwd = one.absolute().relative_to(root)
    bwd = ('..', ) * (len(two.relative_to(root).parents) - 1)

    return Path('.').joinpath(*bwd).joinpath(fwd)


@contextlib.contextmanager
def renamed_upon_completion(destination, tmpdir=None,
                            prefix='', suffix='.tmp'):
    """ Help save/write to file and move upon completion

    Parameters
    ----------
    destination : str or Path
        The final intended location of the file
    tmpdir : str, optional
        By default, this function will yield a temporary filename
        in the same directory as ``destination``, but you may specify
        another location using ``tmpdir``.
    prefix : str, optional
        Characters to prefix the temporary file with
    suffix : str, optional
        Characters to add at the end of the temporary filename

    Yields
    ------
    str
        A temporary filename to use during writing/saving/etc
    """
    destination = Path(destination)
    assert not destination.is_dir()

    if tmpdir is None:
        tmpdir = destination.parent
    else:
        tmpdir = Path(tmpdir)
        assert tmpdir.is_dir()

    tmpfile = tmpdir.joinpath(f'{prefix}{destination.name}{suffix}')

    # Yield tmpfile name for user to use for saving/etc
    logger.debug(f'Providing "{tmpfile}" as write location')
    yield str(tmpfile)

    # We're back -- rename/move the tmpfile to ``destination``
    logger.debug(f'Renaming/moving file {tmpfile}->{destination}')
    # `shutil.move` supports move, or copy if on different device/partition
    shutil.move(str(tmpfile), str(destination))
